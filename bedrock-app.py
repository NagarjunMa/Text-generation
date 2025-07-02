import boto3
import json
import streamlit as st
import time
from datetime import datetime
import pandas as pd

# Configure AWS Bedrock client
@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock Runtime client"""
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1'  # Bedrock is available in specific regions
    )

# Model configurations with pricing info
MODELS = {
    'Claude 3 Sonnet': {
        'id': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'input_price_per_1k': 0.003,
        'output_price_per_1k': 0.015,
        'max_tokens': 4096,
        'description': 'Best for reasoning, analysis, creative writing'
    },
    'Claude 3 Haiku': {
        'id': 'anthropic.claude-3-haiku-20240307-v1:0',
        'input_price_per_1k': 0.00025,
        'output_price_per_1k': 0.00125,
        'max_tokens': 4096,
        'description': 'Fastest and most cost-effective'
    },
    'Llama 2 70B': {
        'id': 'meta.llama2-70b-chat-v1',
        'input_price_per_1k': 0.00195,
        'output_price_per_1k': 0.00256,
        'max_tokens': 2048,
        'description': 'Open-source, good for general tasks'
    },
    'Titan Text G1 - Express': {
        'id': 'amazon.titan-text-express-v1',
        'input_price_per_1k': 0.0008,
        'output_price_per_1k': 0.0016,
        'max_tokens': 8192,
        'description': 'AWS native, cost-effective for basic tasks'
    }
}

def invoke_bedrock_model(client, model_id, prompt, max_tokens=1000, temperature=0.7):
    """
    Invoke Bedrock model - handles different model formats
    """
    try:
        if 'anthropic.claude' in model_id:
            # Claude format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif 'meta.llama2' in model_id:
            # Llama format
            body = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
        elif 'amazon.titan' in model_id:
            # Titan format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 1,
                    "stopSequences": []
                }
            }
        
        # Make the API call
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        # Parse response based on model
        response_body = json.loads(response['body'].read())
        
        if 'anthropic.claude' in model_id:
            text = response_body['content'][0]['text']
            input_tokens = response_body['usage']['input_tokens']
            output_tokens = response_body['usage']['output_tokens']
        elif 'meta.llama2' in model_id:
            text = response_body['generation']
            input_tokens = response_body.get('prompt_token_count', 0)
            output_tokens = response_body.get('generation_token_count', 0)
        elif 'amazon.titan' in model_id:
            text = response_body['results'][0]['outputText']
            input_tokens = response_body['inputTextTokenCount']
            output_tokens = response_body['results'][0]['tokenCount']
        
        return {
            'text': text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'success': True
        }
        
    except Exception as e:
        return {
            'text': f"Error: {str(e)}",
            'input_tokens': 0,
            'output_tokens': 0,
            'success': False,
            'error': str(e)
        }

def invoke_bedrock_streaming(client, model_id, prompt, max_tokens=1000, temperature=0.7):
    """
    Invoke Bedrock model with streaming response
    """
    try:
        if 'anthropic.claude' in model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif 'meta.llama2' in model_id:
            body = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
        elif 'amazon.titan' in model_id:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 1
                }
            }
        
        # Streaming API call
        response = client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(body),
            contentType='application/json'
        )
        
        # Process streaming response
        full_text = ""
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if 'anthropic.claude' in model_id:
                if chunk['type'] == 'content_block_delta':
                    full_text += chunk['delta']['text']
            elif 'meta.llama2' in model_id:
                full_text += chunk.get('generation', '')
            elif 'amazon.titan' in model_id:
                full_text += chunk.get('outputText', '')
                
            yield full_text
            
    except Exception as e:
        yield f"Streaming Error: {str(e)}"

def calculate_cost(input_tokens, output_tokens, model_config):
    """Calculate the cost of the API call"""
    input_cost = (input_tokens / 1000) * model_config['input_price_per_1k']
    output_cost = (output_tokens / 1000) * model_config['output_price_per_1k']
    return input_cost + output_cost

def main():
    st.set_page_config(
        page_title="AWS Bedrock Text Generation",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AWS Bedrock Text Generation App")
    st.markdown("Master Bedrock Runtime API with multiple foundation models")
    
    # Initialize session state
    if 'usage_history' not in st.session_state:
        st.session_state.usage_history = []
    
    # Sidebar - Model Configuration
    st.sidebar.header("🔧 Model Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Choose Foundation Model:",
        list(MODELS.keys()),
        help="Different models have different strengths and pricing"
    )
    
    model_config = MODELS[selected_model]
    
    # Display model info
    st.sidebar.info(f"""
    **{selected_model}**
    
    {model_config['description']}
    
    **Pricing:**
    • Input: ${model_config['input_price_per_1k']:.4f}/1K tokens
    • Output: ${model_config['output_price_per_1k']:.4f}/1K tokens
    
    **Max Tokens:** {model_config['max_tokens']:,}
    """)
    
    # Model parameters
    max_tokens = st.sidebar.slider(
        "Max Tokens:",
        min_value=100,
        max_value=model_config['max_tokens'],
        value=min(1000, model_config['max_tokens']),
        step=100
    )
    
    temperature = st.sidebar.slider(
        "Temperature (Creativity):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative/random"
    )
    
    streaming_mode = st.sidebar.checkbox(
        "Enable Streaming",
        value=True,
        help="Stream response in real-time vs wait for complete response"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Text Generation")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="Ask me anything or give me a creative writing task...",
            help="Try different types of prompts: questions, creative writing, analysis, etc."
        )
        
        # Preset prompts
        st.subheader("🎯 Quick Prompts")
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("📝 Creative Story"):
                prompt = "Write a short science fiction story about an AI that discovers emotions for the first time."
            if st.button("🧠 Explain Concept"):
                prompt = "Explain quantum computing in simple terms that a 12-year-old could understand."
        
        with preset_col2:
            if st.button("💼 Business Email"):
                prompt = "Write a professional email declining a meeting request due to scheduling conflicts."
            if st.button("🔍 Data Analysis"):
                prompt = "Analyze the pros and cons of remote work vs office work for software developers."
        
        # Generate button
        if st.button("🚀 Generate Text", type="primary", disabled=not prompt):
            if prompt:
                client = get_bedrock_client()
                
                # Record start time
                start_time = time.time()
                
                with st.container():
                    st.subheader("📤 Generated Response")
                    
                    if streaming_mode:
                        # Streaming mode
                        response_placeholder = st.empty()
                        
                        try:
                            for partial_response in invoke_bedrock_streaming(
                                client, model_config['id'], prompt, max_tokens, temperature
                            ):
                                response_placeholder.markdown(partial_response)
                                time.sleep(0.05)  # Small delay for better UX
                            
                            final_response = partial_response
                            
                        except Exception as e:
                            st.error(f"Streaming error: {str(e)}")
                            final_response = ""
                    
                    else:
                        # Non-streaming mode
                        with st.spinner("Generating response..."):
                            result = invoke_bedrock_model(
                                client, model_config['id'], prompt, max_tokens, temperature
                            )
                        
                        if result['success']:
                            st.markdown(result['text'])
                            final_response = result['text']
                            
                            # Calculate metrics
                            end_time = time.time()
                            response_time = end_time - start_time
                            cost = calculate_cost(
                                result['input_tokens'], 
                                result['output_tokens'], 
                                model_config
                            )
                            
                            # Store usage data
                            usage_data = {
                                'timestamp': datetime.now(),
                                'model': selected_model,
                                'prompt_length': len(prompt),
                                'response_length': len(result['text']),
                                'input_tokens': result['input_tokens'],
                                'output_tokens': result['output_tokens'],
                                'cost': cost,
                                'response_time': response_time
                            }
                            st.session_state.usage_history.append(usage_data)
                            
                        else:
                            st.error(f"Generation failed: {result['error']}")
    
    with col2:
        st.header("📊 Model Comparison")
        
        # Model comparison table
        comparison_data = []
        for name, config in MODELS.items():
            comparison_data.append({
                'Model': name,
                'Input $/1K': f"${config['input_price_per_1k']:.4f}",
                'Output $/1K': f"${config['output_price_per_1k']:.4f}",
                'Max Tokens': f"{config['max_tokens']:,}",
                'Best For': config['description'][:30] + "..."
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Usage statistics
        if st.session_state.usage_history:
            st.header("📈 Usage Statistics")
            
            total_cost = sum(item['cost'] for item in st.session_state.usage_history)
            total_tokens = sum(
                item['input_tokens'] + item['output_tokens'] 
                for item in st.session_state.usage_history
            )
            avg_response_time = sum(
                item.get('response_time', 0) for item in st.session_state.usage_history
            ) / len(st.session_state.usage_history)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Cost", f"${total_cost:.4f}")
                st.metric("Total Tokens", f"{total_tokens:,}")
            with col_b:
                st.metric("Requests", len(st.session_state.usage_history))
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.usage_history = []
                st.rerun()
    
    # Footer with learning tips
    st.markdown("---")
    st.subheader("🎓 Learning Tips")
    
    tip_col1, tip_col2 = st.columns(2)
    with tip_col1:
        st.info("""
        **API Mastery:**
        • `invoke_model`: Single response
        • `invoke_model_with_response_stream`: Real-time streaming
        • Different models need different request formats
        """)
    
    with tip_col2:
        st.success("""
        **Cost Optimization:**
        • Choose the right model for your task
        • Use Haiku for simple tasks (cheapest)
        • Monitor token usage closely
        • Consider caching for repeated requests
        """)

if __name__ == "__main__":
    main()