# 🤖 AWS Bedrock Text Generation App

A comprehensive Streamlit application for learning and mastering AWS Bedrock foundation models. Perfect for developers preparing for Bedrock interviews or building production AI applications.

![Bedrock App Demo](https://img.shields.io/badge/AWS-Bedrock-orange) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

## 🌟 Features

### Core Capabilities
- **Multiple Foundation Models**: Claude 3 (Sonnet/Haiku), Llama 2 70B, Titan Text Express
- **Streaming & Non-Streaming**: Real-time response streaming with toggle option
- **Cost Tracking**: Real-time pricing calculation and usage analytics
- **Model Comparison**: Side-by-side comparison of capabilities and pricing
- **Interactive Parameters**: Adjustable temperature, max tokens, and model settings

### Learning Features
- **Preset Prompts**: Quick-start templates for different use cases
- **Usage Statistics**: Track tokens, costs, and response times
- **Model Insights**: Understand when to use each foundation model
- **Best Practices**: Built-in tips for cost optimization and API mastery

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- AWS Account with Bedrock access
- AWS CLI configured

### Installation

1. **Clone or download the application files**
   ```bash
   # Save the Python code as bedrock_app.py
   # Save the connection test as connection_test.py
   ```

2. **Install dependencies**
   ```bash
   pip install boto3 streamlit pandas awscli
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```
   Enter your:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region: `us-east-1` (required for Bedrock)
   - Output format: `json`

4. **Enable Bedrock model access**
   - Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock)
   - Navigate to **Model access** in the sidebar
   - Request access for:
     - Claude 3 Sonnet
     - Claude 3 Haiku
     - Llama 2 70B Chat
     - Titan Text Express
   - Wait for approval (typically 5-30 minutes)

5. **Test your connection**
   ```bash
   python connection_test.py
   ```

6. **Launch the application**
   ```bash
   streamlit run bedrock_app.py
   ```

## 🎯 Usage Guide

### Basic Text Generation
1. Select a foundation model from the sidebar
2. Adjust parameters (temperature, max tokens)
3. Enter your prompt or use preset prompts
4. Click "Generate Text" to see results

### Streaming Mode
- Toggle "Enable Streaming" to see real-time response generation
- Compare streaming vs non-streaming user experience
- Ideal for long-form content generation

### Cost Analysis
- Monitor real-time cost calculations
- View usage statistics in the right panel
- Compare costs across different models
- Track token consumption patterns

### Model Comparison
Built-in comparison table showing:
- Input/output pricing per 1K tokens
- Maximum token limits
- Best use cases for each model
- Performance characteristics

## 📊 Supported Models

| Model | Input Cost/1K | Output Cost/1K | Max Tokens | Best For |
|-------|---------------|----------------|------------|----------|
| Claude 3 Sonnet | $0.003 | $0.015 | 4,096 | Complex reasoning, analysis |
| Claude 3 Haiku | $0.00025 | $0.00125 | 4,096 | Fast, cost-effective tasks |
| Llama 2 70B | $0.00195 | $0.00256 | 2,048 | Open-source, general purpose |
| Titan Text Express | $0.0008 | $0.0016 | 8,192 | AWS native, basic tasks |

## 🛠️ Technical Architecture

### Core Components
```python
# Bedrock Runtime Client
client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Standard API Call
response = client.invoke_model(
    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
    body=json.dumps(body)
)

# Streaming API Call
response = client.invoke_model_with_response_stream(
    modelId=model_id,
    body=json.dumps(body)
)
```

### Request Formats
Each model requires different request body formats:

**Claude Models:**
```json
{
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [{"role": "user", "content": "Your prompt"}],
    "max_tokens": 1000,
    "temperature": 0.7
}
```

**Llama Models:**
```json
{
    "prompt": "<s>[INST] Your prompt [/INST]",
    "max_gen_len": 1000,
    "temperature": 0.7,
    "top_p": 0.9
}
```

**Titan Models:**
```json
{
    "inputText": "Your prompt",
    "textGenerationConfig": {
        "maxTokenCount": 1000,
        "temperature": 0.7,
        "topP": 1
    }
}
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### IAM Permissions Required
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

## 🐛 Troubleshooting

### Common Issues

**1. NoCredentialsError**
```bash
# Solution: Configure AWS credentials
aws configure
```

**2. AccessDeniedException**
- Go to AWS Bedrock Console
- Request model access under "Model access"
- Wait for approval

**3. ValidationException**
- Check if model ID is correct
- Verify model access is approved
- Ensure you're using supported region

**4. Region Not Supported**
Bedrock is available in limited regions:
- us-east-1 (N. Virginia) ✅ Recommended
- us-west-2 (Oregon)
- eu-west-1 (Ireland)
- ap-southeast-1 (Singapore)
- ap-northeast-1 (Tokyo)

### Connection Test
Run the connection test script to diagnose issues:
```bash
python connection_test.py
```

This will check:
- ✅ AWS credentials configuration
- ✅ Bedrock service connectivity
- ✅ Model access permissions
- ✅ Available foundation models

## 📚 Learning Objectives

This application helps you master:

### Bedrock Runtime API
- `invoke_model()` for standard requests
- `invoke_model_with_response_stream()` for streaming
- Request/response format handling
- Error handling and retry logic

### Foundation Model Selection
- Understanding model capabilities
- Cost vs. performance trade-offs
- Use case optimization
- Token management strategies

### Production Considerations
- Cost monitoring and optimization
- Response time analysis
- Parameter tuning effects
- Streaming vs non-streaming UX

## 🎓 Interview Preparation

Perfect preparation for questions like:
- "How do you call different Bedrock models in Python?"
- "Explain the difference between streaming and non-streaming responses"
- "How do you optimize costs when using Bedrock?"
- "When would you choose Claude vs Llama vs Titan?"
- "Walk me through implementing a text generation system"

## 📈 Usage Analytics

The app tracks:
- **Total Cost**: Cumulative API call costs
- **Token Usage**: Input and output token consumption
- **Response Times**: Average API response latency
- **Request Count**: Total number of API calls
- **Model Distribution**: Usage patterns across models

## 🔒 Security Best Practices

- ✅ Never hardcode AWS credentials
- ✅ Use IAM roles in production
- ✅ Implement proper error handling
- ✅ Monitor usage and costs
- ✅ Use least privilege access policies

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new foundation models as they become available
- Enhance the UI/UX
- Add more preset prompts
- Implement additional analytics

## 📄 License

This project is provided for educational purposes. AWS usage charges apply for Bedrock API calls.

## 🔗 Useful Links

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Boto3 Bedrock Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
- [Bedrock Model Catalog](https://docs.aws.amazon.com/bedrock/latest/userguide/model-catalog.html)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

## 📞 Support

For AWS Bedrock issues:
- Check the [troubleshooting section](#troubleshooting)
- Run the connection test script
- Refer to AWS Bedrock documentation
- Check AWS Service Health Dashboard

---

**Built for learning AWS Bedrock** 🚀 **Ready for production** 💼 **Interview preparation** 🎯