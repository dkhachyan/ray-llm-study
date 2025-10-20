# Using llm.py with MODELS Environment Variable

## Overview
The `llm.py` file dynamically loads models from the `MODELS` environment variable and provides OpenAI-compatible API endpoints. The model name is specified in the OpenAI request.

## Setting Environment Variable

### Linux/MacOS
```bash
export MODELS="Qwen/Qwen2.5-7B-Instruct,unsloth/Meta-Llama-3.1-8B-Instruct"
```

### Windows
```cmd
set MODELS=Qwen/Qwen2.5-7B-Instruct,unsloth/Meta-Llama-3.1-8B-Instruct
```

### In Python
```python
import os
os.environ['MODELS'] = 'Qwen/Qwen2.5-7B-Instruct,unsloth/Meta-Llama-3.1-8B-Instruct'
```

## Features

### Dynamic Model Loading
- Models are loaded from the `MODELS` environment variable
- Multiple models can be specified as a comma-separated list
- Model name is specified in the OpenAI request

### OpenAI-Compatible API Endpoints
- `/health` - Shows health status and loaded models
- `/v1/chat/completions` - Chat completion endpoint (POST)
- `/v1/models` - List available models (GET)

## Example Usage

### With OpenAI-Compatible API
```python
import openai

# Configure client to use local endpoint
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local deployment
)

# List available models
models = client.models.list()
print([model.id for model in models.data])

# Chat completion with specific model
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",  # Model name from MODELS env var
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Direct HTTP Requests
```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

## Error Handling
- Raises error if `MODELS` environment variable is not set
- Raises error if no valid models are found in the environment variable
- Proper error logging for model initialization failures
- OpenAI-compatible error responses for invalid model requests