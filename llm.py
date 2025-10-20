from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import logging
from typing import Optional, Dict, Any
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI-compatible schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ModelConfig:
    """Configuration class for model settings"""
    DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512
    }
    
    @staticmethod
    def get_models_from_env() -> list:
        """Get list of models from MODELS environment variable"""
        models_env = os.getenv("MODELS", "")
        if not models_env:
            raise ValueError("MODELS environment variable is not set")
        
        models = [model.strip() for model in models_env.split(",") if model.strip()]
        if not models:
            raise ValueError("No valid models found in MODELS environment variable")
        
        logger.info(f"Loaded models from environment: {models}")
        return models

class ModelManager:
    """Manages vLLM model lifecycle and operations"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.sampling_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with sleep mode enabled"""
        try:
            logger.info(f"Initializing model: {self.model_name}")
            self.model = LLM(self.model_name, enable_sleep_mode=True)
            self.sampling_params = SamplingParams(**ModelConfig.DEFAULT_SAMPLING_PARAMS)
            self.model.reset_prefix_cache()
            self.model.sleep(level=1)
            logger.info(f"Model {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {str(e)}")
            raise
    
    
    def generate_chat_completion(self, messages: list[ChatMessage], custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate chat completion with proper error handling and timing"""
        # Convert messages to prompt format
        prompt = self._format_messages_to_prompt(messages)
        
        start_time = time.time()
        
        try:
            # Wake up the model
            self.model.wake_up()
            
            # Use custom sampling parameters if provided
            sampling_params = self.sampling_params
            if custom_params:
                sampling_params = SamplingParams(**custom_params)
            
            # Generate text
            outputs = self.model.generate(prompt, sampling_params)
            
            # Get the generated text
            generated_text = outputs[0].outputs[0].text
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            
            # Put model back to sleep
            self.model.reset_prefix_cache()
            self.model.sleep(level=1)
            
            processing_time = time.time() - start_time
            logger.info(f"Generated {tokens_generated} tokens in {processing_time:.2f}s for model {self.model_name}")
            
            return generated_text, tokens_generated, processing_time
            
        except Exception as e:
            logger.error(f"Error during chat completion for model {self.model_name}: {str(e)}")
            # Ensure model is put back to sleep even if generation fails
            try:
                self.model.reset_prefix_cache()
                self.model.sleep(level=1)
            except Exception as sleep_error:
                logger.error(f"Failed to put model to sleep: {str(sleep_error)}")
            raise
    
    def _format_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert chat messages to a prompt string"""
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n\n"
        
        # Add the final assistant prompt
        prompt += "Assistant:"
        return prompt

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs. Includes OpenAI-compatible endpoints.",
    version="1.0.0"
)

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1}
)
@serve.ingress(api)
class LLMServingAPI:
    """Main API class for serving multiple LLM models"""
    
    def __init__(self):
        # Get models from environment variable
        self.models = ModelConfig.get_models_from_env()
        
        # Create model managers for each model
        self.model_managers = {}
        for model_name in self.models:
            try:
                self.model_managers[model_name] = ModelManager(model_name)
                logger.info(f"Successfully initialized model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                raise
        
        logger.info(f"LLM Serving API initialized with models: {list(self.model_managers.keys())}")
    
    @api.get("/health")
    async def health_check(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy", "models": ",".join(self.models)}
    
    # OpenAI-compatible endpoints
    @api.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """OpenAI-compatible chat completion endpoint"""
        try:
            # Check if requested model is available
            if request.model not in self.model_managers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model} not found. Available models: {list(self.model_managers.keys())}"
                )
            
            # Prepare custom sampling parameters
            custom_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
            
            # Generate chat completion
            generated_text, tokens_generated, processing_time = self.model_managers[request.model].generate_chat_completion(
                request.messages, custom_params
            )
            
            # Create response in OpenAI format
            response_message = ChatMessage(role="assistant", content=generated_text)
            choice = ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason="stop"
            )
            
            usage = ChatCompletionUsage(
                prompt_tokens=0,  # Note: vLLM doesn't provide prompt token count easily
                completion_tokens=tokens_generated,
                total_tokens=tokens_generated
            )
            
            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Chat completion failed for model {request.model}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
    
    @api.get("/v1/models")
    async def list_openai_models(self) -> Dict[str, Any]:
        """OpenAI-compatible models list endpoint"""
        models_list = []
        for model_name in self.models:
            models_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm"
            })
        
        return {
            "object": "list",
            "data": models_list
        }


# Ray Serve deployment
app = LLMServingAPI.bind()