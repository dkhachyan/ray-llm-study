from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import logging
from typing import Optional, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request and Response schemas
class GenerationRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512

class GenerationResponse(BaseModel):
    text: str
    model: str
    processing_time: float
    tokens_generated: int

class ModelConfig:
    """Configuration class for model settings"""
    QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLAMA_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
    DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512
    }

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
    
    def generate_text(self, prompt: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate text with proper error handling and timing"""
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
            logger.error(f"Error during text generation for model {self.model_name}: {str(e)}")
            # Ensure model is put back to sleep even if generation fails
            try:
                self.model.reset_prefix_cache()
                self.model.sleep(level=1)
            except Exception as sleep_error:
                logger.error(f"Failed to put model to sleep: {str(sleep_error)}")
            raise

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs.",
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
        self.qwen_manager = ModelManager(ModelConfig.QWEN_MODEL)
        self.llama_manager = ModelManager(ModelConfig.LLAMA_MODEL)
        logger.info("LLM Serving API initialized with both models")
    
    @api.post("/qwen", response_model=GenerationResponse)
    async def generate_qwen(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Qwen model"""
        try:
            # Prepare custom sampling parameters
            custom_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
            
            # Generate text
            generated_text, tokens_generated, processing_time = self.qwen_manager.generate_text(
                request.prompt, custom_params
            )
            
            return GenerationResponse(
                text=generated_text,
                model="Qwen2.5-7B-Instruct",
                processing_time=processing_time,
                tokens_generated=tokens_generated
            )
            
        except Exception as e:
            logger.error(f"Qwen generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @api.post("/llama", response_model=GenerationResponse)
    async def generate_llama(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Llama model"""
        try:
            # Prepare custom sampling parameters
            custom_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
            
            # Generate text
            generated_text, tokens_generated, processing_time = self.llama_manager.generate_text(
                request.prompt, custom_params
            )
            
            return GenerationResponse(
                text=generated_text,
                model="Llama-3.1-8B-Instruct",
                processing_time=processing_time,
                tokens_generated=tokens_generated
            )
            
        except Exception as e:
            logger.error(f"Llama generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    @api.get("/health")
    async def health_check(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy", "models": "qwen,llama"}
    
    @api.get("/models")
    async def list_models(self) -> Dict[str, list]:
        """List available models"""
        return {
            "available_models": [
                {"name": "Qwen2.5-7B-Instruct", "endpoint": "/qwen"},
                {"name": "Llama-3.1-8B-Instruct", "endpoint": "/llama"}
            ]
        }

# Ray Serve deployment
app = LLMServingAPI.bind()