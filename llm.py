import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "qwen-0.5b",
        "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        }
    },
    # Pass the desired accelerator type (e.g. A10G, L4, etc.)
    # accelerator_type="A10G",
    # You can customize the engine arguments (e.g. vLLM engine kwargs)
    engine_kwargs={
        "tensor_parallel_size": 1,
    },
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)