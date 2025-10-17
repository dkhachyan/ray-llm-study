from ray import serve
from starlette.requests import Request
from vllm import LLM, SamplingParams

@serve.deployment()
class HelloWorld:
    def __init__(self):
        self.model = LLM("Qwen/Qwen2.5-7B-Instruct", enable_sleep_mode=True)
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        self.model.reset_prefix_cache()
        self.model.sleep(level=1)
        self.promt="Hi. Who are you"
        self.model2 = LLM("unsloth/Meta-Llama-3.1-8B-Instruct", enable_sleep_mode=True)
        self.model2.reset_prefix_cache()
        self.model2.sleep(level=1)
    async def __call__(self, request: Request):
        self.model.wake_up()
        outputs = self.model.generate(self.promt, self.sampling_params)
        self.model.reset_prefix_cache()
        self.model.sleep(level=1)
        return outputs[0].outputs[0].text

app = HelloWorld.bind()