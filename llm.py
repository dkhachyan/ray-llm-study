from ray import serve
from starlette.requests import Request
from vllm import LLM, SamplingParams

@serve.deployment()
class HelloWorld:
    def __init__(self):
        self.model = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        self.promt="privet kto kotik"
    async def __call__(self, request: Request):
        self.model.wake_up()
        outputs = self.model.generate(self.promt, self.sampling_params)
        self.model.sleep(level=2)
        return outputs[0].outputs[0].text

app = HelloWorld.bind()