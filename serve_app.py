# serve_app.py
from ray import serve
from starlette.requests import Request

@serve.deployment()
class HelloWorld:
    async def __call__(self, request: Request):
        return "Hello from Ray Serve!"

# This is required so Ray Serve knows what to deploy
app = HelloWorld.bind()