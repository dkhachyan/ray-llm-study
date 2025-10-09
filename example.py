import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

ray.init()
serve.start()

@serve.deployment
class MyFirstDeployment:
    # Take the message to return as an argument to the constructor.
    def __init__(self, msg):
        self.msg = msg

    def __call__(self):
        return self.msg

my_first_deployment = MyFirstDeployment.bind("Hello world!")
handle: DeploymentHandle = serve.run(my_first_deployment)
assert handle.remote().result() == "Hello world!"