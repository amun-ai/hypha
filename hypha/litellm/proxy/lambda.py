from mangum import Mangum
from hypha.litellm.proxy.proxy_server import app

handler = Mangum(app, lifespan="on")
