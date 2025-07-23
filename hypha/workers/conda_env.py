from hypha.workers.conda_env_executor import CondaEnvExecutor

# Create a temporary environment with specific packages
executor = CondaEnvExecutor.create_temp_env(
    packages=['python=3.11', {"pip": ['hypha-rpc']}],
    channels=['conda-forge']
)

# Define code to run - must include an 'execute' function
code = '''
import asyncio
from hypha_rpc import connect_to_server

async def start_server(server_url):
    async with connect_to_server({"server_url": server_url}) as server:
        def hello(name):
            print("Hello " + name)
            return "Hello " + name

        svc = await server.register_service({
            "name": "Hello World",
            "id": "hello-world",
            "config": {
                "visibility": "public"
            },
            "hello": hello
        })

        print(f"Hello world service registered at workspace: {server.config.workspace}, id: {svc.id}")

        print(f'You can use this service using the service id: {svc.id}')

        print(f"You can also test the service via the HTTP proxy: {server_url}/{server.config.workspace}/services/{svc.id.split('/')[1]}/hello?name=John")

        # Keep the server running
        # await server.serve()

def execute(input_data):
    server_url = "https://hypha.aicell.io"
    asyncio.run(start_server(server_url))
'''

# Execute with input data
input_data = {"values": [1, 2, 3, 4, 5]}

with executor:
    result = executor.execute(code, input_data)
    if result.success:
        print(result)
    else:
        print(f"Error: {result.error}")