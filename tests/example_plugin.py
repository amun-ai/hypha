"""Provide an example plugin."""
from imjoy_rpc import api

# pylint: disable=no-self-use


class ImJoyPlugin:
    """Represent an ImJoy plugin."""

    async def setup(self):
        """Set up the plugin."""
        token = await api.generateToken()
        assert "@imjoy@" in token
        print(f"Generated token: {token}")

        service_info = await api.register_service(
            {
                "id": "test-service",
                "name": "echo service",
                "type": "echo",
                "echo": lambda x: print("echo: " + str(x)),
            },
            overwrite=True,
        )
        service = await api.get_service(service_info)
        await service.echo("a message")
        await api.log("initialized")

    async def run(self, ctx):
        """Run the plugin."""
        await api.log("hello world")


api.export(ImJoyPlugin(), config={"name": "test-plugin"})
