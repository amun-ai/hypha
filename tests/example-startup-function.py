"""Example hypha startup function for loading services at startup of Hypha."""


async def hypha_startup(server):
    """Register the services."""

    # The server object passed into this function is the same as the one in the client script
    # You can register more functions or call other functions in the server object
    await server.register_service(
        {
            "id": "test-service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "test": lambda x: x + 99,
        }
    )
