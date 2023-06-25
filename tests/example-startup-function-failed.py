"""Example hypha startup function for loading services at startup of Hypha."""


async def hypha_startup(server):
    """Register the services."""
    await server.non_existing_function()
