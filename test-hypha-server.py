from pydantic import BaseModel, Field
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function


async def main():
    server = await connect_to_server({"server_url": "http://127.0.0.1:9527"})

    class UserInfo(BaseModel):
        name: str = Field(..., description="Name of the user")
        email: str = Field(..., description="Email of the user")
        age: int = Field(..., description="Age of the user")
        address: str = Field(..., description="Address of the user")

    @schema_function
    def register_user(user_info: UserInfo) -> str:
        return f"User {user_info.name} registered"

    info = await server.register_service(
        {
            "name": "User Service",
            "id": "user-service",
            "description": "Service for registering users",
            "register_user": register_user,
        }
    )
    print(
        f"Service registered, you can see it here: http://127.0.0.1:9527/{info.config.workspace}/services/{info.id.split('/')[-1]}"
    )


import asyncio

loop = asyncio.get_event_loop()
loop.create_task(main())
loop.run_forever()
