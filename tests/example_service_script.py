import argparse
import asyncio
import logging

from hypha_rpc import connect_to_server


async def start_service(server_url, service_id, workspace=None, token=None):
    """Start the service."""
    client_id = service_id + "-client"
    print(f"Starting service...")
    server = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": server_url,
            "workspace": workspace,
            "token": token,
        }
    )
    await server.register_service(
        {
            "id": service_id,
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "test": lambda x: x + 99,
        }
    )
    print(
        f"Service (client_id={client_id}, service_id={service_id}) started successfully, available at {server_url}/{server.config.workspace}/services"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test services")
    parser.add_argument(
        "--server-url", type=str, default="https://ai.imjoy.io", help="The server url"
    )
    parser.add_argument(
        "--service-id", type=str, default="test-service", help="The service id"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="The workspace name"
    )
    parser.add_argument("--token", type=str, default=None, help="The token")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()
    loop.create_task(
        start_service(
            args.server_url,
            args.service_id,
            workspace=args.workspace,
            token=args.token,
        )
    )
    loop.run_forever()
