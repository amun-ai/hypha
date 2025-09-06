"""Interactive shell for hypha server."""

import asyncio
import sys
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout
from ptpython.repl import embed
from ptpython.repl import PythonRepl
from fastapi import FastAPI
import uvicorn

from hypha.server import get_argparser, create_application


def configure_ptpython(repl: PythonRepl) -> None:
    """Configure ptpython REPL settings."""
    repl.show_signature = True
    repl.show_docstring = True
    repl.completion_visualisation = "pop-up"
    repl.show_line_numbers = True
    repl.show_status_bar = True
    repl.show_sidebar_help = True
    repl.highlight_matching_parenthesis = True
    repl.use_code_colorscheme("vs")

    # Set the welcome message
    welcome_message = """Welcome to Hypha Interactive Shell!

Available objects:

    - server: The FastAPI server instance
    - app: The FastAPI application instance
    - store: The Redis store instance


You can run async code directly using top-level await.
Type 'exit()' or press Ctrl+D to quit.
"""
    print(welcome_message)


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is in use on the specified interface by attempting to bind to it.

    Args:
        host: Interface/host to check (e.g. 'localhost', '0.0.0.0')
        port: Port number to check

    Returns:
        bool: True if port is in use, False otherwise
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow reuse of address to avoid "Address already in use" after testing
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            # Close and unbind the socket by returning False
            return False
        except socket.error:
            return True


async def start_interactive_shell(app: FastAPI, args: Any) -> None:
    """Start an interactive shell with the hypha store and server app."""

    # Get the store from the app state
    store = app.state.store

    server = None
    server_task = None

    print("Initializing interactive shell...\n")

    async def start_server():
        if is_port_in_use(args.host, int(args.port)):
            print(
                f"\nFailed to start server on port {args.port}. Port is already in use."
            )
            sys.exit(1)
        nonlocal server, server_task
        config = uvicorn.Config(app, host=args.host, port=int(args.port))
        server = uvicorn.Server(config)
        # Run the server in a separate task
        server_task = asyncio.create_task(server.serve())
        await store.get_event_bus().wait_for_local("startup")
        print(f"\nServer started at http://{args.host}:{args.port}")

    if args.enable_server:
        # Start the server in the background
        await start_server()

    # Prepare the local namespace
    local_ns = {
        "app": app,
        "store": store,
    }
    if server:
        local_ns["server"] = server
    else:
        args.startup_functions = args.startup_functions or []
        # Admin terminal is already enabled via constructor if the flag is set
        await store.init(
            reset_redis=args.reset_redis, startup_functions=args.startup_functions
        )
        local_ns["start_server"] = start_server
        
        # Add admin terminal to namespace if available
        if store._admin_utils and store._admin_utils.terminal:
            local_ns["admin_terminal"] = store._admin_utils.terminal
            print("Admin terminal available as 'admin_terminal'")
            print("Use admin_terminal.execute_command() to run commands")

    # Start the interactive shell with stdout patching for better async support
    with patch_stdout():
        await embed(
            globals=None,
            locals=local_ns,
            configure=configure_ptpython,
            return_asyncio_coroutine=True,
            patch_stdout=True,
        )

    # Cleanup when shell exits
    if server:
        print("\nShutting down server...")
        server.should_exit = True
        await server_task

    if store.is_ready():
        print("Cleaning up store...")
        await store.teardown()
        print("Cleanup complete.")


def main() -> None:
    """Main entry point for the interactive shell."""
    arg_parser = get_argparser()
    args = arg_parser.parse_args()

    # Create the FastAPI app instance
    app = create_application(args)

    try:
        asyncio.run(start_interactive_shell(app, args))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Hypha Interactive Shell...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
