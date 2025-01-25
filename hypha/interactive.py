"""Interactive shell for hypha server."""

import asyncio
import sys
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout
from ptpython.repl import embed
from ptpython.repl import PythonRepl
from fastapi import FastAPI
import uvicorn

from hypha.server import get_argparser


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
Type 'exit' to quit.
"""
    print(welcome_message)


async def start_interactive_shell(app: FastAPI, args: Any) -> None:
    """Start an interactive shell with the hypha store and server app."""

    # Get the store from the app state
    store = app.state.store

    # Start the server in the background
    config = uvicorn.Config(app, host=args.host, port=int(args.port))
    server = uvicorn.Server(config)
    # Run the server in a separate task
    server_task = asyncio.create_task(server.serve())
    await store.get_event_bus().wait_for_local("startup")
    print(f"\nServer started at http://{args.host}:{args.port}")
    print("Initializing interactive shell...\n")

    # Prepare the local namespace
    local_ns = {
        "app": app,
        "store": store,
        "server": server,
    }

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

    try:
        asyncio.run(start_interactive_shell(args))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Hypha Interactive Shell...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
