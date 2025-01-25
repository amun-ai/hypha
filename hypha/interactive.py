"""Interactive shell for hypha server."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit.patch_stdout import patch_stdout
from ptpython.repl import embed
from ptpython.repl import PythonRepl
import uvicorn

from hypha.server import get_argparser, create_application
from hypha.core.store import RedisStore


def create_store_from_args(args) -> RedisStore:
    """Create a RedisStore instance from command line arguments."""
    local_base_url = f"http://127.0.0.1:{args.port}/{args.base_path.strip('/')}".strip(
        "/"
    )
    public_base_url = (
        args.public_base_url.strip("/") if args.public_base_url else local_base_url
    )

    store = RedisStore(
        None,  # No FastAPI app needed for interactive mode
        server_id=args.server_id,
        public_base_url=public_base_url,
        local_base_url=local_base_url,
        redis_uri=args.redis_uri,
        database_uri=args.database_uri,
        ollama_host=args.ollama_host,
        cache_dir=args.cache_dir,
        openai_config={
            "base_url": args.openai_base_url,
            "api_key": args.openai_api_key,
        },
        enable_service_search=args.enable_service_search,
    )
    return store


def configure_ptpython(repl: PythonRepl) -> None:
    """Configure ptpython REPL settings."""
    repl.show_signature = True
    repl.show_docstring = True
    repl.completion_visualisation = "pop-up"
    repl.show_line_numbers = True
    repl.show_status_bar = True
    repl.show_sidebar_help = True
    repl.highlight_matching_parenthesis = True
    repl.use_code_colorscheme("monokai")

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


async def start_interactive_shell(args: Any) -> None:
    """Start an interactive shell with the hypha store and server app."""
    # Create the FastAPI application
    app = create_application(args)

    # Get the store from the app state
    store = app.state.store

    # Start the server in the background
    config = uvicorn.Config(app, host=args.host, port=int(args.port))
    server = uvicorn.Server(config)

    # Run the server in a separate task
    server_task = asyncio.create_task(server.serve())

    print(f"Server started at http://{args.host}:{args.port}")
    print("Initializing interactive shell...")

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
