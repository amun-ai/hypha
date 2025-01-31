"""Main module for hypha."""

import asyncio
from hypha.server import get_argparser, create_application
from hypha.interactive import start_interactive_shell
import uvicorn
import os
import sys


def run_interactive_cli():
    """Run the interactive CLI."""

    # Create the app instance
    arg_parser = get_argparser(add_help=True)
    opt = arg_parser.parse_args(sys.argv[1:])

    # Force interactive server mode
    if not opt.interactive:
        opt.interactive = True

    app = create_application(opt)

    # Start the interactive shell
    asyncio.run(start_interactive_shell(app, opt))


def main():
    # Create the app instance
    arg_parser = get_argparser()
    opt = arg_parser.parse_args()
    app = create_application(opt)

    if opt.interactive:
        asyncio.run(start_interactive_shell(app, opt))
    else:
        uvicorn.run(app, host=opt.host, port=int(opt.port))


if __name__ == "__main__":
    main()
else:
    # Create the app instance when imported by uvicorn
    arg_parser = get_argparser(add_help=False)
    opt = arg_parser.parse_args(
        ["--from-env"]
    )  # Parse with from-env flag to support environment variables
    app = create_application(opt)
    __all__ = ["app"]
