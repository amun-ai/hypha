"""Main module for hypha."""

import asyncio
from hypha.server import get_argparser, create_application
from hypha.interactive import start_interactive_shell


def main():
    """Main function."""
    arg_parser = get_argparser()
    arg_parser.add_argument(
        "--interactive",
        action="store_true",
        help="start an interactive shell with the hypha store",
    )
    opt = arg_parser.parse_args()

    if opt.interactive:
        asyncio.run(start_interactive_shell(opt))
    else:
        import uvicorn

        app = create_application(opt)
        uvicorn.run(app, host=opt.host, port=int(opt.port))


if __name__ == "__main__":
    main()
