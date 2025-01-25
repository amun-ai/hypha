"""Main module for hypha."""

import asyncio
from hypha.server import get_argparser, create_application
from hypha.interactive import start_interactive_shell


def main():
    """Main function."""
    arg_parser = get_argparser()
    opt = arg_parser.parse_args()
    app = create_application(opt)

    if opt.interactive:
        asyncio.run(start_interactive_shell(app, opt))
    else:
        import uvicorn
        uvicorn.run(app, host=opt.host, port=int(opt.port))


if __name__ == "__main__":
    main()
