"""Main module for hypha."""

import asyncio
from hypha.server import get_argparser, create_application
from hypha.interactive import start_interactive_shell


if __name__ == "__main__":
    arg_parser = get_argparser()
    opt = arg_parser.parse_args()
    app = create_application(opt)

    if opt.interactive:
        asyncio.run(start_interactive_shell(app, opt))
    else:
        import uvicorn

        uvicorn.run(app, host=opt.host, port=int(opt.port))

else:
    # Create the app instance when imported by uvicorn
    arg_parser = get_argparser(add_help=False)
    opt = arg_parser.parse_args(
        ["--from-env"]
    )  # Parse with from-env flag to support environment variables
    app = create_application(opt)
    __all__ = ["app"]
