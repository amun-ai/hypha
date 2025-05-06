# run_lite_server.py
import uvicorn
from hypha.lite.server import get_argparser, create_application, logger

if __name__ == "__main__":
    arg_parser = get_argparser()
    opt = arg_parser.parse_args()

    # Basic host warning (copied from server.py main block)
    if opt.host in ("127.0.0.1", "localhost") and not opt.public_base_url:
        logger.info(
            "***Note: Server is only accessible locally. "
            "Use --host=0.0.0.0 or set --public-base-url to allow external access.***"
        )

    app = create_application(opt)
    # Use host/port from parsed options
    uvicorn.run(app, host=opt.host, port=int(opt.port), loop="asyncio") 