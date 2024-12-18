"""Plugin runner."""

from . import start_runner

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "app",
        type=str,
        help="the app to run, can be a .py or .imjoy.html file, or `server-app-worker` for launching a server app worker",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="url to the hypha websocket server",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="the app workspace",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="token for the app workspace",
    )

    parser.add_argument(
        "--quit-on-ready",
        action="store_true",
        help="quit the server when the app is ready",
    )

    opt = parser.parse_args()

    start_runner(opt)
