"""Plugin runner."""
from . import start_runner

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="path to a plugin file")
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
        help="the plugin workspace",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="token for the plugin workspace",
    )

    parser.add_argument(
        "--quit-on-ready",
        action="store_true",
        help="quit the server when the plugin is ready",
    )

    opt = parser.parse_args()

    start_runner(opt)
