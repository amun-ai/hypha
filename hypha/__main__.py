"""Main module for hypha."""

import asyncio
import argparse
import sys
from hypha.server import get_argparser, create_application
from hypha.interactive import start_interactive_shell
from hypha.core.auth import (
    generate_auth_token,
    create_scope,
)
from hypha.core import UserInfo, UserPermission
from hypha.utils import random_id
import uvicorn


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


def create_cli_parser():
    """Create the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="hypha-cli", description="Hypha server and CLI tools"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command (default)
    server_parser = subparsers.add_parser("server", help="Start the hypha server")

    # Get the server arguments from get_argparser and add them to server subcommand
    server_arg_parser = get_argparser(add_help=False)
    for action in server_arg_parser._actions:
        if action.dest != "help":
            kwargs = {
                "dest": action.dest,
                "default": action.default,
                "help": action.help,
            }

            # Handle different action types
            if isinstance(action, argparse._StoreTrueAction):
                kwargs["action"] = "store_true"
            elif isinstance(action, argparse._StoreFalseAction):
                kwargs["action"] = "store_false"
            else:
                # For other action types, add appropriate parameters
                if action.type is not None:
                    kwargs["type"] = action.type
                if action.nargs is not None:
                    kwargs["nargs"] = action.nargs
                if hasattr(action, "choices") and action.choices is not None:
                    kwargs["choices"] = action.choices

            server_parser.add_argument(*action.option_strings, **kwargs)

    # Generate token command
    token_parser = subparsers.add_parser(
        "generate-token", help="Generate authentication token"
    )
    token_parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace name (default: anonymous user workspace)",
    )
    token_parser.add_argument(
        "--expires-in",
        type=int,
        default=3600,
        help="Token expiration time in seconds (default: 3600)",
    )
    token_parser.add_argument(
        "--client-id", type=str, help="Optional client ID to restrict the token"
    )
    token_parser.add_argument(
        "--user-id", type=str, help="User ID (default: generated anonymous user)"
    )
    token_parser.add_argument(
        "--role",
        type=str,
        action="append",
        help="User roles (can be specified multiple times, default: ['anonymous'])",
    )
    token_parser.add_argument("--email", type=str, help="User email")
    token_parser.add_argument(
        "--scope",
        type=str,
        help="Token scope in format 'workspace#permission' (default: workspace#read_write)",
    )
    token_parser.add_argument(
        "--permission",
        type=str,
        choices=["read", "read_write", "admin"],
        default="read_write",
        help="Permission level for the workspace (default: read_write)",
    )

    return parser


async def generate_token_command(args):
    """Handle the generate-token command."""
    # Map user-friendly permission names to enum values
    permission_map = {"read": "r", "read_write": "rw", "admin": "a"}

    # Create user info
    if args.user_id:
        user_id = args.user_id
    else:
        user_id = "anonymous-" + random_id(readable=True)

    roles = args.role if args.role else ["anonymous"]

    # Determine workspace
    workspace = args.workspace
    if not workspace:
        if args.user_id:
            workspace = f"ws-user-{args.user_id}"
        else:
            workspace = f"ws-user-{user_id}"

    # Create scope
    if args.scope:
        # Parse custom scope format: workspace#permission
        if "#" in args.scope:
            ws_name, permission = args.scope.split("#", 1)
            # Map permission to enum value if it's a user-friendly name
            mapped_permission = permission_map.get(permission, permission)
            scope = create_scope(
                workspaces={ws_name: UserPermission(mapped_permission)},
                current_workspace=ws_name,
                client_id=args.client_id,
            )
        else:
            # Simple workspace name, use default permission
            mapped_permission = permission_map.get(args.permission, args.permission)
            scope = create_scope(
                workspaces={args.scope: UserPermission(mapped_permission)},
                current_workspace=args.scope,
                client_id=args.client_id,
            )
    else:
        # Create scope with specified workspace and permission
        mapped_permission = permission_map.get(args.permission, args.permission)
        scope = create_scope(
            workspaces={workspace: UserPermission(mapped_permission)},
            current_workspace=workspace,
            client_id=args.client_id,
        )

    # Create user info
    user_info = UserInfo(
        id=user_id,
        roles=roles,
        is_anonymous="anonymous" in roles,
        email=args.email,
        scope=scope,
    )

    # Generate token
    token = await generate_auth_token(user_info, args.expires_in)

    print(f"Generated token for user '{user_id}':")
    print(f"  Workspace: {workspace}")
    print(f"  Permission: {args.permission}")
    print(f"  Expires in: {args.expires_in} seconds")
    if args.client_id:
        print(f"  Client ID: {args.client_id}")
    if args.email:
        print(f"  Email: {args.email}")
    print(f"  Roles: {', '.join(roles)}")
    print(f"\nToken:")
    print(token)

    return token


def main():
    """Main entry point for the CLI."""
    # If no arguments provided, automatically add --from-env
    if len(sys.argv) == 1:
        sys.argv.append("--from-env")

    # Check if this is a subcommand call
    if len(sys.argv) > 1 and sys.argv[1] in ["generate-token"]:
        # Handle subcommands
        parser = create_cli_parser()
        args = parser.parse_args()

        if args.command == "generate-token":
            asyncio.run(generate_token_command(args))
            return

    # Handle server command or legacy mode (no subcommand)
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Remove 'server' from args for legacy compatibility
        sys.argv.pop(1)

    # Create the app instance using original server logic
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
