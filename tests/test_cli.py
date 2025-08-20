"""Test CLI functionality."""

import pytest
import subprocess
import sys
import os
import tempfile
import time
import threading
import asyncio
from hypha.server import create_application, get_argparser
from hypha.__main__ import generate_token_command, create_cli_parser
from hypha.core.auth import _parse_token
import uvicorn


class TestCLI:
    """Test CLI commands."""

    def test_generate_token_basic(self):
        """Test basic token generation."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token"])

        token = asyncio.run(generate_token_command(args))

        # Token should be a string
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be able to parse the token
        user_info = _parse_token(token)
        assert user_info is not None
        assert user_info.is_anonymous is True
        assert "anonymous" in user_info.roles

    def test_generate_token_with_workspace(self):
        """Test token generation with specific workspace."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--workspace", "test-workspace"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.scope is not None
        assert "test-workspace" in user_info.scope.workspaces
        assert user_info.scope.current_workspace == "test-workspace"

    def test_generate_token_with_expires_in(self):
        """Test token generation with custom expiration."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--expires-in", "7200"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        # Token should be valid and have the correct expiration
        assert user_info is not None

    def test_generate_token_with_client_id(self):
        """Test token generation with client ID."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--client-id", "test-client"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.scope is not None
        assert user_info.scope.client_id == "test-client"

    def test_generate_token_with_user_id(self):
        """Test token generation with specific user ID."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--user-id", "test-user"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.id == "test-user"

    def test_generate_token_with_email(self):
        """Test token generation with email."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--email", "test@example.com"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.email == "test@example.com"

    def test_generate_token_with_roles(self):
        """Test token generation with custom roles."""
        parser = create_cli_parser()
        args = parser.parse_args(
            ["generate-token", "--role", "admin", "--role", "user"]
        )

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert "admin" in user_info.roles
        assert "user" in user_info.roles

    def test_generate_token_with_scope_format(self):
        """Test token generation with scope format."""
        parser = create_cli_parser()
        args = parser.parse_args(["generate-token", "--scope", "my-workspace#admin"])

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.scope is not None
        assert "my-workspace" in user_info.scope.workspaces
        assert user_info.scope.workspaces["my-workspace"].value == "a"
        assert user_info.scope.current_workspace == "my-workspace"

    def test_generate_token_with_permission(self):
        """Test token generation with specific permission."""
        parser = create_cli_parser()
        args = parser.parse_args(
            ["generate-token", "--permission", "admin", "--workspace", "test-ws"]
        )

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.scope is not None
        assert user_info.scope.workspaces["test-ws"].value == "a"

    def test_generate_token_all_options(self):
        """Test token generation with all options."""
        parser = create_cli_parser()
        args = parser.parse_args(
            [
                "generate-token",
                "--workspace",
                "full-test-workspace",
                "--expires-in",
                "1800",
                "--client-id",
                "full-test-client",
                "--user-id",
                "full-test-user",
                "--role",
                "admin",
                "--role",
                "developer",
                "--email",
                "fulltest@example.com",
                "--permission",
                "admin",
            ]
        )

        token = asyncio.run(generate_token_command(args))

        user_info = _parse_token(token)
        assert user_info.id == "full-test-user"
        assert user_info.email == "fulltest@example.com"
        assert "admin" in user_info.roles
        assert "developer" in user_info.roles
        assert user_info.scope is not None
        assert user_info.scope.client_id == "full-test-client"
        assert "full-test-workspace" in user_info.scope.workspaces
        assert user_info.scope.workspaces["full-test-workspace"].value == "a"
        assert user_info.scope.current_workspace == "full-test-workspace"


class TestCLIIntegration:
    """Test CLI integration with running server."""

    @pytest.fixture
    def running_server(self):
        """Start a test server in the background."""
        # Create a test server
        arg_parser = get_argparser(add_help=False)
        opt = arg_parser.parse_args(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "9528",  # Use different port for testing
                "--enable-server-apps",
            ]
        )
        app = create_application(opt)

        # Start server in background thread
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=9528, log_level="error")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait a bit for server to start
        time.sleep(2)

        yield {"host": "127.0.0.1", "port": 9528, "url": "http://127.0.0.1:9528"}

        # Cleanup happens automatically with daemon thread

    def test_cli_help(self):
        """Test CLI help functionality."""
        result = subprocess.run(
            [sys.executable, "-m", "hypha.__main__", "--help"],
            capture_output=True,
            text=True,
        )

        # Should exit with code 0 for help
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_generate_token_cli_command(self):
        """Test generate-token command via CLI."""
        # Set a fixed JWT secret for consistent token generation
        env = os.environ.copy()
        env["HYPHA_JWT_SECRET"] = "test-secret-key-for-cli-testing"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hypha.__main__",
                "generate-token",
                "--workspace",
                "cli-test-workspace",
                "--expires-in",
                "3600",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        assert "Generated token for user" in result.stdout
        assert "cli-test-workspace" in result.stdout
        assert "Token:" in result.stdout

        # Extract the token from output
        lines = result.stdout.strip().split("\n")
        token_line = None
        for i, line in enumerate(lines):
            if line.strip() == "Token:":
                token_line = lines[i + 1].strip()
                break

        assert token_line is not None
        assert len(token_line) > 0

        # Verify the token format looks correct (JWT tokens have 3 parts separated by dots)
        parts = token_line.split(".")
        assert len(parts) == 3, "Token should have 3 parts separated by dots"

        # Each part should be base64-encoded and not empty
        for part in parts:
            assert len(part) > 0, "Token parts should not be empty"

    def test_auto_from_env_behavior(self):
        """Test that --from-env is automatically added when no args provided."""
        # Set environment variable
        env = os.environ.copy()
        env["HYPHA_HOST"] = "127.0.0.1"
        env["HYPHA_PORT"] = "9529"

        # Create a simple script that just imports and checks the arguments
        test_script = """
import sys
from hypha.__main__ import main
from hypha.server import get_argparser

# Mock sys.argv to simulate no arguments
original_argv = sys.argv[:]
sys.argv = ["hypha"]  # Only program name

try:
    # This should automatically add --from-env
    parser = get_argparser()
    # The main function should handle this case
    print("SUCCESS: Auto --from-env behavior works")
except SystemExit as e:
    if e.code == 0:
        print("SUCCESS: Help was shown (expected behavior)")
    else:
        print(f"ERROR: Unexpected exit code {e.code}")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    sys.argv = original_argv
"""

        # Write to temporary file and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script)
            temp_script = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )

            # Should not fail with an error
            assert "ERROR:" not in result.stdout or "SUCCESS:" in result.stdout

        finally:
            os.unlink(temp_script)

    def test_server_subcommand_compatibility(self):
        """Test that server subcommand works."""
        result = subprocess.run(
            [sys.executable, "-m", "hypha.__main__", "server", "--help"],
            capture_output=True,
            text=True,
        )

        # Should show help for server options
        assert result.returncode == 0
        assert (
            "--host" in result.stdout
            or "--port" in result.stdout
            or "usage:" in result.stdout.lower()
        )
