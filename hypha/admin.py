"""Admin terminal and utilities for Hypha server."""

import asyncio
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("admin")
logger.setLevel(LOGLEVEL)


class AdminTerminal:
    """Admin terminal for interactive Python REPL."""

    def __init__(self, store):
        """Initialize the admin terminal.
        
        Args:
            store: The RedisStore instance
        """
        self.store = store
        self.app = getattr(store, '_app', None)
        self.output_buffer = []
        self.input_queue = asyncio.Queue()
        self.buffer_lock = asyncio.Lock()
        self.terminal_size = {"rows": 24, "cols": 80}
        self._running = False
        self._repl_task = None
        self.namespace = {}
        self._input_event = asyncio.Event()
        self._current_input = ""
        # Add execution state tracking
        self._execution_complete = asyncio.Event()
        self._is_executing = False
        self._execution_output = []
        # Command history
        self._command_history = []
        self._history_index = -1
        self._max_history = 1000
        # Tab completion state
        self._completion_context = None
        self._cursor_position = 0

    @schema_method
    async def start_terminal(self) -> Dict[str, Any]:
        """Start the Python REPL terminal.
        Returns:
            Dictionary with terminal info
        """
        if self._running and self._repl_task and not self._repl_task.done():
            # Terminal is already running, just return success
            logger.info("Python REPL already running, reusing existing session")
            return {
                "success": True,
                "message": "Terminal already running",
                "terminal_size": self.terminal_size
            }

        try:
            # Clear buffer and start REPL
            async with self.buffer_lock:
                self.output_buffer = []
            self._running = True
            
            # Prepare namespace with store and app objects
            self.namespace = {
                "store": self.store,
                "app": self.app,
                "admin": self,
                "__builtins__": __builtins__,
            }
            
            # Send welcome message
            welcome = (
                "\r\n"
                "=== Hypha Admin Python Terminal ===\r\n"
                "\r\n"
                "Available objects:\r\n"
                "  - store: The Redis store instance\r\n"
                "  - app: The FastAPI application instance\r\n"
                "  - admin: This admin terminal instance\r\n"
                "\r\n"
                "You can run async code directly using await.\r\n"
                "Type help(object) for documentation.\r\n"
                "Use UP/DOWN arrows for command history, TAB for completion.\r\n"
                "\r\n"
                ">>> "
            )
            async with self.buffer_lock:
                self.output_buffer.append(welcome)
            
            # Start the REPL task
            self._repl_task = asyncio.create_task(self._run_repl())
            
            logger.info("Admin Python REPL started")
            return {
                "success": True,
                "message": "Python REPL started",
                "terminal_size": self.terminal_size
            }
            
        except Exception as e:
            logger.error(f"Failed to start admin terminal: {e}")
            return {"success": False, "error": str(e)}

    async def _run_repl(self):
        """Run the ptpython REPL in the background."""
        import ast
        import types
        
        try:
            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create string IO buffers for capturing output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            logger.info("Python REPL background task started")
            
            # Simple REPL loop
            while self._running:
                try:
                    # Wait for input
                    await self._input_event.wait()
                    
                    # Clear the event immediately to prevent infinite loops on error
                    self._input_event.clear()
                    
                    if not self._running:
                        break
                    
                    # Ensure _current_input is a string
                    if not isinstance(self._current_input, str):
                        logger.error(f"Invalid input type: {type(self._current_input)}, expected str")
                        self._current_input = ""
                        async with self.buffer_lock:
                            self.output_buffer.append("Error: Invalid input type\r\n>>> ")
                            if self._is_executing:
                                self._execution_output.append("Error: Invalid input type")
                        # Mark execution as complete if it was in progress
                        if self._is_executing:
                            self._is_executing = False
                            self._execution_complete.set()
                        continue
                        
                    input_code = self._current_input.strip()
                    self._current_input = ""
                    self._cursor_position = 0
                    
                    if not input_code:
                        continue
                    
                    # Add to command history if not duplicate of last command
                    if not self._command_history or self._command_history[-1] != input_code:
                        self._command_history.append(input_code)
                        if len(self._command_history) > self._max_history:
                            self._command_history.pop(0)
                    self._history_index = len(self._command_history)
                    
                    # Mark execution as started
                    self._is_executing = True
                    self._execution_complete.clear()
                    self._execution_output = []
                        
                    # Redirect stdout/stderr to capture output
                    sys.stdout = stdout_buffer
                    sys.stderr = stderr_buffer
                    
                    try:
                        # Parse the code to check if it's an expression or statement
                        try:
                            # Try parsing as an expression first
                            parsed = ast.parse(input_code, mode='eval')
                            # If it's an expression, we can evaluate it
                            result = eval(compile(parsed, '<string>', 'eval'), self.namespace)
                            if asyncio.iscoroutine(result):
                                result = await result
                            if result is not None:
                                print(repr(result))
                        except SyntaxError:
                            # Not an expression, try as statement
                            try:
                                # Check if it contains await (top-level await support)
                                if 'await ' in input_code:
                                    # Wrap in async function for top-level await
                                    wrapped_code = f"async def __async_wrapper():\n    {input_code.replace(chr(10), chr(10)+'    ')}"
                                    exec(wrapped_code, self.namespace)
                                    result = await self.namespace['__async_wrapper']()
                                    if result is not None:
                                        print(repr(result))
                                else:
                                    # Regular statement
                                    exec(input_code, self.namespace)
                            except SyntaxError as se:
                                print(f"SyntaxError: {se}")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    # Restore stdout/stderr
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
                    # Get captured output
                    stdout_output = stdout_buffer.getvalue()
                    stderr_output = stderr_buffer.getvalue()
                    
                    # Clear buffers
                    stdout_buffer.seek(0)
                    stdout_buffer.truncate(0)
                    stderr_buffer.seek(0)
                    stderr_buffer.truncate(0)
                    
                    # Add output to buffer and execution output
                    if stdout_output or stderr_output:
                        output = stdout_output + stderr_output + "\r\n>>> "
                        async with self.buffer_lock:
                            self.output_buffer.append(output)
                            self._execution_output.append(stdout_output + stderr_output)
                    else:
                        async with self.buffer_lock:
                            self.output_buffer.append(">>> ")
                    
                    # Mark execution as complete
                    self._is_executing = False
                    self._execution_complete.set()
                            
                except Exception as e:
                    logger.error(f"REPL error: {e}")
                    # Ensure stdout/stderr are restored
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    
                    async with self.buffer_lock:
                        self.output_buffer.append(f"REPL Error: {e}\r\n>>> ")
                        if self._is_executing:
                            self._execution_output.append(f"REPL Error: {e}")
                    
                    # Mark execution as complete even on error
                    if self._is_executing:
                        self._is_executing = False
                        self._execution_complete.set()
                    
                    # Reset current input to prevent issues
                    self._current_input = ""
                    
        except Exception as e:
            logger.error(f"REPL task failed: {e}")
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Ensure execution is marked as complete if still running
            if self._is_executing:
                self._is_executing = False
                self._execution_complete.set()
            logger.info("Python REPL background task ended")

    @schema_method
    async def resize_terminal(
        self,
        rows: int = Field(..., description="Number of rows"),
        cols: int = Field(..., description="Number of columns"),
    ) -> Dict[str, Any]:
        """Resize the terminal window."""
        try:
            self.terminal_size = {"rows": rows, "cols": cols}
            logger.debug(f"Resized admin terminal to {rows}x{cols}")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @schema_method
    async def read_terminal(
        self,
    ) -> Dict[str, Any]:
        """Read output from the terminal."""
        if not self._running:
            return {"success": False, "error": "Terminal not running"}
        
        try:
            # Get any new output from buffer
            async with self.buffer_lock:
                if self.output_buffer:
                    output = ''.join(self.output_buffer)
                    self.output_buffer.clear()  # Properly clear the list
                    return {"success": True, "output": output}
            
            return {"success": True, "output": ""}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @schema_method
    async def get_screen_content(
        self,
    ) -> Dict[str, Any]:
        """Get the accumulated screen buffer content without clearing it."""
        try:
            async with self.buffer_lock:
                # Don't clear the buffer, just return the current content
                content = ''.join(self.output_buffer) if self.output_buffer else ""
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @schema_method
    async def write_terminal(
        self,
        data: str = Field(..., description="Data to write to terminal"),
    ) -> Dict[str, Any]:
        """Write data to the terminal."""
        if not self._running:
            return {"success": False, "error": "Terminal not running"}
        
        try:
            # Ensure data is a string
            if not isinstance(data, str):
                logger.warning(f"write_terminal received non-string data: {type(data)}")
                data = str(data) if data is not None else ""
            
            # Ensure _current_input is always a string
            if not isinstance(self._current_input, str):
                logger.warning(f"_current_input was not a string: {type(self._current_input)}")
                self._current_input = ""
            
            # Handle special sequences for arrow keys and tab
            if data == '\x1b[A':  # Up arrow
                return await self._handle_history_navigation('up')
            elif data == '\x1b[B':  # Down arrow
                return await self._handle_history_navigation('down')
            elif data == '\x1b[C':  # Right arrow
                if self._cursor_position < len(self._current_input):
                    self._cursor_position += 1
                return {"success": True, "action": "cursor_right"}
            elif data == '\x1b[D':  # Left arrow
                if self._cursor_position > 0:
                    self._cursor_position -= 1
                return {"success": True, "action": "cursor_left"}
            elif data == '\t':  # Tab key
                return await self._handle_tab_completion()
            # Handle input data
            elif data == '\r' or data == '\n' or data == '\r\n':
                # Execute the current input
                if self._current_input.strip():
                    # Frontend already displayed the input, just add newline
                    async with self.buffer_lock:
                        self.output_buffer.append("\r\n")
                    self._input_event.set()
                else:
                    # Just show new prompt
                    async with self.buffer_lock:
                        self.output_buffer.append("\r\n>>> ")
                self._history_index = len(self._command_history)
            elif data == '\x7f' or data == '\b':
                # Backspace - handle with cursor position
                if self._cursor_position > 0 and self._current_input:
                    # Remove character at cursor position
                    self._current_input = (
                        self._current_input[:self._cursor_position-1] + 
                        self._current_input[self._cursor_position:]
                    )
                    self._cursor_position -= 1
            elif data == '\x03':
                # Ctrl+C
                self._current_input = ""
                self._cursor_position = 0
                self._history_index = len(self._command_history)
                async with self.buffer_lock:
                    self.output_buffer.append("^C\r\n>>> ")
            elif data and ord(data[0]) >= 32:  # Printable character
                # Insert character at cursor position
                self._current_input = (
                    self._current_input[:self._cursor_position] + 
                    data + 
                    self._current_input[self._cursor_position:]
                )
                self._cursor_position += 1
                
            return {"success": True, "bytes_written": len(data)}
        except Exception as e:
            logger.error(f"write_terminal error: {e}")
            return {"success": False, "error": str(e)}

    @schema_method
    async def execute_command(
        self,
        command: str = Field(..., description="Command to execute"),
        timeout: float = Field(5.0, description="Timeout in seconds"),
    ) -> Dict[str, Any]:
        """Execute a command in the terminal and wait for output.
        
        This method executes a Python command and waits for its completion,
        collecting all output before returning. It properly tracks execution
        state without injecting any markers into the output.
        
        Returns:
            Dictionary with success status and complete command output
            
        Raises:
            TimeoutError: If command doesn't complete within timeout
        """
        if not self._running:
            return {"success": False, "error": "Terminal not running"}
        
        try:
            # Validate command is a string
            if not isinstance(command, str):
                logger.error(f"execute_command received non-string command: {type(command)}")
                return {"success": False, "error": f"Invalid command type: expected str, got {type(command).__name__}"}
            
            # Wait if another command is executing
            if self._is_executing:
                try:
                    await asyncio.wait_for(self._execution_complete.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    return {"success": False, "error": "Another command is still executing"}
            
            # Clear execution output before starting
            self._execution_output = []
            
            # Set the command and trigger execution (ensure it's a string)
            self._current_input = str(command)  # Ensure it's a string
            self._input_event.set()
            
            # Wait for execution to complete
            try:
                await asyncio.wait_for(self._execution_complete.wait(), timeout=timeout)
                
                # Collect the execution output
                output = ''.join(self._execution_output)
                
                # Clean up the output (remove extra newlines if present)
                if output.endswith('\n'):
                    output = output[:-1]
                
                return {"success": True, "output": output}
                
            except asyncio.TimeoutError:
                # Command timed out
                # Try to collect any partial output
                partial_output = ''.join(self._execution_output) if self._execution_output else ""
                
                # Clear the executing state since we're timing out
                self._is_executing = False
                self._execution_complete.set()
                
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "partial_output": partial_output
                }
                
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            # Ensure we don't leave the terminal in an executing state
            if self._is_executing:
                self._is_executing = False
                self._execution_complete.set()
            return {"success": False, "error": str(e)}

    async def _handle_history_navigation(self, direction: str) -> Dict[str, Any]:
        """Handle up/down arrow key navigation through command history."""
        try:
            if not self._command_history:
                return {"success": True, "action": "no_history"}
            
            if direction == 'up':
                # Move up in history (older commands)
                if self._history_index > 0:
                    self._history_index -= 1
                    command = self._command_history[self._history_index]
                    self._current_input = command
                    self._cursor_position = len(command)
                    
                    # Clear current line and show historical command
                    async with self.buffer_lock:
                        # Clear line and return to start
                        self.output_buffer.append("\r\x1b[K>>> " + command)
                    
                    return {"success": True, "action": "history_up", "command": command}
                    
            elif direction == 'down':
                # Move down in history (newer commands)
                if self._history_index < len(self._command_history) - 1:
                    self._history_index += 1
                    command = self._command_history[self._history_index]
                    self._current_input = command
                    self._cursor_position = len(command)
                    
                    # Clear current line and show historical command
                    async with self.buffer_lock:
                        self.output_buffer.append("\r\x1b[K>>> " + command)
                    
                    return {"success": True, "action": "history_down", "command": command}
                elif self._history_index == len(self._command_history) - 1:
                    # At the end of history, clear the input
                    self._history_index = len(self._command_history)
                    self._current_input = ""
                    self._cursor_position = 0
                    
                    async with self.buffer_lock:
                        self.output_buffer.append("\r\x1b[K>>> ")
                    
                    return {"success": True, "action": "history_clear"}
            
            return {"success": True, "action": "history_unchanged"}
            
        except Exception as e:
            logger.error(f"History navigation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_tab_completion(self) -> Dict[str, Any]:
        """Handle tab key for auto-completion."""
        try:
            if not self._current_input:
                return {"success": True, "action": "no_completion"}
            
            # Get the part to complete (everything up to cursor position)
            text_to_complete = self._current_input[:self._cursor_position]
            
            # Find the last word/identifier to complete
            import re
            match = re.search(r'([\w\.]+)$', text_to_complete)
            if not match:
                return {"success": True, "action": "no_completion"}
            
            partial = match.group(1)
            prefix = text_to_complete[:match.start(1)]
            
            # Get completions
            completions = self._get_completions(partial)
            
            if not completions:
                return {"success": True, "action": "no_completions"}
            elif len(completions) == 1:
                # Single completion - use it
                completion = completions[0]
                new_text = prefix + completion
                remaining = self._current_input[self._cursor_position:]
                self._current_input = new_text + remaining
                self._cursor_position = len(new_text)
                
                # Update display
                async with self.buffer_lock:
                    self.output_buffer.append("\r\x1b[K>>> " + self._current_input)
                    if remaining:
                        # Move cursor back if there's text after cursor
                        self.output_buffer.append("\x1b[" + str(len(remaining)) + "D")
                
                return {"success": True, "action": "completed", "completion": completion}
            else:
                # Multiple completions - show them
                async with self.buffer_lock:
                    self.output_buffer.append("\r\n")
                    # Show completions in columns
                    max_len = max(len(c) for c in completions)
                    cols = max(1, self.terminal_size["cols"] // (max_len + 2))
                    for i in range(0, len(completions), cols):
                        row = completions[i:i+cols]
                        formatted_row = "  ".join(c.ljust(max_len) for c in row)
                        self.output_buffer.append(formatted_row + "\r\n")
                    # Redraw the prompt and current input
                    self.output_buffer.append(">>> " + self._current_input)
                    # Position cursor correctly
                    if len(self._current_input) > self._cursor_position:
                        back_amount = len(self._current_input) - self._cursor_position
                        self.output_buffer.append("\x1b[" + str(back_amount) + "D")
                
                return {"success": True, "action": "show_completions", "completions": completions}
            
        except Exception as e:
            logger.error(f"Tab completion error: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_completions(self, partial: str) -> list:
        """Get possible completions for a partial string."""
        try:
            parts = partial.split('.')
            
            if len(parts) == 1:
                # Complete from namespace
                prefix = parts[0]
                return [name for name in self.namespace.keys() 
                       if name.startswith(prefix) and not name.startswith('_')]
            else:
                # Complete attributes of an object
                obj_path = '.'.join(parts[:-1])
                attr_prefix = parts[-1]
                
                try:
                    # Evaluate the object path
                    obj = eval(obj_path, self.namespace)
                    
                    # Get all attributes
                    attrs = dir(obj)
                    
                    # Filter by prefix and exclude private attributes by default
                    completions = [obj_path + '.' + attr for attr in attrs 
                                 if attr.startswith(attr_prefix) and not attr.startswith('_')]
                    
                    # If the prefix starts with _, include private attributes
                    if attr_prefix.startswith('_'):
                        completions = [obj_path + '.' + attr for attr in attrs 
                                     if attr.startswith(attr_prefix)]
                    
                    return completions
                except:
                    return []
        except:
            return []

    async def stop_terminal(self):
        """Stop the admin terminal."""
        self._running = False
        self._input_event.set()  # Wake up REPL task
        
        if self._repl_task:
            try:
                await asyncio.wait_for(self._repl_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._repl_task.cancel()
                try:
                    await self._repl_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Admin terminal stopped")


class AdminUtilities:
    """Administrative utilities for Hypha server management."""
    
    def __init__(self, store):
        """Initialize admin utilities.
        
        Args:
            store: The RedisStore instance
        """
        self.store = store
        self.terminal = None
        
    def setup_terminal(self):
        """Setup the admin terminal."""
        if not self.terminal:
            self.terminal = AdminTerminal(self.store)
    
    @schema_method
    async def list_servers(self):
        """List all connected Hypha servers."""
        return await self.store.list_servers()
    
    @schema_method
    async def kickout_client(
        self,
        workspace: str = Field(..., description="Workspace name"),
        client_id: str = Field(..., description="Client ID to disconnect"),
        code: int = Field(1000, description="Disconnect code"),
        reason: str = Field("Admin requested disconnect", description="Disconnect reason"),
    ):
        """Force disconnect a client from the server."""
        return await self.store.kickout_client(workspace, client_id, code, reason)
    
    @schema_method
    async def list_workspaces(self):
        """List all workspaces."""
        return await self.store.list_all_workspaces()
    
    @schema_method
    async def unload_workspace(
        self,
        workspace: str = Field(..., description="Workspace to unload"),
        wait: bool = Field(False, description="Wait for unload to complete"),
        timeout: int = Field(10, description="Timeout in seconds if wait=True"),
    ):
        """Unload a workspace from memory."""
        return await self.store.unload_workspace(workspace, wait, timeout)
    
    @schema_method
    async def get_metrics(self):
        """Get server metrics and statistics."""
        return await self.store.get_metrics()
    
    def get_service_api(self):
        """Get the admin service API definition."""
        service_config = {
            "id": "admin-utils",
            "name": "Admin Utilities",
            "config": {
                "visibility": "protected",
                "require_context": False,
            },
            "list_servers": self.list_servers,
            "kickout_client": self.kickout_client,
            "list_workspaces": self.list_workspaces,
            "unload_workspace": self.unload_workspace,
            "get_metrics": self.get_metrics,
        }
        
        # Add terminal methods if terminal is setup
        if self.terminal:
            service_config.update({
                "start_terminal": self.terminal.start_terminal,
                "resize_terminal": self.terminal.resize_terminal,
                "read_terminal": self.terminal.read_terminal,
                "get_screen_content": self.terminal.get_screen_content,
                "write_terminal": self.terminal.write_terminal,
                "execute_command": self.terminal.execute_command,
            })
        
        return service_config


async def register_admin_terminal_web_interface(server, admin_service_id, server_url, workspace):
    """Register the web interface for the admin terminal.
    
    Args:
        server: The Hypha server instance
        admin_service_id: The ID of the registered admin service
        server_url: The server URL to inject into the HTML
        workspace: The workspace name to inject into the HTML
    
    Returns:
        The registered web service or None if registration fails
    """
    try:
        app = FastAPI()
        
        # Get the HTML file path
        html_file_path = Path(__file__).parent / "admin_terminal.html"
        
        @app.get("/", response_class=HTMLResponse)
        async def serve_admin_terminal_ui():
            """Serve the admin terminal UI with injected configuration."""
            try:
                if html_file_path.exists():
                    html_content = html_file_path.read_text()
                    # Replace template variables
                    html_content = html_content.replace(
                        "{{ADMIN_SERVICE_ID}}", 
                        admin_service_id
                    )
                    html_content = html_content.replace(
                        "{{SERVER_URL}}", 
                        server_url
                    )
                    html_content = html_content.replace(
                        "{{WORKSPACE}}", 
                        workspace
                    )
                    return html_content
                else:
                    return """
                    <html>
                    <head><title>Admin Terminal Not Found</title></head>
                    <body>
                    <h1>Error: admin_terminal.html not found</h1>
                    <p>The admin terminal UI file is missing.</p>
                    </body>
                    </html>
                    """
            except Exception as e:
                logger.error(f"Failed to serve admin terminal UI: {e}")
                return f"<html><body><h1>Error loading admin terminal UI</h1><p>{str(e)}</p></body></html>"
        
        async def serve_asgi(args, context=None):
            """ASGI handler for the admin terminal web interface."""
            if context:
                logger.debug(f'{context["user"]["id"]} - {args["scope"]["method"]} - {args["scope"]["path"]}')
            await app(args["scope"], args["receive"], args["send"])
        
        # Register the ASGI service as public (authentication will be handled in the frontend)
        web_service = await server.register_service({
            "id": "hypha-admin-terminal",
            "name": "Hypha Admin Terminal Web Interface",
            "type": "asgi",
            "serve": serve_asgi,
            "config": {"visibility": "public", "require_context": False}
        })
        
        logger.info(f"Admin terminal web interface registered at: {server_url}/{workspace}/apps/{web_service.id.split(':')[1]}")
        return web_service
        
    except ImportError as e:
        error_msg = "FastAPI not available, admin terminal web interface requires: pip install fastapi"
        logger.error(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        logger.error(f"Failed to register admin terminal web interface: {e}")
        raise


async def setup_admin_services(store, enable_terminal=False):
    """Setup admin services for the Hypha server.
    
    Args:
        store: The RedisStore instance
        enable_terminal: Whether to enable the admin terminal
        
    Returns:
        The AdminUtilities instance
    """
    admin = AdminUtilities(store)
    
    if enable_terminal:
        admin.setup_terminal()
        logger.info("Admin terminal enabled")
    
    return admin