"""Terminal Worker for executing commands in isolated terminal sessions."""

import asyncio
import httpx
import inspect
import json
import logging
import os
import ptyprocess
import re
import select
import shutil
import signal
import shortuuid
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
    safe_call_callback,
)

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("terminal")
logger.setLevel(LOGLEVEL)


class MessageEmitter:
    """Event emitter for terminal output events."""

    def __init__(self, logger=None):
        """Set up instance."""
        self._event_handlers = {}
        self._logger = logger

    def on(self, event, handler):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def once(self, event, handler):
        """Register an event handler that should only run once."""
        # wrap the handler function
        def wrap_func(*args, **kwargs):
            return handler(*args, **kwargs)
        wrap_func.___event_run_once = True
        self.on(event, wrap_func)

    def off(self, event=None, handler=None):
        """Reset one or all event handlers."""
        if event is None and handler is None:
            self._event_handlers = {}
        elif event is not None and handler is None:
            if event in self._event_handlers:
                self._event_handlers[event] = []
        else:
            if event in self._event_handlers and handler in self._event_handlers[event]:
                self._event_handlers[event].remove(handler)

    def emit(self, event, data=None):
        """Emit an event to all registered handlers."""
        if event in self._event_handlers:
            handlers_to_remove = []
            for handler in self._event_handlers[event]:
                try:
                    ret = handler(data)
                    if inspect.isawaitable(ret):
                        asyncio.ensure_future(ret)
                except Exception as err:
                    if self._logger:
                        self._logger.exception(f"Event handler error for {event}: {err}")
                finally:
                    if hasattr(handler, "___event_run_once"):
                        handlers_to_remove.append(handler)
            
            # Remove one-time handlers
            for handler in handlers_to_remove:
                self._event_handlers[event].remove(handler)

    async def wait_for(self, event, timeout):
        """Wait for an event to be emitted, or timeout."""
        future = asyncio.get_event_loop().create_future()

        def handler(data):
            if not future.done():
                future.set_result(data)

        self.once(event, handler)

        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as err:
            self.off(event, handler)
            raise err


class TerminalScreenRenderer:
    """Simple terminal screen renderer that removes ANSI escape sequences and control codes."""
    
    def __init__(self, width=80, height=24):
        """Initialize the terminal screen renderer.
        
        Args:
            width: Terminal width in characters
            height: Terminal height in lines
        """
        self.width = width
        self.height = height
        # Regex patterns for cleaning terminal output
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        # Updated to catch OSC sequences that may not have \x1b prefix in some cases
        self.osc_escape = re.compile(r'(?:\x1b)?\][^\x07]*\x07')
        # Also catch sequences like "633;C" without proper escape prefix
        self.osc_simple = re.compile(r'\b\d{3};[A-Z]\b')
        self.control_chars = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')
    
    def clean_terminal_output(self, text: str) -> str:
        """Remove ANSI escape sequences and control codes from text.
        
        Args:
            text: Raw terminal output with escape sequences
            
        Returns:
            Cleaned text without escape sequences
        """
        # Remove ANSI escape sequences
        text = self.ansi_escape.sub('', text)
        
        # Remove OSC sequences (like \x1b]633;... or ]633;...)
        text = self.osc_escape.sub('', text)
        
        # Remove simple OSC patterns like "633;C", "633;D;0" etc
        text = self.osc_simple.sub('', text)
        
        # Remove control characters except newlines, tabs, and carriage returns
        text = self.control_chars.sub('', text)
        
        # Clean up any remaining OSC-like patterns (e.g., "633;E;command")
        text = re.sub(r'\d{3};[A-Z];[^\n]*', '', text)
        text = re.sub(r'\d{3};[A-Z]', '', text)
        # Clean up fragments like ";None;0"
        text = re.sub(r';None;?\d*', '', text)
        
        return text
    
    def render_screen(self, raw_output: str) -> str:
        """Render the terminal screen from raw output.
        
        This simulates what you would see when selecting and copying from a terminal.
        
        Args:
            raw_output: Raw terminal output with all escape sequences
            
        Returns:
            Rendered screen content as plain text
        """
        # Clean the output
        cleaned = self.clean_terminal_output(raw_output)
        
        # Split into lines
        lines = cleaned.split('\n')
        
        # Process each line to handle carriage returns and overwrites
        processed_lines = []
        for line in lines:
            # Handle carriage returns within a line
            if '\r' in line:
                # Split by \r and take the last part (simulating overwrite)
                parts = line.split('\r')
                # Keep the last non-empty part
                line = next((p for p in reversed(parts) if p), '')
            
            processed_lines.append(line)
        
        # Join back together
        return '\n'.join(processed_lines)
    
    def extract_visible_output(self, raw_output: str, command: str = None) -> str:
        """Extract only the visible output from a command execution.
        
        This removes the command echo and prompt, returning just the actual output.
        
        Args:
            raw_output: Raw terminal output
            command: The command that was executed (optional)
            
        Returns:
            Just the command output without echo or prompts
        """
        cleaned = self.render_screen(raw_output)
        lines = cleaned.split('\n')
        
        # If we know the command, try to remove its echo
        if command and lines:
            # Remove the first line if it matches or contains the command
            command_stripped = command.strip()
            if lines[0].strip() == command_stripped or command_stripped in lines[0]:
                lines = lines[1:]
        
        # Remove trailing prompt lines and empty lines
        while lines and (
            '>>>' in lines[-1] or 
            lines[-1].strip().endswith('$') or 
            lines[-1].strip().endswith('#') or
            lines[-1].strip() == '' or
            # Also remove lines that are just leftover prompt fragments
            lines[-1].strip() in ['>>>', '>', '$', '#']
        ):
            lines.pop()
        
        # Also remove any leading empty lines
        while lines and lines[0].strip() == '':
            lines.pop(0)
        
        return '\n'.join(lines).strip()


class TerminalWorker(BaseWorker):
    """Terminal worker for executing commands in isolated terminal sessions."""

    instance_counter: int = 0

    def __init__(self, server_url: str = None, use_local_url: Union[bool, str] = False, working_dir: str = None):
        """Initialize the terminal worker.
        
        Args:
            server_url: The Hypha server URL
            use_local_url: Whether to use local URLs for server communication
            working_dir: Base directory for session working directories (defaults to /tmp/hypha_sessions)
        """
        super().__init__()
        self.instance_id = f"terminal-{shortuuid.uuid()}"
        self.controller_id = str(TerminalWorker.instance_counter)
        TerminalWorker.instance_counter += 1
        
        # Terminal configuration constants
        self._READ_BUFFER_SIZE = 65536  # 64KB read buffer
        self._MAX_BUFFER_ENTRIES = 2000  # ~200KB total buffer
        self._MAX_LOG_ENTRIES = 100
        self._ALLOWED_CONTROL_CHARS = {7, 8, 9, 10, 11, 12, 13, 27}  # BEL, BS, TAB, LF, VT, FF, CR, ESC
        
        # convert true/false string to bool, and keep the string if it's not a bool
        if isinstance(use_local_url, str):
            if use_local_url.lower() == "true":
                self._use_local_url = True
            elif use_local_url.lower() == "false":
                self._use_local_url = False
            else:
                self._use_local_url = use_local_url
        else:
            self._use_local_url = use_local_url
        self._server_url = server_url
        
        # Set up working directory base path
        if working_dir:
            self._working_dir_base = Path(working_dir)
        else:
            # Default to /tmp with random subfolder for this worker instance
            self._working_dir_base = Path(tempfile.gettempdir()) / f"hypha_sessions_{shortuuid.uuid()}"
        
        # Ensure base working directory exists
        self._working_dir_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using working directory base: {self._working_dir_base}")

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
        self._session_working_dirs: Dict[str, Path] = {}  # Track working directories per session
        self._capture_tasks: Dict[str, asyncio.Task] = {}  # Background capture tasks for each session
        self._session_emitters: Dict[str, MessageEmitter] = {}  # Event emitters for each session
        
        # Create a screen renderer for each worker instance
        self._screen_renderer = TerminalScreenRenderer()

    def _filter_problematic_characters(self, text: str) -> str:
        """Filter out problematic characters that cause xterm.js parsing errors.
        
        This removes DEL characters and other control characters that can cause
        xterm.js to fail parsing, while preserving essential terminal sequences.
        
        Args:
            text: Raw terminal output text
            
        Returns:
            Filtered text safe for xterm.js
        """
        if not text:
            return text
        
        # Use optimized list comprehension with instance constants
        filtered_chars = [
            char for char in text
            if (ord(char) >= 32 and ord(char) <= 126) or  # Printable ASCII
               (ord(char) in self._ALLOWED_CONTROL_CHARS) or  # Essential control chars
               (ord(char) > 127)                              # Unicode characters
        ]
        
        return ''.join(filtered_chars)

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["terminal"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return "Terminal Worker"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return "A worker for executing commands in isolated terminal sessions"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    @property
    def use_local_url(self) -> Union[bool, str]:
        """Return whether the worker should use local URLs."""
        return self._use_local_url

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
        """Compile terminal application - validate manifest."""
        # Set entry point if not set
        if "entry_point" not in manifest:
            manifest["entry_point"] = "main.sh"

        # Set startup command if not specified
        if "startup_command" not in manifest:
            manifest["startup_command"] = "/bin/bash"

        return manifest, files

    async def _capture_terminal_output(self, session_id: str):
        """Background task to continuously capture and filter terminal output.
        
        This runs independently of any attached clients and ensures we never lose
        terminal output. All output is filtered and buffered for later retrieval
        and emitted as events for attached listeners.
        """
        logger.info(f"Starting capture task for session {session_id}")
        
        session_data = self._session_data.get(session_id)
        if not session_data:
            logger.error(f"No session data found for {session_id}")
            return
            
        terminal_process = session_data.get("process")
        if not terminal_process:
            logger.error(f"No terminal process found for {session_id}")
            return
            
        fd = terminal_process.fd
        emitter = self._session_emitters.get(session_id)
        
        try:
            while session_id in self._sessions and terminal_process.isalive():
                # Non-blocking read with timeout
                ready, _, _ = select.select([fd], [], [], 0.1)
                
                if ready:
                    try:
                        # Read and process output
                        output = os.read(fd, self._READ_BUFFER_SIZE)
                        if output:
                            # Decode and filter output in one step
                            decoded_output = output.decode('utf-8', errors='ignore')
                            filtered_output = self._filter_problematic_characters(decoded_output)
                            
                            # Update buffer atomically
                            async with session_data["buffer_lock"]:
                                session_data["screen_buffer"].append(filtered_output)
                                # Maintain buffer size
                                if len(session_data["screen_buffer"]) > self._MAX_BUFFER_ENTRIES:
                                    session_data["screen_buffer"] = session_data["screen_buffer"][-self._MAX_BUFFER_ENTRIES:]
                            
                            # Emit filtered event to all listeners
                            if emitter and filtered_output:
                                emitter.emit(f"output_{session_id}", {
                                    "type": "update",
                                    "content": filtered_output,
                                    "session_id": session_id,
                                    "timestamp": time.time()
                                })
                            
                            # Maintain logs (using filtered output)
                            session_data["logs"]["stdout"].append(filtered_output)
                            if len(session_data["logs"]["stdout"]) > self._MAX_LOG_ENTRIES:
                                session_data["logs"]["stdout"] = session_data["logs"]["stdout"][-self._MAX_LOG_ENTRIES:]
                                
                    except OSError as e:
                        if e.errno == 5:  # I/O error during terminal resize
                            await asyncio.sleep(0.01)
                        else:
                            logger.debug(f"Terminal read error for {session_id}: {e}")
                            
                # Prevent CPU spinning
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Capture task error for session {session_id}: {e}")
        finally:
            # Notify listeners that session ended
            if emitter:
                emitter.emit(f"session_ended_{session_id}", {"session_id": session_id})
            logger.info(f"Capture task ended for session {session_id}")

    async def _prepare_staged_files(
        self,
        files_to_stage: List[str],
        working_dir: Path,
        config: WorkerConfig,
        progress_callback=None,
    ) -> None:
        """Prepare and download files from artifact manager to the working directory.
        
        Args:
            files_to_stage: List of file paths from artifact manager. Can include:
                - Simple file paths: "data.csv"
                - Folder paths (ending with /): "models/"
                - Renamed files: "source.txt:target.txt"
                - Renamed folders: "source/:target/"
            working_dir: Directory where files should be placed
            config: Worker configuration with server/auth details
            progress_callback: Optional callback for progress updates
            
        Raises:
            WorkerError: If any file or directory cannot be downloaded
        """
        if not files_to_stage:
            return
        
        # Validate files_to_stage format
        if not isinstance(files_to_stage, list):
            raise WorkerError(f"files_to_stage must be a list, got {type(files_to_stage).__name__}")
        
        for item in files_to_stage:
            if not isinstance(item, str):
                raise WorkerError(f"Each item in files_to_stage must be a string, got {type(item).__name__}: {item}")
            if not item or item.strip() == "":
                raise WorkerError("Empty or whitespace-only path in files_to_stage")
            
        await safe_call_callback(progress_callback,
            {"type": "info", "message": f"Preparing {len(files_to_stage)} files/folders from artifact manager..."}
        )
        
        async def download_directory_recursive(client, source_dir, target_dir, processed_dirs=None):
            """Recursively download all files from a directory."""
            if processed_dirs is None:
                processed_dirs = set()
            
            # Avoid infinite recursion
            if source_dir in processed_dirs:
                return
            processed_dirs.add(source_dir)
            
            # Get directory listing - use the files endpoint with trailing slash
            dir_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source_dir}/?use_proxy=true"
            
            try:
                response = await client.get(
                    dir_url,
                    headers={"Authorization": f"Bearer {config.token}"}
                )
                response.raise_for_status()
                items = response.json()
                
                # Process each item in the directory
                for item in items:
                    item_name = item.get("name", "")
                    item_type = item.get("type", "file")
                    
                    if item_type == "directory":
                        # Recursively download subdirectory
                        sub_source = f"{source_dir}/{item_name}".strip("/")
                        sub_target = target_dir / item_name
                        sub_target.mkdir(parents=True, exist_ok=True)
                        await download_directory_recursive(client, sub_source, sub_target, processed_dirs)
                    else:
                        # Download file
                        source_file = f"{source_dir}/{item_name}".strip("/")
                        target_file = target_dir / item_name
                        
                        file_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source_file}?use_proxy=true"
                        file_response = await client.get(
                            file_url,
                            headers={"Authorization": f"Bearer {config.token}"}
                        )
                        file_response.raise_for_status()
                        
                        # Ensure parent directory exists
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            # Write file to working directory
                            target_file.write_bytes(file_response.content)
                            logger.debug(f"Downloaded {source_file} to {target_file}")
                        except OSError as write_error:
                            error_msg = f"Failed to write file {target_file}: {write_error}"
                            logger.error(error_msg)
                            raise WorkerError(error_msg) from write_error
                
                return len(items)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    error_msg = f"Directory not found in artifact manager: {source_dir}"
                    logger.error(error_msg)
                    raise WorkerError(error_msg) from e
                else:
                    raise
        
        async with httpx.AsyncClient(verify=not config.disable_ssl, timeout=30.0) as client:
            for item in files_to_stage:
                # Parse source and target from the item
                if ":" in item:
                    source, target = item.split(":", 1)
                else:
                    source = target = item
                
                # Check if it's a folder (ends with /)
                is_folder = source.endswith("/")
                
                if is_folder:
                    # Handle folder download recursively
                    source = source.rstrip("/")
                    target = target.rstrip("/")
                    
                    await safe_call_callback(progress_callback,
                        {"type": "info", "message": f"Downloading folder recursively: {source} -> {target}"}
                    )
                    
                    # Create target folder
                    target_folder = working_dir / target
                    try:
                        target_folder.mkdir(parents=True, exist_ok=True)
                    except OSError as mkdir_error:
                        error_msg = f"Failed to create directory {target_folder}: {mkdir_error}"
                        logger.error(error_msg)
                        await safe_call_callback(progress_callback,
                            {"type": "error", "message": error_msg}
                        )
                        raise WorkerError(error_msg) from mkdir_error
                    
                    # Recursively download all files
                    file_count = await download_directory_recursive(client, source, target_folder)
                    
                    await safe_call_callback(progress_callback,
                        {"type": "success", "message": f"Downloaded folder {source} ({file_count} items)"}
                    )
                else:
                    # Handle single file download
                    await safe_call_callback(progress_callback,
                        {"type": "info", "message": f"Downloading file: {source} -> {target}"}
                    )
                    
                    file_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source}?use_proxy=true"
                    
                    try:
                        response = await client.get(
                            file_url,
                            headers={"Authorization": f"Bearer {config.token}"}
                        )
                        response.raise_for_status()
                        
                        # Create target file path
                        target_file = working_dir / target
                        
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            # Write file to working directory
                            target_file.write_bytes(response.content)
                            
                            logger.info(f"Downloaded {source} to {target_file}")
                            await safe_call_callback(progress_callback,
                                {"type": "success", "message": f"Downloaded {source}"}
                            )
                        except OSError as write_error:
                            error_msg = f"Failed to write file {target_file}: {write_error}"
                            logger.error(error_msg)
                            await safe_call_callback(progress_callback,
                                {"type": "error", "message": error_msg}
                            )
                            raise WorkerError(error_msg) from write_error
                        
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            error_msg = f"File not found in artifact manager: {source}"
                            logger.error(error_msg)
                            await safe_call_callback(progress_callback,
                                {"type": "error", "message": error_msg}
                            )
                            raise WorkerError(error_msg) from e
                        else:
                            raise

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new terminal session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)

        session_id = config.id
        logger.info(f"Starting terminal session {session_id}")
        
        async def progress_callback(message: dict):
            """Invoke optional progress callback if provided."""
            callback = getattr(config, "progress_callback", None)
            await safe_call_callback(callback, message)

        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")

        # Report initial progress
        await progress_callback(
            {
                "type": "info",
                "message": f"Starting terminal session {session_id}",
            }
        )

        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.manifest.get("type", "terminal"),
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.manifest,
        )

        self._sessions[session_id] = session_info

        try:
            # Create session-specific working directory
            session_working_dir = self._working_dir_base / config.id
            session_working_dir.mkdir(parents=True, exist_ok=True)
            self._session_working_dirs[config.id] = session_working_dir
            logger.info(f"Created session working directory: {session_working_dir}")
            
            # Prepare staged files if specified
            files_to_stage = config.manifest.get("files_to_stage", [])
            if files_to_stage:
                try:
                    await self._prepare_staged_files(
                        files_to_stage,
                        session_working_dir,
                        config,
                        progress_callback
                    )
                except Exception as e:
                    error_msg = f"Failed to prepare staged files: {str(e)}"
                    logger.error(error_msg)
                    await progress_callback(
                        {"type": "error", "message": error_msg}
                    )
                    raise WorkerError(error_msg) from e
            
            # Get startup command from manifest
            startup_command = config.manifest.get("startup_command", "/bin/bash")
            
            # Parse command if it's a string
            if isinstance(startup_command, str):
                import shlex
                startup_command = shlex.split(startup_command)
            
            await progress_callback(
                {
                    "type": "info",
                    "message": f"Spawning terminal with command: {' '.join(startup_command)}",
                }
            )

            # Create a new pseudo-terminal with working directory
            try:
                child = ptyprocess.PtyProcess.spawn(startup_command, cwd=str(session_working_dir))
            except Exception as e:
                raise WorkerError(f"Failed to spawn terminal process: {e}")

            # Store session data with terminal size
            self._session_data[session_id] = {
                "process": child,
                "created_at": time.time(),
                "screen_buffer": [],
                "buffer_lock": asyncio.Lock(),  # Lock for thread-safe buffer access
                "startup_command": startup_command,
                "terminal_size": {"rows": 24, "cols": 80},  # Default size
                "logs": {
                    "stdout": [],
                    "stderr": [],
                    "info": [f"Terminal session started with command: {' '.join(startup_command)}"],
                    "error": [],
                },
            }

            # Create event emitter for this session
            self._session_emitters[session_id] = MessageEmitter(logger=logger)

            # Update session status
            session_info.status = SessionStatus.RUNNING
            
            # Start the background capture task for this session
            capture_task = asyncio.create_task(self._capture_terminal_output(session_id))
            self._capture_tasks[session_id] = capture_task
            logger.info(f"Started background capture task for session {session_id}")

            await progress_callback(
                {
                    "type": "success",
                    "message": f"Terminal session {session_id} started successfully",
                }
            )

            logger.info(f"Started terminal session {session_id}")
            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)

            await progress_callback(
                {
                    "type": "error",
                    "message": f"Failed to start terminal session: {str(e)}",
                }
            )

            logger.error(f"Failed to start terminal session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def execute_stream(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Execute a command and yield output as an async generator.
        
        This is a generator version of execute that yields output chunks as they arrive,
        similar to subprocess streaming. It's useful for long-running commands where you
        want to process output incrementally.
        
        Args:
            session_id: The session to execute in
            script: The command to execute
            config: Optional execution configuration with keys:
                - timeout: Max time to wait for command completion (default: 30.0)
                - raw_output: Return raw output with ANSI codes (default: False)
                - chunk_size: Max bytes to read per chunk (default: 4096)
            context: Optional context information
            
        Yields:
            Dictionaries with output chunks:
                - type: "output" for stdout data, "error" for errors, "complete" when done
                - data: The output text (for "output" type)
                - status: Final status (for "complete" type)
                
        Example:
            async for chunk in worker.execute_stream(session_id, "ls -la"):
                if chunk["type"] == "output":
                    print(chunk["data"], end="")
                elif chunk["type"] == "complete":
                    print(f"\\nCommand completed with status: {chunk['status']}")
        """
        if session_id not in self._sessions:
            yield {"type": "error", "message": f"Terminal session {session_id} not found"}
            return

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("process"):
            yield {"type": "error", "message": f"No terminal process available for session {session_id}"}
            return

        terminal_process = session_data["process"]
        
        # Check if process is still alive
        if not terminal_process.isalive():
            yield {"type": "error", "message": f"Terminal process for session {session_id} is not alive"}
            return

        try:
            # Get configuration
            timeout = config.get("timeout", 30.0) if config else 30.0
            raw_output = config.get("raw_output", False) if config else False
            chunk_size = config.get("chunk_size", 4096) if config else 4096
            
            # Capture the screen state before command execution
            screen_before = ''.join(session_data.get("screen_buffer", []))
            
            # Clear any pending output first
            fd = terminal_process.fd
            while True:
                ready, _, _ = select.select([fd], [], [], 0.01)
                if ready:
                    try:
                        os.read(fd, chunk_size)
                    except OSError:
                        break
                else:
                    break
            
            # Write the command to the terminal
            terminal_process.write(script.encode())
            if not script.endswith('\n'):
                terminal_process.write(b'\n')
            
            # Read output in chunks
            start_time = time.time()
            command_seen = False
            prompt_count = 0
            full_output = []
            
            while time.time() - start_time < timeout:
                # Use select for non-blocking read
                ready, _, _ = select.select([fd], [], [], 0.1)
                
                if ready:
                    try:
                        output = os.read(fd, chunk_size)
                        if output:
                            decoded_output = output.decode('utf-8', errors='ignore')
                            full_output.append(decoded_output)
                            
                            # Store in screen buffer
                            async with session_data["buffer_lock"]:
                                session_data["screen_buffer"].append(decoded_output)
                                if len(session_data["screen_buffer"]) > 1000:
                                    session_data["screen_buffer"] = session_data["screen_buffer"][-1000:]
                            
                            # Yield the output chunk
                            if raw_output:
                                yield {"type": "output", "data": decoded_output}
                            else:
                                # Clean the output before yielding
                                cleaned = self._screen_renderer.clean_terminal_output(decoded_output)
                                if cleaned.strip():  # Only yield non-empty cleaned output
                                    yield {"type": "output", "data": cleaned}
                            
                            # Check if we've seen the command echo
                            if not command_seen and script.strip() in decoded_output:
                                command_seen = True
                            
                            # Check for completion patterns
                            if ">>>" in decoded_output:
                                prompt_count += 1
                                if command_seen and prompt_count > 1:
                                    # Give a tiny bit more time to collect any trailing output
                                    await asyncio.sleep(0.05)
                                    ready, _, _ = select.select([fd], [], [], 0.01)
                                    if ready:
                                        try:
                                            extra = os.read(fd, chunk_size)
                                            if extra:
                                                decoded_extra = extra.decode('utf-8', errors='ignore')
                                                full_output.append(decoded_extra)
                                                if raw_output:
                                                    yield {"type": "output", "data": decoded_extra}
                                                else:
                                                    cleaned = self._screen_renderer.clean_terminal_output(decoded_extra)
                                                    if cleaned.strip():
                                                        yield {"type": "output", "data": cleaned}
                                        except OSError:
                                            pass
                                    break
                            
                            # For shell, look for $ or # prompt
                            elif ('$' in decoded_output or '#' in decoded_output) and command_seen:
                                lines = decoded_output.split('\n')
                                for line in lines:
                                    if line.strip().endswith('$') or line.strip().endswith('#'):
                                        # This looks like a prompt, we're done
                                        break
                                else:
                                    continue
                                break
                                
                    except OSError as e:
                        yield {"type": "error", "message": f"Error reading from terminal: {e}"}
                        break
                else:
                    # No data ready - if we've seen output after the command, we might be done
                    if command_seen and len(full_output) > 1:
                        # Give one more chance to read
                        await asyncio.sleep(0.1)
                        ready, _, _ = select.select([fd], [], [], 0.01)
                        if ready:
                            try:
                                output = os.read(fd, chunk_size)
                                if output:
                                    decoded_output = output.decode('utf-8', errors='ignore')
                                    full_output.append(decoded_output)
                                    if raw_output:
                                        yield {"type": "output", "data": decoded_output}
                                    else:
                                        cleaned = self._screen_renderer.clean_terminal_output(decoded_output)
                                        if cleaned.strip():
                                            yield {"type": "output", "data": cleaned}
                            except OSError:
                                pass
                        break
            
            # Update logs
            combined_output = ''.join(full_output)
            logs = session_data.get("logs", {})
            logs.setdefault("stdout", []).append(combined_output)
            
            # Yield completion status
            yield {"type": "complete", "status": "ok"}
            
        except Exception as e:
            error_msg = f"Failed to execute command: {str(e)}"
            logger.error(f"Failed to execute command in session {session_id}: {e}")
            yield {"type": "error", "message": error_msg}

    async def execute(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Any] = None,
        output_callback: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a command in the running terminal session.
        
        Args:
            session_id: The session to execute in
            script: The command to execute
            config: Optional execution configuration
            progress_callback: Optional callback for execution progress
            output_callback: Optional callback for output
            context: Optional context information
            
        Returns:
            Dictionary containing execution results including screen changes
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("process"):
            raise WorkerError(f"No terminal process available for session {session_id}")

        terminal_process = session_data["process"]
        
        # Check if process is still alive
        if not terminal_process.isalive():
            raise WorkerError(f"Terminal process for session {session_id} is not alive")

        await safe_call_callback(progress_callback,
            {"type": "info", "message": "Executing command in terminal..."}
        )

        try:
            # Capture the screen state before command execution
            screen_before = ''.join(session_data.get("screen_buffer", []))
            
            # Clear any pending output first
            fd = terminal_process.fd
            while True:
                ready, _, _ = select.select([fd], [], [], 0.01)
                if ready:
                    try:
                        os.read(fd, 4096)
                    except OSError:
                        break
                else:
                    break
            
            # Write the command to the terminal
            terminal_process.write(script.encode())
            if not script.endswith('\n'):
                terminal_process.write(b'\n')
            
            # Read the output with a timeout
            timeout = config.get("timeout", 5.0) if config else 5.0
            output_lines = []
            start_time = time.time()
            command_seen = False
            prompt_count = 0
            
            while time.time() - start_time < timeout:
                # Use select for non-blocking read
                ready, _, _ = select.select([fd], [], [], 0.1)
                
                if ready:
                    try:
                        output = os.read(fd, 4096)
                        if output:
                            decoded_output = output.decode('utf-8', errors='ignore')
                            output_lines.append(decoded_output)
                            
                            # Store in screen buffer
                            session_data["screen_buffer"].append(decoded_output)
                            if len(session_data["screen_buffer"]) > 1000:
                                session_data["screen_buffer"] = session_data["screen_buffer"][-1000:]
                            
                            # Send to output callback if provided
                            if output_callback:
                                await safe_call_callback(output_callback, {
                                    "type": "stream",
                                    "name": "stdout",
                                    "text": decoded_output
                                })
                            
                            # Check if we've seen the command echo
                            if not command_seen and script.strip() in decoded_output:
                                command_seen = True
                            
                            # For Python interactive mode, look for >>> prompt after seeing output
                            if ">>>" in decoded_output:
                                prompt_count += 1
                                # If we've seen the command and now see a prompt again, we're done
                                if command_seen and prompt_count > 1:
                                    # Give a tiny bit more time to collect any trailing output
                                    await asyncio.sleep(0.05)
                                    # Read any remaining output
                                    ready, _, _ = select.select([fd], [], [], 0.01)
                                    if ready:
                                        try:
                                            extra = os.read(fd, 4096)
                                            if extra:
                                                decoded_extra = extra.decode('utf-8', errors='ignore')
                                                output_lines.append(decoded_extra)
                                                session_data["screen_buffer"].append(decoded_extra)
                                                if output_callback:
                                                    await safe_call_callback(output_callback, {
                                                        "type": "stream",
                                                        "name": "stdout",
                                                        "text": decoded_extra
                                                    })
                                        except OSError:
                                            pass
                                    break
                            
                            # For shell, look for $ or # prompt
                            elif ('$' in decoded_output or '#' in decoded_output) and command_seen:
                                # Check if this looks like a shell prompt at the end
                                lines = decoded_output.split('\n')
                                for line in lines:
                                    if line.strip().endswith('$') or line.strip().endswith('#'):
                                        # This looks like a prompt, we're done
                                        break
                                else:
                                    continue
                                break
                                
                    except OSError:
                        # No more data available
                        break
                else:
                    # No data ready - if we've seen output after the command, we might be done
                    if command_seen and len(output_lines) > 1:
                        # Give one more chance to read
                        await asyncio.sleep(0.1)
                        ready, _, _ = select.select([fd], [], [], 0.01)
                        if ready:
                            try:
                                output = os.read(fd, 4096)
                                if output:
                                    decoded_output = output.decode('utf-8', errors='ignore')
                                    output_lines.append(decoded_output)
                                    session_data["screen_buffer"].append(decoded_output)
                            except OSError:
                                pass
                        break
            
            # Combine output
            full_output = ''.join(output_lines)
            
            # Check if user wants raw output (default is cleaned/rendered)
            raw_output = config.get("raw_output", False) if config else False
            
            # Capture the screen state after command execution
            screen_after = ''.join(session_data.get("screen_buffer", []))
            
            # Calculate the screen difference (what changed)
            if len(screen_after) > len(screen_before):
                screen_diff_raw = screen_after[len(screen_before):]
            else:
                # If screen was cleared or scrolled, just use the full output
                screen_diff_raw = full_output
            
            # Update logs
            logs = session_data.get("logs", {})
            logs.setdefault("stdout", []).append(full_output)
            
            await safe_call_callback(progress_callback,
                {"type": "success", "message": "Command executed successfully"}
            )
            
            # Prepare outputs based on raw_output config
            outputs = []
            
            if raw_output:
                # Return raw output with all ANSI codes and control sequences
                outputs.append({
                    "type": "stream",
                    "name": "stdout",
                    "text": full_output
                })
                
                # Add raw screen diff
                if screen_diff_raw and screen_diff_raw.strip():
                    outputs.append({
                        "type": "screen",
                        "name": "screen_diff",
                        "text": screen_diff_raw
                    })
            else:
                # Return cleaned/rendered output (default behavior)
                # Extract just the visible output without command echo and prompts
                cleaned_output = self._screen_renderer.extract_visible_output(full_output, script)
                outputs.append({
                    "type": "stream",
                    "name": "stdout",
                    "text": cleaned_output
                })
                
                # Add cleaned screen diff
                if screen_diff_raw and screen_diff_raw.strip():
                    cleaned_diff = self._screen_renderer.extract_visible_output(screen_diff_raw, script)
                    if cleaned_diff:
                        outputs.append({
                            "type": "screen",
                            "name": "screen_diff",
                            "text": cleaned_diff
                        })
            
            return {
                "status": "ok",
                "outputs": outputs
            }
            
        except Exception as e:
            error_msg = f"Failed to execute command: {str(e)}"
            logger.error(f"Failed to execute command in session {session_id}: {e}")
            
            await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
            
            logs = session_data.get("logs", {})
            logs.setdefault("error", []).append(error_msg)
            
            return {
                "status": "error",
                "outputs": [],
                "error": {
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [error_msg]
                }
            }

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a terminal session."""
        if session_id not in self._sessions:
            logger.warning(f"Terminal session {session_id} not found for stopping")
            return

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            # Cancel the capture task first
            capture_task = self._capture_tasks.get(session_id)
            if capture_task and not capture_task.done():
                logger.info(f"Cancelling capture task for session {session_id}")
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup session data
            session_data = self._session_data.get(session_id)
            if session_data:
                # Shutdown the terminal process
                terminal_process = session_data.get("process")
                if terminal_process:
                    logger.info(f"Shutting down terminal process for session {session_id}")
                    try:
                        terminal_process.kill(signal.SIGTERM)
                    except Exception as e:
                        logger.warning(
                            f"Error shutting down terminal for session {session_id}: {e}"
                        )

            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped terminal session {session_id}")

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop terminal session {session_id}: {e}")
            raise
        finally:
            # Cleanup working directory
            session_working_dir = self._session_working_dirs.get(session_id)
            if session_working_dir and session_working_dir.exists():
                try:
                    shutil.rmtree(session_working_dir)
                    logger.info(f"Removed session working directory: {session_working_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove working directory {session_working_dir}: {e}")
            
            # Cleanup session data
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)
            self._session_working_dirs.pop(session_id, None)
            self._capture_tasks.pop(session_id, None)
            self._session_emitters.pop(session_id, None)

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], str]:
        """Get logs for a terminal session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        
        Special type 'screen' returns the rendered screen content (cleaned of ANSI codes) as a string.
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            if type == "screen":
                return ""
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

        # Handle special 'screen' type - return rendered/cleaned screen
        if type == "screen":
            screen_buffer = session_data.get("screen_buffer", [])
            raw_screen = ''.join(screen_buffer)
            # Return the rendered screen (cleaned of ANSI codes)
            return self._screen_renderer.render_screen(raw_screen)

        logs = session_data.get("logs", {})
        
        # Convert logs to items format
        all_items = []
        for log_type, log_entries in logs.items():
            for entry in log_entries:
                all_items.append({"type": log_type, "content": entry})
        
        # Filter by type if specified (excluding 'screen' which is handled above)
        if type:
            filtered_items = [item for item in all_items if item["type"] == type]
        else:
            filtered_items = all_items
        
        total = len(filtered_items)
        
        # Apply pagination
        if limit is None:
            paginated_items = filtered_items[offset:]
        else:
            paginated_items = filtered_items[offset:offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    async def resize_terminal(
        self,
        session_id: str,
        rows: int,
        cols: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resize the terminal window."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("process"):
            raise WorkerError(f"No terminal process available for session {session_id}")

        terminal_process = session_data["process"]
        
        try:
            terminal_process.setwinsize(rows, cols)
            # Store the new size for later restoration
            session_data["terminal_size"] = {"rows": rows, "cols": cols}
            logger.debug(f"Resized terminal {session_id} to {rows}x{cols}")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_terminal(
        self,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read output from the terminal."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("process"):
            raise WorkerError(f"No terminal process available for session {session_id}")

        terminal_process = session_data["process"]
        
        try:
            # Check if process is alive
            if not terminal_process.isalive():
                return {"success": False, "error": "Process not alive"}
            
            # Use select for non-blocking read
            fd = terminal_process.fd
            ready, _, _ = select.select([fd], [], [], 0.1)  # 0.1 second timeout
            
            if ready:
                try:
                    output = os.read(fd, 4096)
                    if output:
                        decoded_output = output.decode('utf-8', errors='ignore')
                        # Store output in screen buffer
                        session_data["screen_buffer"].append(decoded_output)
                        # Keep buffer size reasonable
                        if len(session_data["screen_buffer"]) > 1000:
                            session_data["screen_buffer"] = session_data["screen_buffer"][-1000:]
                        return {"success": True, "output": decoded_output}
                except OSError:
                    pass
            
            return {"success": True, "output": ""}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_screen_content(
        self,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get the accumulated screen buffer content."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {"success": False, "error": "Session data not found"}

        try:
            # Get screen content atomically and filter it
            async with session_data["buffer_lock"]:
                screen_content = ''.join(session_data.get("screen_buffer", []))
            
            # Filter content to prevent xterm.js parsing errors
            filtered_content = self._filter_problematic_characters(screen_content)
            
            return {"success": True, "content": filtered_content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def write_terminal(
        self,
        session_id: str,
        data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write data to the terminal (for interactive input from xterm.js).
        
        Args:
            session_id: The session to write to
            data: The data to write (can be single characters or strings)
            context: Optional context information
            
        Returns:
            Dictionary with success status
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("process"):
            raise WorkerError(f"No terminal process available for session {session_id}")

        terminal_process = session_data["process"]
        
        # Check if process is still alive
        if not terminal_process.isalive():
            raise WorkerError(f"Terminal process for session {session_id} is not alive")

        try:
            # Write data directly to terminal
            # Note: Character filtering happens on output capture, not input
            terminal_process.write(data.encode('utf-8'))
            
            return {"success": True, "bytes_written": len(data)}
        except Exception as e:
            error_msg = f"Failed to write to terminal: {str(e)}"
            logger.error(f"Failed to write to terminal session {session_id}: {e}")
            return {"success": False, "error": error_msg}

    async def attach(
        self,
        session_id: str,
        on_screen_update: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach to a terminal session with a callback for screen updates.
        
        This method will block and continuously listen for terminal output events until:
        - The session is stopped
        - The terminal process dies
        - The callback fails or times out
        
        Args:
            session_id: The session to attach to
            on_screen_update: Callback function for screen updates
            context: Optional context information
            
        Returns:
            Dictionary with final status when attachment ends
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")
        
        session_data = self._session_data.get(session_id)
        if not session_data:
            raise WorkerError(f"Session data not found for {session_id}")
        
        terminal_process = session_data.get("process")
        if not terminal_process:
            raise WorkerError(f"No terminal process for session {session_id}")
        
        emitter = self._session_emitters.get(session_id)
        if not emitter:
            raise WorkerError(f"No event emitter for session {session_id}")
        
        # Get terminal info and initial screen content
        terminal_size = session_data.get("terminal_size", {"rows": 24, "cols": 80})
        
        async with session_data["buffer_lock"]:
            initial_screen = ''.join(session_data.get("screen_buffer", []))
        
        # Filter initial screen content (already filtered at capture, but ensure consistency)
        initial_screen = self._filter_problematic_characters(initial_screen)
        
        # If no callback provided, return initial state
        if not on_screen_update:
            return {
                "success": True,
                "session_id": session_id,
                "initial_screen": initial_screen,
                "terminal_size": terminal_size
            }
        
        # Send initial screen content
        try:
            await safe_call_callback(on_screen_update, {
                "type": "initial",
                "content": initial_screen,
                "session_id": session_id,
                "terminal_size": terminal_size
            })
        except Exception as e:
            logger.warning(f"Failed to send initial screen content: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": f"Callback failed on initial content: {e}"
            }
        
        # Set up attachment for continuous updates
        attachment_id = f"attach_{session_id}_{shortuuid.uuid()}"
        logger.info(f"Starting attach loop for session {session_id} with attachment {attachment_id}")
        
        # Attachment state management
        attachment_active = True
        session_ended = False
        update_queue = asyncio.Queue(maxsize=50)  # Prevent memory issues
        
        def handle_output(data):
            """Handle output events from the terminal."""
            if not attachment_active or not data.get("content"):
                return
            
            try:
                update_queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning(f"Update queue full for attachment {attachment_id}, dropping update")
        
        def handle_session_ended(data):
            """Handle session ended event."""
            nonlocal session_ended
            session_ended = True
            logger.info(f"Session ended event received for {session_id}")
        
        async def process_updates():
            """Process queued updates with intelligent batching."""
            nonlocal attachment_active
            
            while attachment_active and not session_ended:
                try:
                    # Wait for updates with timeout
                    try:
                        first_update = await asyncio.wait_for(update_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    
                    # Batch multiple updates for efficiency
                    updates = [first_update]
                    batch_deadline = time.time() + 0.05  # 50ms batch window
                    
                    # Collect additional updates
                    while time.time() < batch_deadline and len(updates) < 10:
                        try:
                            updates.append(update_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    
                    # Prepare update data
                    if len(updates) == 1:
                        data = updates[0]
                    else:
                        # Combine multiple updates
                        combined_content = ''.join(u.get("content", "") for u in updates if u.get("content"))
                        data = {
                            "type": "update",
                            "content": combined_content,
                            "session_id": session_id,
                            "batched": len(updates)
                        }
                    
                    # Send update with dynamic timeout
                    if data.get("content"):
                        content_size = len(data.get("content", ""))
                        timeout = min(30.0, max(5.0, content_size / 10000))
                        
                        await asyncio.wait_for(
                            safe_call_callback(on_screen_update, data),
                            timeout=timeout
                        )
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Callback timed out for attachment {attachment_id}")
                    attachment_active = False
                except Exception as e:
                    logger.warning(f"Callback failed for attachment {attachment_id}: {e}")
                    attachment_active = False
        
        # Register event handlers
        emitter.on(f"output_{session_id}", handle_output)
        emitter.on(f"session_ended_{session_id}", handle_session_ended)
        
        # Start update processor
        update_processor = asyncio.create_task(process_updates())
        
        try:
            # Keep attachment alive
            while (attachment_active and 
                   session_id in self._sessions and 
                   terminal_process.isalive() and 
                   not session_ended):
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Attach loop error for attachment {attachment_id}: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": f"Attach loop failed: {e}"
            }
        finally:
            # Cleanup
            attachment_active = False
            update_processor.cancel()
            try:
                await update_processor
            except asyncio.CancelledError:
                pass
            
            emitter.off(f"output_{session_id}", handle_output)
            emitter.off(f"session_ended_{session_id}", handle_session_ended)
            logger.info(f"Attach loop ended for attachment {attachment_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "detached": True
        }

    async def list_sessions(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """List all active terminal sessions.
        
        Args:
            context: Optional context information
            
        Returns:
            Dictionary containing list of sessions with their metadata
        """
        try:
            sessions = []
            for session_id, session_info in self._sessions.items():
                session_data = self._session_data.get(session_id, {})
                sessions.append({
                    "session_id": session_id,
                    "app_id": session_info.app_id,
                    "status": session_info.status.value,
                    "created_at": session_info.created_at,
                    "workspace": session_info.workspace,
                    "startup_command": session_data.get("startup_command", []),
                    "is_alive": session_data.get("process", {}).isalive() if session_data.get("process") else False
                })
            return {"success": True, "sessions": sessions}
        except Exception as e:
            error_msg = f"Failed to list sessions: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "sessions": []}

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the terminal worker."""
        logger.info("Shutting down terminal worker...")

        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop terminal session {session_id}: {e}")

        logger.info("Terminal worker shutdown complete")

    def get_worker_service(self) -> Dict[str, Any]:
        """Get the service configuration for registration with terminal-specific methods."""
        service_config = super().get_worker_service()
        # Add terminal specific methods
        service_config["resize_terminal"] = self.resize_terminal
        service_config["read_terminal"] = self.read_terminal
        service_config["get_screen_content"] = self.get_screen_content
        service_config["write_terminal"] = self.write_terminal
        service_config["list_sessions"] = self.list_sessions
        service_config["attach"] = self.attach
        service_config["execute_stream"] = self.execute_stream
        return service_config


async def register_web_interface(server, worker_service_id, server_url, workspace):
    """Register the web interface for the terminal worker.
    
    Args:
        server: The Hypha server instance
        worker_service_id: The ID of the registered worker service
        server_url: The server URL to inject into the HTML
        workspace: The workspace name to inject into the HTML
    
    Returns:
        The registered web service or None if registration fails
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        
        app = FastAPI()
        
        # Get the HTML file path relative to this module
        html_file_path = Path(__file__).parent / "terminal.html"
        
        @app.get("/", response_class=HTMLResponse)
        async def serve_terminal_ui():
            """Serve the terminal UI with injected configuration."""
            try:
                if html_file_path.exists():
                    html_content = html_file_path.read_text()
                    # Replace template variables
                    html_content = html_content.replace(
                        "{{TERMINAL_SERVICE_ID}}", 
                        worker_service_id
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
                    <head><title>Terminal UI Not Found</title></head>
                    <body>
                    <h1>Error: terminal.html not found</h1>
                    <p>The terminal UI file is missing.</p>
                    </body>
                    </html>
                    """
            except Exception as e:
                logger.error(f"Failed to serve terminal UI: {e}")
                return f"<html><body><h1>Error loading terminal UI</h1><p>{str(e)}</p></body></html>"
        
        async def serve_asgi(args, context=None):
            """ASGI handler for the terminal web interface."""
            if context:
                logger.debug(f'{context["user"]["id"]} - {args["scope"]["method"]} - {args["scope"]["path"]}')
            await app(args["scope"], args["receive"], args["send"])
        
        # Register the ASGI service
        web_service = await server.register_service({
            "id": "hypha-terminal",
            "name": "Hypha Terminal Web Interface",
            "type": "asgi",
            "serve": serve_asgi,
            "config": {"visibility": "public", "require_context": True}
        })
        
        logger.info(f"Terminal web interface registered at: {server_url}/{workspace}/apps/{web_service.id.split(':')[1]}")
        return web_service
        
    except ImportError as e:
        error_msg = "FastAPI not available, terminal web interface requires: pip install fastapi"
        logger.error(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        logger.error(f"Failed to register terminal web interface: {e}")
        raise


async def hypha_startup(server):
    """Hypha startup function to initialize terminal worker."""
    try:
        # Built-in worker should use local URLs and a specific working directory
        working_dir = os.environ.get("TERMINAL_WORKING_DIR")
        authorized_workspaces = [w.strip() for w in os.environ.get("TERMINAL_AUTHORIZED_WORKSPACES", "").strip().split(",") if w.strip()]
        worker = TerminalWorker(server_url=server.config.local_base_url, use_local_url=True, working_dir=working_dir)
        service = worker.get_worker_service()
        
        # Log what we're registering
        logger.info(f"Registering terminal worker with supported_types: {service.get('supported_types')}")
        
        # If authorized_workspaces is specified, use protected visibility
        # Otherwise, use public visibility for wider access
        if authorized_workspaces:
            service["config"]["visibility"] = "protected"
            service["config"]["authorized_workspaces"] = authorized_workspaces
        else:
            # No authorized workspaces means public access
            service["config"]["visibility"] = "public"
        
        # Register using the standard register_service method
        result = await server.register_service(service)
        logger.info(f"Terminal worker initialized and registered with id: {result.id}, visibility: {service['config']['visibility']}, supported_types: {service.get('supported_types')}")

        # Register web interface if enabled (default: true for built-in workers)
        enable_web = os.environ.get("TERMINAL_ENABLE_WEB", "true").lower() == "true"
        if enable_web:
            try:
                await register_web_interface(
                    server,
                    result.id,
                    server.config.public_base_url,
                    server.config.workspace
                )
            except ImportError:
                # For built-in workers, web interface is optional
                logger.warning("FastAPI not available, terminal web interface disabled. Install with: pip install fastapi")
            except Exception as e:
                # For other errors, still log but don't fail the worker
                logger.error(f"Failed to register terminal web interface: {e}")
                # Optionally re-raise if you want the built-in worker to fail on web interface errors
                # raise
        
        return service
    except Exception as e:
        logger.error(f"Failed to register terminal worker: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function for command line execution."""
    import argparse
    import sys

    def get_env_var(name: str, default: str = None) -> str:
        """Get environment variable with HYPHA_ prefix."""
        return os.environ.get(f"HYPHA_{name.upper()}", default)

    parser = argparse.ArgumentParser(
        description="Hypha Terminal Worker - Execute commands in isolated terminal sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (with HYPHA_ prefix):
  HYPHA_SERVER_URL     Hypha server URL (e.g., https://hypha.aicell.io)
  HYPHA_WORKSPACE      Workspace name (e.g., my-workspace)
  HYPHA_TOKEN          Authentication token
  HYPHA_SERVICE_ID     Service ID for the worker (optional)
  HYPHA_VISIBILITY     Service visibility: public or protected (default: protected)
  TERMINAL_AUTHORIZED_WORKSPACES  Comma-separated list of authorized workspaces (default: root)
  
Examples:
  # Using command line arguments
  python -m hypha.workers.terminal --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN

  # With web interface enabled
  python -m hypha.workers.terminal --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN --enable-web

  # Using environment variables
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  export HYPHA_WORKSPACE=my-workspace
  export HYPHA_TOKEN=your-token-here
  python -m hypha.workers.terminal

  # Mixed usage (command line overrides environment variables)
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  python -m hypha.workers.terminal --workspace my-workspace --token TOKEN --enable-web
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default=get_env_var("SERVER_URL"),
        help="Hypha server URL (default: from HYPHA_SERVER_URL env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_var("WORKSPACE"),
        help="Workspace name (default: from HYPHA_WORKSPACE env var)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=get_env_var("TOKEN"),
        help="Authentication token (default: from HYPHA_TOKEN env var)",
    )
    parser.add_argument(
        "--service-id",
        type=str,
        default=get_env_var("SERVICE_ID"),
        help="Service ID for the worker (default: from HYPHA_SERVICE_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=get_env_var("CLIENT_ID"),
        help="Client ID for the worker (default: from HYPHA_CLIENT_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--visibility",
        type=str,
        choices=["public", "protected"],
        default=get_env_var("VISIBILITY", "protected"),
        help="Service visibility (default: protected, from HYPHA_VISIBILITY env var)",
    )
    parser.add_argument(
        "--disable-ssl",
        action="store_true",
        help="Disable SSL verification (default: false)",
    )
    parser.add_argument(
        "--authorized-workspaces",
        type=str,
        default=get_env_var("TERMINAL_AUTHORIZED_WORKSPACES", "root"),
        help="Comma-separated list of authorized workspaces (default: root)",
    )
    parser.add_argument(
        "--use-local-url",
        default="false",
        help="Use local URLs for server communication (default: false for CLI workers, true for built-in workers)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="Base directory for session working directories (default: /tmp/hypha_sessions_<uuid>)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--enable-web",
        action="store_true",
        default=os.environ.get("TERMINAL_ENABLE_WEB", "false").lower() == "true",
        help="Enable the web interface for terminal access (default: false for CLI)"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.server_url:
        print(
            "Error: --server-url is required (or set HYPHA_SERVER_URL environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.workspace:
        print(
            "Error: --workspace is required (or set HYPHA_WORKSPACE environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.token:
        print(
            "Error: --token is required (or set HYPHA_TOKEN environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(logging.INFO)

    print(f"Starting Hypha Terminal Worker...")
    print(f"  Server URL: {args.server_url}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Client ID: {args.client_id}")
    print(f"  Service ID: {args.service_id}")
    print(f"  Visibility: {args.visibility}")
    print(f"  Use Local URL: {args.use_local_url}")
    print(f"  Working Dir: {args.working_dir or 'Auto-generated in /tmp'}")
    print(f"  Authorized Workspaces: {args.authorized_workspaces}")
    print(f"  Web Interface: {'Enabled' if args.enable_web else 'Disabled'}")

    async def run_worker():
        """Run the terminal worker."""
        try:
            from hypha_rpc import connect_to_server

            # Connect to server
            server = await connect_to_server(
                server_url=args.server_url,
                workspace=args.workspace,
                token=args.token,
                client_id=args.client_id,
                ssl=False if args.disable_ssl else None,
            )

            # Create and register worker
            worker = TerminalWorker(
                server_url=args.server_url, 
                use_local_url=args.use_local_url,
                working_dir=args.working_dir,
            )

            # Get service config and set custom properties
            service_config = worker.get_worker_service()
            if args.service_id:
                service_config["id"] = args.service_id
            # Set visibility in the correct location (inside config)
            service_config["config"]["visibility"] = args.visibility
            if args.authorized_workspaces:
                service_config["config"]["authorized_workspaces"] = args.authorized_workspaces.split(",")

            # Register the service
            print(f"🔄 Registering terminal worker with config:")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Type: {service_config['type']}")
            print(f"   Supported types: {service_config['supported_types']}")
            print(f"   Visibility: {service_config.get('config', {}).get('visibility', 'N/A')}")
            print(f"   Workspace: {args.workspace}")
            
            registration_result = await server.register_service(service_config)
            print(f"   Registered service id: {registration_result.id}")

            print(f"✅ Terminal Worker registered successfully!")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Supported types: {worker.supported_types}")
            print(f"   Visibility: {args.visibility}")
            
            # Register web interface if enabled
            if args.enable_web:
                print(f"")
                print(f"🔄 Registering web interface...")
                try:
                    web_service = await register_web_interface(
                        server,
                        registration_result.id,
                        args.server_url,
                        args.workspace
                    )
                    print(f"✅ Web interface registered successfully!")
                    print(f"   Access URL: {args.server_url}/{args.workspace}/apps/{web_service.id.split('/')[1]}")
                except ImportError as e:
                    print(f"❌ Failed to register web interface: {e}", file=sys.stderr)
                    print(f"   Install FastAPI with: pip install fastapi", file=sys.stderr)
                    raise
                except Exception as e:
                    print(f"❌ Failed to register web interface: {e}", file=sys.stderr)
                    raise
            
            print(f"")
            print(f"Worker is ready to process terminal requests...")
            print(f"Press Ctrl+C to stop the worker.")

            # Keep the worker running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down Terminal Worker...")
                await worker.shutdown()
                print(f"✅ Worker shutdown complete.")

        except Exception as e:
            print(f"❌ Failed to start Terminal Worker: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the worker
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()