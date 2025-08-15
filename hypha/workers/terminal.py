"""Terminal Worker for executing commands in isolated terminal sessions."""

import asyncio
import httpx
import json
import logging
import os
import ptyprocess
import re
import select
import shutil
import signal
import shortuuid
import subprocess
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

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("terminal")
logger.setLevel(LOGLEVEL)


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
        
        # Create a screen renderer for each worker instance
        self._screen_renderer = TerminalScreenRenderer()

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

            # Store session data
            self._session_data[session_id] = {
                "process": child,
                "created_at": time.time(),
                "screen_buffer": [],
                "startup_command": startup_command,
                "logs": {
                    "stdout": [],
                    "stderr": [],
                    "info": [f"Terminal session started with command: {' '.join(startup_command)}"],
                    "error": [],
                },
            }

            # Update session status
            session_info.status = SessionStatus.RUNNING

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

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, List[str]], List[str], str]:
        """Get logs for a terminal session.
        
        Special type 'screen' returns the rendered screen content (cleaned of ANSI codes).
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Terminal session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if type is None else [] if type != "screen" else ""

        # Handle special 'screen' type - return rendered/cleaned screen
        if type == "screen":
            screen_buffer = session_data.get("screen_buffer", [])
            raw_screen = ''.join(screen_buffer)
            # Return the rendered screen (cleaned of ANSI codes)
            return self._screen_renderer.render_screen(raw_screen)

        logs = session_data.get("logs", {})

        if type:
            target_logs = logs.get(type, [])
            end_idx = (
                len(target_logs)
                if limit is None
                else min(offset + limit, len(target_logs))
            )
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = (
                    len(log_entries)
                    if limit is None
                    else min(offset + limit, len(log_entries))
                )
                result[log_type_key] = log_entries[offset:end_idx]
            return result

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
            screen_content = ''.join(session_data.get("screen_buffer", []))
            return {"success": True, "content": screen_content}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        return service_config


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

  # Using environment variables
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  export HYPHA_WORKSPACE=my-workspace
  export HYPHA_TOKEN=your-token-here
  python -m hypha.workers.terminal

  # Mixed usage (command line overrides environment variables)
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  python -m hypha.workers.terminal --workspace my-workspace --token TOKEN
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
                service_config["config"]["authorized_workspaces"] = args.authorized_workspaces

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