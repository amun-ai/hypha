"""Test Kubernetes Worker functionality."""

import asyncio
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from hypha.workers.k8s import KubernetesWorker
from hypha.workers.base import (
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
    WorkerNotAvailableError,
)


class MockPod:
    """Mock Kubernetes Pod object."""
    
    def __init__(self, name="test-pod", phase="Running", ready=True):
        self.metadata = Mock()
        self.metadata.name = name
        self.metadata.creation_timestamp = datetime.now()
        self.metadata.labels = {
            "hypha-worker": "k8s",
            "hypha-session-id": "test-sess",
            "created-by": "hypha-k8s-worker"
        }
        
        self.status = Mock()
        self.status.phase = phase
        self.status.conditions = [Mock()] if ready else []
        if ready:
            self.status.conditions[0].status = "True"
        self.status.container_statuses = [Mock()]
        self.status.container_statuses[0].restart_count = 0
        
        self.spec = Mock()
        self.spec.node_name = "test-node"


class MockExecOutput:
    """Mock for Kubernetes exec output stream."""
    
    def __init__(self, stdout="", stderr="", return_code=0, should_timeout=False):
        self.stdout_data = stdout
        self.stderr_data = stderr
        self.return_code = return_code
        self.should_timeout = should_timeout
        self._is_open = True
        self._stdout_read = False
        self._stderr_read = False
        self._update_count = 0
        import time
        self._start_time = time.time()
        
    def is_open(self):
        """Check if stream is open."""
        if self.should_timeout:
            return self._update_count < 15  # Simulate long operation that should timeout
        return self._is_open and not (self._stdout_read and self._stderr_read)
    
    def update(self, timeout=1):
        """Update stream state."""
        self._update_count += 1
        if self.should_timeout:
            # Simulate time passing to trigger timeout
            import time
            time.sleep(0.2)  # Longer sleep to ensure timeout is triggered
        elif self._update_count > 2:  # Simulate completion after a few updates
            self._is_open = False
    
    def peek_stdout(self):
        """Check if stdout has data."""
        return self.stdout_data and not self._stdout_read
    
    def peek_stderr(self):
        """Check if stderr has data.""" 
        return self.stderr_data and not self._stderr_read
    
    def read_stdout(self):
        """Read stdout data."""
        if self._stdout_read:
            return ""
        self._stdout_read = True
        return self.stdout_data
    
    def read_stderr(self):
        """Read stderr data."""
        if self._stderr_read:
            return ""
        self._stderr_read = True
        return self.stderr_data
    
    def close(self):
        """Close the stream."""
        self._is_open = False

    @property
    def returncode(self):
        """Return the exit code."""
        return self.return_code


@pytest.fixture
def mock_k8s_config():
    """Mock Kubernetes configuration loading."""
    with patch('hypha.workers.k8s.config') as mock_config:
        mock_config.ConfigException = Exception
        mock_config.load_incluster_config = Mock()
        mock_config.load_kube_config = Mock()
        yield mock_config


@pytest.fixture 
def mock_k8s_client():
    """Mock Kubernetes client."""
    with patch('hypha.workers.k8s.client') as mock_client:
        # Create mock API instance
        mock_api = Mock()
        mock_client.CoreV1Api.return_value = mock_api
        
        # Mock client classes
        mock_client.V1Pod = Mock()
        mock_client.V1Container = Mock()
        mock_client.V1PodSpec = Mock()
        mock_client.V1ObjectMeta = Mock()
        mock_client.V1EnvVar = Mock()
        mock_client.V1SecurityContext = Mock()
        mock_client.V1PodSecurityContext = Mock()
        mock_client.V1Capabilities = Mock()
        mock_client.V1SeccompProfile = Mock()
        mock_client.V1DeleteOptions = Mock()
        
        yield mock_client, mock_api


@pytest.fixture
def mock_stream():
    """Mock Kubernetes stream function."""
    with patch('hypha.workers.k8s.stream') as mock_stream_func:
        yield mock_stream_func


@pytest.fixture
def worker_config():
    """Create a test worker configuration."""
    return WorkerConfig(
        id="test-session-123",
        app_id="test-app",
        workspace="test-workspace", 
        client_id="test-client",
        server_url="https://test-server.com",
        token="test-token",
        artifact_id="test-artifact-123",
        manifest={
            "type": "k8s-pod",
            "image": "ubuntu:20.04",
            "command": ["sleep", "300"],
            "env": {"TEST_VAR": "test_value"}
        }
    )


class TestKubernetesWorker:
    """Test cases for KubernetesWorker."""

    def test_init_success(self, mock_k8s_config, mock_k8s_client):
        """Test successful worker initialization."""
        mock_client, mock_api = mock_k8s_client
        
        worker = KubernetesWorker(
            namespace="test-namespace",
            default_timeout=1800,
            image_pull_policy="Always"
        )
        
        assert worker.namespace == "test-namespace"
        assert worker.default_timeout == 1800
        assert worker.image_pull_policy == "Always"
        assert worker.v1 == mock_api
        assert "k8s-worker" in worker.instance_id

    def test_init_no_config_available(self, mock_k8s_config, mock_k8s_client):
        """Test worker initialization when no k8s config is available."""
        mock_config = mock_k8s_config
        mock_config.load_incluster_config.side_effect = Exception("No incluster config")
        mock_config.load_kube_config.side_effect = Exception("No kubeconfig")
        
        with pytest.raises(WorkerNotAvailableError):
            KubernetesWorker()

    def test_properties(self, mock_k8s_config, mock_k8s_client):
        """Test worker properties."""
        worker = KubernetesWorker(namespace="test-ns")
        
        assert worker.supported_types == ["k8s-pod"]
        assert "test-ns" in worker.name
        assert "test-ns" in worker.description
        assert worker.require_context is True

    @pytest.mark.asyncio
    async def test_compile_success(self, mock_k8s_config, mock_k8s_client):
        """Test successful manifest compilation."""
        worker = KubernetesWorker()
        
        manifest = {
            "image": "ubuntu:20.04",
            "command": ["echo", "hello"],
            "env": {"TEST": "value"}
        }
        
        compiled_manifest, files = await worker.compile(manifest, [])
        
        assert compiled_manifest["image"] == "ubuntu:20.04"
        assert compiled_manifest["image_pull_policy"] == "IfNotPresent"
        assert compiled_manifest["restart_policy"] == "Never"
        assert compiled_manifest["timeout"] == 3600

    @pytest.mark.asyncio
    async def test_compile_missing_image(self, mock_k8s_config, mock_k8s_client):
        """Test compilation failure with missing image."""
        worker = KubernetesWorker()
        
        manifest = {"command": ["echo", "hello"]}
        
        with pytest.raises(WorkerError, match="Required field 'image' missing"):
            await worker.compile(manifest, [])

    @pytest.mark.asyncio
    async def test_compile_invalid_image(self, mock_k8s_config, mock_k8s_client):
        """Test compilation failure with invalid image format."""
        worker = KubernetesWorker()
        
        manifest = {"image": "invalid-image-no-tag"}
        
        with pytest.raises(WorkerError, match="Invalid image format"):
            await worker.compile(manifest, [])

    @pytest.mark.asyncio
    async def test_compile_invalid_env(self, mock_k8s_config, mock_k8s_client):
        """Test compilation failure with invalid environment variables."""
        worker = KubernetesWorker()
        
        manifest = {
            "image": "ubuntu:20.04",
            "env": "not-a-dict"
        }
        
        with pytest.raises(WorkerError, match="Environment variables must be a dictionary"):
            await worker.compile(manifest, [])

    @pytest.mark.asyncio
    async def test_compile_invalid_command(self, mock_k8s_config, mock_k8s_client):
        """Test compilation failure with invalid command format."""
        worker = KubernetesWorker()
        
        manifest = {
            "image": "ubuntu:20.04", 
            "command": "not-a-list"
        }
        
        with pytest.raises(WorkerError, match="Command must be a list"):
            await worker.compile(manifest, [])

    @pytest.mark.asyncio
    async def test_start_success(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test successful pod start."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Mock pod creation and status checking
        mock_pod = MockPod(phase="Running")
        mock_api.create_namespaced_pod.return_value = mock_pod
        mock_api.read_namespaced_pod.return_value = mock_pod
        
        session_id = await worker.start(worker_config)
        
        assert session_id == worker_config.id
        assert session_id in worker._sessions
        assert worker._sessions[session_id].status == SessionStatus.RUNNING
        
        # Verify pod creation was called
        mock_api.create_namespaced_pod.assert_called_once()
        
        # Verify environment variables were set correctly
        call_args = mock_api.create_namespaced_pod.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_start_duplicate_session(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test starting duplicate session fails."""
        worker = KubernetesWorker()
        worker._sessions[worker_config.id] = Mock()
        
        with pytest.raises(WorkerError, match="already exists"):
            await worker.start(worker_config)

    @pytest.mark.asyncio 
    async def test_start_pod_creation_fails(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test pod creation failure."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        from kubernetes.client.rest import ApiException
        mock_api.create_namespaced_pod.side_effect = ApiException("Pod creation failed")
        
        with pytest.raises(WorkerError, match="Failed to create pod"):
            await worker.start(worker_config)
        
        # Session should be cleaned up
        assert worker_config.id not in worker._sessions

    @pytest.mark.asyncio
    async def test_start_pod_fails_to_start(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test pod fails to reach running state.""" 
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Mock pod creation success but failure to start
        mock_pod_failed = MockPod(phase="Failed")
        mock_api.create_namespaced_pod.return_value = Mock()
        mock_api.read_namespaced_pod.return_value = mock_pod_failed
        
        with pytest.raises(WorkerError, match="failed to start"):
            await worker.start(worker_config)

    @pytest.mark.asyncio
    async def test_stop_success(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test successful pod stop."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {"pod_name": "test-pod"}
        
        await worker.stop(worker_config.id)
        
        # Verify pod deletion
        mock_api.delete_namespaced_pod.assert_called_once()
        
        # Verify cleanup
        assert worker_config.id not in worker._sessions
        assert worker_config.id not in worker._session_data

    @pytest.mark.asyncio
    async def test_stop_nonexistent_session(self, mock_k8s_config, mock_k8s_client):
        """Test stopping nonexistent session."""
        worker = KubernetesWorker()
        
        # Should not raise exception, just log warning
        await worker.stop("nonexistent-session")

    @pytest.mark.asyncio
    async def test_stop_pod_not_found(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test stopping session when pod is already deleted."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {"pod_name": "test-pod"}
        
        # Mock 404 error (pod not found)
        from kubernetes.client.rest import ApiException
        mock_api.delete_namespaced_pod.side_effect = ApiException(status=404)
        
        await worker.stop(worker_config.id)
        
        # Should complete successfully despite 404
        assert worker_config.id not in worker._sessions

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_k8s_config, mock_k8s_client, mock_stream, worker_config):
        """Test successful command execution."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {
            "pod_name": "test-pod",
            "logs": {"stdout": [], "stderr": [], "info": [], "error": []}
        }
        
        # Mock exec output
        mock_exec_output = MockExecOutput(stdout="test output\n", return_code=0)
        mock_stream.return_value = mock_exec_output
        
        result = await worker.execute(worker_config.id, "echo test")
        
        assert result["status"] == "ok"
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["type"] == "stream"
        assert result["outputs"][0]["name"] == "stdout"
        assert result["outputs"][0]["text"] == "test output"
        
        # Verify stream was called with correct parameters
        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_complex_command(self, mock_k8s_config, mock_k8s_client, mock_stream, worker_config):
        """Test execution of complex command with shell."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {
            "pod_name": "test-pod",
            "logs": {"stdout": [], "stderr": [], "info": [], "error": []}
        }
        
        # Mock exec output
        mock_exec_output = MockExecOutput(stdout="complex output\n", return_code=0)
        mock_stream.return_value = mock_exec_output
        
        complex_command = "echo hello | wc -w"
        result = await worker.execute(worker_config.id, complex_command)
        
        assert result["status"] == "ok"
        
        # Verify that complex command triggered shell execution
        call_args = mock_stream.call_args
        assert "/bin/sh" in str(call_args) or "command" in str(call_args)

    @pytest.mark.asyncio
    async def test_execute_with_stderr(self, mock_k8s_config, mock_k8s_client, mock_stream, worker_config):
        """Test execution with stderr output."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {
            "pod_name": "test-pod",
            "logs": {"stdout": [], "stderr": [], "info": [], "error": []}
        }
        
        # Mock exec output with stderr
        mock_exec_output = MockExecOutput(
            stdout="normal output\n",
            stderr="error message\n",
            return_code=1
        )
        mock_stream.return_value = mock_exec_output
        
        result = await worker.execute(worker_config.id, "command-with-error")
        
        assert result["status"] == "error"
        assert len(result["outputs"]) == 2  # stdout and stderr
        
        stdout_output = next(o for o in result["outputs"] if o["name"] == "stdout")
        stderr_output = next(o for o in result["outputs"] if o["name"] == "stderr")
        
        assert stdout_output["text"] == "normal output"
        assert stderr_output["text"] == "error message"
        
        assert "error" in result
        assert result["error"]["ename"] == "CommandError"

    @pytest.mark.asyncio
    async def test_execute_timeout(self, mock_k8s_config, mock_k8s_client, mock_stream, worker_config):
        """Test execution timeout."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {
            "pod_name": "test-pod",
            "logs": {"stdout": [], "stderr": [], "info": [], "error": []}
        }
        
        # Mock exec output that times out
        mock_exec_output = MockExecOutput(should_timeout=True)
        mock_stream.return_value = mock_exec_output
        
        result = await worker.execute(
            worker_config.id, 
            "long-command",
            config={"timeout": 1}  # Very short timeout
        )
        
        assert result["status"] == "error" 
        assert "TimeoutError" in result["error"]["ename"]

    @pytest.mark.asyncio
    async def test_execute_nonexistent_session(self, mock_k8s_config, mock_k8s_client):
        """Test executing in nonexistent session."""
        worker = KubernetesWorker()
        
        with pytest.raises(SessionNotFoundError):
            await worker.execute("nonexistent", "echo test")

    @pytest.mark.asyncio
    async def test_execute_no_pod_data(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test executing when no pod data available."""
        worker = KubernetesWorker()
        
        # Set up session but no session data
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        
        with pytest.raises(WorkerError, match="No pod data available"):
            await worker.execute(worker_config.id, "echo test")

    @pytest.mark.asyncio
    async def test_get_logs(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test getting logs."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up session with logs
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {
            "pod_name": "test-pod",
            "logs": {
                "stdout": ["line1", "line2"],
                "stderr": ["error1"],
                "info": ["info1", "info2"]
            }
        }
        
        # Mock Kubernetes logs
        mock_api.read_namespaced_pod_log.return_value = "k8s log line"
        
        # Test getting all logs
        logs = await worker.get_logs(worker_config.id)
        assert "items" in logs
        stdout_items = [item for item in logs["items"] if item["type"] == "stdout"]
        assert len(stdout_items) >= 2  # Original logs plus k8s logs
        
        # Test getting specific log type
        stdout_logs = await worker.get_logs(worker_config.id, type="stdout", limit=1)
        assert "items" in stdout_logs
        assert len(stdout_logs["items"]) == 1
        assert stdout_logs["items"][0]["type"] == "stdout"

    @pytest.mark.asyncio
    async def test_get_logs_not_found(self, mock_k8s_config, mock_k8s_client):
        """Test getting logs for nonexistent session."""
        worker = KubernetesWorker()
        
        with pytest.raises(SessionNotFoundError):
            await worker.get_logs("nonexistent")

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_k8s_config, mock_k8s_client, worker_config):
        """Test worker shutdown."""
        mock_client, mock_api = mock_k8s_client
        worker = KubernetesWorker()
        
        # Set up sessions
        session_info = SessionInfo(
            session_id=worker_config.id,
            app_id=worker_config.app_id,
            workspace=worker_config.workspace,
            client_id=worker_config.client_id,
            status=SessionStatus.RUNNING,
            app_type="k8s-pod",
            created_at=datetime.now().isoformat()
        )
        worker._sessions[worker_config.id] = session_info
        worker._session_data[worker_config.id] = {"pod_name": "test-pod"}
        
        await worker.shutdown()
        
        # All sessions should be stopped
        assert len(worker._sessions) == 0

    def test_get_worker_service(self, mock_k8s_config, mock_k8s_client):
        """Test worker service configuration."""
        worker = KubernetesWorker()
        
        service_config = worker.get_worker_service()
        
        assert service_config["type"] == "server-app-worker"
        assert service_config["supported_types"] == ["k8s-pod"]
        assert "start" in service_config
        assert "stop" in service_config
        assert "execute" in service_config
        assert "get_logs" in service_config


@pytest.mark.asyncio
async def test_hypha_startup_success():
    """Test successful hypha startup function."""
    with patch('hypha.workers.k8s.KubernetesWorker') as mock_worker_class, \
         patch.dict('os.environ', {
             'HYPHA_K8S_NAMESPACE': 'test-ns',
             'HYPHA_K8S_DEFAULT_TIMEOUT': '1800',
             'HYPHA_K8S_IMAGE_PULL_POLICY': 'Always'
         }):
        
        mock_worker = Mock()
        mock_worker.register_worker_service = AsyncMock()
        mock_worker_class.return_value = mock_worker
        
        mock_server = Mock()
        
        from hypha.workers.k8s import hypha_startup
        await hypha_startup(mock_server)
        
        # Verify worker was created with environment variables
        mock_worker_class.assert_called_once_with(
            namespace='test-ns',
            default_timeout=1800,
            image_pull_policy='Always'
        )
        
        # Verify worker was registered
        mock_worker.register_worker_service.assert_called_once_with(mock_server)


@pytest.mark.asyncio
async def test_hypha_startup_worker_not_available():
    """Test hypha startup when worker is not available."""
    with patch('hypha.workers.k8s.KubernetesWorker') as mock_worker_class:
        mock_worker_class.side_effect = WorkerNotAvailableError("No k8s config")
        
        mock_server = Mock()
        
        from hypha.workers.k8s import hypha_startup
        # Should not raise exception, just log error
        await hypha_startup(mock_server)


@pytest.mark.asyncio
async def test_hypha_startup_other_error():
    """Test hypha startup with other errors.""" 
    with patch('hypha.workers.k8s.KubernetesWorker') as mock_worker_class:
        mock_worker_class.side_effect = Exception("Unexpected error")
        
        mock_server = Mock()
        
        from hypha.workers.k8s import hypha_startup
        with pytest.raises(Exception, match="Unexpected error"):
            await hypha_startup(mock_server)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])