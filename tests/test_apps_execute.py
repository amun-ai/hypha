"""Test the execute method in server apps controller."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from hypha.apps import ServerAppController
from hypha.core import UserInfo, UserPermission


@pytest.fixture
def mock_store():
    """Create a mock store."""
    store = Mock()
    store.get_event_bus.return_value = Mock(on_local=Mock(), off_local=Mock())
    store.local_base_url = "http://localhost:9527"
    store.public_base_url = "http://localhost:9527"
    store.register_public_service = Mock()
    store.set_server_app_controller = Mock()
    store.get_root_user = Mock(return_value=Mock(model_dump=Mock(return_value={"id": "root"})))
    return store


@pytest.fixture
def mock_artifact_manager():
    """Create a mock artifact manager."""
    return Mock()


@pytest.fixture
def app_controller(mock_store, mock_artifact_manager):
    """Create a ServerAppController instance."""
    controller = ServerAppController(
        store=mock_store,
        in_docker=False,
        port=9527,
        artifact_manager=mock_artifact_manager
    )
    return controller


@pytest.mark.asyncio
async def test_execute_method_success(app_controller):
    """Test successful execution via the execute method."""
    # Set up mock session with a worker that supports execute
    mock_worker = AsyncMock()
    mock_worker.execute = AsyncMock(return_value={
        "status": "ok",
        "outputs": [
            {"type": "stream", "name": "stdout", "text": "Hello, World!"},
            {"type": "execute_result", "data": {"text/plain": "42"}}
        ],
        "execution_count": 1
    })
    
    session_id = "test-workspace/test-client"
    app_controller._sessions[session_id] = {
        "id": session_id,
        "app_id": "test-app",
        "workspace": "test-workspace",
        "client_id": "test-client",
        "_worker": mock_worker
    }
    
    # Create test context
    context = {
        "ws": "test-workspace",
        "user": {
            "id": "test-user",
            "email": "test@example.com",
            "roles": ["user"]
        }
    }
    
    # Mock permission check
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=True)
    ))
    
    # Execute script
    result = await app_controller.execute(
        session_id=session_id,
        script="print('Hello, World!')",
        config={"timeout": 30.0},
        context=context
    )
    
    # Verify results
    assert result["status"] == "ok"
    assert len(result["outputs"]) == 2
    assert result["outputs"][0]["text"] == "Hello, World!"
    
    # Verify worker was called correctly
    mock_worker.execute.assert_called_once_with(
        session_id,
        script="print('Hello, World!')",
        config={"timeout": 30.0},
        progress_callback=None,
        context=context
    )


@pytest.mark.asyncio
async def test_execute_method_with_progress_callback(app_controller):
    """Test execution with progress callback."""
    # Set up mock session
    mock_worker = AsyncMock()
    mock_worker.execute = AsyncMock(return_value={
        "status": "ok",
        "outputs": [],
        "execution_count": 1
    })
    
    session_id = "test-workspace/test-client"
    app_controller._sessions[session_id] = {
        "id": session_id,
        "_worker": mock_worker
    }
    
    # Create progress callback
    progress_messages = []
    def progress_callback(msg):
        progress_messages.append(msg)
    
    context = {
        "ws": "test-workspace",
        "user": {"id": "test-user", "roles": ["user"]}
    }
    
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=True)
    ))
    
    # Execute with progress callback
    await app_controller.execute(
        session_id=session_id,
        script="x = 1",
        progress_callback=progress_callback,
        context=context
    )
    
    # Verify progress callback was passed through
    mock_worker.execute.assert_called_once()
    call_args = mock_worker.execute.call_args
    assert call_args[1]["progress_callback"] == progress_callback


@pytest.mark.asyncio
async def test_execute_method_permission_denied(app_controller):
    """Test execution denied due to permissions."""
    session_id = "test-workspace/test-client"
    app_controller._sessions[session_id] = {"id": session_id}
    
    context = {
        "ws": "test-workspace",
        "user": {"id": "test-user", "roles": ["viewer"]}
    }
    
    # Mock permission check to return False
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=False),
        id="test-user"
    ))
    
    # Should raise permission error
    with pytest.raises(Exception) as exc_info:
        await app_controller.execute(
            session_id=session_id,
            script="print('test')",
            context=context
        )
    
    assert "does not have permission" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_method_session_not_found(app_controller):
    """Test execution with non-existent session."""
    context = {
        "ws": "test-workspace",
        "user": {"id": "test-user", "roles": ["user"]}
    }
    
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=True)
    ))
    
    # Should raise session not found error
    with pytest.raises(Exception) as exc_info:
        await app_controller.execute(
            session_id="non-existent-session",
            script="print('test')",
            context=context
        )
    
    assert "Server app instance not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_method_worker_not_supported(app_controller):
    """Test execution with worker that doesn't support execute."""
    # Worker without execute method
    mock_worker = Mock(spec=[])  # No execute attribute
    
    session_id = "test-workspace/test-client"
    app_controller._sessions[session_id] = {
        "id": session_id,
        "_worker": mock_worker
    }
    
    context = {
        "ws": "test-workspace",
        "user": {"id": "test-user", "roles": ["user"]}
    }
    
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=True)
    ))
    
    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        await app_controller.execute(
            session_id=session_id,
            script="print('test')",
            context=context
        )
    
    assert "does not support the execute method" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_method_error_handling(app_controller):
    """Test execution error handling."""
    # Worker that returns error status
    mock_worker = AsyncMock()
    mock_worker.execute = AsyncMock(return_value={
        "status": "error",
        "outputs": [
            {"type": "error", "ename": "SyntaxError", "evalue": "invalid syntax"}
        ],
        "error": {
            "ename": "SyntaxError",
            "evalue": "invalid syntax",
            "traceback": ["  File '<stdin>', line 1", "    print("]
        }
    })
    
    session_id = "test-workspace/test-client"
    app_controller._sessions[session_id] = {
        "id": session_id,
        "_worker": mock_worker
    }
    
    context = {
        "ws": "test-workspace",
        "user": {"id": "test-user", "roles": ["user"]}
    }
    
    UserInfo.model_validate = Mock(return_value=Mock(
        check_permission=Mock(return_value=True)
    ))
    
    # Execute invalid code
    result = await app_controller.execute(
        session_id=session_id,
        script="print(",  # Invalid syntax
        context=context
    )
    
    # Should return error result, not raise exception
    assert result["status"] == "error"
    assert result["error"]["ename"] == "SyntaxError"


@pytest.mark.asyncio
async def test_execute_service_api_included(app_controller):
    """Test that execute is included in the service API."""
    service_api = app_controller.get_service_api()
    
    # Verify execute is in the service API
    assert "execute" in service_api
    assert service_api["execute"] == app_controller.execute
    
    # Verify it's placed between take_screenshot and edit_file
    api_keys = list(service_api.keys())
    execute_index = api_keys.index("execute")
    assert api_keys[execute_index - 1] == "take_screenshot"
    assert api_keys[execute_index + 1] == "edit_file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])