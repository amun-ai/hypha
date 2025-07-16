"""Test the new application type system."""

import pytest
from hypha.runner.python_eval import PythonEvalRunner
from hypha.runner.browser import BrowserAppRunner
from hypha.apps import ServerAppController
from hypha.core.store import RedisStore


@pytest.fixture
def python_eval_runner():
    """Create a Python eval runner for testing."""
    # Mock store
    class MockStore:
        def register_public_service(self, service):
            pass
        
        def get_artifact_manager(self):
            return MockArtifactManager()
    
    class MockArtifactManager:
        async def get_file(self, app_id, file_path, version=None, context=None):
            # Return a mock URL that will return Python code
            return "mock://python-code"
    
    store = MockStore()
    runner = PythonEvalRunner(store)
    return runner


@pytest.fixture
def browser_runner():
    """Create a browser runner for testing."""
    # Mock store
    class MockStore:
        def register_public_service(self, service):
            pass
    
    store = MockStore()
    runner = BrowserAppRunner(store)
    return runner


class TestPythonEvalRunner:
    """Test the Python eval runner."""
    
    def test_supported_types(self, python_eval_runner):
        """Test that the Python eval runner supports the correct types."""
        service = python_eval_runner.get_service()
        assert "supported_types" in service
        assert "python-eval" in service["supported_types"]
        assert len(service["supported_types"]) == 1
    
    @pytest.mark.asyncio
    async def test_start_python_eval_session(self, python_eval_runner):
        """Test starting a Python eval session."""
        # Mock the httpx client to return Python code
        import httpx
        from unittest.mock import AsyncMock, patch
        
        python_code = '''
print("Hello from Python eval!")
result = 2 + 2
print(f"2 + 2 = {result}")
'''
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = python_code
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await python_eval_runner.start(
                client_id="test-client",
                app_id="test-app",
                server_url="http://localhost:8080",
                public_base_url="http://localhost:8080",
                local_base_url="http://localhost:8080",
                workspace="test-workspace",
                entry_point="main.py",
                app_type="python-eval",
            )
            
            assert result["session_id"] == "test-workspace/test-client"
            assert result["status"] == "completed"
            assert len(result["logs"]) == 2
            assert "Hello from Python eval!" in result["logs"][0]
            assert "2 + 2 = 4" in result["logs"][1]
    
    @pytest.mark.asyncio
    async def test_python_eval_with_error(self, python_eval_runner):
        """Test Python eval with code that has an error."""
        import httpx
        from unittest.mock import AsyncMock, patch
        
        python_code = '''
print("This will work")
undefined_variable
print("This won't be reached")
'''
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = python_code
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await python_eval_runner.start(
                client_id="test-client",
                app_id="test-app",
                server_url="http://localhost:8080",
                public_base_url="http://localhost:8080",
                local_base_url="http://localhost:8080",
                workspace="test-workspace",
                entry_point="main.py",
                app_type="python-eval",
            )
            
            assert result["session_id"] == "test-workspace/test-client"
            assert result["status"] == "error"
            assert result["error"] is not None
            assert "undefined_variable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_stop_session(self, python_eval_runner):
        """Test stopping a Python eval session."""
        # First start a session
        session_id = "test-workspace/test-client"
        python_eval_runner._eval_sessions[session_id] = {
            "session_id": session_id,
            "status": "completed",
            "logs": ["Test log"],
        }
        
        # Stop the session
        await python_eval_runner.stop(session_id)
        
        # Verify session is removed
        assert session_id not in python_eval_runner._eval_sessions
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, python_eval_runner):
        """Test listing Python eval sessions."""
        # Add some test sessions
        python_eval_runner._eval_sessions["workspace1/client1"] = {
            "session_id": "workspace1/client1",
            "status": "completed",
            "logs": ["Test log 1"],
        }
        python_eval_runner._eval_sessions["workspace2/client2"] = {
            "session_id": "workspace2/client2",
            "status": "completed",
            "logs": ["Test log 2"],
        }
        
        # List sessions for workspace1
        sessions = await python_eval_runner.list("workspace1")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "workspace1/client1"
        
        # List sessions for workspace2
        sessions = await python_eval_runner.list("workspace2")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "workspace2/client2"


class TestBrowserRunner:
    """Test the browser runner."""
    
    def test_supported_types(self, browser_runner):
        """Test that the browser runner supports the correct types."""
        service = browser_runner.get_service()
        assert "supported_types" in service
        expected_types = ["web-python", "web-worker", "window", "iframe"]
        assert service["supported_types"] == expected_types


class TestAppTypeSystem:
    """Test the overall app type system."""
    
    def test_type_filtering(self):
        """Test that runners are filtered by supported types."""
        # Mock runners
        class MockRunner1:
            supported_types = ["web-python", "web-worker"]
        
        class MockRunner2:
            supported_types = ["python-eval"]
        
        class MockRunner3:
            supported_types = ["window", "iframe"]
        
        runners = [MockRunner1(), MockRunner2(), MockRunner3()]
        
        # Test filtering for web-python
        web_python_runners = [r for r in runners if "web-python" in r.supported_types]
        assert len(web_python_runners) == 1
        assert isinstance(web_python_runners[0], MockRunner1)
        
        # Test filtering for python-eval
        python_eval_runners = [r for r in runners if "python-eval" in r.supported_types]
        assert len(python_eval_runners) == 1
        assert isinstance(python_eval_runners[0], MockRunner2)
        
        # Test filtering for window
        window_runners = [r for r in runners if "window" in r.supported_types]
        assert len(window_runners) == 1
        assert isinstance(window_runners[0], MockRunner3)


@pytest.mark.asyncio
async def test_simple_python_eval_use_case():
    """Test a simple Python eval use case."""
    # Create a simple Python code that does basic calculation
    python_code = '''
# Simple calculator
a = 10
b = 20
result = a + b
print(f"The result of {a} + {b} is {result}")

# Test some basic Python features
numbers = [1, 2, 3, 4, 5]
sum_numbers = sum(numbers)
print(f"Sum of {numbers} is {sum_numbers}")
'''
    
    # Mock store and artifact manager
    class MockStore:
        def register_public_service(self, service):
            pass
        
        def get_artifact_manager(self):
            return MockArtifactManager()
    
    class MockArtifactManager:
        async def get_file(self, app_id, file_path, version=None, context=None):
            return "mock://python-code"
    
    # Create runner
    store = MockStore()
    runner = PythonEvalRunner(store)
    
    # Mock the httpx client to return our Python code
    import httpx
    from unittest.mock import AsyncMock, patch
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = python_code
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Start the Python eval session
        result = await runner.start(
            client_id="calc-client",
            app_id="simple-calculator",
            server_url="http://localhost:8080",
            public_base_url="http://localhost:8080",
            local_base_url="http://localhost:8080",
            workspace="test-workspace",
            entry_point="calculator.py",
            app_type="python-eval",
        )
        
        # Verify the results
        assert result["session_id"] == "test-workspace/calc-client"
        assert result["status"] == "completed"
        assert result["error"] is None
        assert len(result["logs"]) == 2
        assert "The result of 10 + 20 is 30" in result["logs"][0]
        assert "Sum of [1, 2, 3, 4, 5] is 15" in result["logs"][1]
        
        # Test getting logs
        logs = await runner.logs(result["session_id"])
        assert "log" in logs
        assert len(logs["log"]) == 2
        
        # Clean up
        await runner.stop(result["session_id"])