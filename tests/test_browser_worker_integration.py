"""Comprehensive integration tests for browser worker functionality."""

import pytest
import pytest_asyncio
import asyncio
import base64
from hypha.workers.browser import BrowserWorker
from hypha.workers.base import WorkerConfig, SessionNotFoundError, WorkerError

# Mark all async functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def browser_worker():
    """Create a real browser worker instance."""
    worker = BrowserWorker(in_docker=False)
    await worker.initialize()
    try:
        yield worker
    finally:
        await worker.shutdown()


async def test_localstorage_and_sessionstorage_configuration(browser_worker):
    """Test that localStorage and sessionStorage are properly preloaded."""
    
    # Start a simple HTTP server to serve our test page
    import asyncio
    from aiohttp import web
    
    # Create a simple HTML page
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Storage Test</title></head>
    <body>
        <h1>Storage Test Page</h1>
        <div id="status">Page loaded</div>
    </body>
    </html>
    """
    
    # Create a simple HTTP server
    app = web.Application()
    async def handle_test_page(request):
        return web.Response(text=test_html, content_type='text/html')
    
    app.router.add_get('/test-storage.html', handle_test_page)
    app.router.add_get('/', handle_test_page)  # Also handle root
    
    # Start the server on a free port
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)  # Port 0 = auto-select free port
    await site.start()
    
    # Get the actual port
    port = site._server.sockets[0].getsockname()[1]
    
    try:
        config = WorkerConfig(
            id="test-storage-session",
            app_id="test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://localhost:8000",
            token="test-token",
            artifact_id="test-workspace/test-app",
            manifest={
                "type": "web-app",
                "entry_point": f"http://127.0.0.1:{port}/test-storage.html",
                "local_storage": {
                    "api_key": "test-api-key-123",
                    "user_preferences": '{"theme": "dark", "language": "en"}',
                    "auth_token": "bearer-token-xyz"
                },
                "session_storage": {
                    "temp_session": "session-789",
                    "csrf_token": "csrf-abc-123"
                }
            },
            entry_point="index.html"
        )
    
        # Start the session
        session_id = await browser_worker.start(config)
        assert session_id == "test-storage-session"
        
        # Wait a moment for page to fully load
        await asyncio.sleep(0.5)
        
        # Use execute to validate both storage types
        # With localhost URL, storage should be available and pre-populated
        result = await browser_worker.execute(session_id, """
            (() => {
                // Get all localStorage items
                const localData = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    localData[key] = localStorage.getItem(key);
                }
                
                // Get all sessionStorage items
                const sessionData = {};
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    sessionData[key] = sessionStorage.getItem(key);
                }
                
                return {
                    localStorage: localData,
                    sessionStorage: sessionData
                };
            })()
        """)
        
        # Validate localStorage
        assert result["localStorage"]["api_key"] == "test-api-key-123"
        assert result["localStorage"]["user_preferences"] == '{"theme": "dark", "language": "en"}'
        assert result["localStorage"]["auth_token"] == "bearer-token-xyz"
        
        # Validate sessionStorage
        assert result["sessionStorage"]["temp_session"] == "session-789"
        assert result["sessionStorage"]["csrf_token"] == "csrf-abc-123"
        
        print("✓ localStorage and sessionStorage correctly configured")
        
        # Stop the session
        await browser_worker.stop(session_id)
        
    finally:
        # Clean up the HTTP server
        await runner.cleanup()


async def test_execute_script_functionality(browser_worker):
    """Test the execute method for running JavaScript in browser."""
    
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Execute Test</title></head>
    <body>
        <h1 id="title">Original Title</h1>
        <div id="content">Original Content</div>
        <script>
            window.testData = {value: 42};
        </script>
    </body>
    </html>
    """
    
    data_url = f"data:text/html;base64,{base64.b64encode(test_html.encode()).decode()}"
    
    config = WorkerConfig(
        id="test-execute-session",
        app_id="test-app",
        workspace="test-workspace",
        client_id="test-client",
        server_url="http://localhost:8000",
        token="test-token",
        artifact_id="test-workspace/test-app",
        manifest={
            "type": "web-app",
            "entry_point": data_url
        },
        entry_point="index.html"
    )
    
    # Start the session
    session_id = await browser_worker.start(config)
    assert session_id == "test-execute-session"
    
    # Wait for page to load
    await asyncio.sleep(0.5)
    
    # Execute script to read page data
    result = await browser_worker.execute(session_id, """
        ({
            title: document.getElementById('title').textContent,
            content: document.getElementById('content').textContent,
            testData: window.testData
        })
    """)
    
    assert result["title"] == "Original Title"
    assert result["content"] == "Original Content"
    assert result["testData"]["value"] == 42
    
    # Execute script to modify the page
    await browser_worker.execute(session_id, """
        document.getElementById('title').textContent = 'Modified Title';
        document.getElementById('content').textContent = 'Modified Content';
        window.testData.value = 100;
    """)
    
    # Verify modifications
    result = await browser_worker.execute(session_id, """
        ({
            title: document.getElementById('title').textContent,
            content: document.getElementById('content').textContent,
            testData: window.testData
        })
    """)
    
    assert result["title"] == "Modified Title"
    assert result["content"] == "Modified Content"
    assert result["testData"]["value"] == 100
    
    print("✓ Execute method works correctly")
    
    # Stop the session
    await browser_worker.stop(session_id)


async def test_execute_error_handling(browser_worker):
    """Test error handling in execute method."""
    
    # Test with non-existent session - should raise SessionNotFoundError
    with pytest.raises(SessionNotFoundError) as exc_info:
        await browser_worker.execute("non-existent-session", "console.log('test');")
    assert "Browser session non-existent-session not found" in str(exc_info.value)
    
    # Test with JavaScript error
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Error Test</title></head>
    <body><h1>Error Test</h1></body>
    </html>
    """
    
    data_url = f"data:text/html;base64,{base64.b64encode(test_html.encode()).decode()}"
    
    config = WorkerConfig(
        id="test-error-session",
        app_id="test-app",
        workspace="test-workspace",
        client_id="test-client",
        server_url="http://localhost:8000",
        token="test-token",
        artifact_id="test-workspace/test-app",
        manifest={
            "type": "web-app",
            "entry_point": data_url
        },
        entry_point="index.html"
    )
    
    session_id = await browser_worker.start(config)
    assert session_id == "test-error-session"
    
    await asyncio.sleep(0.5)
    
    # This should raise a WorkerError
    with pytest.raises(WorkerError) as exc_info:
        await browser_worker.execute(session_id, """
            // This will throw an error
            throw new Error('Intentional test error');
        """)
    
    assert "Failed to execute script" in str(exc_info.value)
    print("✓ Error handling works correctly")
    
    # Stop the session
    await browser_worker.stop(session_id)


async def test_screenshot_functionality(browser_worker):
    """Test taking screenshots of browser sessions."""
    
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Screenshot Test</title></head>
    <body style="background: linear-gradient(to right, #667eea 0%, #764ba2 100%); height: 100vh; margin: 0;">
        <h1 style="color: white; text-align: center; padding-top: 40vh;">Screenshot Test Page</h1>
    </body>
    </html>
    """
    
    data_url = f"data:text/html;base64,{base64.b64encode(test_html.encode()).decode()}"
    
    config = WorkerConfig(
        id="test-screenshot-session",
        app_id="test-app",
        workspace="test-workspace",
        client_id="test-client",
        server_url="http://localhost:8000",
        token="test-token",
        artifact_id="test-workspace/test-app",
        manifest={
            "type": "web-app",
            "entry_point": data_url
        },
        entry_point="index.html"
    )
    
    session_id = await browser_worker.start(config)
    assert session_id == "test-screenshot-session"
    
    await asyncio.sleep(0.5)
    
    # Take PNG screenshot
    screenshot_png = await browser_worker.take_screenshot(session_id, format="png")
    assert isinstance(screenshot_png, bytes)
    assert len(screenshot_png) > 1000  # Should be a non-trivial image
    
    # Take JPEG screenshot
    screenshot_jpeg = await browser_worker.take_screenshot(session_id, format="jpeg")
    assert isinstance(screenshot_jpeg, bytes)
    assert len(screenshot_jpeg) > 1000
    
    print("✓ Screenshot functionality works correctly")
    
    # Stop the session
    await browser_worker.stop(session_id)


async def test_multiple_concurrent_sessions(browser_worker):
    """Test running multiple browser sessions concurrently."""
    
    sessions = []
    
    # Start 3 concurrent sessions
    for i in range(3):
        test_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Session {i}</title></head>
        <body>
            <h1 id="title">Session {i}</h1>
            <script>
                window.sessionNumber = {i};
            </script>
        </body>
        </html>
        """
        
        data_url = f"data:text/html;base64,{base64.b64encode(test_html.encode()).decode()}"
        
        config = WorkerConfig(
            id=f"test-multi-session-{i}",
            app_id="test-app",
            workspace="test-workspace",
            client_id=f"test-client-{i}",
            server_url="http://localhost:8000",
            token="test-token",
            artifact_id="test-workspace/test-app",
            manifest={
                "type": "web-app",
                "entry_point": data_url
            },
            entry_point="index.html"
        )
        
        session_id = await browser_worker.start(config)
        assert session_id == f"test-multi-session-{i}"
        sessions.append(session_id)
    
    # Wait for all sessions to load
    await asyncio.sleep(1)
    
    # Verify each session has its own isolated context
    for i, session_id in enumerate(sessions):
        result = await browser_worker.execute(session_id, """
            ({
                sessionNumber: window.sessionNumber,
                title: document.getElementById('title').textContent
            })
        """)
        
        assert result["sessionNumber"] == i
        assert result["title"] == f"Session {i}"
    
    print("✓ Multiple concurrent sessions work correctly")
    
    # Clean up all sessions
    for session_id in sessions:
        await browser_worker.stop(session_id)


async def test_complex_javascript_operations(browser_worker):
    """Test executing complex JavaScript operations."""
    
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Complex Operations</title></head>
    <body>
        <div id="container"></div>
        <script>
            window.initialData = {"numbers": [1, 2, 3, 4, 5]};
        </script>
    </body>
    </html>
    """
    
    data_url = f"data:text/html;base64,{base64.b64encode(test_html.encode()).decode()}"
    
    config = WorkerConfig(
        id="test-complex-ops",
        app_id="test-app",
        workspace="test-workspace",
        client_id="test-client",
        server_url="http://localhost:8000",
        token="test-token",
        artifact_id="test-workspace/test-app",
        manifest={
            "type": "web-app",
            "entry_point": data_url
        },
        entry_point="index.html"
    )
    
    session_id = await browser_worker.start(config)
    assert session_id == "test-complex-ops"
    
    await asyncio.sleep(0.5)
    
    # Execute complex operations
    result = await browser_worker.execute(session_id, """
        (() => {
            // Use window.initialData instead of localStorage
            const data = window.initialData;
            
            // Perform array operations
            const doubled = data.numbers.map(n => n * 2);
            const sum = data.numbers.reduce((a, b) => a + b, 0);
            const filtered = data.numbers.filter(n => n > 2);
            
            // Create DOM elements
            const container = document.getElementById('container');
            for (let i = 0; i < 3; i++) {
                const div = document.createElement('div');
                div.id = 'item-' + i;
                div.textContent = 'Item ' + i;
                container.appendChild(div);
            }
            
            // Count created elements
            const elementCount = container.children.length;
            
            // Create and manipulate objects
            const complexObject = {
                nested: {
                    deep: {
                        value: 'found'
                    }
                },
                array: [{ id: 1 }, { id: 2 }]
            };
            
            // Return comprehensive results
            return {
                arrayOps: {
                    original: data.numbers,
                    doubled: doubled,
                    sum: sum,
                    filtered: filtered
                },
                domOps: {
                    elementCount: elementCount,
                    firstItemText: document.getElementById('item-0')?.textContent
                },
                complexData: {
                    deepValue: complexObject.nested.deep.value,
                    arrayLength: complexObject.array.length
                }
            };
        })()
    """)
    
    # Validate complex operations
    assert result["arrayOps"]["doubled"] == [2, 4, 6, 8, 10]
    assert result["arrayOps"]["sum"] == 15
    assert result["arrayOps"]["filtered"] == [3, 4, 5]
    assert result["domOps"]["elementCount"] == 3
    assert result["domOps"]["firstItemText"] == "Item 0"
    assert result["complexData"]["deepValue"] == "found"
    assert result["complexData"]["arrayLength"] == 2
    
    print("✓ Complex JavaScript operations work correctly")
    
    # Stop the session
    await browser_worker.stop(session_id)


print("Comprehensive browser worker integration tests ready")