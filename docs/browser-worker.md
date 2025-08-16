# Hypha Browser Worker

The Hypha Browser Worker is a standalone service that enables executing web applications in isolated browser environments using Playwright. This worker provides a headless Chromium browser that can run web-based applications such as JavaScript apps, Python web apps (via Pyodide), web workers, and external web applications, all while being deployed separately from the main Hypha server for flexible distributed computing setups.

## Overview

The Browser Worker provides:
- **Isolated browser environments**: Each application runs in a separate browser context
- **Multiple app type support**: JavaScript, Python (web-python), web workers, iframes, and external web apps
- **Screenshot capabilities**: Take screenshots of running applications
- **Caching support**: Optional response caching for improved performance
- **Remote execution**: Can run on different machines from the Hypha server
- **Authentication support**: Handle cookies, localStorage, and authorization headers
- **Docker support**: Easy deployment in containerized environments

### Supported Application Types

The Browser Worker supports the following application types:

- **`web-python`**: Python applications running via Pyodide in the browser
- **`web-worker`**: JavaScript Web Worker applications
- **`window`**: JavaScript applications running in a browser window context
- **`iframe`**: Applications designed to run in iframe containers
- **`hypha`**: Legacy ImJoy plugin format applications
- **`web-app`**: External web applications accessed via URL

## Prerequisites

- Python 3.9 or higher
- `hypha` package installed (`pip install hypha` or `pip install hypha[server]`)
- Playwright browser dependencies (automatically installed with hypha)
- Network access to the Hypha server
- Valid authentication token for the target workspace

## Installation Methods

### Method 1: Manual Python Setup

This method involves setting up your own Python environment and installing the required dependencies.

#### Step 1: Create a Dedicated Environment

```bash
# Create a new Python environment
python -m venv hypha-browser-worker
source hypha-browser-worker/bin/activate  # On Windows: hypha-browser-worker\Scripts\activate

# Install required dependencies
pip install hypha hypha-rpc

# Install Playwright browsers (this may take a few minutes)
playwright install chromium
```

#### Step 2: Verify Installation

```bash
# Check Playwright installation
playwright --version

# Check hypha installation
python -c "import hypha; print(f'Hypha version: {hypha.__version__}')"

# Test browser worker import
python -c "from hypha.workers.browser import BrowserWorker; print('Browser worker available')"
```

#### Step 3: Start the Worker

```bash
# Using command line arguments
python -m hypha.workers.browser \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token your-token-here \
  --service-id my-browser-worker \
  --visibility public

# Or using environment variables
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace
export HYPHA_TOKEN=your-token-here
export HYPHA_SERVICE_ID=my-browser-worker
export HYPHA_VISIBILITY=public
python -m hypha.workers.browser
```

### Method 2: Using Hypha Docker Container

This method uses the official Hypha Docker container which includes all necessary dependencies including Playwright browsers.

#### Step 1: Pull the Docker Image

```bash
# Pull the latest Hypha image
docker pull ghcr.io/amun-ai/hypha:0.20.71

# Or pull a specific version
docker pull ghcr.io/amun-ai/hypha:latest
```

#### Step 2: Run the Browser Worker in Docker

```bash
# Run with command line arguments
docker run -it --rm \
  -e HYPHA_SERVER_URL=https://hypha.aicell.io \
  -e HYPHA_WORKSPACE=my-workspace \
  -e HYPHA_TOKEN=your-token-here \
  -e HYPHA_SERVICE_ID=my-browser-worker \
  -e HYPHA_VISIBILITY=public \
  -e HYPHA_IN_DOCKER=true \
  --shm-size=2gb \
  ghcr.io/amun-ai/hypha:0.20.71 \
  python -m hypha.workers.browser --verbose
```

**Note**: The `--shm-size=2gb` flag is important for browser applications as it increases shared memory size needed by Chromium.

#### Step 3: Docker Compose Setup (Optional)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  browser-worker:
    image: ghcr.io/amun-ai/hypha:0.20.71
    command: python -m hypha.workers.browser --verbose
    environment:
      - HYPHA_SERVER_URL=https://hypha.aicell.io
      - HYPHA_WORKSPACE=my-workspace
      - HYPHA_TOKEN=your-token-here
      - HYPHA_SERVICE_ID=my-browser-worker
      - HYPHA_VISIBILITY=public
      - HYPHA_IN_DOCKER=true
      # Optional: Playwright configuration
      - PLAYWRIGHT_PERMISSIONS=camera,microphone,geolocation
      - PLAYWRIGHT_BYPASS_CSP=false
      - PLAYWRIGHT_IGNORE_HTTPS_ERRORS=false
      - PLAYWRIGHT_LOCALE=en-US
      - PLAYWRIGHT_TIMEZONE_ID=America/New_York
      - PLAYWRIGHT_VIEWPORT_WIDTH=1920
      - PLAYWRIGHT_VIEWPORT_HEIGHT=1080
      - PLAYWRIGHT_COLOR_SCHEME=light
    shm_size: 2gb
    restart: unless-stopped
    volumes:
      # Optional: mount cache directory for browser data
      - ./browser_cache:/tmp/.cache
```

Then start with:

```bash
docker-compose up -d
```

## Configuration Options

The Browser Worker supports various configuration options through command line arguments or environment variables.

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--server-url` | Hypha server URL (e.g., https://hypha.aicell.io) | Yes | - |
| `--workspace` | Workspace name | Yes | - |
| `--token` | Authentication token | Yes | - |
| `--service-id` | Service ID for the worker | No | Auto-generated |
| `--visibility` | Service visibility: `public` or `protected` | No | `protected` |
| `--in-docker` | Set if running in Docker container | No | False |
| `--verbose` / `-v` | Enable verbose logging | No | False |

### Environment Variables

All command line arguments can be set using environment variables with the `HYPHA_` prefix:

| Environment Variable | Equivalent Argument |
|---------------------|-------------------|
| `HYPHA_SERVER_URL` | `--server-url` |
| `HYPHA_WORKSPACE` | `--workspace` |
| `HYPHA_TOKEN` | `--token` |
| `HYPHA_SERVICE_ID` | `--service-id` |
| `HYPHA_VISIBILITY` | `--visibility` |
| `HYPHA_IN_DOCKER` | `--in-docker` |

#### Playwright Browser Configuration

The Browser Worker also supports extensive Playwright-specific configuration through environment variables with the `PLAYWRIGHT_` prefix:

##### Security & CSP Settings

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_BYPASS_CSP` | Bypass Content Security Policy restrictions | `true`, `false` |
| `PLAYWRIGHT_IGNORE_HTTPS_ERRORS` | Ignore HTTPS certificate errors | `true`, `false` |
| `PLAYWRIGHT_ACCEPT_DOWNLOADS` | Allow file downloads (default: false) | `true`, `false` |
| `PLAYWRIGHT_PERMISSIONS` | Comma-separated list of permissions to grant | `camera,microphone,geolocation` |

##### Browser Identity & Localization

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_USER_AGENT` | Custom user agent string | `Mozilla/5.0 (compatible; MyApp/1.0)` |
| `PLAYWRIGHT_LOCALE` | Locale for the browser context | `en-US`, `fr-FR`, `ja-JP` |
| `PLAYWRIGHT_TIMEZONE_ID` | Timezone ID | `America/New_York`, `Europe/London`, `Asia/Tokyo` |

##### Display & Viewport Settings

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_VIEWPORT_WIDTH` | Viewport width in pixels | `1920`, `1366`, `375` |
| `PLAYWRIGHT_VIEWPORT_HEIGHT` | Viewport height in pixels | `1080`, `768`, `812` |
| `PLAYWRIGHT_DEVICE_SCALE_FACTOR` | Device pixel ratio | `1`, `2`, `3` |
| `PLAYWRIGHT_IS_MOBILE` | Enable mobile mode | `true`, `false` |
| `PLAYWRIGHT_HAS_TOUCH` | Enable touch events | `true`, `false` |

##### Accessibility & Preferences

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_COLOR_SCHEME` | Color scheme preference | `dark`, `light`, `no-preference`, `null` |
| `PLAYWRIGHT_REDUCED_MOTION` | Reduced motion preference | `reduce`, `no-preference`, `null` |

##### Geolocation Settings

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_GEOLOCATION_LATITUDE` | Geolocation latitude (requires longitude) | `40.7128`, `-34.6037` |
| `PLAYWRIGHT_GEOLOCATION_LONGITUDE` | Geolocation longitude (requires latitude) | `-74.0060`, `-58.3816` |
| `PLAYWRIGHT_GEOLOCATION_ACCURACY` | Geolocation accuracy in meters (optional) | `10`, `100`, `1000` |

##### Network & Authentication

| Environment Variable | Description | Example Values |
|---------------------|-------------|----------------|
| `PLAYWRIGHT_JAVASCRIPT_ENABLED` | Enable/disable JavaScript | `true`, `false` |
| `PLAYWRIGHT_OFFLINE` | Enable offline mode | `true`, `false` |
| `PLAYWRIGHT_HTTP_USERNAME` | HTTP basic auth username | `myuser` |
| `PLAYWRIGHT_HTTP_PASSWORD` | HTTP basic auth password | `mypassword` |
| `PLAYWRIGHT_EXTRA_HTTP_HEADERS` | Extra HTTP headers as JSON | `{"X-API-Key": "secret", "Custom-Header": "value"}` |
| `PLAYWRIGHT_BASE_URL` | Base URL for relative URLs | `https://api.example.com` |

##### Available Permissions

When using `PLAYWRIGHT_PERMISSIONS`, you can specify any combination of:
- `camera` - Access to camera devices
- `microphone` - Access to microphone devices  
- `geolocation` - Access to geolocation services
- `notifications` - Permission to show notifications
- `background-sync` - Background synchronization
- `persistent-storage` - Persistent storage access
- `accessibility-events` - Accessibility events
- `clipboard-read` - Read from clipboard
- `clipboard-write` - Write to clipboard
- `payment-handler` - Payment handling capabilities

##### Configuration Examples

**Basic Mobile Testing Setup:**
```bash
export PLAYWRIGHT_IS_MOBILE=true
export PLAYWRIGHT_HAS_TOUCH=true
export PLAYWRIGHT_VIEWPORT_WIDTH=375
export PLAYWRIGHT_VIEWPORT_HEIGHT=812
export PLAYWRIGHT_DEVICE_SCALE_FACTOR=3
export PLAYWRIGHT_USER_AGENT="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)"

python -m hypha.workers.browser --verbose
```

**Geolocation-Enabled App (New York City):**
```bash
export PLAYWRIGHT_PERMISSIONS="geolocation"
export PLAYWRIGHT_GEOLOCATION_LATITUDE=40.7128
export PLAYWRIGHT_GEOLOCATION_LONGITUDE=-74.0060
export PLAYWRIGHT_GEOLOCATION_ACCURACY=10

python -m hypha.workers.browser --verbose
```

**Dark Mode with French Localization:**
```bash
export PLAYWRIGHT_COLOR_SCHEME=dark
export PLAYWRIGHT_LOCALE=fr-FR
export PLAYWRIGHT_TIMEZONE_ID=Europe/Paris
export PLAYWRIGHT_REDUCED_MOTION=reduce

python -m hypha.workers.browser --verbose
```

**API Testing with Authentication:**
```bash
export PLAYWRIGHT_BASE_URL=https://api.myapp.com
export PLAYWRIGHT_HTTP_USERNAME=testuser
export PLAYWRIGHT_HTTP_PASSWORD=testpass
export PLAYWRIGHT_EXTRA_HTTP_HEADERS='{"X-API-Version": "v2", "X-Test-Mode": "true"}'

python -m hypha.workers.browser --verbose
```

**Development/Testing Configuration:**
```bash
export PLAYWRIGHT_BYPASS_CSP=true
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=true
export PLAYWRIGHT_ACCEPT_DOWNLOADS=true
export PLAYWRIGHT_PERMISSIONS="camera,microphone,notifications,clipboard-read,clipboard-write"
export PLAYWRIGHT_JAVASCRIPT_ENABLED=true

python -m hypha.workers.browser --verbose
```

**File Download Testing:**
```bash
# Enable downloads for testing file download functionality
export PLAYWRIGHT_ACCEPT_DOWNLOADS=true
export PLAYWRIGHT_PERMISSIONS="persistent-storage"

python -m hypha.workers.browser --verbose
```

### Example Configurations

#### Basic Setup
```bash
python -m hypha.workers.browser \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token abc123...
```

#### Production Setup
```bash
python -m hypha.workers.browser \
  --server-url https://my-hypha-server.com \
  --workspace production \
  --token prod-token-123 \
  --service-id prod-browser-worker-01 \
  --visibility protected \
  --verbose
```

#### Environment Variables Setup
```bash
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace
export HYPHA_TOKEN=your-token-here
export HYPHA_SERVICE_ID=my-browser-worker

python -m hypha.workers.browser --verbose
```

#### Docker Environment Setup
```bash
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace
export HYPHA_TOKEN=your-token-here
export HYPHA_IN_DOCKER=true

# Optional: Configure Playwright settings for production
export PLAYWRIGHT_PERMISSIONS="camera,microphone,geolocation"
export PLAYWRIGHT_BYPASS_CSP=false
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=false
export PLAYWRIGHT_LOCALE=en-US
export PLAYWRIGHT_TIMEZONE_ID=America/New_York
export PLAYWRIGHT_VIEWPORT_WIDTH=1920
export PLAYWRIGHT_VIEWPORT_HEIGHT=1080

docker run -it --rm \
  -e HYPHA_SERVER_URL \
  -e HYPHA_WORKSPACE \
  -e HYPHA_TOKEN \
  -e HYPHA_IN_DOCKER \
  -e PLAYWRIGHT_PERMISSIONS \
  -e PLAYWRIGHT_BYPASS_CSP \
  -e PLAYWRIGHT_IGNORE_HTTPS_ERRORS \
  -e PLAYWRIGHT_LOCALE \
  -e PLAYWRIGHT_TIMEZONE_ID \
  -e PLAYWRIGHT_VIEWPORT_WIDTH \
  -e PLAYWRIGHT_VIEWPORT_HEIGHT \
  --shm-size=2gb \
  ghcr.io/amun-ai/hypha:0.20.71 \
  python -m hypha.workers.browser --verbose
```

## Usage

Once the worker is running, it can execute various types of web applications. Here's how to use it:

### 1. Create a Web Python Application

Create a Python script that runs in the browser via Pyodide:

```python
# main.py
def setup():
    from js import document
    import numpy as np
    
    # Create a simple web interface
    div = document.createElement("div")
    div.innerHTML = """
    <h2>Web Python App</h2>
    <button id="calculate">Calculate</button>
    <div id="result"></div>
    """
    document.body.appendChild(div)
    
    def calculate(event):
        # Perform calculation
        data = np.random.rand(100)
        mean = np.mean(data)
        std = np.std(data)
        
        result_div = document.getElementById("result")
        result_div.innerHTML = f"<p>Mean: {mean:.3f}</p><p>Std: {std:.3f}</p>"
    
    document.getElementById("calculate").addEventListener("click", calculate)

# Auto-run setup when app loads
setup()
```

### 2. Create an Application Manifest

```yaml
# manifest.yaml
name: My Web Python App
type: web-python
version: 1.0.0
description: A Python app running in the browser
entry_point: main.py
dependencies:
  - numpy
```

### 3. Create a JavaScript Web Worker App

```javascript
// worker.js
// This runs in a Web Worker context
self.addEventListener('message', function(e) {
    const data = e.data;
    
    // Perform computation
    const result = data.map(x => x * 2);
    
    // Send result back
    self.postMessage({
        type: 'result',
        data: result
    });
});

// Export API for Hypha
if (typeof api !== 'undefined') {
    api.export({
        name: "My Web Worker",
        process_data: function(data) {
            return new Promise((resolve) => {
                self.postMessage(data);
                self.addEventListener('message', function(e) {
                    if (e.data.type === 'result') {
                        resolve(e.data.data);
                    }
                });
            });
        }
    });
}
```

### 4. Deploy and Execute

```python
# Client code to use the browser worker
import asyncio
from hypha_rpc import connect_to_server

async def main():
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    # Get the browser worker service
    browser_worker = await server.get_service("my-browser-worker")
    
    # Start a session
    session_id = await browser_worker.start({
        "id": "my-session",
        "app_id": "my-web-app",
        "workspace": "my-workspace",
        "client_id": "client-123",
        "server_url": "https://hypha.aicell.io",
        "token": "your-token-here",
        "artifact_id": "my-workspace/my-web-app",
        "manifest": {
            "type": "web-python",
            "name": "My Web Python App",
            "entry_point": "main.py"
        },
        "entry_point": "main.py"
    })
    
    print(f"Browser session started: {session_id}")
    
    # Take a screenshot
    screenshot = await browser_worker.take_screenshot(session_id, format="png")
    with open("screenshot.png", "wb") as f:
        f.write(screenshot)
    
    # Get logs
    logs = await browser_worker.get_logs(session_id)
    print(f"Session logs: {logs}")
    
    # Stop the session
    await browser_worker.stop(session_id)

asyncio.run(main())
```

## Advanced Features

### Startup Scripts

The browser worker supports executing JavaScript startup scripts automatically after the page loads. This is useful for:
- Injecting utility functions
- Setting up event listeners
- Modifying page behavior
- Pre-configuring the application environment

#### Configuration

Add a `startup_script` field to your manifest pointing to a JavaScript file in the artifact manager:

```yaml
# manifest.yaml
name: My App with Startup Script
type: web-app
entry_point: https://example.com
startup_script: scripts/initialize.js  # Path to JS file in artifact manager
```

#### Example Startup Script

```javascript
// scripts/initialize.js
console.log('Startup script executing...');

// Add utility functions to the window
window.utils = {
    getData: async function() {
        const response = await fetch('/api/data');
        return response.json();
    },
    
    formatDate: function(date) {
        return new Intl.DateTimeFormat('en-US').format(date);
    }
};

// Set up event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page fully loaded');
    
    // Auto-fill forms, set defaults, etc.
    const form = document.querySelector('#myForm');
    if (form) {
        form.querySelector('#apiKey').value = localStorage.getItem('apiKey') || '';
    }
});

// Modify page behavior
if (window.location.hostname === 'example.com') {
    // Remove ads or unwanted elements
    document.querySelectorAll('.ad-banner').forEach(el => el.remove());
    
    // Add custom styles
    const style = document.createElement('style');
    style.textContent = `
        body { font-family: 'Monaco', monospace; }
        .highlight { background-color: yellow; }
    `;
    document.head.appendChild(style);
}

// Return status for logging
'Startup script completed successfully';
```

#### Important Notes

1. **Execution Timing**: The startup script runs after the page loads but before the application is considered ready
2. **Error Handling**: Errors in the startup script are logged but won't fail the session
3. **File Location**: The script must be uploaded to the artifact manager alongside your application
4. **Security**: The script runs with full page access, so ensure it's from a trusted source

### Script Execution API

You can execute JavaScript code in a running browser session using the `execute` method:

```python
# Connect to browser worker
browser_worker = await server.get_service("my-browser-worker")

# Start a session
session_id = await browser_worker.start(config)

# Execute JavaScript code
result = await browser_worker.execute(session_id, """
    // Interact with the page
    const data = {
        title: document.title,
        url: window.location.href,
        cookies: document.cookie,
        localStorage: Object.keys(localStorage)
    };
    
    // Perform actions
    document.querySelector('#submit-button')?.click();
    
    // Return data
    return data;
""")

print(f"Page info: {result}")

# Execute more complex operations
await browser_worker.execute(session_id, """
    // Wait for an element to appear
    await new Promise(resolve => {
        const observer = new MutationObserver((mutations, obs) => {
            if (document.querySelector('#dynamic-content')) {
                obs.disconnect();
                resolve();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    });
    
    // Extract and return the content
    return document.querySelector('#dynamic-content').textContent;
""")
```

### Screenshot Capabilities

The browser worker can take screenshots of running applications:

```python
# Take a PNG screenshot
screenshot_png = await browser_worker.take_screenshot(session_id, format="png")

# Take a JPEG screenshot
screenshot_jpg = await browser_worker.take_screenshot(session_id, format="jpeg")

# Save screenshot
with open("app_screenshot.png", "wb") as f:
    f.write(screenshot_png)
```

### Response Caching

The browser worker supports caching of HTTP responses to improve performance:

```yaml
# In your application manifest
name: My Cached App
type: web-python
enable_cache: true
cache_routes:
  - "*.js"
  - "*.css"
  - "*/api/data/*"
```

### Authentication Support

The browser worker provides comprehensive authentication support with cookies, localStorage, sessionStorage, and HTTP headers that are set BEFORE the page loads:

#### Cookie Configuration

Cookies can be configured in two ways:

**Simple Format (name-value pairs):**
```yaml
# In your application manifest
name: Authenticated App
type: web-app
entry_point: https://my-app.example.com
cookies:
  session_id: "abc123"
  user_token: "xyz789"
```

**Advanced Format (full cookie objects):**
```yaml
# In your application manifest
name: Authenticated App
type: web-app
entry_point: https://my-app.example.com
cookies:
  - name: "session_id"
    value: "abc123"
    domain: "example.com"
    path: "/"
    expires: 1735689600  # Unix timestamp
    httpOnly: true
    secure: true
    sameSite: "Lax"  # Can be: Strict, Lax, or None
  - name: "user_token"
    value: "xyz789"
    domain: ".example.com"  # Subdomain wildcard
    path: "/api"
```

#### Storage Configuration

Both localStorage and sessionStorage are preloaded BEFORE the page loads using initialization scripts:

```yaml
# In your application manifest
name: Storage Example
type: web-app
entry_point: https://my-app.example.com

# localStorage - persists across browser sessions
local_storage:
  api_key: "my-api-key"
  user_preferences: '{"theme": "dark", "language": "en"}'
  auth_token: "stored-token-123"

# sessionStorage - cleared when tab closes
session_storage:
  temp_data: "temporary-value"
  session_id: "current-session-123"
```

#### HTTP Headers Configuration

Custom HTTP headers can be set through the manifest or environment variables:

```yaml
# In your application manifest
name: API Client App
type: web-app
entry_point: https://api.example.com

# Authorization header shorthand
authorization_token: "Bearer jwt-token-here"

# Custom HTTP headers
extra_http_headers:
  X-API-Key: "api-key-123"
  X-Client-Version: "2.0.0"
  Custom-Header: "custom-value"
```

#### Complete Authentication Example

```yaml
# manifest.yaml - Full authentication configuration
name: Enterprise App
type: web-app
entry_point: https://enterprise.example.com

# Cookies with full configuration
cookies:
  - name: "auth_session"
    value: "secure-session-id"
    domain: ".example.com"
    path: "/"
    expires: 1735689600
    httpOnly: true
    secure: true
    sameSite: "Strict"

# Preloaded localStorage
local_storage:
  api_endpoint: "https://api.example.com/v2"
  user_id: "user-123"
  preferences: '{"notifications": true}'

# Preloaded sessionStorage
session_storage:
  csrf_token: "csrf-token-abc"
  temp_state: '{"step": 2, "data": {}}'

# Authorization header
authorization_token: "Bearer eyJhbGciOiJIUzI1NiIs..."

# Additional HTTP headers
extra_http_headers:
  X-API-Version: "2.0"
  X-Request-ID: "request-123"

# Security settings
bypass_csp: false
ignore_https_errors: false
```

#### Important Notes

1. **Storage Preloading**: localStorage and sessionStorage are set using `page.add_init_script()`, ensuring they're available immediately when the page loads
2. **Cookie Domains**: Cookies automatically use the target URL's domain if not specified
3. **Security**: All values are properly escaped to prevent injection attacks
4. **Persistence**: localStorage persists across sessions, sessionStorage is cleared when the tab closes
5. **Order of Operations**: Authentication is set up in this order:
   - Cookies are added to the context
   - HTTP headers are configured
   - Storage initialization scripts are added
   - Page navigation occurs with all auth already in place

## Monitoring and Logging

### Log Levels

- **INFO**: General operational information
- **DEBUG**: Detailed debugging information
- **WARNING**: Warning messages
- **ERROR**: Error messages

### Verbose Output

Enable verbose logging with `--verbose` or `-v`:

```bash
python -m hypha.workers.browser --verbose
```

### Sample Output

```
Starting Hypha Browser Worker...
  Server URL: https://hypha.aicell.io
  Workspace: my-workspace
  Service ID: my-browser-worker
  Visibility: public
  In Docker: false

✅ Browser Worker registered successfully!
   Service ID: my-browser-worker
   Supported types: ['web-python', 'web-worker', 'window', 'iframe', 'hypha', 'web-app']
   Visibility: public

Worker is ready to process browser application requests...
Press Ctrl+C to stop the worker.
```

## Troubleshooting

### Common Issues

#### 1. Playwright Not Found
```
Error: Playwright not properly installed
```
**Solution**: Install Playwright browsers
```bash
playwright install chromium
```

#### 2. Shared Memory Issues in Docker
```
Error: Page crashed
```
**Solution**: Increase shared memory size
```bash
docker run --shm-size=2gb ...
```

#### 3. Permission Errors in Docker
```
Error: Permission denied accessing browser
```
**Solution**: Run with proper flags
```bash
docker run -e HYPHA_IN_DOCKER=true ...
```

#### 4. Browser Launch Failures
```
Error: Failed to launch browser
```
**Solution**: Check system dependencies
```bash
# On Ubuntu/Debian
sudo apt-get install -y \
    libnss3 \
    libnspr4 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0
```

#### 5. Network Connection Issues
```
Failed to connect to server
```
**Solution**: 
- Verify server URL is correct
- Check network connectivity
- Ensure token is valid and not expired

### Performance Tips

1. **Use caching**: Enable response caching for frequently accessed resources
2. **Optimize viewport**: Set appropriate viewport size for your applications
3. **Resource management**: Monitor memory usage, especially for long-running sessions
4. **Docker optimization**: Use `--shm-size=2gb` or higher for better performance

### Security Architecture

The Browser Worker implements multiple layers of security to ensure safe execution of web applications in multi-tenant environments:

#### 1. Browser Context Isolation

Each application session runs in a **completely isolated browser context**, which provides:

- **Separate storage**: Each context has its own cookies, localStorage, sessionStorage, and IndexedDB
- **Separate cache**: HTTP cache is isolated per context
- **Separate permissions**: Each context can have different browser permissions
- **Memory isolation**: JavaScript heap and execution contexts are completely separated
- **No cross-context access**: Applications in different contexts cannot access each other's data, even if served from the same origin

This is equivalent to running applications in completely separate browser profiles, ensuring strong isolation between different users' applications.

#### 2. CORS and Web Security Configuration

The Browser Worker provides flexible CORS handling to balance functionality with security:

##### Default Configuration (CORS Bypass Enabled)
By default, the browser runs with `--disable-web-security` to allow applications to access external APIs without CORS restrictions:

```bash
# Default behavior - CORS checks disabled
export PLAYWRIGHT_DISABLE_WEB_SECURITY=true  # This is the default
```

This configuration:
- Allows cross-origin requests to any API without CORS headers
- Maintains process isolation with `--site-per-process` for defense in depth
- Keeps browser context isolation intact (critical for multi-tenant security)

##### Strict Security Mode
For production environments requiring strict CORS enforcement:

```bash
# Enable strict CORS checking
export PLAYWRIGHT_DISABLE_WEB_SECURITY=false
```

This configuration:
- Enforces standard browser CORS policies
- Requires proper CORS headers from servers
- Provides maximum security but may limit API access

#### 3. Process Isolation

The browser maintains process-level isolation using `--site-per-process`:
- Different origins run in separate OS processes
- Provides defense against Spectre-style attacks
- Adds an additional security boundary beyond context isolation

#### 4. Application-Level Security Controls

##### Per-Application HTTP Headers
Applications can configure custom HTTP headers through the manifest:

```yaml
# In application manifest
extra_http_headers:
  X-API-Key: "your-api-key"
  Authorization: "Bearer token"
  Custom-Header: "value"
```

##### Content Security Policy (CSP)
Control CSP enforcement per application:

```yaml
# In application manifest
bypass_csp: false  # Enforce CSP (default)
# or
bypass_csp: true   # Bypass CSP for development/testing
```

##### HTTPS Certificate Validation
Configure certificate validation per application:

```yaml
# In application manifest
ignore_https_errors: false  # Validate certificates (default)
# or
ignore_https_errors: true   # Skip certificate validation for testing
```

#### 5. Security Configuration Options

| Configuration | Environment Variable | Manifest Field | Description | Security Impact |
|--------------|---------------------|----------------|-------------|-----------------|
| Web Security | `PLAYWRIGHT_DISABLE_WEB_SECURITY` | - | Disable CORS checks | Allows cross-origin API access |
| CSP Bypass | `PLAYWRIGHT_BYPASS_CSP` | `bypass_csp` | Bypass Content Security Policy | Allows inline scripts/styles |
| HTTPS Errors | `PLAYWRIGHT_IGNORE_HTTPS_ERRORS` | `ignore_https_errors` | Ignore certificate errors | Allows self-signed certificates |
| Downloads | `PLAYWRIGHT_ACCEPT_DOWNLOADS` | - | Enable file downloads | Allows saving files locally |
| HTTP Headers | `PLAYWRIGHT_EXTRA_HTTP_HEADERS` | `extra_http_headers` | Custom request headers | Adds authentication/API keys |

#### 6. Best Practices for Different Environments

##### Development Environment
Maximum flexibility for testing and development:

```bash
export PLAYWRIGHT_DISABLE_WEB_SECURITY=true  # Allow cross-origin requests
export PLAYWRIGHT_BYPASS_CSP=true            # Allow inline scripts
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=true   # Allow self-signed certs
export PLAYWRIGHT_ACCEPT_DOWNLOADS=true      # Allow file downloads
```

##### Production Environment
Balanced security with necessary functionality:

```bash
export PLAYWRIGHT_DISABLE_WEB_SECURITY=true  # May be needed for API access
export PLAYWRIGHT_BYPASS_CSP=false           # Enforce CSP
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=false  # Validate certificates
export PLAYWRIGHT_ACCEPT_DOWNLOADS=false     # Restrict downloads
```

##### High-Security Environment
Maximum security with restricted functionality:

```bash
export PLAYWRIGHT_DISABLE_WEB_SECURITY=false # Enforce CORS
export PLAYWRIGHT_BYPASS_CSP=false           # Enforce CSP
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=false  # Validate certificates
export PLAYWRIGHT_ACCEPT_DOWNLOADS=false     # No downloads
export PLAYWRIGHT_PERMISSIONS=""             # No special permissions
```

#### 7. Multi-Tenant Security Guarantees

Even with CORS disabled, the Browser Worker maintains strong isolation between tenants:

1. **Context Isolation**: Each user's application runs in a separate browser context
2. **Storage Isolation**: No shared cookies, localStorage, or cache between contexts
3. **Memory Isolation**: Separate JavaScript execution environments
4. **Network Isolation**: Each context can have different proxy settings and authentication

This architecture ensures that:
- User A's application **cannot** access User B's application data
- Even if both applications are served from the same origin
- Even if web security is disabled for CORS bypass
- Each context acts as a completely independent browser instance

#### 8. Security Considerations

1. **Token Security**: 
   - Store tokens securely using environment variables or secrets management
   - Never commit tokens to version control
   - Rotate tokens regularly

2. **Network Security**: 
   - Always use HTTPS for server connections in production
   - Configure proper TLS certificates
   - Consider using VPN or private networks for sensitive deployments

3. **Resource Limits**: 
   - Monitor CPU and memory usage
   - Set appropriate resource limits in container orchestration
   - Implement rate limiting for API calls

4. **Audit Logging**: 
   - Enable verbose logging for security auditing
   - Monitor for unusual access patterns
   - Keep logs for compliance requirements

5. **Regular Updates**: 
   - Keep Hypha and Playwright updated
   - Apply security patches promptly
   - Monitor security advisories

## API Reference

### Worker Service Methods

#### `start(config: WorkerConfig) -> str`
Start a new browser session.

#### `stop(session_id: str) -> None`
Stop a browser session.

#### `execute(session_id: str, script: str) -> Any`
Execute JavaScript code in a browser session and return the result.

**Parameters:**
- `session_id`: The session ID to execute the script in
- `script`: JavaScript code to execute in the page context

**Returns:**
The result of the script execution (can be any JSON-serializable value)

**Example:**
```python
# Execute script in running session
result = await browser_worker.execute(session_id, """
    // Access page elements and return data
    const title = document.title;
    const links = Array.from(document.querySelectorAll('a')).map(a => a.href);
    return { title, linkCount: links.length };
""")
print(f"Page title: {result['title']}")
print(f"Number of links: {result['linkCount']}")
```

#### `take_screenshot(session_id: str, format: str = "png") -> bytes`
Take a screenshot of a browser session.

#### `get_logs(session_id: str, type: str = None) -> Dict[str, List[str]]`
Get logs for a browser session.

#### `clear_app_cache(workspace: str, app_id: str) -> Dict[str, Any]`
Clear cache for an application.

#### `get_app_cache_stats(workspace: str, app_id: str) -> Dict[str, Any]`
Get cache statistics for an application.

## Advanced Configuration

### Custom Docker Setup

For advanced Docker deployments, you can customize the container:

```dockerfile
FROM ghcr.io/amun-ai/hypha:0.20.71

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    fonts-liberation \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Set custom environment variables
ENV HYPHA_VISIBILITY=protected
ENV HYPHA_IN_DOCKER=true

# Set the default command
CMD ["python", "-m", "hypha.workers.browser", "--verbose"]
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypha-browser-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hypha-browser-worker
  template:
    metadata:
      labels:
        app: hypha-browser-worker
    spec:
      containers:
      - name: browser-worker
        image: ghcr.io/amun-ai/hypha:0.20.71
        command: ["python", "-m", "hypha.workers.browser", "--verbose"]
        env:
        - name: HYPHA_SERVER_URL
          value: "https://hypha.aicell.io"
        - name: HYPHA_WORKSPACE
          value: "production"
        - name: HYPHA_TOKEN
          valueFrom:
            secretKeyRef:
              name: hypha-secrets
              key: token
        - name: HYPHA_VISIBILITY
          value: "protected"
        - name: HYPHA_IN_DOCKER
          value: "true"
        # Optional: Playwright configuration
        - name: PLAYWRIGHT_PERMISSIONS
          value: "camera,microphone"
        - name: PLAYWRIGHT_BYPASS_CSP
          value: "false"
        - name: PLAYWRIGHT_IGNORE_HTTPS_ERRORS
          value: "false"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
```

## Support

For issues and questions:
- GitHub Issues: [https://github.com/amun-ai/hypha/issues](https://github.com/amun-ai/hypha/issues)
- Documentation: [https://amun-ai.github.io/hypha/](https://amun-ai.github.io/hypha/)
- Community: Join our Discord or forum discussions 