# WebRTC Streaming with Hypha

This document covers how to use WebRTC for real-time peer-to-peer communication in Hypha applications. WebRTC enables high-performance streaming of audio, video, and data between clients with minimal latency.

## Client Usage

### Prerequisites

Before using WebRTC features, you need to:

1. **Login to Hypha**: Anonymous users cannot access ICE servers for security reasons
2. **Install WebRTC dependencies**: For Python clients, install `aiortc` and `av`

```bash
pip install aiortc av
```

### Python WebRTC Client

Here's a complete example of streaming video using Python:

```python
import asyncio
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCConfiguration, RTCIceServer
from av import VideoFrame
from hypha_rpc import connect_to_server, login

# Login and connect to Hypha server
token = await login({"server_url": "https://hypha.aicell.io"})
api = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": token
})

class VideoTransformTrack(MediaStreamTrack):
    """Example video track that applies transformations."""
    
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.counter = 0
        
    async def recv(self):
        """Generate or transform video frames."""
        pts, time_base = await self.next_timestamp()
        
        # Create a test pattern
        width, height = 640, 480
        frame_data = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add animated pattern
        self.counter += 1
        color = (self.counter % 255, (self.counter * 2) % 255, (self.counter * 3) % 255)
        frame_data[height//4:3*height//4, width//4:3*width//4] = color
        
        # Create VideoFrame
        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        
        return frame

async def main():
    # Get ICE servers from Hypha
    ice_servers_config = await api.get_rtc_ice_servers()
    
    # Convert to aiortc format
    ice_servers = []
    for server in ice_servers_config:
        ice_servers.append(RTCIceServer(
            urls=server["urls"],
            username=server["username"],
            credential=server["credential"],
            credentialType="password"
        ))
    
    # Create peer connection
    config = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=config)
    
    # Add video track
    video_track = VideoTransformTrack()
    pc.addTrack(video_track)
    
    # Set up event handlers
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
    
    @pc.on("track")
    def on_track(track):
        print(f"Received track: {track.kind}")
    
    # Create and set local description
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    print(f"Offer SDP: {pc.localDescription.sdp}")
    
    # In a real application, you would exchange SDP with the remote peer
    # For example, through Hypha's messaging system

asyncio.run(main())
```

### JavaScript WebRTC Client

Here's the equivalent JavaScript implementation:

```html
<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Streaming Example</title>
    <script src="https://cdn.jsdelivr.net/npm/hypha-rpc@latest/dist/hypha-rpc-websocket.min.js"></script>
</head>
<body>
    <video id="localVideo" autoplay muted width="320" height="240"></video>
    <video id="remoteVideo" autoplay width="320" height="240"></video>
    
    <script>
        async function setupWebRTC() {
            // Login and connect to Hypha
            const token = await hyphaWebsocketClient.login({
                server_url: "https://hypha.aicell.io"
            });
            
            const api = await hyphaWebsocketClient.connectToServer({
                server_url: "https://hypha.aicell.io",
                token: token
            });
            
            // Get ICE servers from Hypha
            const iceServersConfig = await api.get_rtc_ice_servers();
            
            // Convert to WebRTC format
            const iceServers = iceServersConfig.map(server => ({
                urls: server.urls,
                username: server.username,
                credential: server.credential,
                credentialType: "password"
            }));
            
            // Create peer connection
            const pc = new RTCPeerConnection({ iceServers });
            
            // Get user media (camera/microphone)
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            
            // Display local video
            document.getElementById('localVideo').srcObject = stream;
            
            // Add tracks to peer connection
            stream.getTracks().forEach(track => {
                pc.addTrack(track, stream);
            });
            
            // Handle remote stream
            pc.ontrack = (event) => {
                console.log('Received remote track:', event.track.kind);
                const remoteVideo = document.getElementById('remoteVideo');
                remoteVideo.srcObject = event.streams[0];
            };
            
            // Handle connection state changes
            pc.onconnectionstatechange = () => {
                console.log('Connection state:', pc.connectionState);
            };
            
            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            console.log('Offer SDP:', pc.localDescription.sdp);
            
            // In a real application, exchange SDP with remote peer
        }
        
        setupWebRTC().catch(console.error);
    </script>
</body>
</html>
```

### Data Channels

WebRTC also supports data channels for real-time data exchange:

```python
# Python data channel example
@pc.on("datachannel")
def on_datachannel(channel):
    @channel.on("message")
    def on_message(message):
        print(f"Received: {message}")
        # Echo the message back
        channel.send(f"Echo: {message}")

# Create data channel
data_channel = pc.createDataChannel("chat")

@data_channel.on("open")
def on_open():
    data_channel.send("Hello from Python!")
```

## Peer-to-Peer WebRTC Services

Hypha provides built-in support for WebRTC services that enable direct peer-to-peer communication between clients. This is ideal for high-bandwidth applications like video streaming, real-time control, and large data transfers.

### Registering WebRTC Services

Use `register_rtc_service` to create a WebRTC-enabled service:

```python
from hypha_rpc import connect_to_server, register_rtc_service

# Connect and login first
token = await login({"server_url": "https://hypha.aicell.io"})
server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": token
})

# Register a WebRTC service
await register_rtc_service(server, "webrtc-streaming")
```

For more complex services with custom functions:

```python
def stream_video():
    """Generate video frames."""
    # Your video generation logic here
    return {"frame": "video_data", "timestamp": time.time()}

def control_device(command, **params):
    """Control connected devices."""
    if command == "move":
        return {"status": "moved", "position": params}
    elif command == "capture":
        return {"status": "captured", "image": "image_data"}

await register_rtc_service(
    server,
    service_id="microscope-control",
    config={
        "visibility": "public",
        "stream_video": stream_video,
        "control_device": control_device,
        "on_init": lambda pc: print(f"WebRTC connection established: {pc}")
    }
)
```

### Accessing WebRTC Services

Use `get_rtc_service` to connect to WebRTC services:

```python
from hypha_rpc import get_rtc_service

# Connect to the WebRTC service
rtc_service = await get_rtc_service(server, "microscope-control")

# Use the service functions (now via peer-to-peer WebRTC)
frame = await rtc_service.stream_video()
result = await rtc_service.control_device("move", x=100, y=200)
```

### JavaScript WebRTC Services

WebRTC services work seamlessly in JavaScript:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@latest/dist/hypha-rpc-websocket.min.js"></script>
<script>
async function setupWebRTCService() {
    // Connect to server
    const token = await hyphaWebsocketClient.login({
        server_url: "https://hypha.aicell.io"
    });
    
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://hypha.aicell.io",
        token: token
    });
    
    // Get WebRTC service
    const rtcService = await hyphaWebsocketClient.getRTCService(server, "microscope-control");
    
    // Access service functions via peer-to-peer connection
    const frame = await rtcService.stream_video();
    const result = await rtcService.control_device("capture");
    
    console.log("Frame received:", frame);
    console.log("Control result:", result);
}

setupWebRTCService();
</script>
```

### Automatic WebRTC Enable

Enable WebRTC automatically when connecting:

```python
# Python - automatic WebRTC
server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": token,
    "webrtc": True  # Automatically register WebRTC service
})

# All services registered on this server are now WebRTC-enabled
my_service = await server.register_service({
    "id": "auto-webrtc-service",
    "process_data": lambda data: f"Processed: {data}"
})
```

```javascript
// JavaScript - automatic WebRTC
const server = await hyphaWebsocketClient.connectToServer({
    server_url: "https://hypha.aicell.io", 
    token: token,
    webrtc: true
});

// Services are automatically WebRTC-enabled
```

### Getting Services via WebRTC

Access any service through WebRTC by using the `webrtc=True` option:

```python
# Get service via WebRTC peer-to-peer connection
service = await server.get_service("my-service", webrtc=True, webrtc_config={
    "timeout": 30,
    "ice_servers": custom_ice_servers  # Optional custom ICE servers
})

# All function calls now go through WebRTC
result = await service.process_data("large_dataset")
```

### WebRTC Configuration Options

Both `register_rtc_service` and `get_rtc_service` accept configuration options:

```python
config = {
    "visibility": "public",  # Service visibility
    "timeout": 30,           # Connection timeout
    "ice_servers": None,     # Custom ICE servers (auto-fetched if None)
    "on_init": callback_fn,  # Called when connection established
    "on_close": close_fn,    # Called when connection closes
    "data_channels": True,   # Enable data channels
}

await register_rtc_service(server, "my-service", config=config)
```

### Real-World Example: Remote Microscopy

Here's a complete example for remote microscopy control:

```python
import asyncio
import numpy as np
from hypha_rpc import connect_to_server, register_rtc_service

async def setup_microscopy_service():
    # Connect with authentication
    token = await login({"server_url": "https://hypha.aicell.io"})
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token
    })
    
    # Microscope state
    stage_position = {"x": 0, "y": 0, "z": 0}
    
    def move_stage(x, y, z=None):
        """Move microscope stage."""
        stage_position["x"] = x
        stage_position["y"] = y
        if z is not None:
            stage_position["z"] = z
        return {"position": stage_position.copy(), "status": "moved"}
    
    def capture_image():
        """Capture microscope image."""
        # Simulate image capture
        image_data = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        return {
            "image": image_data.tolist(),
            "position": stage_position.copy(),
            "timestamp": time.time()
        }
    
    def start_live_stream():
        """Start live video streaming."""
        return {"status": "streaming", "fps": 30, "resolution": "512x512"}
    
    # Register WebRTC service
    await register_rtc_service(
        server,
        service_id="remote-microscope",
        config={
            "visibility": "public",
            "move_stage": move_stage,
            "capture_image": capture_image,
            "start_live_stream": start_live_stream,
            "get_position": lambda: stage_position.copy(),
            "on_init": lambda pc: print("Microscope WebRTC connection established")
        }
    )
    
    print("Remote microscopy service ready!")
    print("Clients can connect via: get_rtc_service(server, 'remote-microscope')")

# Run the service
asyncio.run(setup_microscopy_service())
```

Client code to control the microscope:

```python
async def control_microscope():
    # Connect as a client
    token = await login({"server_url": "https://hypha.aicell.io"})
    client = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": token
    })
    
    # Get the microscope service via WebRTC
    microscope = await get_rtc_service(client, "remote-microscope")
    
    # Control the microscope (all via peer-to-peer WebRTC)
    await microscope.move_stage(100, 200, 50)
    position = await microscope.get_position()
    print(f"Current position: {position}")
    
    # Capture an image
    image_data = await microscope.capture_image()
    print(f"Captured image at {image_data['position']}")
    
    # Start streaming
    stream_info = await microscope.start_live_stream()
    print(f"Live stream: {stream_info}")

# Control the microscope
asyncio.run(control_microscope())
```

### Synchronous WebRTC API

For synchronous Python code, use the sync wrappers:

```python
from hypha_rpc.sync import connect_to_server, register_rtc_service, get_rtc_service, login

# Synchronous WebRTC service registration
token = login({"server_url": "https://hypha.aicell.io"})
server = connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})

register_rtc_service(
    server,
    service_id="sync-webrtc-service",
    config={
        "visibility": "public",
        "process": lambda data: f"Processed: {data}"
    }
)

# Synchronous WebRTC service access
rtc_service = get_rtc_service(server, "sync-webrtc-service")
result = rtc_service.process("test data")
print(result)  # "Processed: test data"
```

## Server Setup

### COTURN Server Configuration

WebRTC requires a TURN/STUN server for NAT traversal. We recommend using COTURN:

#### Docker Setup

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  coturn:
    image: coturn/coturn:4
    ports:
      - "3478:3478/tcp"
      - "3478:3478/udp"
      - "49152-49200:49152-49200/udp"
    environment:
      - TURN_SHARED_SECRET=your-secret-key
    volumes:
      - ./turnserver.conf:/etc/turnserver.conf:ro
    restart: unless-stopped
    
  hypha:
    image: your-hypha-image
    ports:
      - "9527:9527"
    environment:
      - HYPHA_COTURN_SECRET=your-secret-key
      - HYPHA_COTURN_URI=coturn:3478
      - HYPHA_PUBLIC_COTURN_URI=your-domain.com:3478
    depends_on:
      - coturn
```

#### COTURN Configuration

Create `turnserver.conf`:

```conf
# Basic COTURN configuration
listening-port=3478
tls-listening-port=5349

# Relay port range
min-port=49152
max-port=49200

# Authentication
use-auth-secret
static-auth-secret=your-secret-key
realm=hypha.io

# Performance settings
tcp-relay
fingerprint
lt-cred-mech

# Security
no-rfc5780
no-multicast-peers
no-loopback-peers

# External IP (set to your server's public IP)
external-ip=YOUR_PUBLIC_IP

# Logging
verbose
log-file=/var/log/turnserver.log
```

### Hypha Server Configuration

#### Basic Configuration

Start Hypha with COTURN support:

```bash
python -m hypha.server \
    --host=0.0.0.0 \
    --port=9527 \
    --coturn-secret=your-secret-key \
    --coturn-uri=localhost:3478 \
    --public-coturn-uri=your-domain.com:3478
```

#### Environment Variables

You can also use environment variables:

```bash
export HYPHA_COTURN_SECRET=your-secret-key
export HYPHA_COTURN_URI=localhost:3478
export HYPHA_PUBLIC_COTURN_URI=your-domain.com:3478
python -m hypha.server --from-env
```

#### Separate Internal/External URIs

The `--coturn-uri` is used for internal connectivity testing, while `--public-coturn-uri` is provided to clients for actual connections. This is useful in Docker deployments where internal and external hostnames differ.

### Docker Deployment

For production deployment with Docker:

```yaml
version: '3.8'
services:
  coturn:
    image: coturn/coturn:4
    network_mode: host
    volumes:
      - ./turnserver.conf:/etc/turnserver.conf:ro
    restart: unless-stopped
    
  hypha:
    image: hypha:latest
    ports:
      - "9527:9527"
    environment:
      - HYPHA_COTURN_SECRET=your-secret-key
      - HYPHA_COTURN_URI=localhost:3478
      - HYPHA_PUBLIC_COTURN_URI=your-domain.com:3478
      - HYPHA_ENABLE_S3=true
      - HYPHA_START_MINIO_SERVER=true
    depends_on:
      - coturn
    restart: unless-stopped
```

## Security Considerations

1. **Authentication Required**: Only authenticated users can access ICE servers
2. **Secret Management**: Use strong, unique secrets for COTURN authentication
3. **Firewall Configuration**: Ensure proper port configuration for UDP relay ports
4. **TLS/SSL**: Use HTTPS for Hypha and configure TLS for COTURN in production
5. **Rate Limiting**: Consider implementing rate limiting for ICE server requests

## Troubleshooting

### Common Issues

#### ICE Connection Failures

```python
# Check ICE server connectivity
ice_servers = await api.get_rtc_ice_servers(test_connectivity=True)
```

#### Anonymous User Errors

Ensure users login before accessing WebRTC features:

```python
# Always login first
token = await login({"server_url": "https://hypha.aicell.io"})
api = await connect_to_server({
    "server_url": "https://hypha.aicell.io", 
    "token": token
})
```

#### COTURN Configuration Issues

Check COTURN logs:

```bash
docker logs coturn_container_name
```

#### Network Connectivity

Test COTURN server directly:

```bash
# Test STUN
stunclient your-coturn-server.com 3478

# Test TURN (requires authentication)
turnutils_uclient -t -u username -w password your-coturn-server.com
```

### Debug Logging

Enable debug logging in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

In JavaScript:

```javascript
// Enable WebRTC debug logging
window.localStorage.setItem('debug', 'aiortc:*');
```

### API Reference

#### `get_rtc_ice_servers(ttl=43200, test_connectivity=True)`

- **ttl**: Token lifetime in seconds (default: 12 hours)
- **test_connectivity**: Whether to test COTURN connectivity (default: True)
- **Returns**: List of ICE server configurations

**Example Response:**
```json
[
  {
    "username": "1234567890:user123",
    "credential": "base64-encoded-hmac",
    "urls": [
      "turn:your-domain.com:3478",
      "stun:your-domain.com:3478"
    ]
  }
]
``` 