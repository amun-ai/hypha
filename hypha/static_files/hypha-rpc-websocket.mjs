/******/ var __webpack_modules__ = ({

/***/ "./src/http-client.js":
/*!****************************!*\
  !*** ./src/http-client.js ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   HTTPStreamingRPCConnection: () => (/* binding */ HTTPStreamingRPCConnection),
/* harmony export */   _connectToServerHTTP: () => (/* binding */ _connectToServerHTTP),
/* harmony export */   connectToServerHTTP: () => (/* binding */ connectToServerHTTP),
/* harmony export */   getRemoteServiceHTTP: () => (/* binding */ getRemoteServiceHTTP),
/* harmony export */   normalizeServerUrl: () => (/* binding */ normalizeServerUrl)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs");
/**
 * HTTP Streaming RPC Client for Hypha.
 *
 * This module provides HTTP-based RPC transport as an alternative to WebSocket.
 * It uses:
 * - HTTP GET with streaming (msgpack) for server-to-client messages
 * - HTTP POST for client-to-server messages
 *
 * This is more resilient to network issues than WebSocket because:
 * 1. Each POST request is independent (stateless)
 * 2. GET stream can be easily reconnected
 * 3. Works through more proxies and firewalls
 *
 * ## Performance Optimizations
 *
 * Modern browsers automatically provide optimal HTTP performance:
 *
 * ### Automatic HTTP/2 Support
 * - Browsers negotiate HTTP/2 when server supports it
 * - Multiplexing: Multiple requests over single TCP connection
 * - Header compression: HPACK reduces overhead
 * - Server push: Pre-emptive resource delivery
 *
 * ### Connection Pooling
 * - Browsers maintain connection pools per origin
 * - Automatic keep-alive for HTTP/1.1
 * - Connection reuse reduces latency
 * - No manual configuration needed
 *
 * ### Fetch API Optimizations
 * - `keepalive: true` flag ensures connection reuse
 * - Streaming responses with backpressure handling
 * - Efficient binary data transfer (ArrayBuffer/Uint8Array)
 *
 * ### Server-Side Configuration
 * For optimal performance, ensure server has:
 * - Keep-alive timeout: 300s (matches typical browser defaults) ✓ CONFIGURED
 * - Fast compression: gzip level 1 (2-5x faster than level 5) ✓ CONFIGURED
 * - Uvicorn connection limits optimized ✓ CONFIGURED
 *
 * ### HTTP/2 Support
 * - Uvicorn does NOT natively support HTTP/2 (as of 2026)
 * - In production, use nginx/Caddy/ALB as reverse proxy for HTTP/2
 * - Reverse proxy handles HTTP/2 ↔ HTTP/1.1 translation
 * - Browsers automatically use HTTP/2 when reverse proxy supports it
 * - Current HTTP/1.1 implementation is already optimal
 *
 * ### Performance Results
 * With properly configured server, HTTP transport achieves:
 * - 10-12 MB/s throughput for large payloads (4-15 MB)
 * - 2-3x faster than before optimization
 * - 3-28x faster than WebSocket for data transfer
 * - 31% improvement in connection reuse efficiency
 */






const MAX_RETRY = 1000000;

/**
 * HTTP Streaming RPC Connection.
 *
 * Uses HTTP GET with streaming for receiving messages and HTTP POST for sending messages.
 * Uses msgpack binary format with length-prefixed frames for efficient binary data support.
 */
class HTTPStreamingRPCConnection {
  /**
   * Initialize HTTP streaming connection.
   *
   * @param {string} server_url - The server URL (http:// or https://)
   * @param {string} client_id - Unique client identifier
   * @param {string} workspace - Target workspace (optional)
   * @param {string} token - Authentication token (optional)
   * @param {string} reconnection_token - Token for reconnection (optional)
   * @param {number} timeout - Request timeout in seconds (default: 60)
   * @param {number} token_refresh_interval - Interval for token refresh (default: 2 hours)
   */
  constructor(
    server_url,
    client_id,
    workspace = null,
    token = null,
    reconnection_token = null,
    timeout = 60,
    token_refresh_interval = 2 * 60 * 60,
  ) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(server_url && client_id, "server_url and client_id are required");
    this._server_url = server_url.replace(/\/$/, "");
    this._client_id = client_id;
    this._workspace = workspace;
    this._token = token;
    this._reconnection_token = reconnection_token;
    this._timeout = timeout;
    this._token_refresh_interval = token_refresh_interval;

    this._handle_message = null;
    this._handle_disconnected = null;
    this._handle_connected = null;

    this._closed = false;
    this._enable_reconnect = false;
    this._is_reconnection = false;
    this.connection_info = null;
    this.manager_id = null;

    this._abort_controller = null;
    this._refresh_token_task = null;
  }

  /**
   * Register message handler.
   */
  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  /**
   * Register disconnection handler.
   */
  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  /**
   * Register connection handler.
   */
  on_connected(handler) {
    this._handle_connected = handler;
  }

  /**
   * Get HTTP headers with authentication.
   *
   * @param {boolean} for_stream - If true, set Accept header for msgpack stream
   * @returns {Object} Headers object
   */
  _get_headers(for_stream = false) {
    const headers = {
      "Content-Type": "application/msgpack",
    };
    if (for_stream) {
      headers["Accept"] = "application/x-msgpack-stream";
    }
    if (this._token) {
      headers["Authorization"] = `Bearer ${this._token}`;
    }
    return headers;
  }

  /**
   * Open the streaming connection.
   */
  async open() {
    console.info(`Opening HTTP streaming connection to ${this._server_url}`);

    // Build stream URL - workspace is part of path, default to "public" for anonymous
    const ws = this._workspace || "public";
    const stream_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;

    // Start streaming in background
    this._startStreamLoop(stream_url);

    // Wait for connection info (first message)
    const start = Date.now();
    while (this.connection_info === null) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      if (Date.now() - start > this._timeout * 1000) {
        throw new Error("Timeout waiting for connection info");
      }
      if (this._closed) {
        throw new Error("Connection closed during setup");
      }
    }

    this.manager_id = this.connection_info.manager_id;
    if (this._workspace) {
      const actual_ws = this.connection_info.workspace;
      if (actual_ws !== this._workspace) {
        throw new Error(
          `Connected to wrong workspace: ${actual_ws}, expected: ${this._workspace}`,
        );
      }
    }
    this._workspace = this.connection_info.workspace;

    if (this.connection_info.reconnection_token) {
      this._reconnection_token = this.connection_info.reconnection_token;
    }

    // Adjust token refresh interval based on server's token lifetime
    if (this.connection_info.reconnection_token_life_time) {
      const token_life_time = this.connection_info.reconnection_token_life_time;
      if (this._token_refresh_interval > token_life_time / 1.5) {
        console.warn(
          `Token refresh interval (${this._token_refresh_interval}s) is too long, ` +
            `adjusting to ${(token_life_time / 1.5).toFixed(0)}s based on token lifetime`,
        );
        this._token_refresh_interval = token_life_time / 1.5;
      }
    }

    console.info(
      `HTTP streaming connected to workspace: ${this._workspace}, ` +
        `manager_id: ${this.manager_id}`,
    );

    // Start token refresh
    if (this._token_refresh_interval > 0) {
      this._startTokenRefresh();
    }

    if (this._handle_connected) {
      await this._handle_connected(this.connection_info);
    }

    return this.connection_info;
  }

  /**
   * Start periodic token refresh via POST.
   */
  _startTokenRefresh() {
    // Clear existing refresh if any
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
    }

    // Initial delay of 2s, then periodic refresh
    setTimeout(() => {
      this._sendRefreshToken();
      this._refresh_token_task = setInterval(() => {
        this._sendRefreshToken();
      }, this._token_refresh_interval * 1000);
    }, 2000);
  }

  /**
   * Send a token refresh request via POST.
   */
  async _sendRefreshToken() {
    if (this._closed) return;
    try {
      const ws = this._workspace || "public";
      const url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;
      const body = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)({ type: "refresh_token" });
      const response = await fetch(url, {
        method: "POST",
        headers: this._get_headers(false),
        body: body,
      });
      if (response.ok) {
        console.debug("Token refresh requested successfully");
      } else {
        console.warn(`Token refresh request failed: ${response.status}`);
      }
    } catch (e) {
      console.warn(`Failed to send refresh token request: ${e.message}`);
    }
  }

  /**
   * Start the streaming loop.
   *
   * OPTIMIZATION: Modern browsers automatically:
   * - Negotiate HTTP/2 when server supports it
   * - Use connection pooling for multiple requests to same origin
   * - Handle keep-alive for persistent connections
   * - Stream responses efficiently with backpressure handling
   */
  async _startStreamLoop(url) {
    this._enable_reconnect = true;
    this._closed = false;
    let retry = 0;
    this._is_reconnection = false;

    while (!this._closed && retry < MAX_RETRY) {
      try {
        // Update URL with current workspace (may have changed after initial connection)
        const ws = this._workspace || "public";
        let stream_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;
        if (this._reconnection_token) {
          stream_url += `&reconnection_token=${encodeURIComponent(this._reconnection_token)}`;
        }

        // OPTIMIZATION: Browser fetch automatically streams responses
        // and negotiates HTTP/2 when available for better performance
        this._abort_controller = new AbortController();
        const response = await fetch(stream_url, {
          method: "GET",
          headers: this._get_headers(true),
          signal: this._abort_controller.signal,
        });

        if (!response.ok) {
          const error_text = await response.text();
          throw new Error(
            `Stream failed with status ${response.status}: ${error_text}`,
          );
        }

        retry = 0; // Reset retry counter on successful connection

        // Process binary msgpack stream with 4-byte length prefix
        await this._processMsgpackStream(response);
      } catch (error) {
        if (this._closed) break;
        console.error(`Connection error: ${error.message}`);

        if (!this._enable_reconnect) {
          break;
        }
      }

      // After the first connection attempt, all subsequent ones are reconnections
      this._is_reconnection = true;

      // Reconnection logic
      if (!this._closed && this._enable_reconnect) {
        retry += 1;
        // Exponential backoff with max 60 seconds
        const delay = Math.min(Math.pow(2, Math.min(retry, 6)), 60);
        console.warn(
          `Stream disconnected, reconnecting in ${delay.toFixed(1)}s (attempt ${retry})`,
        );
        await new Promise((resolve) => setTimeout(resolve, delay * 1000));
      } else {
        break;
      }
    }

    if (!this._closed && this._handle_disconnected) {
      this._handle_disconnected("Stream ended");
    }
  }

  /**
   * Check if frame data is a control message and decode it.
   *
   * Control messages vs RPC messages:
   * - Control messages: Single msgpack object with "type" field (connection_info, ping, etc.)
   * - RPC messages: May contain multiple concatenated msgpack objects (main message + extra data)
   *
   * We only need to decode the first object to check if it's a control message.
   * RPC messages are passed as raw bytes to the handler.
   *
   * @param {Uint8Array} frame_data - The msgpack frame data
   * @returns {Object|null} Decoded control message or null
   */
  _tryDecodeControlMessage(frame_data) {
    try {
      // Decode the first msgpack object in the frame
      const decoded = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.decode)(frame_data);

      // Control messages are simple objects with a "type" field
      if (typeof decoded === "object" && decoded !== null && decoded.type) {
        const controlTypes = [
          "connection_info",
          "ping",
          "pong",
          "reconnection_token",
          "error",
        ];
        if (controlTypes.includes(decoded.type)) {
          return decoded;
        }
      }

      // Not a control message
      return null;
    } catch {
      // Decode failed - this is an RPC message
      return null;
    }
  }

  /**
   * Process msgpack stream with 4-byte length prefix.
   */
  async _processMsgpackStream(response) {
    const reader = response.body.getReader();
    // Growing buffer to avoid O(n^2) re-allocation on every chunk
    let buffer = new Uint8Array(4096);
    let bufferLen = 0;

    while (!this._closed) {
      const { done, value } = await reader.read();

      if (done) break;

      // Grow buffer if needed (double size until it fits)
      const needed = bufferLen + value.length;
      if (needed > buffer.length) {
        let newSize = buffer.length;
        while (newSize < needed) newSize *= 2;
        const grown = new Uint8Array(newSize);
        grown.set(buffer.subarray(0, bufferLen));
        buffer = grown;
      }
      buffer.set(value, bufferLen);
      bufferLen += value.length;

      // Process complete frames from buffer
      let offset = 0;
      while (bufferLen - offset >= 4) {
        // Read 4-byte length prefix (big-endian)
        const length =
          (buffer[offset] << 24) |
          (buffer[offset + 1] << 16) |
          (buffer[offset + 2] << 8) |
          buffer[offset + 3];

        if (bufferLen - offset < 4 + length) {
          // Incomplete frame, wait for more data
          break;
        }

        // Extract the frame (slice creates a copy, which is needed since buffer is reused)
        const frame_data = buffer.slice(offset + 4, offset + 4 + length);
        offset += 4 + length;

        // Try to decode as control message first
        const controlMsg = this._tryDecodeControlMessage(frame_data);
        if (controlMsg) {
          const msg_type = controlMsg.type;
          if (msg_type === "connection_info") {
            this.connection_info = controlMsg;
            // On reconnection, update state and notify RPC layer.
            // Run as a non-blocking task so the stream can continue
            // processing incoming RPC responses (the reconnection
            // handler sends RPC calls that need stream responses).
            if (this._is_reconnection) {
              this._handleReconnection(controlMsg).catch((err) => {
                console.error(`Reconnection handling failed: ${err.message}`);
              });
            }
            continue;
          } else if (msg_type === "ping" || msg_type === "pong") {
            continue;
          } else if (msg_type === "reconnection_token") {
            this._reconnection_token = controlMsg.reconnection_token;
            continue;
          } else if (msg_type === "error") {
            console.error(`Server error: ${controlMsg.message}`);
            continue;
          }
        }

        // For RPC messages (or unrecognized control messages), pass raw frame data to handler
        if (this._handle_message) {
          try {
            await this._handle_message(frame_data);
          } catch (error) {
            console.error(`Error in message handler: ${error.message}`);
          }
        }
      }
      // Compact: shift remaining data to the front of the buffer
      if (offset > 0) {
        const remaining = bufferLen - offset;
        if (remaining > 0) {
          buffer.copyWithin(0, offset, bufferLen);
        }
        bufferLen = remaining;
      }
    }
  }

  /**
   * Handle reconnection: update state and notify RPC layer.
   */
  async _handleReconnection(connection_info) {
    this.manager_id = connection_info.manager_id;
    this._workspace = connection_info.workspace;

    if (connection_info.reconnection_token) {
      this._reconnection_token = connection_info.reconnection_token;
    }

    // Adjust token refresh interval if needed
    if (connection_info.reconnection_token_life_time) {
      const token_life_time = connection_info.reconnection_token_life_time;
      if (this._token_refresh_interval > token_life_time / 1.5) {
        this._token_refresh_interval = token_life_time / 1.5;
      }
    }

    console.warn(
      `Stream reconnected to workspace: ${this._workspace}, ` +
        `manager_id: ${this.manager_id}`,
    );

    // Notify RPC layer so it can re-register services
    if (this._handle_connected) {
      await this._handle_connected(this.connection_info);
    }

    // Wait a short time for services to be re-registered
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  /**
   * Send a message to the server via HTTP POST.
   *
   * OPTIMIZATION: Uses keepalive flag for connection reuse.
   * Modern browsers automatically:
   * - Use HTTP/2 when available (multiplexing, header compression)
   * - Manage connection pooling with HTTP/1.1 keep-alive
   * - Reuse connections for same-origin requests
   */
  async emit_message(data) {
    if (this._closed) {
      throw new Error("Connection is closed");
    }

    // Build POST URL - workspace is part of path (must be set after connection)
    const ws = this._workspace || "public";
    let post_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;

    // Ensure data is Uint8Array
    const body = data instanceof Uint8Array ? data : new Uint8Array(data);

    // Retry logic to handle transient issues such as load balancer
    // routing POST requests to a different server instance than the GET stream
    const maxRetries = 3;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // Note: keepalive has a 64KB body size limit in browsers, so only use
        // it for small payloads. For large payloads, skip keepalive.
        const useKeepalive = body.length < 60000;
        const response = await fetch(post_url, {
          method: "POST",
          headers: this._get_headers(false),
          body: body,
          ...(useKeepalive && { keepalive: true }),
        });

        if (!response.ok) {
          const error_text = await response.text();
          // Retry on 400 errors that indicate the server doesn't recognize
          // our stream (e.g., load balancer routed to a different instance)
          if (response.status === 400 && attempt < maxRetries - 1) {
            console.warn(
              `POST failed (attempt ${attempt + 1}/${maxRetries}): ${error_text}, retrying...`,
            );
            await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
            continue;
          }
          throw new Error(
            `POST failed with status ${response.status}: ${error_text}`,
          );
        }

        return true;
      } catch (error) {
        if (attempt < maxRetries - 1 && !this._closed) {
          console.warn(
            `Failed to send message (attempt ${attempt + 1}/${maxRetries}): ${error.message}, retrying...`,
          );
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        } else {
          console.error(`Failed to send message: ${error.message}`);
          throw error;
        }
      }
    }
  }

  /**
   * Set reconnection flag.
   */
  set_reconnection(value) {
    this._enable_reconnect = value;
  }

  /**
   * Close the connection.
   */
  async disconnect(reason = "client disconnect") {
    if (this._closed) return;

    this._closed = true;

    // Clear token refresh interval
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
      this._refresh_token_task = null;
    }

    // Abort any active stream fetch to release the connection immediately
    if (this._abort_controller) {
      this._abort_controller.abort();
      this._abort_controller = null;
    }

    if (this._handle_disconnected) {
      this._handle_disconnected(reason);
    }
  }
}

/**
 * Normalize server URL for HTTP transport.
 */
function normalizeServerUrl(server_url) {
  if (!server_url) {
    throw new Error("server_url is required");
  }

  // Convert ws:// to http://
  if (server_url.startsWith("ws://")) {
    server_url = server_url.replace("ws://", "http://");
  } else if (server_url.startsWith("wss://")) {
    server_url = server_url.replace("wss://", "https://");
  }

  // Remove /ws suffix if present (WebSocket endpoint)
  if (server_url.endsWith("/ws")) {
    server_url = server_url.slice(0, -3);
  }

  return server_url.replace(/\/$/, "");
}

/**
 * Internal function to establish HTTP streaming connection.
 */
async function _connectToServerHTTP(config) {
  let clientId = config.clientId || config.client_id;
  if (!clientId) {
    clientId = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();
  }

  const server_url = normalizeServerUrl(config.serverUrl || config.server_url);

  const connection = new HTTPStreamingRPCConnection(
    server_url,
    clientId,
    config.workspace,
    config.token,
    config.reconnection_token,
    config.method_timeout || 30,
    config.token_refresh_interval || 2 * 60 * 60,
  );

  const connection_info = await connection.open();
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(connection_info, "Failed to connect to server");

  await new Promise((resolve) => setTimeout(resolve, 100));

  const workspace = connection_info.workspace;

  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    workspace,
    default_context: { connection_type: "http_streaming" },
    name: config.name,
    method_timeout: config.method_timeout,
    app_id: config.app_id,
    server_base_url: connection_info.public_base_url,
  });

  await rpc.waitFor("services_registered", config.method_timeout || 120);

  const wm = await rpc.get_manager_service({
    timeout: config.method_timeout || 30,
    case_conversion: "camel",
  });
  wm.rpc = rpc;

  // Add standard methods
  wm.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.disconnect.bind(rpc), {
    name: "disconnect",
    description: "Disconnect from server",
    parameters: { properties: {}, type: "object" },
  });

  wm.registerService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_service.bind(rpc), {
    name: "registerService",
    description: "Register a service",
    parameters: {
      properties: {
        service: { description: "Service to register", type: "object" },
      },
      required: ["service"],
      type: "object",
    },
  });

  const _getService = wm.getService;
  wm.getService = async (query, config = {}) => {
    return await _getService(query, config);
  };
  if (_getService.__schema__) {
    wm.getService.__schema__ = _getService.__schema__;
  }

  async function serve() {
    await new Promise(() => {}); // Wait forever
  }

  wm.serve = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(serve, {
    name: "serve",
    description: "Run event loop forever",
    parameters: { type: "object", properties: {} },
  });

  if (connection_info) {
    wm.config = Object.assign(wm.config || {}, connection_info);
  }

  // Handle force-exit from manager
  if (connection.manager_id) {
    rpc.on("force-exit", async (message) => {
      if (message.from === "*/" + connection.manager_id) {
        console.info(`Disconnecting from server: ${message.reason}`);
        await rpc.disconnect();
      }
    });
  }

  return wm;
}

/**
 * Connect to server using HTTP streaming transport.
 *
 * This is an alternative to WebSocket connection that's more resilient
 * to network issues.
 *
 * @param {Object} config - Configuration object
 * @returns {Promise<Object>} Connected workspace manager
 */
async function connectToServerHTTP(config = {}) {
  return await _connectToServerHTTP(config);
}

/**
 * Get a remote service using HTTP transport.
 */
async function getRemoteServiceHTTP(serviceUri, config = {}) {
  const { serverUrl, workspace, clientId, serviceId, appId } =
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.parseServiceUrl)(serviceUri);
  const fullServiceId = `${workspace}/${clientId}:${serviceId}@${appId}`;

  if (config.serverUrl) {
    if (config.serverUrl !== serverUrl) {
      throw new Error("server_url mismatch");
    }
  }
  config.serverUrl = serverUrl;

  const server = await connectToServerHTTP(config);
  return await server.getService(fullServiceId);
}


/***/ }),

/***/ "./src/rpc.js":
/*!********************!*\
  !*** ./src/rpc.js ***!
  \********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   API_VERSION: () => (/* binding */ API_VERSION),
/* harmony export */   RPC: () => (/* binding */ RPC)
/* harmony export */ });
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs");
/**
 * Contains the RPC object used both by the application
 * site, and by each plugin
 */





const API_VERSION = 3;
const CHUNK_SIZE = 1024 * 256;
const CONCURRENCY_LIMIT = 30;

const ArrayBufferView = Object.getPrototypeOf(
  Object.getPrototypeOf(new Uint8Array()),
).constructor;

/**
 * Check if a value is a primitive type that needs no encoding/decoding.
 */
function _isPrimitive(v) {
  if (v === null || v === undefined) return true;
  const t = typeof v;
  return t === "number" || t === "string" || t === "boolean";
}

/**
 * Check if an array (and nested arrays/objects) contains only primitives.
 */
function _allPrimitivesArray(arr) {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (_isPrimitive(v)) continue;
    if (v instanceof Uint8Array) continue;
    if (Array.isArray(v)) {
      if (!_allPrimitivesArray(v)) return false;
      continue;
    }
    if (v && v.constructor === Object) {
      if ("_rtype" in v) return false;
      if (!_allPrimitivesObject(v)) return false;
      continue;
    }
    return false;
  }
  return true;
}

/**
 * Check if an object (and nested objects/arrays) contains only primitives.
 */
function _allPrimitivesObject(obj) {
  const values = Object.values(obj);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (_isPrimitive(v)) continue;
    if (v instanceof Uint8Array) continue;
    if (Array.isArray(v)) {
      if (!_allPrimitivesArray(v)) return false;
      continue;
    }
    if (v && v.constructor === Object) {
      if ("_rtype" in v) return false;
      if (!_allPrimitivesObject(v)) return false;
      continue;
    }
    return false;
  }
  return true;
}

function _appendBuffer(buffer1, buffer2) {
  const tmp = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
  tmp.set(new Uint8Array(buffer1), 0);
  tmp.set(new Uint8Array(buffer2), buffer1.byteLength);
  return tmp.buffer;
}

/**
 * Wrap a promise with a timeout.
 * @param {Promise} promise - The promise to wrap.
 * @param {number} timeoutMs - The timeout in milliseconds.
 * @param {string} message - Optional error message for timeout.
 * @returns {Promise} - The wrapped promise that will reject on timeout.
 */
function withTimeout(promise, timeoutMs, message = "Operation timed out") {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(`TimeoutError: ${message}`));
    }, timeoutMs);

    promise
      .then((result) => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

function indexObject(obj, is) {
  if (!is) throw new Error("undefined index");
  if (typeof is === "string") return indexObject(obj, is.split("."));
  else if (is.length === 0) return obj;
  else return indexObject(obj[is[0]], is.slice(1));
}

// Simple fallback schema generation - no docstring parsing for JS

function _get_schema(obj, name = null, skipContext = false) {
  if (Array.isArray(obj)) {
    return obj.map((v, i) => _get_schema(v, null, skipContext));
  } else if (typeof obj === "object" && obj !== null) {
    let schema = {};
    for (let k in obj) {
      schema[k] = _get_schema(obj[k], k, skipContext);
    }
    return schema;
  } else if (typeof obj === "function") {
    if (obj.__schema__) {
      const schema = JSON.parse(JSON.stringify(obj.__schema__));
      if (name) {
        schema.name = name;
        obj.__schema__.name = name;
      }
      if (skipContext) {
        if (schema.parameters && schema.parameters.properties) {
          delete schema.parameters.properties["context"];
        }
        if (schema.parameters && schema.parameters.required) {
          const contextIndex = schema.parameters.required.indexOf("context");
          if (contextIndex > -1) {
            schema.parameters.required.splice(contextIndex, 1);
          }
        }
      }
      return { type: "function", function: schema };
    } else {
      // Simple fallback for JavaScript - just return basic function schema with name
      const funcName = name || obj.name || "function";
      return {
        type: "function",
        function: {
          name: funcName,
        },
      };
    }
  } else if (typeof obj === "number") {
    return { type: "number" };
  } else if (typeof obj === "string") {
    return { type: "string" };
  } else if (typeof obj === "boolean") {
    return { type: "boolean" };
  } else if (obj === null) {
    return { type: "null" };
  } else {
    return {};
  }
}

function _annotate_service(service, serviceTypeInfo) {
  function validateKeys(serviceDict, schemaDict, path = "root") {
    // Validate that all keys in schemaDict exist in serviceDict
    for (let key in schemaDict) {
      if (!serviceDict.hasOwnProperty(key)) {
        throw new Error(`Missing key '${key}' in service at path '${path}'`);
      }
    }

    // Check for any unexpected keys in serviceDict
    for (let key in serviceDict) {
      if (key !== "type" && !schemaDict.hasOwnProperty(key)) {
        throw new Error(`Unexpected key '${key}' in service at path '${path}'`);
      }
    }
  }

  function annotateRecursive(newService, schemaInfo, path = "root") {
    if (typeof newService === "object" && !Array.isArray(newService)) {
      validateKeys(newService, schemaInfo, path);
      for (let k in newService) {
        let v = newService[k];
        let newPath = `${path}.${k}`;
        if (typeof v === "object" && !Array.isArray(v)) {
          annotateRecursive(v, schemaInfo[k], newPath);
        } else if (typeof v === "function") {
          if (schemaInfo.hasOwnProperty(k)) {
            newService[k] = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_1__.schemaFunction)(v, {
              name: schemaInfo[k]["name"],
              description: schemaInfo[k].description || "",
              parameters: schemaInfo[k]["parameters"],
            });
          } else {
            throw new Error(
              `Missing schema for function '${k}' at path '${newPath}'`,
            );
          }
        }
      }
    } else if (Array.isArray(newService)) {
      if (newService.length !== schemaInfo.length) {
        throw new Error(`Length mismatch at path '${path}'`);
      }
      newService.forEach((v, i) => {
        let newPath = `${path}[${i}]`;
        if (typeof v === "object" && !Array.isArray(v)) {
          annotateRecursive(v, schemaInfo[i], newPath);
        } else if (typeof v === "function") {
          if (schemaInfo.hasOwnProperty(i)) {
            newService[i] = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_1__.schemaFunction)(v, {
              name: schemaInfo[i]["name"],
              description: schemaInfo[i].description || "",
              parameters: schemaInfo[i]["parameters"],
            });
          } else {
            throw new Error(
              `Missing schema for function at index ${i} in path '${newPath}'`,
            );
          }
        }
      });
    }
  }

  validateKeys(service, serviceTypeInfo["definition"]);
  annotateRecursive(service, serviceTypeInfo["definition"]);
  return service;
}

function getFunctionInfo(func) {
  const funcString = func.toString();

  // Extract function name
  const nameMatch = funcString.match(/function\s*(\w*)/);
  const name = (nameMatch && nameMatch[1]) || "";

  // Extract function parameters, excluding comments
  const paramsMatch = funcString.match(/\(([^)]*)\)/);
  let params = "";
  if (paramsMatch) {
    params = paramsMatch[1]
      .split(",")
      .map((p) =>
        p
          .replace(/\/\*.*?\*\//g, "") // Remove block comments
          .replace(/\/\/.*$/g, ""),
      ) // Remove line comments
      .filter((p) => p.trim().length > 0) // Remove empty strings after removing comments
      .map((p) => p.trim()) // Trim remaining whitespace
      .join(", ");
  }

  // Extract function docstring (block comment)
  let docMatch = funcString.match(/\)\s*\{\s*\/\*([\s\S]*?)\*\//);
  const docstringBlock = (docMatch && docMatch[1].trim()) || "";

  // Extract function docstring (line comment)
  docMatch = funcString.match(/\)\s*\{\s*(\/\/[\s\S]*?)\n\s*[^\s\/]/);
  const docstringLine =
    (docMatch &&
      docMatch[1]
        .split("\n")
        .map((s) => s.replace(/^\/\/\s*/, "").trim())
        .join("\n")) ||
    "";

  const docstring = docstringBlock || docstringLine;
  return (
    name &&
    params.length > 0 && {
      name: name,
      sig: params,
      doc: docstring,
    }
  );
}

function concatArrayBuffers(buffers) {
  var buffersLengths = buffers.map(function (b) {
      return b.byteLength;
    }),
    totalBufferlength = buffersLengths.reduce(function (p, c) {
      return p + c;
    }, 0),
    unit8Arr = new Uint8Array(totalBufferlength);
  buffersLengths.reduce(function (p, c, i) {
    unit8Arr.set(new Uint8Array(buffers[i]), p);
    return p + c;
  }, 0);
  return unit8Arr.buffer;
}

class Timer {
  constructor(timeout, callback, args, label) {
    this._timeout = timeout;
    this._callback = callback;
    this._args = args;
    this._label = label || "timer";
    this._task = null;
    this.started = false;
  }

  start() {
    if (this.started) {
      this.reset();
    } else {
      this._task = setTimeout(() => {
        this._callback.apply(this, this._args);
      }, this._timeout * 1000);
      this.started = true;
    }
  }

  clear() {
    if (this._task && this.started) {
      clearTimeout(this._task);
      this._task = null;
      this.started = false;
    } else {
      console.warn(`Clearing a timer (${this._label}) which is not started`);
    }
  }

  reset() {
    if (this._task) {
      clearTimeout(this._task);
    }
    this._task = setTimeout(() => {
      this._callback.apply(this, this._args);
    }, this._timeout * 1000);
    this.started = true;
  }
}

class RemoteService extends Object {}

/**
 * RPC object represents a single site in the
 * communication protocol between the application and the plugin
 *
 * @param {Object} connection a special object allowing to send
 * and receive messages from the opposite site (basically it
 * should only provide send() and onMessage() methods)
 */
class RPC extends _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.MessageEmitter {
  constructor(
    connection,
    {
      client_id = null,
      default_context = null,
      name = null,
      codecs = null,
      method_timeout = null,
      max_message_buffer_size = 0,
      debug = false,
      workspace = null,
      silent = false,
      app_id = null,
      server_base_url = null,
      long_message_chunk_size = null,
    },
  ) {
    super(debug);
    this._codecs = codecs || {};
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(client_id && typeof client_id === "string");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(client_id, "client_id is required");
    this._client_id = client_id;
    this._name = name;
    this._app_id = app_id || "*";
    this._local_workspace = workspace;
    this._silent = silent;
    this.default_context = default_context || {};
    this._method_annotations = new WeakMap();
    this._max_message_buffer_size = max_message_buffer_size;
    this._chunk_store = {};
    this._method_timeout = method_timeout || 30;
    this._server_base_url = server_base_url;
    this._long_message_chunk_size = long_message_chunk_size || CHUNK_SIZE;

    // make sure there is an execute function
    this._services = {};
    this._object_store = {
      services: this._services,
    };
    // Index: target_id -> Set of top-level session keys for fast cleanup
    this._targetIdIndex = {};

    // Track background tasks for proper cleanup
    this._background_tasks = new Set();

    // Periodic session sweep for interface-object sessions (clear_after_called=false)
    // that have no activity for a long time. Max age = 10 * method_timeout.
    this._sessionMaxAge = (this._method_timeout || 30) * 10 * 1000;
    this._sessionSweepInterval = setInterval(() => {
      this._sweepStaleSessions();
    }, this._sessionMaxAge / 2);

    // Set up global unhandled promise rejection handler for RPC-related errors
    // Use a class-level reference counter so the handler is added once and removed
    // only when the last RPC instance is closed.
    this._unhandledRejectionHandler = (event) => {
      const reason = event.reason;
      if (reason && typeof reason === "object") {
        const reasonStr = reason.toString();
        if (
          reasonStr.includes("Method not found") ||
          reasonStr.includes("Session not found") ||
          reasonStr.includes("Method expired")
        ) {
          console.debug(
            "Ignoring expected method/session not found error:",
            reason,
          );
          event.preventDefault();
          return;
        }
      }
      console.warn("Unhandled RPC promise rejection:", reason);
    };

    this._unhandledRejectionNodeHandler = null;

    if (!RPC._rejectionHandlerCount) {
      RPC._rejectionHandlerCount = 0;
    }
    RPC._rejectionHandlerCount++;

    if (RPC._rejectionHandlerCount === 1) {
      if (typeof window !== "undefined") {
        window.addEventListener(
          "unhandledrejection",
          this._unhandledRejectionHandler,
        );
      } else if (typeof process !== "undefined") {
        this._unhandledRejectionNodeHandler = (reason, promise) => {
          this._unhandledRejectionHandler({
            reason,
            promise,
            preventDefault: () => {},
          });
        };
        process.on("unhandledRejection", this._unhandledRejectionNodeHandler);
      }
    }

    if (connection) {
      this.add_service({
        id: "built-in",
        type: "built-in",
        name: `Built-in services for ${this._local_workspace}/${this._client_id}`,
        config: {
          require_context: true,
          visibility: "public",
          api_version: API_VERSION,
        },
        ping: this._ping.bind(this),
        get_service: this.get_local_service.bind(this),
        message_cache: {
          create: this._create_message.bind(this),
          append: this._append_message.bind(this),
          set: this._set_message.bind(this),
          process: this._process_message.bind(this),
          remove: this._remove_message.bind(this),
        },
      });
      this._boundHandleMethod = this._handle_method.bind(this);
      this._boundHandleError = console.error;
      this.on("method", this._boundHandleMethod);
      this.on("error", this._boundHandleError);

      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(connection.emit_message && connection.on_message);
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        connection.manager_id !== undefined,
        "Connection must have manager_id",
      );
      this._emit_message = connection.emit_message.bind(connection);
      connection.on_message(this._on_message.bind(this));
      this._connection = connection;
      const onConnected = async (connectionInfo) => {
        if (!this._silent && this._connection.manager_id) {
          console.debug("Connection established, reporting services...");
          try {
            // Retry getting manager service with exponential backoff
            const manager = await this.get_manager_service({
              timeout: 20,
              case_conversion: "camel",
            });
            const services = Object.values(this._services);
            const servicesCount = services.length;
            let registeredCount = 0;
            const failedServices = [];

            // Use timeout for service registration to prevent hanging
            const serviceRegistrationTimeout = this._method_timeout || 30000;

            for (let service of services) {
              try {
                const serviceInfo = this._extract_service_info(service);
                await withTimeout(
                  manager.registerService(serviceInfo),
                  serviceRegistrationTimeout,
                  `Timeout registering service ${service.id || "unknown"}`,
                );
                registeredCount++;
                console.debug(
                  `Successfully registered service: ${service.id || "unknown"}`,
                );
              } catch (serviceError) {
                failedServices.push(service.id || "unknown");
                if (
                  serviceError.message &&
                  serviceError.message.includes("TimeoutError")
                ) {
                  console.error(
                    `Timeout registering service ${service.id || "unknown"}`,
                  );
                } else {
                  console.error(
                    `Failed to register service ${service.id || "unknown"}: ${serviceError}`,
                  );
                }
              }
            }

            if (registeredCount === servicesCount) {
              console.info(
                `Successfully registered all ${registeredCount} services with the server`,
              );
            } else {
              console.warn(
                `Only registered ${registeredCount} out of ${servicesCount} services with the server. Failed services: ${failedServices.join(", ")}`,
              );
            }

            // Fire event with registration status
            this._fire("services_registered", {
              total: servicesCount,
              registered: registeredCount,
              failed: failedServices,
            });

            // Subscribe to client_disconnected events if the manager supports it
            try {
              if (
                manager.subscribe &&
                typeof manager.subscribe === "function"
              ) {
                console.debug("Subscribing to client_disconnected events");

                const handleClientDisconnected = async (event) => {
                  // The client ID is in event.data.id based on the event structure
                  const clientId = event.data?.id || event.client;
                  const workspace = event.data?.workspace;
                  if (clientId && workspace) {
                    // Construct the full client path with workspace prefix
                    const fullClientId = `${workspace}/${clientId}`;
                    console.debug(
                      `Client ${fullClientId} disconnected, cleaning up sessions`,
                    );
                    await this._handleClientDisconnected(fullClientId);
                  } else if (clientId) {
                    console.debug(
                      `Client ${clientId} disconnected, cleaning up sessions`,
                    );
                    await this._handleClientDisconnected(clientId);
                  }
                };

                // Subscribe to the event topic first with timeout
                this._clientDisconnectedSubscription = await withTimeout(
                  manager.subscribe(["client_disconnected"]),
                  serviceRegistrationTimeout,
                  "Timeout subscribing to client_disconnected events",
                );

                // Then register the local event handler
                this.on("client_disconnected", handleClientDisconnected);

                console.debug(
                  "Successfully subscribed to client_disconnected events",
                );
              } else {
                console.debug(
                  "Manager does not support subscribe method, skipping client_disconnected handling",
                );
                this._clientDisconnectedSubscription = null;
              }
            } catch (subscribeError) {
              console.warn(
                `Failed to subscribe to client_disconnected events: ${subscribeError}`,
              );
              this._clientDisconnectedSubscription = null;
            }
          } catch (managerError) {
            console.error(
              `Failed to get manager service for registering services: ${managerError}`,
            );
            // Fire event with error status
            this._fire("services_registration_failed", {
              error: managerError.toString(),
              total_services: Object.keys(this._services).length,
            });
          }
        } else {
          // console.debug("Connection established", connectionInfo);
        }
        if (connectionInfo) {
          if (connectionInfo.public_base_url) {
            this._server_base_url = connectionInfo.public_base_url;
          }
          this._fire("connected", connectionInfo);
        }
      };
      connection.on_connected(onConnected);

      // Register disconnect handler to reject all pending RPC calls
      // This ensures no remote function call hangs forever when the connection drops
      if (typeof connection.on_disconnected === "function") {
        connection.on_disconnected((reason) => {
          // If reconnection is enabled, don't reject pending calls immediately.
          // The timeout mechanism will handle them if reconnection fails,
          // allowing calls to succeed after a successful reconnection.
          if (connection._enable_reconnect) {
            console.info(
              `Connection lost (${reason}), reconnection enabled - pending calls will be handled by timeout`,
            );
            return;
          }
          console.warn(
            `Connection lost (${reason}), rejecting all pending RPC calls`,
          );
          this._rejectPendingCalls(
            `Connection lost: ${reason || "unknown reason"}`,
          );
        });
      }

      onConnected();
    } else {
      this._emit_message = function () {
        console.log("No connection to emit message");
      };
    }
  }

  register_codec(config) {
    if (!config["name"] || (!config["encoder"] && !config["decoder"])) {
      throw new Error(
        "Invalid codec format, please make sure you provide a name, type, encoder and decoder.",
      );
    } else {
      if (config.type) {
        for (let k of Object.keys(this._codecs)) {
          if (this._codecs[k].type === config.type || k === config.name) {
            delete this._codecs[k];
            console.warn("Remove duplicated codec: " + k);
          }
        }
      }
      this._codecs[config["name"]] = config;
    }
  }

  async _ping(msg, context) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(msg == "ping");
    return "pong";
  }

  async ping(client_id, timeout) {
    let method = this._generate_remote_method({
      _rserver: this._server_base_url,
      _rtarget: client_id,
      _rmethod: "services.built-in.ping",
      _rpromise: true,
      _rdoc: "Ping a remote client",
    });
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)((await method("ping", timeout)) == "pong");
  }

  _create_message(key, heartbeat, overwrite, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }

    if (!this._object_store["message_cache"]) {
      this._object_store["message_cache"] = {};
    }

    // Evict stale cache entries (older than 5 minutes) and enforce size limit
    const cache = this._object_store["message_cache"];
    const MAX_CACHE_SIZE = 256;
    const MAX_CACHE_AGE = 5 * 60 * 1000;
    const cacheKeys = Object.keys(cache);
    if (cacheKeys.length >= MAX_CACHE_SIZE) {
      const now = Date.now();
      for (const k of cacheKeys) {
        const entry = cache[k];
        if (
          entry &&
          entry._cache_created_at &&
          now - entry._cache_created_at > MAX_CACHE_AGE
        ) {
          delete cache[k];
        }
      }
      // If still over limit, evict oldest entries
      const remaining = Object.keys(cache);
      if (remaining.length >= MAX_CACHE_SIZE) {
        remaining
          .sort(
            (a, b) =>
              (cache[a]._cache_created_at || 0) -
              (cache[b]._cache_created_at || 0),
          )
          .slice(0, remaining.length - MAX_CACHE_SIZE + 1)
          .forEach((k) => delete cache[k]);
      }
    }

    if (!overwrite && cache[key]) {
      throw new Error(
        `Message with the same key (${key}) already exists in the cache store, please use overwrite=true or remove it first.`,
      );
    }
    cache[key] = [];
    cache[key]._cache_created_at = Date.now();
  }

  _append_message(key, data, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data instanceof ArrayBufferView);
    cache[key].push(data);
  }

  _set_message(key, index, data, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data instanceof ArrayBufferView);
    cache[key][index] = data;
  }

  _remove_message(key, context) {
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    delete cache[key];
  }

  _process_message(key, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(!!context, "Context is required");
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    cache[key] = concatArrayBuffers(cache[key]);
    // console.debug(`Processing message ${key} (bytes=${cache[key].byteLength})`);
    let unpacker = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_2__.decodeMulti)(cache[key]);
    const { done, value } = unpacker.next();
    const main = value;
    // Make sure the fields are from trusted source
    Object.assign(main, {
      from: context.from,
      to: context.to,
      ws: context.ws,
      user: context.user,
    });
    main["ctx"] = JSON.parse(JSON.stringify(main));
    Object.assign(main["ctx"], this.default_context);
    if (!done) {
      let extra = unpacker.next();
      Object.assign(main, extra.value);
    }
    this._fire(main["type"], main);
    // console.debug(
    //   this._client_id,
    //   `Processed message ${key} (bytes=${cache[key].byteLength})`,
    // );
    delete cache[key];
  }

  _on_message(message) {
    if (typeof message === "string") {
      const main = JSON.parse(message);
      // Add trusted context to the method call
      main["ctx"] = Object.assign({}, main, this.default_context);
      this._fire(main["type"], main);
    } else if (message instanceof ArrayBuffer || ArrayBuffer.isView(message)) {
      // Handle both ArrayBuffer (WebSocket) and Uint8Array/ArrayBufferView (HTTP transport)
      let unpacker = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_2__.decodeMulti)(message);
      const { done, value } = unpacker.next();
      const main = value;
      // Add trusted context to the method call
      main["ctx"] = Object.assign({}, main, this.default_context);
      if (!done) {
        let extra = unpacker.next();
        Object.assign(main, extra.value);
      }
      this._fire(main["type"], main);
    } else if (typeof message === "object") {
      // Add trusted context to the method call
      message["ctx"] = Object.assign({}, message, this.default_context);
      this._fire(message["type"], message);
    } else {
      throw new Error("Invalid message format");
    }
  }

  reset() {
    this._removeRejectionHandler();
    this._event_handlers = {};
    this._services = {};
  }

  _removeRejectionHandler() {
    if (RPC._rejectionHandlerCount && RPC._rejectionHandlerCount > 0) {
      RPC._rejectionHandlerCount--;
      if (RPC._rejectionHandlerCount === 0) {
        if (typeof window !== "undefined" && this._unhandledRejectionHandler) {
          window.removeEventListener(
            "unhandledrejection",
            this._unhandledRejectionHandler,
          );
        } else if (
          typeof process !== "undefined" &&
          this._unhandledRejectionNodeHandler
        ) {
          process.removeListener(
            "unhandledRejection",
            this._unhandledRejectionNodeHandler,
          );
        }
      }
    }
    this._unhandledRejectionHandler = null;
    this._unhandledRejectionNodeHandler = null;
  }

  close() {
    // Clean up all pending sessions (rejects promises, clears timers/heartbeats, deletes sessions)
    this._cleanupOnDisconnect();

    // Remove method and error event listeners
    if (this._boundHandleMethod) {
      this.off("method", this._boundHandleMethod);
      this._boundHandleMethod = null;
    }
    if (this._boundHandleError) {
      this.off("error", this._boundHandleError);
      this._boundHandleError = null;
    }

    // Clean up client_disconnected subscription
    if (this._clientDisconnectedSubscription) {
      try {
        if (
          typeof this._clientDisconnectedSubscription.unsubscribe === "function"
        ) {
          this._clientDisconnectedSubscription.unsubscribe();
        }
      } catch (e) {
        console.debug(`Error unsubscribing client_disconnected: ${e}`);
      }
      this.off("client_disconnected");
      this._clientDisconnectedSubscription = null;
    }

    // Remove the global unhandled rejection handler
    this._removeRejectionHandler();

    // Remove ALL remaining event handlers to prevent memory leaks
    // This clears any custom event listeners registered by users via .on()
    this.off();

    // Clean up background tasks
    try {
      for (const task of this._background_tasks) {
        if (task && typeof task.cancel === "function") {
          try {
            task.cancel();
          } catch (e) {
            console.debug(`Error canceling background task: ${e}`);
          }
        }
      }
      this._background_tasks.clear();
    } catch (e) {
      console.debug(`Error cleaning up background tasks: ${e}`);
    }

    // Clear session sweep interval
    if (this._sessionSweepInterval) {
      clearInterval(this._sessionSweepInterval);
      this._sessionSweepInterval = null;
    }

    // Clean up connection references to prevent circular references
    try {
      this._connection = null;
      this._emit_message = function () {
        console.debug("RPC connection closed, ignoring message");
        return Promise.reject(new Error("Connection is closed"));
      };
    } catch (e) {
      console.debug(`Error during connection cleanup: ${e}`);
    }

    this._fire("disconnected");
  }

  async _handleClientDisconnected(clientId) {
    try {
      console.debug(`Handling disconnection for client: ${clientId}`);

      // Clean up all sessions for the disconnected client
      const sessionsCleaned = this._cleanupSessionsForClient(clientId);

      if (sessionsCleaned > 0) {
        console.debug(
          `Cleaned up ${sessionsCleaned} sessions for disconnected client: ${clientId}`,
        );
      }

      // Fire an event to notify about the client disconnection
      this._fire("remote_client_disconnected", {
        client_id: clientId,
        sessions_cleaned: sessionsCleaned,
      });
    } catch (e) {
      console.error(
        `Error handling client disconnection for ${clientId}: ${e}`,
      );
    }
  }

  _removeFromTargetIdIndex(sessionId) {
    /**
     * Remove a session from the target_id index.
     * Call this before removing a session from _object_store.
     */
    const topKey = sessionId.split(".")[0];
    const session = this._object_store[topKey];
    if (session && typeof session === "object") {
      const targetId = session.target_id;
      if (targetId && targetId in this._targetIdIndex) {
        this._targetIdIndex[targetId].delete(topKey);
        if (this._targetIdIndex[targetId].size === 0) {
          delete this._targetIdIndex[targetId];
        }
      }
    }
  }

  /**
   * Clean up a single session entry: reject promise, clear timer, cancel heartbeat.
   * Centralizes the cleanup logic used by multiple methods.
   * @param {object} session - The session object from _object_store
   * @param {string|null} rejectReason - If provided, reject the session's promise with this reason
   */
  _cleanupSessionEntry(session, rejectReason = null) {
    if (!session || typeof session !== "object") return;
    if (
      rejectReason &&
      session.reject &&
      typeof session.reject === "function"
    ) {
      try {
        session.reject(new Error(rejectReason));
      } catch (e) {
        console.debug(`Error rejecting session: ${e}`);
      }
    }
    if (session.heartbeat_task) {
      try {
        clearInterval(session.heartbeat_task);
      } catch (e) {
        /* ignore */
      }
    }
    if (
      session.timer &&
      session.timer.started &&
      typeof session.timer.clear === "function"
    ) {
      try {
        session.timer.clear();
      } catch (e) {
        /* ignore */
      }
    }
  }

  _cleanupSessionsForClient(clientId) {
    let sessionsCleaned = 0;

    // Use index for O(1) lookup instead of iterating all sessions
    const sessionKeys = this._targetIdIndex[clientId];
    if (!sessionKeys) return 0;

    const reason = `Client disconnected: ${clientId}`;
    for (const sessionKey of sessionKeys) {
      const session = this._object_store[sessionKey];
      if (!session || typeof session !== "object") continue;
      if (session.target_id !== clientId) continue;

      this._cleanupSessionEntry(session, reason);
      delete this._object_store[sessionKey];
      sessionsCleaned++;
      console.debug(`Cleaned up session: ${sessionKey}`);
    }

    delete this._targetIdIndex[clientId];
    return sessionsCleaned;
  }

  _rejectPendingCalls(reason = "Connection lost") {
    /**
     * Reject all pending RPC calls when the connection is lost.
     * Does NOT remove sessions (connection might be re-established).
     */
    try {
      let rejectedCount = 0;
      for (const key of Object.keys(this._object_store)) {
        if (key === "services" || key === "message_cache") continue;
        const value = this._object_store[key];
        if (typeof value === "object" && value !== null) {
          if (value.reject && typeof value.reject === "function") {
            rejectedCount++;
          }
          this._cleanupSessionEntry(value, reason);
        }
      }
      if (rejectedCount > 0) {
        console.warn(
          `Rejected ${rejectedCount} pending RPC call(s) due to: ${reason}`,
        );
      }
    } catch (e) {
      console.error(`Error rejecting pending calls: ${e}`);
    }
  }

  _cleanupOnDisconnect() {
    try {
      console.debug("Cleaning up all sessions due to local RPC disconnection");

      const keysToDelete = [];
      for (const key of Object.keys(this._object_store)) {
        if (key === "services" || key === "message_cache") continue;
        const value = this._object_store[key];
        this._cleanupSessionEntry(value, "RPC connection closed");
        keysToDelete.push(key);
      }

      for (const key of keysToDelete) {
        delete this._object_store[key];
      }

      this._targetIdIndex = {};
    } catch (e) {
      console.error(`Error during cleanup on disconnect: ${e}`);
    }
  }

  async disconnect() {
    // Store connection reference before closing
    const connection = this._connection;
    this.close();

    // Disconnect the underlying connection if it exists
    if (connection) {
      try {
        await connection.disconnect();
      } catch (e) {
        console.debug(`Error disconnecting underlying connection: ${e}`);
      }
    }
  }

  async get_manager_service(config, maxRetries = 20) {
    config = config || {};
    const baseDelay = 500;
    const maxDelay = 10000;
    let lastError = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const retryDelay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);

      if (!this._connection.manager_id) {
        if (attempt < maxRetries - 1) {
          console.warn(
            `Manager ID not set, retrying in ${retryDelay}ms (attempt ${attempt + 1}/${maxRetries})`,
          );
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
          continue;
        } else {
          throw new Error("Manager ID not set after maximum retries");
        }
      }

      try {
        const svc = await this.get_remote_service(
          `*/${this._connection.manager_id}:default`,
          config,
        );
        return svc;
      } catch (e) {
        lastError = e;
        console.warn(
          `Failed to get manager service (attempt ${attempt + 1}/${maxRetries}): ${e.message}`,
        );
        if (attempt < maxRetries - 1) {
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
        }
      }
    }

    throw lastError;
  }

  get_all_local_services() {
    return this._services;
  }
  get_local_service(service_id, context) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(context, "Context is required");

    const [ws, client_id] = context["to"].split("/");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      client_id === this._client_id,
      "Services can only be accessed locally",
    );

    const service = this._services[service_id];
    if (!service) {
      throw new Error("Service not found: " + service_id);
    }

    // Note: Do NOT mutate service.config.workspace here!
    // Doing so would corrupt the stored service config when called from
    // a different workspace (e.g., "public"), causing reconnection to fail
    // because _extract_service_info would use the wrong workspace value.

    // allow access for the same workspace
    if (
      service.config.visibility == "public" ||
      service.config.visibility == "unlisted"
    ) {
      return service;
    }

    // allow access for the same workspace
    if (context["ws"] === ws) {
      return service;
    }

    // Check if user is from an authorized workspace
    const authorized_workspaces = service.config.authorized_workspaces;
    if (
      authorized_workspaces &&
      authorized_workspaces.includes(context["ws"])
    ) {
      return service;
    }

    throw new Error(
      `Permission denied for getting protected service: ${service_id}, workspace mismatch: ${ws} != ${context["ws"]}`,
    );
  }
  async get_remote_service(service_uri, config) {
    let { timeout, case_conversion, kwargs_expansion } = config || {};
    timeout = timeout === undefined ? this._method_timeout : timeout;
    if (!service_uri && this._connection.manager_id) {
      service_uri = "*/" + this._connection.manager_id;
    } else if (!service_uri.includes(":")) {
      service_uri = this._client_id + ":" + service_uri;
    }
    const provider = service_uri.split(":")[0];
    let service_id = service_uri.split(":")[1];
    if (service_id.includes("@")) {
      service_id = service_id.split("@")[0];
      const app_id = service_uri.split("@")[1];
      if (this._app_id && this._app_id !== "*")
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          app_id === this._app_id,
          `Invalid app id: ${app_id} != ${this._app_id}`,
        );
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(provider, `Invalid service uri: ${service_uri}`);

    try {
      const method = this._generate_remote_method({
        _rserver: this._server_base_url,
        _rtarget: provider,
        _rmethod: "services.built-in.get_service",
        _rpromise: true,
        _rdoc: "Get a remote service",
      });
      let svc = await (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.waitFor)(
        method(service_id),
        timeout,
        "Timeout Error: Failed to get remote service: " + service_uri,
      );
      svc.id = `${provider}:${service_id}`;
      if (kwargs_expansion) {
        svc = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.expandKwargs)(svc);
      }
      if (case_conversion)
        return Object.assign(
          new RemoteService(),
          (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.convertCase)(svc, case_conversion),
        );
      else return Object.assign(new RemoteService(), svc);
    } catch (e) {
      console.warn("Failed to get remote service: " + service_uri, e);
      throw e;
    }
  }
  _annotate_service_methods(
    aObject,
    object_id,
    require_context,
    run_in_executor,
    visibility,
    authorized_workspaces,
  ) {
    if (typeof aObject === "function") {
      // mark the method as a remote method that requires context
      let method_name = object_id.split(".")[1];
      this._method_annotations.set(aObject, {
        require_context: Array.isArray(require_context)
          ? require_context.includes(method_name)
          : !!require_context,
        run_in_executor: run_in_executor,
        method_id: "services." + object_id,
        visibility: visibility,
        authorized_workspaces: authorized_workspaces,
      });
    } else if (aObject instanceof Array || aObject instanceof Object) {
      for (let key of Object.keys(aObject)) {
        let val = aObject[key];
        if (typeof val === "function" && val.__rpc_object__) {
          let client_id = val.__rpc_object__._rtarget;
          if (client_id.includes("/")) {
            client_id = client_id.split("/")[1];
          }
          if (this._client_id === client_id) {
            if (aObject instanceof Array) {
              aObject = aObject.slice();
            }
            // recover local method
            aObject[key] = indexObject(
              this._object_store,
              val.__rpc_object__._rmethod,
            );
            val = aObject[key]; // make sure it's annotated later
          } else {
            throw new Error(
              `Local method not found: ${val.__rpc_object__._rmethod}, client id mismatch ${this._client_id} != ${client_id}`,
            );
          }
        }
        this._annotate_service_methods(
          val,
          object_id + "." + key,
          require_context,
          run_in_executor,
          visibility,
          authorized_workspaces,
        );
      }
    }
  }
  add_service(api, overwrite) {
    if (!api || Array.isArray(api)) throw new Error("Invalid service object");
    if (api.constructor === Object) {
      api = Object.assign({}, api);
    } else {
      const normApi = {};
      const props = Object.getOwnPropertyNames(api).concat(
        Object.getOwnPropertyNames(Object.getPrototypeOf(api)),
      );
      for (let k of props) {
        if (k !== "constructor") {
          if (typeof api[k] === "function") normApi[k] = api[k].bind(api);
          else normApi[k] = api[k];
        }
      }
      // For class instance, we need set a default id
      api.id = api.id || "default";
      api = normApi;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      api.id && typeof api.id === "string",
      `Service id not found: ${api}`,
    );
    if (!api.name) {
      api.name = api.id;
    }
    if (!api.config) {
      api.config = {};
    }
    if (!api.type) {
      api.type = "generic";
    }
    // require_context only applies to the top-level functions
    let require_context = false,
      run_in_executor = false;
    if (api.config.require_context)
      require_context = api.config.require_context;
    if (api.config.run_in_executor) run_in_executor = true;
    const visibility = api.config.visibility || "protected";
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(["protected", "public", "unlisted"].includes(visibility));

    // Validate authorized_workspaces
    const authorized_workspaces = api.config.authorized_workspaces;
    if (authorized_workspaces !== undefined) {
      if (visibility !== "protected") {
        throw new Error(
          `authorized_workspaces can only be set when visibility is 'protected', got visibility='${visibility}'`,
        );
      }
      if (!Array.isArray(authorized_workspaces)) {
        throw new Error(
          "authorized_workspaces must be an array of workspace ids",
        );
      }
      for (const ws_id of authorized_workspaces) {
        if (typeof ws_id !== "string") {
          throw new Error(
            `Each workspace id in authorized_workspaces must be a string, got ${typeof ws_id}`,
          );
        }
      }
    }
    this._annotate_service_methods(
      api,
      api["id"],
      require_context,
      run_in_executor,
      visibility,
      authorized_workspaces,
    );

    if (this._services[api.id]) {
      if (overwrite) {
        delete this._services[api.id];
      } else {
        throw new Error(
          `Service already exists: ${api.id}, please specify a different id (not ${api.id}) or overwrite=true`,
        );
      }
    }
    this._services[api.id] = api;
    return api;
  }

  _extract_service_info(service) {
    const config = service.config || {};
    config.workspace =
      config.workspace || this._local_workspace || this._connection.workspace;
    if (!config.workspace) {
      throw new Error(
        "Workspace is not set. Please ensure the connection has a workspace or set local_workspace.",
      );
    }
    const skipContext = config.require_context;
    const excludeKeys = [
      "id",
      "config",
      "name",
      "description",
      "type",
      "docs",
      "app_id",
      "service_schema",
    ];
    const filteredService = {};
    for (const key of Object.keys(service)) {
      if (!excludeKeys.includes(key)) {
        filteredService[key] = service[key];
      }
    }
    const serviceSchema = _get_schema(filteredService, null, skipContext);
    const serviceInfo = {
      config: config,
      id: `${config.workspace}/${this._client_id}:${service["id"]}`,
      name: service.name || service["id"],
      description: service.description || "",
      type: service.type || "generic",
      docs: service.docs || null,
      app_id: this._app_id,
      service_schema: serviceSchema,
    };
    return serviceInfo;
  }

  async get_service_schema(service) {
    const skipContext = service.config.require_context;
    return _get_schema(service, null, skipContext);
  }

  async register_service(api, config) {
    let { check_type, notify, overwrite } = config || {};
    notify = notify === undefined ? true : notify;
    let manager;
    if (check_type && api.type) {
      try {
        manager = await this.get_manager_service({
          timeout: 10,
          case_conversion: "camel",
        });
        const type_info = await manager.get_service_type(api.type);
        api = _annotate_service(api, type_info);
      } catch (e) {
        throw new Error(`Failed to get service type ${api.type}, error: ${e}`);
      }
    }

    const service = this.add_service(api, overwrite);
    const serviceInfo = this._extract_service_info(service);
    if (notify) {
      try {
        manager =
          manager ||
          (await this.get_manager_service({
            timeout: 10,
            case_conversion: "camel",
          }));
        await manager.registerService(serviceInfo);
      } catch (e) {
        throw new Error(`Failed to notify workspace manager: ${e}`);
      }
    }
    return serviceInfo;
  }

  async unregister_service(service, notify) {
    notify = notify === undefined ? true : notify;
    let service_id;
    if (typeof service === "string") {
      service_id = service;
    } else {
      service_id = service.id;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      service_id && typeof service_id === "string",
      `Invalid service id: ${service_id}`,
    );
    if (service_id.includes(":")) {
      service_id = service_id.split(":")[1];
    }
    if (service_id.includes("@")) {
      service_id = service_id.split("@")[0];
    }
    if (!this._services[service_id]) {
      throw new Error(`Service not found: ${service_id}`);
    }
    if (notify) {
      const manager = await this.get_manager_service({
        timeout: 10,
        case_conversion: "camel",
      });
      await manager.unregisterService(service_id);
    }
    delete this._services[service_id];
  }

  _ndarray(typedArray, shape, dtype) {
    const _dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(typedArray);
    if (dtype && dtype !== _dtype) {
      throw (
        "dtype doesn't match the type of the array: " + _dtype + " != " + dtype
      );
    }
    shape = shape || [typedArray.length];
    return {
      _rtype: "ndarray",
      _rvalue: typedArray.buffer,
      _rshape: shape,
      _rdtype: _dtype,
    };
  }

  _encode_callback(
    name,
    callback,
    session_id,
    clear_after_called,
    timer,
    local_workspace,
    description,
  ) {
    let method_id = `${session_id}.${name}`;
    let encoded = {
      _rtype: "method",
      _rtarget: local_workspace
        ? `${local_workspace}/${this._client_id}`
        : this._client_id,
      _rmethod: method_id,
      _rpromise: false,
    };

    const self = this;
    let wrapped_callback = function () {
      try {
        callback.apply(null, Array.prototype.slice.call(arguments));
      } catch (error) {
        console.error(
          `Error in callback(${method_id}, ${description}): ${error}`,
        );
      } finally {
        // Clear the timer first if it exists
        if (timer && timer.started) {
          timer.clear();
        }

        // Clean up the entire session when resolve/reject is called
        if (clear_after_called && self._object_store[session_id]) {
          // For promise callbacks (resolve/reject), clean up the entire session
          if (name === "resolve" || name === "reject") {
            self._removeFromTargetIdIndex(session_id);
            delete self._object_store[session_id];
          } else {
            // For other callbacks, just clean up this specific callback
            self._cleanup_session_if_needed(session_id, name);
          }
        }
      }
    };
    wrapped_callback.__name__ = `callback(${method_id})`;
    return [encoded, wrapped_callback];
  }

  _cleanup_session_if_needed(session_id, callback_name) {
    /**
     * Clean session management - all logic in one place.
     */
    if (!session_id) {
      console.debug("Cannot cleanup session: session_id is empty");
      return;
    }

    try {
      const store = this._get_session_store(session_id, false);
      if (!store) {
        console.debug(`Session ${session_id} not found for cleanup`);
        return;
      }

      let should_cleanup = false;

      // Promise sessions: let the promise manager decide cleanup
      if (store._promise_manager) {
        try {
          const promise_manager = store._promise_manager;
          if (
            promise_manager.should_cleanup_on_callback &&
            promise_manager.should_cleanup_on_callback(callback_name)
          ) {
            if (promise_manager.settle) {
              promise_manager.settle();
            }
            should_cleanup = true;
            console.debug(
              `Promise session ${session_id} settled and marked for cleanup`,
            );
          }
        } catch (e) {
          console.warn(
            `Error in promise manager cleanup for session ${session_id}:`,
            e,
          );
        }
      } else {
        // Regular sessions: only cleanup temporary method call sessions
        // Don't cleanup service registration sessions or persistent sessions
        // Only cleanup sessions that are clearly temporary promises for method calls
        if (
          (callback_name === "resolve" || callback_name === "reject") &&
          store._callbacks &&
          Object.keys(store._callbacks).includes(callback_name)
        ) {
          should_cleanup = true;
          console.debug(
            `Regular session ${session_id} marked for cleanup after ${callback_name}`,
          );
        }
      }

      if (should_cleanup) {
        this._cleanup_session_completely(session_id);
      }
    } catch (error) {
      console.warn(`Error during session cleanup for ${session_id}:`, error);
    }
  }

  _cleanup_session_completely(session_id) {
    /**
     * Complete session cleanup with resource management.
     */
    try {
      // Clean up target_id index before deleting the session
      this._removeFromTargetIdIndex(session_id);

      const store = this._get_session_store(session_id, false);
      if (!store) {
        console.debug(`Session ${session_id} already cleaned up`);
        return;
      }

      // Clean up resources before removing session
      if (
        store.timer &&
        store.timer.started &&
        typeof store.timer.clear === "function"
      ) {
        try {
          store.timer.clear();
        } catch (error) {
          console.warn(
            `Error clearing timer for session ${session_id}:`,
            error,
          );
        }
      }

      if (
        store.heartbeat_task &&
        typeof store.heartbeat_task.cancel === "function"
      ) {
        try {
          store.heartbeat_task.cancel();
        } catch (error) {
          console.warn(
            `Error canceling heartbeat for session ${session_id}:`,
            error,
          );
        }
      }

      // Navigate and clean session path
      const levels = session_id.split(".");
      let current_store = this._object_store;

      // Navigate to parent of target level
      for (let i = 0; i < levels.length - 1; i++) {
        const level = levels[i];
        if (!current_store[level]) {
          console.debug(
            `Session path ${session_id} not found at level ${level}`,
          );
          return;
        }
        current_store = current_store[level];
      }

      // Delete the final level
      const final_key = levels[levels.length - 1];
      if (current_store[final_key]) {
        delete current_store[final_key];
        console.debug(`Cleaned up session ${session_id}`);

        // Clean up empty parent containers
        this._cleanup_empty_containers(levels.slice(0, -1));
      }
    } catch (error) {
      console.warn(
        `Error in complete session cleanup for ${session_id}:`,
        error,
      );
    }
  }

  _cleanup_empty_containers(path_levels) {
    /**
     * Clean up empty parent containers to prevent memory leaks.
     */
    try {
      // Work backwards from the deepest level
      for (let depth = path_levels.length - 1; depth >= 0; depth--) {
        let current_store = this._object_store;

        // Navigate to parent of current depth
        for (let i = 0; i < depth; i++) {
          current_store = current_store[path_levels[i]];
          if (!current_store) return; // Path doesn't exist
        }

        // Check if container at current depth is empty
        const container_key = path_levels[depth];
        const container = current_store[container_key];

        if (
          container &&
          typeof container === "object" &&
          Object.keys(container).length === 0
        ) {
          delete current_store[container_key];
          console.debug(
            `Cleaned up empty container at depth ${depth}: ${path_levels.slice(0, depth + 1).join(".")}`,
          );
        } else {
          // Container is not empty, stop cleanup
          break;
        }
      }
    } catch (error) {
      console.warn("Error cleaning up empty containers:", error);
    }
  }

  get_session_stats() {
    /**
     * Get detailed session statistics.
     */
    const stats = {
      total_sessions: 0,
      promise_sessions: 0,
      regular_sessions: 0,
      sessions_with_timers: 0,
      sessions_with_heartbeat: 0,
      system_stores: {},
      session_ids: [],
      memory_usage: 0,
    };

    if (!this._object_store) {
      return stats;
    }

    for (const key in this._object_store) {
      const value = this._object_store[key];

      if (["services", "message_cache"].includes(key)) {
        // System stores - don't count these as sessions
        stats.system_stores[key] = {
          size:
            typeof value === "object" && value ? Object.keys(value).length : 0,
        };
        continue;
      }

      // Count all non-system non-empty objects as sessions
      if (value && typeof value === "object") {
        const sessionKeys = Object.keys(value);

        // Only skip completely empty objects
        if (sessionKeys.length > 0) {
          stats.total_sessions++;
          stats.session_ids.push(key);

          if (value._promise_manager) {
            stats.promise_sessions++;
          } else {
            stats.regular_sessions++;
          }

          if (value._timer || value.timer) stats.sessions_with_timers++;
          if (value._heartbeat || value.heartbeat)
            stats.sessions_with_heartbeat++;

          // Estimate memory usage
          stats.memory_usage += JSON.stringify(value).length;
        }
      }
    }

    return stats;
  }

  _force_cleanup_all_sessions() {
    /**
     * Force cleanup all sessions (for testing purposes).
     */
    if (!this._object_store) {
      console.debug("Force cleaning up 0 sessions");
      return;
    }

    let cleaned_count = 0;
    const keys_to_delete = [];

    for (const key in this._object_store) {
      // Don't delete system stores
      if (!["services", "message_cache"].includes(key)) {
        const value = this._object_store[key];
        if (
          value &&
          typeof value === "object" &&
          Object.keys(value).length > 0
        ) {
          keys_to_delete.push(key);
          cleaned_count++;
        }
      }
    }

    // Delete the sessions
    for (const key of keys_to_delete) {
      delete this._object_store[key];
    }

    // Clear the target_id index since all sessions are removed
    this._targetIdIndex = {};

    console.debug(`Force cleaning up ${cleaned_count} sessions`);
  }

  _sweepStaleSessions() {
    const now = Date.now();
    let swept = 0;
    for (const key of Object.keys(this._object_store)) {
      if (key === "services" || key === "message_cache") continue;
      const session = this._object_store[key];
      if (
        session &&
        typeof session === "object" &&
        session._created_at &&
        now - session._created_at > this._sessionMaxAge
      ) {
        // Only sweep sessions that have no timer (active timers mean they are in use)
        // and no active promise callbacks (resolve/reject mean the session is awaiting a response)
        if (!session.timer || !session.timer.started) {
          if (
            typeof session.resolve === "function" ||
            typeof session.reject === "function"
          ) {
            // Session still has active promise callbacks, skip it
            continue;
          }
          this._removeFromTargetIdIndex(key);
          if (session.heartbeat_task) clearInterval(session.heartbeat_task);
          delete this._object_store[key];
          swept++;
        }
      }
    }
    if (swept > 0) {
      console.debug(`Swept ${swept} stale session(s)`);
    }
  }

  // Clean helper to identify promise method calls by session type
  _is_promise_method_call(method_path) {
    const session_id = method_path.split(".")[0];
    const session = this._get_session_store(session_id, false);
    return session && session._promise_manager;
  }

  // Simplified Promise Manager - enhanced version
  _create_promise_manager() {
    /**
     * Create a promise manager to track promise state and decide cleanup.
     */
    return {
      should_cleanup_on_callback: (callback_name) => {
        return ["resolve", "reject"].includes(callback_name);
      },
      settle: () => {
        // Promise is settled (resolved or rejected)
        console.debug("Promise settled");
      },
    };
  }

  async _encode_promise(
    resolve,
    reject,
    session_id,
    clear_after_called,
    timer,
    local_workspace,
    description,
  ) {
    let store = this._get_session_store(session_id, true);
    if (!store) {
      console.warn(
        `Failed to create session store ${session_id}, session management may be impaired`,
      );
      store = {};
    }

    // Clean promise lifecycle management - TYPE-BASED, not string-based
    store._promise_manager = this._create_promise_manager();

    let encoded = {};

    if (timer && reject && this._method_timeout) {
      [encoded.heartbeat, store.heartbeat] = this._encode_callback(
        "heartbeat",
        timer.reset.bind(timer),
        session_id,
        false,
        null,
        local_workspace,
      );
      store.timer = timer;
      encoded.interval = this._method_timeout / 2;
    } else {
      timer = null;
    }

    [encoded.resolve, store.resolve] = this._encode_callback(
      "resolve",
      resolve,
      session_id,
      clear_after_called,
      timer,
      local_workspace,
      `resolve (${description})`,
    );
    [encoded.reject, store.reject] = this._encode_callback(
      "reject",
      reject,
      session_id,
      clear_after_called,
      timer,
      local_workspace,
      `reject (${description})`,
    );
    return encoded;
  }

  async _send_chunks(data, target_id, session_id) {
    // 1) Get the remote service
    const remote_services = await this.get_remote_service(
      `${target_id}:built-in`,
    );
    if (!remote_services.message_cache) {
      throw new Error(
        "Remote client does not support message caching for large messages.",
      );
    }

    const message_cache = remote_services.message_cache;
    const message_id = session_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
    const total_size = data.length;
    const start_time = Date.now(); // measure time
    const chunk_num = Math.ceil(total_size / this._long_message_chunk_size);
    if (remote_services.config.api_version >= 3) {
      await message_cache.create(message_id, !!session_id);
      const semaphore = new _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.Semaphore(CONCURRENCY_LIMIT);

      const tasks = [];
      for (let idx = 0; idx < chunk_num; idx++) {
        const startByte = idx * this._long_message_chunk_size;
        const chunk = data.slice(
          startByte,
          startByte + this._long_message_chunk_size,
        );

        const taskFn = async () => {
          await message_cache.set(message_id, idx, chunk, !!session_id);
          // console.debug(
          //   `Sending chunk ${idx + 1}/${chunk_num} (total=${total_size} bytes)`,
          // );
        };

        // Push into an array, each one runs under the semaphore
        tasks.push(semaphore.run(taskFn));
      }

      // Wait for all chunk uploads to finish
      try {
        await Promise.all(tasks);
      } catch (error) {
        // If any chunk fails, clean up the message cache
        try {
          await message_cache.remove(message_id);
        } catch (cleanupError) {
          console.error(
            `Failed to clean up message cache after error: ${cleanupError}`,
          );
        }
        throw error;
      }
    } else {
      // 3) Legacy version (sequential appends):
      await message_cache.create(message_id, !!session_id);
      for (let idx = 0; idx < chunk_num; idx++) {
        const startByte = idx * this._long_message_chunk_size;
        const chunk = data.slice(
          startByte,
          startByte + this._long_message_chunk_size,
        );
        await message_cache.append(message_id, chunk, !!session_id);
        // console.debug(
        //   `Sending chunk ${idx + 1}/${chunk_num} (total=${total_size} bytes)`,
        // );
      }
    }
    await message_cache.process(message_id, !!session_id);
    const durationSec = ((Date.now() - start_time) / 1000).toFixed(2);
    // console.debug(`All chunks (${total_size} bytes) sent in ${durationSec} s`);
  }

  emit(main_message, extra_data) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      typeof main_message === "object" && main_message.type,
      "Invalid message, must be an object with a `type` fields.",
    );
    if (!main_message.to) {
      this._fire(main_message.type, main_message);
      return;
    }
    let message_package = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)(main_message);
    if (extra_data) {
      const extra = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)(extra_data);
      const combined = new Uint8Array(message_package.length + extra.length);
      combined.set(message_package);
      combined.set(extra, message_package.length);
      message_package = combined;
    }
    const total_size = message_package.length;
    if (total_size > this._long_message_chunk_size + 1024) {
      console.warn(`Sending large message (size=${total_size})`);
    }
    return this._emit_message(message_package);
  }

  _generate_remote_method(
    encoded_method,
    remote_parent,
    local_parent,
    remote_workspace,
    local_workspace,
  ) {
    let target_id = encoded_method._rtarget;
    if (remote_workspace && !target_id.includes("/")) {
      // Don't modify target_id if it starts with */ (workspace manager service)
      if (!target_id.startsWith("*/")) {
        if (remote_workspace !== target_id) {
          target_id = remote_workspace + "/" + target_id;
        }
        // Fix the target id to be an absolute id
        encoded_method._rtarget = target_id;
      }
    }
    let method_id = encoded_method._rmethod;
    let with_promise = encoded_method._rpromise || false;
    const description = `method: ${method_id}, docs: ${encoded_method._rdoc}`;
    const self = this;

    function remote_method() {
      return new Promise(async (resolve, reject) => {
        try {
          let local_session_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
          if (local_parent) {
            // Store the children session under the parent
            local_session_id = local_parent + "." + local_session_id;
          }
          let store = self._get_session_store(local_session_id, true);
          if (!store) {
            reject(
              new Error(
                `Runtime Error: Failed to get session store ${local_session_id} (context: ${description})`,
              ),
            );
            return;
          }
          store["target_id"] = target_id;
          // Update target_id index for fast session cleanup
          const topKey = local_session_id.split(".")[0];
          if (!(target_id in self._targetIdIndex)) {
            self._targetIdIndex[target_id] = new Set();
          }
          self._targetIdIndex[target_id].add(topKey);
          const args = await self._encode(
            Array.prototype.slice.call(arguments),
            local_session_id,
            local_workspace,
          );
          const argLength = args.length;
          // if the last argument is an object, mark it as kwargs
          const withKwargs =
            argLength > 0 &&
            typeof args[argLength - 1] === "object" &&
            args[argLength - 1] !== null &&
            args[argLength - 1]._rkwargs;
          if (withKwargs) delete args[argLength - 1]._rkwargs;

          let from_client;
          if (!self._local_workspace) {
            from_client = self._client_id;
          } else {
            from_client = self._local_workspace + "/" + self._client_id;
          }

          let main_message = {
            type: "method",
            from: from_client,
            to: target_id,
            method: method_id,
          };
          let extra_data = {};
          if (args) {
            extra_data["args"] = args;
          }
          if (withKwargs) {
            extra_data["with_kwargs"] = withKwargs;
          }

          // console.log(
          //   `Calling remote method ${target_id}:${method_id}, session: ${local_session_id}`
          // );
          if (remote_parent) {
            // Set the parent session
            // Note: It's a session id for the remote, not the current client
            main_message["parent"] = remote_parent;
          }

          let timer = null;
          if (with_promise) {
            // Only pass the current session id to the remote
            // if we want to received the result
            // I.e. the session id won't be passed for promises themselves
            main_message["session"] = local_session_id;
            let method_name = `${target_id}:${method_id}`;

            // Create a timer that gets reset by heartbeat
            // Methods can run indefinitely as long as heartbeat keeps resetting the timer
            // IMPORTANT: When timeout occurs, we must clean up the session to prevent memory leaks
            const timeoutCallback = function (error_msg) {
              // First reject the promise - wrap in Error for proper stack traces
              reject(new Error(error_msg));
              // Then clean up the entire session to stop all callbacks
              if (self._object_store[local_session_id]) {
                // Clean up target_id index before deleting the session
                self._removeFromTargetIdIndex(local_session_id);
                delete self._object_store[local_session_id];
                console.debug(
                  `Cleaned up session ${local_session_id} after timeout`,
                );
              }
            };

            timer = new Timer(
              self._method_timeout,
              timeoutCallback,
              [
                `Method call timed out: ${method_name}, context: ${description}`,
              ],
              method_name,
            );
            // By default, hypha will clear the session after the method is called
            // However, if the args contains _rintf === true, we will not clear the session

            // Helper function to recursively check for _rintf objects
            function hasInterfaceObject(obj) {
              if (!obj || typeof obj !== "object") return false;
              if (obj._rintf === true) return true;
              if (Array.isArray(obj)) {
                return obj.some((item) => hasInterfaceObject(item));
              }
              if (obj.constructor === Object) {
                return Object.values(obj).some((value) =>
                  hasInterfaceObject(value),
                );
              }
              return false;
            }

            let clear_after_called = !hasInterfaceObject(args);

            const promiseData = await self._encode_promise(
              resolve,
              reject,
              local_session_id,
              clear_after_called,
              timer,
              local_workspace,
              description,
            );

            if (with_promise === true) {
              extra_data["promise"] = promiseData;
            } else if (with_promise === "*") {
              extra_data["promise"] = "*";
              extra_data["t"] = self._method_timeout / 2;
            } else {
              throw new Error(`Unsupported promise type: ${with_promise}`);
            }
          }
          // The message consists of two segments, the main message and extra data
          let message_package = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)(main_message);
          if (extra_data) {
            const extra = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)(extra_data);
            const combined = new Uint8Array(
              message_package.length + extra.length,
            );
            combined.set(message_package);
            combined.set(extra, message_package.length);
            message_package = combined;
          }
          const total_size = message_package.length;
          if (
            total_size <= self._long_message_chunk_size + 1024 ||
            remote_method.__no_chunk__
          ) {
            self
              ._emit_message(message_package)
              .then(function () {
                if (timer) {
                  // Start the timer after message is sent successfully
                  timer.start();
                }
                if (!with_promise) {
                  // Fire-and-forget: resolve immediately after message is sent.
                  // Without this, the promise never resolves because no response
                  // is expected. This is critical for heartbeat callbacks which
                  // use _rpromise=false and are awaited in a loop.
                  resolve(null);
                }
              })
              .catch(function (err) {
                const error_msg = `Failed to send the request when calling method (${target_id}:${method_id}), error: ${err}`;
                if (reject) {
                  reject(new Error(error_msg));
                } else {
                  // No reject callback available, log the error to prevent unhandled promise rejections
                  console.warn("Unhandled RPC method call error:", error_msg);
                }
                if (timer && timer.started) {
                  timer.clear();
                }
              });
          } else {
            // send chunk by chunk
            self
              ._send_chunks(message_package, target_id, remote_parent)
              .then(function () {
                if (timer) {
                  // Start the timer after message is sent successfully
                  timer.start();
                }
                if (!with_promise) {
                  // Fire-and-forget: resolve immediately after message is sent
                  resolve(null);
                }
              })
              .catch(function (err) {
                const error_msg = `Failed to send the request when calling method (${target_id}:${method_id}), error: ${err}`;
                if (reject) {
                  reject(new Error(error_msg));
                } else {
                  // No reject callback available, log the error to prevent unhandled promise rejections
                  console.warn("Unhandled RPC method call error:", error_msg);
                }
                if (timer && timer.started) {
                  timer.clear();
                }
              });
          }
        } catch (err) {
          reject(err);
        }
      });
    }

    // Generate debugging information for the method
    remote_method.__rpc_object__ = encoded_method;
    const parts = method_id.split(".");

    remote_method.__name__ = encoded_method._rname || parts[parts.length - 1];
    if (remote_method.__name__.includes("#")) {
      remote_method.__name__ = remote_method.__name__.split("#")[1];
    }
    remote_method.__doc__ =
      encoded_method._rdoc || `Remote method: ${method_id}`;
    remote_method.__schema__ = encoded_method._rschema;
    // Prevent circular chunk sending
    remote_method.__no_chunk__ =
      encoded_method._rmethod === "services.built-in.message_cache.append";
    return remote_method;
  }

  get_client_info() {
    const services = [];
    for (let service of Object.values(this._services)) {
      services.push(this._extract_service_info(service));
    }

    return {
      id: this._client_id,
      services: services,
    };
  }

  async _handle_method(data) {
    let resolve = null;
    let reject = null;
    let heartbeat_task = null;
    try {
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data.method && data.ctx && data.from);
      const method_name = data.from + ":" + data.method;
      const remote_workspace = data.from.split("/")[0];
      const remote_client_id = data.from.split("/")[1];
      // Make sure the target id is an absolute id
      data["to"] = data["to"].includes("/")
        ? data["to"]
        : remote_workspace + "/" + data["to"];
      data["ctx"]["to"] = data["to"];
      let local_workspace;
      if (!this._local_workspace) {
        local_workspace = data["to"].split("/")[0];
      } else {
        if (this._local_workspace && this._local_workspace !== "*") {
          (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
            data["to"].split("/")[0] === this._local_workspace,
            "Workspace mismatch: " +
              data["to"].split("/")[0] +
              " != " +
              this._local_workspace,
          );
        }
        local_workspace = this._local_workspace;
      }
      const local_parent = data.parent;

      if (data.promise) {
        // Decode the promise with the remote session id
        // Such that the session id will be passed to the remote as a parent session id
        const promise = await this._decode(
          data.promise === "*" ? this._expand_promise(data) : data.promise,
          data.session,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        resolve = promise.resolve;
        reject = promise.reject;
        if (promise.heartbeat && promise.interval) {
          async function heartbeat() {
            try {
              // console.debug("Reset heartbeat timer: " + data.method);
              await promise.heartbeat();
            } catch (err) {
              console.error(err);
            }
          }
          heartbeat_task = setInterval(heartbeat, promise.interval * 1000);
          // Store the heartbeat task in the session store for cleanup
          if (data.session) {
            const session_store = this._get_session_store(data.session, false);
            if (session_store) {
              session_store.heartbeat_task = heartbeat_task;
            }
          }
        }
      }

      let method;

      try {
        method = indexObject(this._object_store, data["method"]);
      } catch (e) {
        // Clean promise method detection - TYPE-BASED, not string-based
        if (this._is_promise_method_call(data["method"])) {
          console.debug(
            `Promise method ${data["method"]} not available (detected by session type), ignoring: ${method_name}`,
          );
          return;
        }

        // Check if this is a session-based method call that might have expired
        const method_parts = data["method"].split(".");
        if (method_parts.length > 1) {
          const session_id = method_parts[0];
          // Check if the session exists but the specific method doesn't
          if (session_id in this._object_store) {
            console.debug(
              `Session ${session_id} exists but method ${data["method"]} not found, likely expired callback: ${method_name}`,
            );
            // For expired callbacks, don't throw an exception, just log and return
            if (typeof reject === "function") {
              reject(new Error(`Method expired or not found: ${method_name}`));
            }
            return;
          } else {
            console.debug(
              `Session ${session_id} not found for method ${data["method"]}, likely cleaned up: ${method_name}`,
            );
            // For cleaned up sessions, just log and return without throwing
            if (typeof reject === "function") {
              reject(new Error(`Session not found: ${method_name}`));
            }
            return;
          }
        }

        console.debug(
          `Failed to find method ${method_name} at ${this._client_id}`,
        );
        const error = new Error(
          `Method not found: ${method_name} at ${this._client_id}`,
        );
        if (typeof reject === "function") {
          reject(error);
        } else {
          // Log the error instead of throwing to prevent unhandled exceptions
          console.warn(
            "Method not found and no reject callback:",
            error.message,
          );
        }
        return;
      }

      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        method && typeof method === "function",
        "Invalid method: " + method_name,
      );

      // Check permission
      if (this._method_annotations.has(method)) {
        // For services, it should not be protected
        if (this._method_annotations.get(method).visibility === "protected") {
          // Allow access from same workspace
          if (local_workspace === remote_workspace) {
            // Access granted
          }
          // Check if remote workspace is in authorized_workspaces list
          else if (
            this._method_annotations.get(method).authorized_workspaces &&
            this._method_annotations
              .get(method)
              .authorized_workspaces.includes(remote_workspace)
          ) {
            // Access granted
          }
          // Allow manager access
          else if (
            remote_workspace === "*" &&
            remote_client_id === this._connection.manager_id
          ) {
            // Access granted
          } else {
            throw new Error(
              "Permission denied for invoking protected method " +
                method_name +
                ", workspace mismatch: " +
                local_workspace +
                " != " +
                remote_workspace,
            );
          }
        }
      } else {
        // For sessions, the target_id should match exactly
        let session_target_id =
          this._object_store[data.method.split(".")[0]].target_id;
        if (
          local_workspace === remote_workspace &&
          session_target_id &&
          session_target_id.indexOf("/") === -1
        ) {
          session_target_id = local_workspace + "/" + session_target_id;
        }
        if (session_target_id !== data.from) {
          throw new Error(
            "Access denied for method call (" +
              method_name +
              ") from " +
              data.from +
              " to target " +
              session_target_id,
          );
        }
      }

      // Make sure the parent session is still open
      if (local_parent) {
        // The parent session should be a session that generate the current method call
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          this._get_session_store(local_parent, true) !== null,
          "Parent session was closed: " + local_parent,
        );
      }
      let args;
      if (data.args) {
        args = await this._decode(
          data.args,
          data.session,
          null,
          remote_workspace,
          null,
        );
      } else {
        args = [];
      }
      if (
        this._method_annotations.has(method) &&
        this._method_annotations.get(method).require_context
      ) {
        // if args.length + 1 is less than the required number of arguments we will pad with undefined
        // so we make sure the last argument is the context
        if (args.length + 1 < method.length) {
          for (let i = args.length; i < method.length - 1; i++) {
            args.push(undefined);
          }
        }
        args.push(data.ctx);
        // assert(
        //   args.length === method.length,
        //   `Runtime Error: Invalid number of arguments for method ${method_name}, expected ${method.length} but got ${args.length}`,
        // );
      }
      // console.debug(`Executing method: ${method_name} (${data.method})`);
      if (data.promise) {
        const result = method.apply(null, args);
        if (result instanceof Promise) {
          result
            .then((result) => {
              resolve(result);
            })
            .catch((err) => {
              reject(err);
            })
            .finally(() => {
              clearInterval(heartbeat_task);
            });
        } else {
          resolve(result);
          clearInterval(heartbeat_task);
        }
      } else {
        method.apply(null, args);
        clearInterval(heartbeat_task);
      }
    } catch (err) {
      if (reject) {
        reject(err);
      } else {
        console.error("Error during calling method: ", err);
      }
      clearInterval(heartbeat_task);
    }
  }

  encode(aObject, session_id) {
    return this._encode(aObject, session_id);
  }

  _get_session_store(session_id, create) {
    if (!session_id) {
      return null;
    }
    let store = this._object_store;
    const levels = session_id.split(".");
    if (create) {
      const last_index = levels.length - 1;
      for (let level of levels.slice(0, last_index)) {
        if (!store[level]) {
          // Instead of returning null, create intermediate sessions as needed
          store[level] = {};
        }
        store = store[level];
      }
      // Create the last level
      if (!store[levels[last_index]]) {
        store[levels[last_index]] = {};
        store[levels[last_index]]._created_at = Date.now();
      }
      return store[levels[last_index]];
    } else {
      for (let level of levels) {
        if (!store[level]) {
          return null;
        }
        store = store[level];
      }
      return store;
    }
  }

  /**
   * Prepares the provided set of remote method arguments for
   * sending to the remote site, replaces all the callbacks with
   * identifiers
   *
   * @param {Array} args to wrap
   *
   * @returns {Array} wrapped arguments
   */
  async _encode(aObject, session_id, local_workspace) {
    const aType = typeof aObject;
    if (
      aType === "number" ||
      aType === "string" ||
      aType === "boolean" ||
      aObject === null ||
      aObject === undefined ||
      aObject instanceof Uint8Array
    ) {
      return aObject;
    }
    if (aObject instanceof ArrayBuffer) {
      return {
        _rtype: "memoryview",
        _rvalue: new Uint8Array(aObject),
      };
    }
    // Reuse the remote object
    if (aObject.__rpc_object__) {
      const _server = aObject.__rpc_object__._rserver || this._server_base_url;
      if (_server === this._server_base_url) {
        return aObject.__rpc_object__;
      } // else {
      //   console.debug(
      //     `Encoding remote function from a different server ${_server}, current server: ${this._server_base_url}`,
      //   );
      // }
    }

    let bObject;

    // skip if already encoded
    if (aObject.constructor instanceof Object && aObject._rtype) {
      // make sure the interface functions are encoded
      const temp = aObject._rtype;
      delete aObject._rtype;
      bObject = await this._encode(aObject, session_id, local_workspace);
      bObject._rtype = temp;
      return bObject;
    }

    if ((0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isGenerator)(aObject) || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncGenerator)(aObject)) {
      // Handle generator functions and generator objects
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        session_id && typeof session_id === "string",
        "Session ID is required for generator encoding",
      );
      const object_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();

      // Get the session store
      const store = this._get_session_store(session_id, true);
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        store !== null,
        `Failed to create session store ${session_id} due to invalid parent`,
      );

      // Check if it's an async generator
      const isAsync = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncGenerator)(aObject);

      // Define method to get next item from the generator
      const nextItemMethod = async () => {
        if (isAsync) {
          const iterator = aObject;
          const result = await iterator.next();
          if (result.done) {
            delete store[object_id];
            return { _rtype: "stop_iteration" };
          }
          return result.value;
        } else {
          const iterator = aObject;
          const result = iterator.next();
          if (result.done) {
            delete store[object_id];
            return { _rtype: "stop_iteration" };
          }
          return result.value;
        }
      };

      // Store the next_item method in the session
      store[object_id] = nextItemMethod;

      // Create a method that will be used to fetch the next item from the generator
      bObject = {
        _rtype: "generator",
        _rserver: this._server_base_url,
        _rtarget: this._client_id,
        _rmethod: `${session_id}.${object_id}`,
        _rpromise: "*",
        _rdoc: "Remote generator",
      };
      return bObject;
    } else if (typeof aObject === "function") {
      if (this._method_annotations.has(aObject)) {
        let annotation = this._method_annotations.get(aObject);
        bObject = {
          _rtype: "method",
          _rserver: this._server_base_url,
          _rtarget: this._client_id,
          _rmethod: annotation.method_id,
          _rpromise: "*",
          _rname: aObject.name,
        };
      } else {
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(typeof session_id === "string");
        let object_id;
        if (aObject.__name__) {
          object_id = `${(0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)()}#${aObject.__name__}`;
        } else {
          object_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
        }
        bObject = {
          _rtype: "method",
          _rserver: this._server_base_url,
          _rtarget: this._client_id,
          _rmethod: `${session_id}.${object_id}`,
          _rpromise: "*",
          _rname: aObject.name,
        };
        let store = this._get_session_store(session_id, true);
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          store !== null,
          `Failed to create session store ${session_id} due to invalid parent`,
        );
        store[object_id] = aObject;
      }
      bObject._rdoc = aObject.__doc__;
      if (!bObject._rdoc) {
        try {
          const funcInfo = getFunctionInfo(aObject);
          if (funcInfo && !bObject._rdoc) {
            bObject._rdoc = `${funcInfo.doc}`;
          }
        } catch (e) {
          console.error("Failed to extract function docstring:", aObject);
        }
      }
      bObject._rschema = aObject.__schema__;
      return bObject;
    }
    const isarray = Array.isArray(aObject);

    for (let tp of Object.keys(this._codecs)) {
      const codec = this._codecs[tp];
      if (codec.encoder && aObject instanceof codec.type) {
        // TODO: what if multiple encoders found
        let encodedObj = await Promise.resolve(codec.encoder(aObject));
        if (encodedObj && !encodedObj._rtype) encodedObj._rtype = codec.name;
        // encode the functions in the interface object
        if (typeof encodedObj === "object") {
          const temp = encodedObj._rtype;
          delete encodedObj._rtype;
          encodedObj = await this._encode(
            encodedObj,
            session_id,
            local_workspace,
          );
          encodedObj._rtype = temp;
        }
        bObject = encodedObj;
        return bObject;
      }
    }

    if (
      /*global tf*/
      typeof tf !== "undefined" &&
      tf.Tensor &&
      aObject instanceof tf.Tensor
    ) {
      const v_buffer = aObject.dataSync();
      bObject = {
        _rtype: "ndarray",
        _rvalue: new Uint8Array(v_buffer.buffer),
        _rshape: aObject.shape,
        _rdtype: aObject.dtype,
      };
    } else if (
      /*global nj*/
      typeof nj !== "undefined" &&
      nj.NdArray &&
      aObject instanceof nj.NdArray
    ) {
      if (!aObject.selection || !aObject.selection.data) {
        throw new Error("Invalid NumJS array: missing selection or data");
      }
      const dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(aObject.selection.data);
      bObject = {
        _rtype: "ndarray",
        _rvalue: new Uint8Array(aObject.selection.data.buffer),
        _rshape: aObject.shape,
        _rdtype: dtype,
      };
    } else if (aObject instanceof Error) {
      console.error(aObject);
      bObject = {
        _rtype: "error",
        _rvalue: aObject.toString(),
        _rtrace: aObject.stack,
      };
    }
    // send objects supported by structure clone algorithm
    // https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm
    else if (
      aObject !== Object(aObject) ||
      aObject instanceof Boolean ||
      aObject instanceof String ||
      aObject instanceof Date ||
      aObject instanceof RegExp ||
      (typeof ImageData !== "undefined" && aObject instanceof ImageData) ||
      (typeof FileList !== "undefined" && aObject instanceof FileList) ||
      (typeof FileSystemDirectoryHandle !== "undefined" &&
        aObject instanceof FileSystemDirectoryHandle) ||
      (typeof FileSystemFileHandle !== "undefined" &&
        aObject instanceof FileSystemFileHandle) ||
      (typeof FileSystemHandle !== "undefined" &&
        aObject instanceof FileSystemHandle) ||
      (typeof FileSystemWritableFileStream !== "undefined" &&
        aObject instanceof FileSystemWritableFileStream)
    ) {
      bObject = aObject;
      // TODO: avoid object such as DynamicPlugin instance.
    } else if (aObject instanceof Blob) {
      let _current_pos = 0;
      async function read(length) {
        let blob;
        if (length) {
          blob = aObject.slice(_current_pos, _current_pos + length);
        } else {
          blob = aObject.slice(_current_pos);
        }
        const ret = new Uint8Array(await blob.arrayBuffer());
        _current_pos = _current_pos + ret.byteLength;
        return ret;
      }
      function seek(pos) {
        _current_pos = pos;
      }
      bObject = {
        _rtype: "iostream",
        _rnative: "js:blob",
        type: aObject.type,
        name: aObject.name,
        size: aObject.size,
        path: aObject._path || aObject.webkitRelativePath,
        read: await this._encode(read, session_id, local_workspace),
        seek: await this._encode(seek, session_id, local_workspace),
      };
    } else if (aObject instanceof ArrayBufferView) {
      const dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(aObject);
      bObject = {
        _rtype: "typedarray",
        _rvalue: new Uint8Array(aObject.buffer),
        _rdtype: dtype,
      };
    } else if (aObject instanceof DataView) {
      bObject = {
        _rtype: "memoryview",
        _rvalue: new Uint8Array(aObject.buffer),
      };
    } else if (aObject instanceof Set) {
      bObject = {
        _rtype: "set",
        _rvalue: await this._encode(
          Array.from(aObject),
          session_id,
          local_workspace,
        ),
      };
    } else if (aObject instanceof Map) {
      bObject = {
        _rtype: "orderedmap",
        _rvalue: await this._encode(
          Array.from(aObject),
          session_id,
          local_workspace,
        ),
      };
    } else if (
      aObject.constructor === Object ||
      Array.isArray(aObject) ||
      aObject instanceof RemoteService
    ) {
      // Fast path: if all values are primitives, return as-is
      if (isarray) {
        if (_allPrimitivesArray(aObject)) return aObject;
      } else if (
        !("_rtype" in aObject) &&
        !(aObject instanceof RemoteService)
      ) {
        if (_allPrimitivesObject(aObject)) return aObject;
      }
      bObject = isarray ? [] : {};
      const keys = Object.keys(aObject);
      for (let k of keys) {
        bObject[k] = await this._encode(
          aObject[k],
          session_id,
          local_workspace,
        );
      }
    } else {
      throw `hypha-rpc: Unsupported data type: ${aObject}, you can register a custom codec to encode/decode the object.`;
    }

    if (!bObject) {
      throw new Error("Failed to encode object");
    }
    return bObject;
  }

  async decode(aObject) {
    return await this._decode(aObject);
  }

  async _decode(
    aObject,
    remote_parent,
    local_parent,
    remote_workspace,
    local_workspace,
  ) {
    if (!aObject) {
      return aObject;
    }
    let bObject;
    if (aObject._rtype) {
      if (
        this._codecs[aObject._rtype] &&
        this._codecs[aObject._rtype].decoder
      ) {
        const temp = aObject._rtype;
        delete aObject._rtype;
        aObject = await this._decode(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        aObject._rtype = temp;

        bObject = await Promise.resolve(
          this._codecs[aObject._rtype].decoder(aObject),
        );
      } else if (aObject._rtype === "method") {
        bObject = this._generate_remote_method(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
      } else if (aObject._rtype === "generator") {
        // Create a method to fetch next items from the remote generator
        const gen_method = this._generate_remote_method(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );

        // Create an async generator proxy
        async function* asyncGeneratorProxy() {
          while (true) {
            try {
              const next_item = await gen_method();
              // Check for StopIteration signal
              if (next_item && next_item._rtype === "stop_iteration") {
                break;
              }
              yield next_item;
            } catch (error) {
              console.error("Error in generator:", error);
              throw error;
            }
          }
        }
        bObject = asyncGeneratorProxy();
      } else if (aObject._rtype === "ndarray") {
        /*global nj tf*/
        //create build array/tensor if used in the plugin
        if (typeof nj !== "undefined" && nj.array) {
          if (Array.isArray(aObject._rvalue)) {
            aObject._rvalue = aObject._rvalue.reduce(_appendBuffer);
          }
          bObject = nj
            .array(new Uint8(aObject._rvalue), aObject._rdtype)
            .reshape(aObject._rshape);
        } else if (typeof tf !== "undefined" && tf.Tensor) {
          if (Array.isArray(aObject._rvalue)) {
            aObject._rvalue = aObject._rvalue.reduce(_appendBuffer);
          }
          const arraytype = _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.dtypeToTypedArray[aObject._rdtype];
          bObject = tf.tensor(
            new arraytype(aObject._rvalue),
            aObject._rshape,
            aObject._rdtype,
          );
        } else {
          //keep it as regular if transfered to the main app
          bObject = aObject;
        }
      } else if (aObject._rtype === "error") {
        bObject = new Error(
          "RemoteError: " + aObject._rvalue + "\n" + (aObject._rtrace || ""),
        );
      } else if (aObject._rtype === "typedarray") {
        const arraytype = _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.dtypeToTypedArray[aObject._rdtype];
        if (!arraytype)
          throw new Error("unsupported dtype: " + aObject._rdtype);
        const buffer = aObject._rvalue.buffer.slice(
          aObject._rvalue.byteOffset,
          aObject._rvalue.byteOffset + aObject._rvalue.byteLength,
        );
        bObject = new arraytype(buffer);
      } else if (aObject._rtype === "memoryview") {
        bObject = aObject._rvalue.buffer.slice(
          aObject._rvalue.byteOffset,
          aObject._rvalue.byteOffset + aObject._rvalue.byteLength,
        ); // ArrayBuffer
      } else if (aObject._rtype === "iostream") {
        if (aObject._rnative === "js:blob") {
          const read = await this._generate_remote_method(
            aObject.read,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          );
          const bytes = await read();
          bObject = new Blob([bytes], {
            type: aObject.type,
            name: aObject.name,
          });
        } else {
          bObject = {};
          for (let k of Object.keys(aObject)) {
            if (!k.startsWith("_")) {
              bObject[k] = await this._decode(
                aObject[k],
                remote_parent,
                local_parent,
                remote_workspace,
                local_workspace,
              );
            }
          }
        }
        bObject["__rpc_object__"] = aObject;
      } else if (aObject._rtype === "orderedmap") {
        bObject = new Map(
          await this._decode(
            aObject._rvalue,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          ),
        );
      } else if (aObject._rtype === "set") {
        bObject = new Set(
          await this._decode(
            aObject._rvalue,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          ),
        );
      } else {
        const temp = aObject._rtype;
        delete aObject._rtype;
        bObject = await this._decode(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        bObject._rtype = temp;
      }
    } else if (aObject.constructor === Object || Array.isArray(aObject)) {
      const isarray = Array.isArray(aObject);
      // Fast path: skip recursive descent if all values are primitives
      if (isarray) {
        if (_allPrimitivesArray(aObject)) return aObject;
      } else {
        if (_allPrimitivesObject(aObject)) return aObject;
      }
      bObject = isarray ? [] : {};
      for (let k of Object.keys(aObject)) {
        if (isarray || aObject.hasOwnProperty(k)) {
          const v = aObject[k];
          bObject[k] = await this._decode(
            v,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          );
        }
      }
    } else {
      bObject = aObject;
    }
    if (bObject === undefined) {
      throw new Error("Failed to decode object");
    }
    return bObject;
  }

  _expand_promise(data) {
    return {
      heartbeat: {
        _rtype: "method",
        _rtarget: data.from.split("/")[1],
        _rmethod: data.session + ".heartbeat",
        _rdoc: `heartbeat callback for method: ${data.method}`,
      },
      resolve: {
        _rtype: "method",
        _rtarget: data.from.split("/")[1],
        _rmethod: data.session + ".resolve",
        _rdoc: `resolve callback for method: ${data.method}`,
      },
      reject: {
        _rtype: "method",
        _rtarget: data.from.split("/")[1],
        _rmethod: data.session + ".reject",
        _rdoc: `reject callback for method: ${data.method}`,
      },
      interval: data.t,
    };
  }
}


/***/ }),

/***/ "./src/utils/index.js":
/*!****************************!*\
  !*** ./src/utils/index.js ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   MessageEmitter: () => (/* binding */ MessageEmitter),
/* harmony export */   Semaphore: () => (/* binding */ Semaphore),
/* harmony export */   assert: () => (/* binding */ assert),
/* harmony export */   cacheRequirements: () => (/* binding */ cacheRequirements),
/* harmony export */   convertCase: () => (/* binding */ convertCase),
/* harmony export */   dtypeToTypedArray: () => (/* binding */ dtypeToTypedArray),
/* harmony export */   expandKwargs: () => (/* binding */ expandKwargs),
/* harmony export */   isAsyncGenerator: () => (/* binding */ isAsyncGenerator),
/* harmony export */   isGenerator: () => (/* binding */ isGenerator),
/* harmony export */   loadRequirements: () => (/* binding */ loadRequirements),
/* harmony export */   loadRequirementsInWebworker: () => (/* binding */ loadRequirementsInWebworker),
/* harmony export */   loadRequirementsInWindow: () => (/* binding */ loadRequirementsInWindow),
/* harmony export */   normalizeConfig: () => (/* binding */ normalizeConfig),
/* harmony export */   parseServiceUrl: () => (/* binding */ parseServiceUrl),
/* harmony export */   randId: () => (/* binding */ randId),
/* harmony export */   toCamelCase: () => (/* binding */ toCamelCase),
/* harmony export */   toSnakeCase: () => (/* binding */ toSnakeCase),
/* harmony export */   typedArrayToDtype: () => (/* binding */ typedArrayToDtype),
/* harmony export */   typedArrayToDtypeMapping: () => (/* binding */ typedArrayToDtypeMapping),
/* harmony export */   urlJoin: () => (/* binding */ urlJoin),
/* harmony export */   waitFor: () => (/* binding */ waitFor)
/* harmony export */ });
function randId() {
  return Math.random().toString(36).substr(2, 10) + new Date().getTime();
}

function toCamelCase(str) {
  // Check if the string is already in camelCase
  if (!str.includes("_")) {
    return str;
  }
  // Convert from snake_case to camelCase
  return str.replace(/_./g, (match) => match[1].toUpperCase());
}

function toSnakeCase(str) {
  // Convert from camelCase to snake_case
  return str.replace(/([A-Z])/g, "_$1").toLowerCase();
}

function expandKwargs(obj) {
  if (typeof obj !== "object" || obj === null) {
    return obj; // Return the value if obj is not an object
  }

  const newObj = Array.isArray(obj) ? [] : {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key];

      if (typeof value === "function") {
        newObj[key] = (...args) => {
          if (args.length === 0) {
            throw new Error(`Function "${key}" expects at least one argument.`);
          }

          // Check if the last argument is an object
          const lastArg = args[args.length - 1];
          let kwargs = {};

          if (
            typeof lastArg === "object" &&
            lastArg !== null &&
            !Array.isArray(lastArg)
          ) {
            // Extract kwargs from the last argument
            kwargs = { ...lastArg, _rkwarg: true };
            args = args.slice(0, -1); // Remove the last argument from args
          }

          // Call the original function with positional args followed by kwargs
          return value(...args, kwargs);
        };

        // Preserve metadata like __name__ and __schema__
        newObj[key].__name__ = key;
        if (value.__schema__) {
          newObj[key].__schema__ = { ...value.__schema__ };
          newObj[key].__schema__.name = key;
        }
      } else {
        newObj[key] = expandKwargs(value); // Recursively process nested objects
      }
    }
  }

  return newObj;
}

function convertCase(obj, caseType) {
  if (typeof obj !== "object" || obj === null || !caseType) {
    return obj; // Return the value if obj is not an object
  }

  const newObj = Array.isArray(obj) ? [] : {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key];
      const camelKey = toCamelCase(key);
      const snakeKey = toSnakeCase(key);

      if (caseType === "camel") {
        newObj[camelKey] = convertCase(value, caseType);
        if (typeof value === "function") {
          newObj[camelKey].__name__ = camelKey;
          if (value.__schema__) {
            newObj[camelKey].__schema__ = { ...value.__schema__ };
            newObj[camelKey].__schema__.name = camelKey;
          }
        }
      } else if (caseType === "snake") {
        newObj[snakeKey] = convertCase(value, caseType);
        if (typeof value === "function") {
          newObj[snakeKey].__name__ = snakeKey;
          if (value.__schema__) {
            newObj[snakeKey].__schema__ = { ...value.__schema__ };
            newObj[snakeKey].__schema__.name = snakeKey;
          }
        }
      } else {
        // TODO handle schema for camel + snake
        if (caseType.includes("camel")) {
          newObj[camelKey] = convertCase(value, "camel");
        }
        if (caseType.includes("snake")) {
          newObj[snakeKey] = convertCase(value, "snake");
        }
      }
    }
  }

  return newObj;
}

function parseServiceUrl(url) {
  // Ensure no trailing slash
  url = url.replace(/\/$/, "");

  // Regex pattern to match the URL structure
  const pattern = new RegExp(
    "^(https?:\\/\\/[^/]+)" + // server_url (http or https followed by domain)
      "\\/([a-z0-9_-]+)" + // workspace (lowercase letters, numbers, - or _)
      "\\/services\\/" + // static part of the URL
      "(?:(?<clientId>[a-zA-Z0-9_-]+):)?" + // optional client_id
      "(?<serviceId>[a-zA-Z0-9_-]+)" + // service_id
      "(?:@(?<appId>[a-zA-Z0-9_-]+))?", // optional app_id
  );

  const match = url.match(pattern);
  if (!match) {
    throw new Error("URL does not match the expected pattern");
  }

  const serverUrl = match[1];
  const workspace = match[2];
  const clientId = match.groups?.clientId || "*";
  const serviceId = match.groups?.serviceId;
  const appId = match.groups?.appId || "*";

  return { serverUrl, workspace, clientId, serviceId, appId };
}

const dtypeToTypedArray = {
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  float32: Float32Array,
  float64: Float64Array,
  array: Array,
};

async function loadRequirementsInWindow(requirements) {
  function _importScript(url) {
    //url is URL of external file, implementationCode is the code
    //to be called from the file, location is the location to
    //insert the <script> element
    return new Promise((resolve, reject) => {
      var scriptTag = document.createElement("script");
      scriptTag.src = url;
      scriptTag.type = "text/javascript";
      scriptTag.onload = resolve;
      scriptTag.onreadystatechange = function () {
        if (this.readyState === "loaded" || this.readyState === "complete") {
          resolve();
        }
      };
      scriptTag.onerror = reject;
      document.head.appendChild(scriptTag);
    });
  }

  // support importScripts outside web worker
  async function importScripts() {
    var args = Array.prototype.slice.call(arguments),
      len = args.length,
      i = 0;
    for (; i < len; i++) {
      await _importScript(args[i]);
    }
  }

  if (
    requirements &&
    (Array.isArray(requirements) || typeof requirements === "string")
  ) {
    try {
      var link_node;
      requirements =
        typeof requirements === "string" ? [requirements] : requirements;
      if (Array.isArray(requirements)) {
        for (var i = 0; i < requirements.length; i++) {
          if (
            requirements[i].toLowerCase().endsWith(".css") ||
            requirements[i].startsWith("css:")
          ) {
            if (requirements[i].startsWith("css:")) {
              requirements[i] = requirements[i].slice(4);
            }
            link_node = document.createElement("link");
            link_node.rel = "stylesheet";
            link_node.href = requirements[i];
            document.head.appendChild(link_node);
          } else if (
            requirements[i].toLowerCase().endsWith(".mjs") ||
            requirements[i].startsWith("mjs:")
          ) {
            // import esmodule
            if (requirements[i].startsWith("mjs:")) {
              requirements[i] = requirements[i].slice(4);
            }
            await new Function("url", "return import(url)")(requirements[i]);
          } else if (
            requirements[i].toLowerCase().endsWith(".js") ||
            requirements[i].startsWith("js:")
          ) {
            if (requirements[i].startsWith("js:")) {
              requirements[i] = requirements[i].slice(3);
            }
            await importScripts(requirements[i]);
          } else if (requirements[i].startsWith("http")) {
            await importScripts(requirements[i]);
          } else if (requirements[i].startsWith("cache:")) {
            //ignore cache
          } else {
            console.log("Unprocessed requirements url: " + requirements[i]);
          }
        }
      } else {
        throw "unsupported requirements definition";
      }
    } catch (e) {
      throw "failed to import required scripts: " + requirements.toString();
    }
  }
}

async function loadRequirementsInWebworker(requirements) {
  if (
    requirements &&
    (Array.isArray(requirements) || typeof requirements === "string")
  ) {
    try {
      if (!Array.isArray(requirements)) {
        requirements = [requirements];
      }
      for (var i = 0; i < requirements.length; i++) {
        if (
          requirements[i].toLowerCase().endsWith(".css") ||
          requirements[i].startsWith("css:")
        ) {
          throw "unable to import css in a webworker";
        } else if (
          requirements[i].toLowerCase().endsWith(".js") ||
          requirements[i].startsWith("js:")
        ) {
          if (requirements[i].startsWith("js:")) {
            requirements[i] = requirements[i].slice(3);
          }
          importScripts(requirements[i]);
        } else if (requirements[i].startsWith("http")) {
          importScripts(requirements[i]);
        } else if (requirements[i].startsWith("cache:")) {
          //ignore cache
        } else {
          console.log("Unprocessed requirements url: " + requirements[i]);
        }
      }
    } catch (e) {
      throw "failed to import required scripts: " + requirements.toString();
    }
  }
}

function loadRequirements(requirements) {
  if (
    typeof WorkerGlobalScope !== "undefined" &&
    self instanceof WorkerGlobalScope
  ) {
    return loadRequirementsInWebworker(requirements);
  } else {
    return loadRequirementsInWindow(requirements);
  }
}

function normalizeConfig(config) {
  config.version = config.version || "0.1.0";
  config.description =
    config.description || `[TODO: add description for ${config.name} ]`;
  config.type = config.type || "rpc-window";
  config.id = config.id || randId();
  config.target_origin = config.target_origin || "*";
  config.allow_execution = config.allow_execution || false;
  // remove functions
  config = Object.keys(config).reduce((p, c) => {
    if (typeof config[c] !== "function") p[c] = config[c];
    return p;
  }, {});
  return config;
}
const typedArrayToDtypeMapping = {
  Int8Array: "int8",
  Int16Array: "int16",
  Int32Array: "int32",
  Uint8Array: "uint8",
  Uint16Array: "uint16",
  Uint32Array: "uint32",
  Float32Array: "float32",
  Float64Array: "float64",
  Array: "array",
};

const typedArrayToDtypeKeys = [];
for (const arrType of Object.keys(typedArrayToDtypeMapping)) {
  typedArrayToDtypeKeys.push(eval(arrType));
}

function typedArrayToDtype(obj) {
  let dtype = typedArrayToDtypeMapping[obj.constructor.name];
  if (!dtype) {
    const pt = Object.getPrototypeOf(obj);
    for (const arrType of typedArrayToDtypeKeys) {
      if (pt instanceof arrType) {
        dtype = typedArrayToDtypeMapping[arrType.name];
        break;
      }
    }
  }
  return dtype;
}

function cacheUrlInServiceWorker(url) {
  return new Promise(function (resolve, reject) {
    const message = {
      command: "add",
      url: url,
    };
    if (!navigator.serviceWorker || !navigator.serviceWorker.register) {
      reject("Service worker is not supported.");
      return;
    }
    const messageChannel = new MessageChannel();
    messageChannel.port1.onmessage = function (event) {
      if (event.data && event.data.error) {
        reject(event.data.error);
      } else {
        resolve(event.data && event.data.result);
      }
    };

    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage(message, [
        messageChannel.port2,
      ]);
    } else {
      reject("Service worker controller is not available");
    }
  });
}

async function cacheRequirements(requirements) {
  requirements = requirements || [];
  if (!Array.isArray(requirements)) {
    requirements = [requirements];
  }
  for (let req of requirements) {
    //remove prefix
    if (req.startsWith("js:")) req = req.slice(3);
    if (req.startsWith("css:")) req = req.slice(4);
    if (req.startsWith("cache:")) req = req.slice(6);
    if (!req.startsWith("http")) continue;

    await cacheUrlInServiceWorker(req).catch((e) => {
      console.error(e);
    });
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || "Assertion failed");
  }
}

//#Source https://bit.ly/2neWfJ2
function urlJoin(...args) {
  return args
    .join("/")
    .replace(/[\/]+/g, "/")
    .replace(/^(.+):\//, "$1://")
    .replace(/^file:/, "file:/")
    .replace(/\/(\?|&|#[^!])/g, "$1")
    .replace(/\?/g, "&")
    .replace("&", "?");
}

function waitFor(prom, time, error) {
  let timer;
  return Promise.race([
    prom,
    new Promise(
      (_r, rej) =>
        (timer = setTimeout(() => {
          rej(error || "Timeout Error");
        }, time * 1000)),
    ),
  ]).finally(() => clearTimeout(timer));
}

class MessageEmitter {
  constructor(debug) {
    this._event_handlers = {};
    this._once_handlers = {};
    this._debug = debug;
  }
  emit() {
    throw new Error("emit is not implemented");
  }
  on(event, handler) {
    if (!this._event_handlers[event]) {
      this._event_handlers[event] = [];
    }
    this._event_handlers[event].push(handler);
  }
  once(event, handler) {
    handler.___event_run_once = true;
    this.on(event, handler);
  }
  off(event, handler) {
    if (!event && !handler) {
      // remove all events handlers
      this._event_handlers = {};
    } else if (event && !handler) {
      // remove all hanlders for the event
      if (this._event_handlers[event]) this._event_handlers[event] = [];
    } else {
      // remove a specific handler
      if (this._event_handlers[event]) {
        const idx = this._event_handlers[event].indexOf(handler);
        if (idx >= 0) {
          this._event_handlers[event].splice(idx, 1);
        }
      }
    }
  }
  _fire(event, data) {
    if (this._event_handlers[event]) {
      var i = this._event_handlers[event].length;
      while (i--) {
        const handler = this._event_handlers[event][i];
        try {
          handler(data);
        } catch (e) {
          console.error(e);
        } finally {
          if (handler.___event_run_once) {
            this._event_handlers[event].splice(i, 1);
          }
        }
      }
    } else {
      if (this._debug) {
        console.warn("unhandled event", event, data);
      }
    }
  }

  waitFor(event, timeout) {
    return new Promise((resolve, reject) => {
      const handler = (data) => {
        clearTimeout(timer);
        resolve(data);
      };
      this.once(event, handler);
      const timer = setTimeout(() => {
        this.off(event, handler);
        reject(new Error("Timeout"));
      }, timeout * 1000);
    });
  }
}

class Semaphore {
  constructor(max) {
    this.max = max;
    this.queue = [];
    this.current = 0;
  }
  async run(task) {
    if (this.current >= this.max) {
      // Wait until a slot is free
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.current++;
    try {
      return await task();
    } finally {
      this.current--;
      if (this.queue.length > 0) {
        // release one waiter
        this.queue.shift()();
      }
    }
  }
}

/**
 * Check if the object is a generator
 * @param {Object} obj - Object to check
 * @returns {boolean} - True if the object is a generator
 */
function isGenerator(obj) {
  if (!obj) return false;

  return (
    typeof obj === "object" &&
    typeof obj.next === "function" &&
    typeof obj.throw === "function" &&
    typeof obj.return === "function"
  );
}

/**
 * Check if an object is an async generator object
 * @param {any} obj - Object to check
 * @returns {boolean} True if object is an async generator object
 */
function isAsyncGenerator(obj) {
  if (!obj) return false;
  // Check if it's an async generator object
  return (
    typeof obj === "object" &&
    typeof obj.next === "function" &&
    typeof obj.throw === "function" &&
    typeof obj.return === "function" &&
    Symbol.asyncIterator in Object(obj) &&
    obj[Symbol.toStringTag] === "AsyncGenerator"
  );
}


/***/ }),

/***/ "./src/utils/schema.js":
/*!*****************************!*\
  !*** ./src/utils/schema.js ***!
  \*****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   schemaFunction: () => (/* binding */ schemaFunction),
/* harmony export */   z: () => (/* binding */ z)
/* harmony export */ });
/* harmony import */ var _index_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./index.js */ "./src/utils/index.js");


// Schema builder utility inspired by Zod for consistent API with Python
const z = {
  object: (properties) => ({
    type: "object",
    properties,
    required: Object.keys(properties).filter(
      (key) => !properties[key]._optional,
    ),
  }),

  string: () => ({ type: "string", _optional: false }),
  number: () => ({ type: "number", _optional: false }),
  integer: () => ({ type: "integer", _optional: false }),
  boolean: () => ({ type: "boolean", _optional: false }),
  array: (items) => ({ type: "array", items, _optional: false }),

  // Make field optional
  optional: (schema) => ({ ...schema, _optional: true }),
};

// Add description method to schema types
["string", "number", "integer", "boolean", "array"].forEach((type) => {
  z[type] = () => {
    const schema = {
      type: type === "integer" ? "integer" : type,
      _optional: false,
    };
    schema.describe = (description) => ({ ...schema, description });
    return schema;
  };
});

function schemaFunction(
  func,
  { schema_type = "auto", name = null, description = null, parameters = null },
) {
  if (!func || typeof func !== "function") {
    throw Error("func should be a function");
  }
  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(schema_type === "auto", "schema_type should be auto");

  // If no name provided, try to get it from function
  const funcName = name || func.name;
  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(funcName, "name should not be null");

  // If parameters is a z.object result, convert it properly
  let processedParameters = parameters;
  if (
    parameters &&
    typeof parameters === "object" &&
    parameters.type === "object"
  ) {
    processedParameters = {
      type: "object",
      properties: parameters.properties || {},
      required: parameters.required || [],
    };

    // Clean up internal _optional flags
    for (const [key, schema] of Object.entries(
      processedParameters.properties,
    )) {
      if (schema._optional !== undefined) {
        delete schema._optional;
      }
    }
  }

  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
    processedParameters && processedParameters.type === "object",
    "parameters should be an object schema",
  );

  func.__schema__ = {
    name: funcName,
    description: description || "",
    parameters: processedParameters,
  };
  return func;
}


/***/ }),

/***/ "./src/webrtc-client.js":
/*!******************************!*\
  !*** ./src/webrtc-client.js ***!
  \******************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   getRTCService: () => (/* binding */ getRTCService),
/* harmony export */   registerRTCService: () => (/* binding */ registerRTCService)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");




class WebRTCConnection {
  constructor(channel) {
    this._data_channel = channel;
    this._handle_message = null;
    this._reconnection_token = null;
    this._handle_disconnected = null;
    this._handle_connected = () => {};
    this.manager_id = null;
    this._data_channel.onopen = async () => {
      this._handle_connected &&
        this._handle_connected({ channel: this._data_channel });
    };
    this._data_channel.onmessage = async (event) => {
      let data = event.data;
      if (data instanceof Blob) {
        data = await data.arrayBuffer();
      }
      if (this._handle_message) {
        this._handle_message(data);
      }
    };
    this._data_channel.onclose = () => {
      if (this._handle_disconnected) this._handle_disconnected("closed");
      console.log("data channel closed");
      this._data_channel = null;
    };
  }

  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  on_connected(handler) {
    this._handle_connected = handler;
  }

  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  async emit_message(data) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(this._handle_message, "No handler for message");
    try {
      this._data_channel.send(data);
    } catch (exp) {
      console.error(`Failed to send data, error: ${exp}`);
      throw exp;
    }
  }

  async disconnect(reason) {
    this._data_channel = null;
    console.info(`data channel connection disconnected (${reason})`);
  }
}

async function _setupRPC(config) {
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(config.channel, "No channel provided");
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(config.workspace, "No workspace provided");
  const channel = config.channel;
  const clientId = config.client_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();
  const connection = new WebRTCConnection(channel);
  config.context = config.context || {};
  config.context.connection_type = "webrtc";
  config.context.ws = config.workspace;
  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    default_context: config.context,
    name: config.name,
    method_timeout: config.method_timeout || 10.0,
    workspace: config.workspace,
    app_id: config.app_id,
    long_message_chunk_size: config.long_message_chunk_size,
  });
  return rpc;
}

async function _createOffer(params, server, config, onInit, context) {
  config = config || {};
  let offer = new RTCSessionDescription({
    sdp: params.sdp,
    type: params.type,
  });

  let pc = new RTCPeerConnection({
    iceServers: config.ice_servers || [
      { urls: ["stun:stun.l.google.com:19302"] },
    ],
    sdpSemantics: "unified-plan",
  });

  if (server) {
    pc.addEventListener("datachannel", async (event) => {
      const channel = event.channel;
      let ctx = null;
      if (context && context.user) ctx = { user: context.user, ws: context.ws };
      const rpc = await _setupRPC({
        channel: channel,
        client_id: channel.label,
        workspace: server.config.workspace,
        context: ctx,
      });
      // Map all the local services to the webrtc client
      rpc._services = server.rpc._services;
    });
  }

  if (onInit) {
    await onInit(pc);
  }

  await pc.setRemoteDescription(offer);

  let answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);

  // Wait for ICE candidates to be gathered (important for Firefox)
  await new Promise((resolveIce) => {
    if (pc.iceGatheringState === "complete") {
      resolveIce();
    } else {
      pc.addEventListener("icegatheringstatechange", () => {
        if (pc.iceGatheringState === "complete") {
          resolveIce();
        }
      });
      // Don't wait forever for ICE gathering
      setTimeout(resolveIce, 5000);
    }
  });

  return {
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    workspace: server.config.workspace,
  };
}

async function getRTCService(server, service_id, config) {
  config = config || {};
  config.peer_id = config.peer_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();

  const pc = new RTCPeerConnection({
    iceServers: config.ice_servers || [
      { urls: ["stun:stun.l.google.com:19302"] },
    ],
    sdpSemantics: "unified-plan",
  });

  return new Promise(async (resolve, reject) => {
    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        pc.close();
        reject(new Error("WebRTC Connection timeout"));
      }
    }, 30000); // Increase timeout to 30 seconds

    try {
      pc.addEventListener(
        "connectionstatechange",
        () => {
          console.log("WebRTC Connection state: ", pc.connectionState);
          if (pc.connectionState === "failed") {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              pc.close();
              reject(new Error("WebRTC Connection failed"));
            }
          } else if (pc.connectionState === "closed") {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              reject(new Error("WebRTC Connection closed"));
            }
          } else if (pc.connectionState === "connected") {
            console.log("WebRTC Connection established successfully");
          }
        },
        false,
      );

      // Add ICE connection state change handler for better debugging
      pc.addEventListener("iceconnectionstatechange", () => {
        console.log("ICE Connection state: ", pc.iceConnectionState);
        if (pc.iceConnectionState === "failed") {
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            pc.close();
            reject(new Error("ICE Connection failed"));
          }
        }
      });

      if (config.on_init) {
        await config.on_init(pc);
        delete config.on_init;
      }

      let channel = pc.createDataChannel(config.peer_id, { ordered: true });
      channel.binaryType = "arraybuffer";

      // Wait for ICE gathering to complete before creating offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE candidates to be gathered (important for Firefox)
      await new Promise((resolveIce) => {
        if (pc.iceGatheringState === "complete") {
          resolveIce();
        } else {
          pc.addEventListener("icegatheringstatechange", () => {
            if (pc.iceGatheringState === "complete") {
              resolveIce();
            }
          });
          // Don't wait forever for ICE gathering
          setTimeout(resolveIce, 5000);
        }
      });

      const svc = await server.getService(service_id);
      const answer = await svc.offer({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      });

      channel.onopen = () => {
        config.channel = channel;
        config.workspace = answer.workspace;
        // Increase timeout for Firefox compatibility
        setTimeout(async () => {
          if (!resolved) {
            try {
              const rpc = await _setupRPC(config);
              pc.rpc = rpc;
              async function get_service(name, ...args) {
                (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
                  !name.includes(":"),
                  "WebRTC service name should not contain ':'",
                );
                (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
                  !name.includes("/"),
                  "WebRTC service name should not contain '/'",
                );
                return await rpc.get_remote_service(
                  config.workspace + "/" + config.peer_id + ":" + name,
                  ...args,
                );
              }
              async function disconnect() {
                await rpc.disconnect();
                pc.close();
              }
              pc.getService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(get_service, {
                name: "getService",
                description: "Get a remote service via webrtc",
                parameters: {
                  type: "object",
                  properties: {
                    service_id: {
                      type: "string",
                      description:
                        "Service ID. This should be a service id in the format: 'workspace/service_id', 'workspace/client_id:service_id' or 'workspace/client_id:service_id@app_id'",
                    },
                    config: {
                      type: "object",
                      description: "Options for the service",
                    },
                  },
                  required: ["id"],
                },
              });
              pc.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(disconnect, {
                name: "disconnect",
                description: "Disconnect from the webrtc connection via webrtc",
                parameters: { type: "object", properties: {} },
              });
              pc.registerCodec = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_codec, {
                name: "registerCodec",
                description: "Register a codec for the webrtc connection",
                parameters: {
                  type: "object",
                  properties: {
                    codec: {
                      type: "object",
                      description: "Codec to register",
                      properties: {
                        name: { type: "string" },
                        type: {},
                        encoder: { type: "function" },
                        decoder: { type: "function" },
                      },
                    },
                  },
                },
              });
              resolved = true;
              clearTimeout(timeout);
              resolve(pc);
            } catch (e) {
              if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                reject(e);
              }
            }
          }
        }, 1000); // Increase timeout to 1 second for Firefox
      };

      channel.onclose = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          reject(new Error("Data channel closed"));
        }
      };

      channel.onerror = (error) => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          reject(new Error(`Data channel error: ${error}`));
        }
      };

      await pc.setRemoteDescription(
        new RTCSessionDescription({
          sdp: answer.sdp,
          type: answer.type,
        }),
      );
    } catch (e) {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(e);
      }
    }
  });
}

async function registerRTCService(server, service_id, config) {
  config = config || {
    visibility: "protected",
    require_context: true,
  };
  const onInit = config.on_init;
  delete config.on_init;
  return await server.registerService({
    id: service_id,
    config,
    offer: (params, context) =>
      _createOffer(params, server, config, onInit, context),
  });
}




/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs":
/*!*************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs ***!
  \*************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   CachedKeyDecoder: () => (/* binding */ CachedKeyDecoder)
/* harmony export */ });
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");

var DEFAULT_MAX_KEY_LENGTH = 16;
var DEFAULT_MAX_LENGTH_PER_KEY = 16;
var CachedKeyDecoder = /** @class */ (function () {
    function CachedKeyDecoder(maxKeyLength, maxLengthPerKey) {
        if (maxKeyLength === void 0) { maxKeyLength = DEFAULT_MAX_KEY_LENGTH; }
        if (maxLengthPerKey === void 0) { maxLengthPerKey = DEFAULT_MAX_LENGTH_PER_KEY; }
        this.maxKeyLength = maxKeyLength;
        this.maxLengthPerKey = maxLengthPerKey;
        this.hit = 0;
        this.miss = 0;
        // avoid `new Array(N)`, which makes a sparse array,
        // because a sparse array is typically slower than a non-sparse array.
        this.caches = [];
        for (var i = 0; i < this.maxKeyLength; i++) {
            this.caches.push([]);
        }
    }
    CachedKeyDecoder.prototype.canBeCached = function (byteLength) {
        return byteLength > 0 && byteLength <= this.maxKeyLength;
    };
    CachedKeyDecoder.prototype.find = function (bytes, inputOffset, byteLength) {
        var records = this.caches[byteLength - 1];
        FIND_CHUNK: for (var _i = 0, records_1 = records; _i < records_1.length; _i++) {
            var record = records_1[_i];
            var recordBytes = record.bytes;
            for (var j = 0; j < byteLength; j++) {
                if (recordBytes[j] !== bytes[inputOffset + j]) {
                    continue FIND_CHUNK;
                }
            }
            return record.str;
        }
        return null;
    };
    CachedKeyDecoder.prototype.store = function (bytes, value) {
        var records = this.caches[bytes.length - 1];
        var record = { bytes: bytes, str: value };
        if (records.length >= this.maxLengthPerKey) {
            // `records` are full!
            // Set `record` to an arbitrary position.
            records[(Math.random() * records.length) | 0] = record;
        }
        else {
            records.push(record);
        }
    };
    CachedKeyDecoder.prototype.decode = function (bytes, inputOffset, byteLength) {
        var cachedValue = this.find(bytes, inputOffset, byteLength);
        if (cachedValue != null) {
            this.hit++;
            return cachedValue;
        }
        this.miss++;
        var str = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_0__.utf8DecodeJs)(bytes, inputOffset, byteLength);
        // Ensure to copy a slice of bytes because the byte may be NodeJS Buffer and Buffer#slice() returns a reference to its internal ArrayBuffer.
        var slicedCopyOfBytes = Uint8Array.prototype.slice.call(bytes, inputOffset, inputOffset + byteLength);
        this.store(slicedCopyOfBytes, str);
        return str;
    };
    return CachedKeyDecoder;
}());

//# sourceMappingURL=CachedKeyDecoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs":
/*!********************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs ***!
  \********************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DecodeError: () => (/* binding */ DecodeError)
/* harmony export */ });
var __extends = (undefined && undefined.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var DecodeError = /** @class */ (function (_super) {
    __extends(DecodeError, _super);
    function DecodeError(message) {
        var _this = _super.call(this, message) || this;
        // fix the prototype chain in a cross-platform way
        var proto = Object.create(DecodeError.prototype);
        Object.setPrototypeOf(_this, proto);
        Object.defineProperty(_this, "name", {
            configurable: true,
            enumerable: false,
            value: DecodeError.name,
        });
        return _this;
    }
    return DecodeError;
}(Error));

//# sourceMappingURL=DecodeError.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DataViewIndexOutOfBoundsError: () => (/* binding */ DataViewIndexOutOfBoundsError),
/* harmony export */   Decoder: () => (/* binding */ Decoder)
/* harmony export */ });
/* harmony import */ var _utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./utils/prettyByte.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs");
/* harmony import */ var _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./ExtensionCodec.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");
/* harmony import */ var _utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./utils/typedArrays.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs");
/* harmony import */ var _CachedKeyDecoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./CachedKeyDecoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs");
/* harmony import */ var _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./DecodeError.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs");
var __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (undefined && undefined.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __asyncValues = (undefined && undefined.__asyncValues) || function (o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
};
var __await = (undefined && undefined.__await) || function (v) { return this instanceof __await ? (this.v = v, this) : new __await(v); }
var __asyncGenerator = (undefined && undefined.__asyncGenerator) || function (thisArg, _arguments, generator) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var g = generator.apply(thisArg, _arguments || []), i, q = [];
    return i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i;
    function verb(n) { if (g[n]) i[n] = function (v) { return new Promise(function (a, b) { q.push([n, v, a, b]) > 1 || resume(n, v); }); }; }
    function resume(n, v) { try { step(g[n](v)); } catch (e) { settle(q[0][3], e); } }
    function step(r) { r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r); }
    function fulfill(value) { resume("next", value); }
    function reject(value) { resume("throw", value); }
    function settle(f, v) { if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]); }
};







var isValidMapKeyType = function (key) {
    var keyType = typeof key;
    return keyType === "string" || keyType === "number";
};
var HEAD_BYTE_REQUIRED = -1;
var EMPTY_VIEW = new DataView(new ArrayBuffer(0));
var EMPTY_BYTES = new Uint8Array(EMPTY_VIEW.buffer);
// IE11: Hack to support IE11.
// IE11: Drop this hack and just use RangeError when IE11 is obsolete.
var DataViewIndexOutOfBoundsError = (function () {
    try {
        // IE11: The spec says it should throw RangeError,
        // IE11: but in IE11 it throws TypeError.
        EMPTY_VIEW.getInt8(0);
    }
    catch (e) {
        return e.constructor;
    }
    throw new Error("never reached");
})();
var MORE_DATA = new DataViewIndexOutOfBoundsError("Insufficient data");
var sharedCachedKeyDecoder = new _CachedKeyDecoder_mjs__WEBPACK_IMPORTED_MODULE_0__.CachedKeyDecoder();
var Decoder = /** @class */ (function () {
    function Decoder(extensionCodec, context, maxStrLength, maxBinLength, maxArrayLength, maxMapLength, maxExtLength, keyDecoder) {
        if (extensionCodec === void 0) { extensionCodec = _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtensionCodec.defaultCodec; }
        if (context === void 0) { context = undefined; }
        if (maxStrLength === void 0) { maxStrLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxBinLength === void 0) { maxBinLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxArrayLength === void 0) { maxArrayLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxMapLength === void 0) { maxMapLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxExtLength === void 0) { maxExtLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (keyDecoder === void 0) { keyDecoder = sharedCachedKeyDecoder; }
        this.extensionCodec = extensionCodec;
        this.context = context;
        this.maxStrLength = maxStrLength;
        this.maxBinLength = maxBinLength;
        this.maxArrayLength = maxArrayLength;
        this.maxMapLength = maxMapLength;
        this.maxExtLength = maxExtLength;
        this.keyDecoder = keyDecoder;
        this.totalPos = 0;
        this.pos = 0;
        this.view = EMPTY_VIEW;
        this.bytes = EMPTY_BYTES;
        this.headByte = HEAD_BYTE_REQUIRED;
        this.stack = [];
    }
    Decoder.prototype.reinitializeState = function () {
        this.totalPos = 0;
        this.headByte = HEAD_BYTE_REQUIRED;
        this.stack.length = 0;
        // view, bytes, and pos will be re-initialized in setBuffer()
    };
    Decoder.prototype.setBuffer = function (buffer) {
        this.bytes = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.ensureUint8Array)(buffer);
        this.view = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.createDataView)(this.bytes);
        this.pos = 0;
    };
    Decoder.prototype.appendBuffer = function (buffer) {
        if (this.headByte === HEAD_BYTE_REQUIRED && !this.hasRemaining(1)) {
            this.setBuffer(buffer);
        }
        else {
            var remainingData = this.bytes.subarray(this.pos);
            var newData = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.ensureUint8Array)(buffer);
            // concat remainingData + newData
            var newBuffer = new Uint8Array(remainingData.length + newData.length);
            newBuffer.set(remainingData);
            newBuffer.set(newData, remainingData.length);
            this.setBuffer(newBuffer);
        }
    };
    Decoder.prototype.hasRemaining = function (size) {
        return this.view.byteLength - this.pos >= size;
    };
    Decoder.prototype.createExtraByteError = function (posToShow) {
        var _a = this, view = _a.view, pos = _a.pos;
        return new RangeError("Extra ".concat(view.byteLength - pos, " of ").concat(view.byteLength, " byte(s) found at buffer[").concat(posToShow, "]"));
    };
    /**
     * @throws {@link DecodeError}
     * @throws {@link RangeError}
     */
    Decoder.prototype.decode = function (buffer) {
        this.reinitializeState();
        this.setBuffer(buffer);
        var object = this.doDecodeSync();
        if (this.hasRemaining(1)) {
            throw this.createExtraByteError(this.pos);
        }
        return object;
    };
    Decoder.prototype.decodeMulti = function (buffer) {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    this.reinitializeState();
                    this.setBuffer(buffer);
                    _a.label = 1;
                case 1:
                    if (!this.hasRemaining(1)) return [3 /*break*/, 3];
                    return [4 /*yield*/, this.doDecodeSync()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 1];
                case 3: return [2 /*return*/];
            }
        });
    };
    Decoder.prototype.decodeAsync = function (stream) {
        var stream_1, stream_1_1;
        var e_1, _a;
        return __awaiter(this, void 0, void 0, function () {
            var decoded, object, buffer, e_1_1, _b, headByte, pos, totalPos;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        decoded = false;
                        _c.label = 1;
                    case 1:
                        _c.trys.push([1, 6, 7, 12]);
                        stream_1 = __asyncValues(stream);
                        _c.label = 2;
                    case 2: return [4 /*yield*/, stream_1.next()];
                    case 3:
                        if (!(stream_1_1 = _c.sent(), !stream_1_1.done)) return [3 /*break*/, 5];
                        buffer = stream_1_1.value;
                        if (decoded) {
                            throw this.createExtraByteError(this.totalPos);
                        }
                        this.appendBuffer(buffer);
                        try {
                            object = this.doDecodeSync();
                            decoded = true;
                        }
                        catch (e) {
                            if (!(e instanceof DataViewIndexOutOfBoundsError)) {
                                throw e; // rethrow
                            }
                            // fallthrough
                        }
                        this.totalPos += this.pos;
                        _c.label = 4;
                    case 4: return [3 /*break*/, 2];
                    case 5: return [3 /*break*/, 12];
                    case 6:
                        e_1_1 = _c.sent();
                        e_1 = { error: e_1_1 };
                        return [3 /*break*/, 12];
                    case 7:
                        _c.trys.push([7, , 10, 11]);
                        if (!(stream_1_1 && !stream_1_1.done && (_a = stream_1.return))) return [3 /*break*/, 9];
                        return [4 /*yield*/, _a.call(stream_1)];
                    case 8:
                        _c.sent();
                        _c.label = 9;
                    case 9: return [3 /*break*/, 11];
                    case 10:
                        if (e_1) throw e_1.error;
                        return [7 /*endfinally*/];
                    case 11: return [7 /*endfinally*/];
                    case 12:
                        if (decoded) {
                            if (this.hasRemaining(1)) {
                                throw this.createExtraByteError(this.totalPos);
                            }
                            return [2 /*return*/, object];
                        }
                        _b = this, headByte = _b.headByte, pos = _b.pos, totalPos = _b.totalPos;
                        throw new RangeError("Insufficient data in parsing ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte), " at ").concat(totalPos, " (").concat(pos, " in the current buffer)"));
                }
            });
        });
    };
    Decoder.prototype.decodeArrayStream = function (stream) {
        return this.decodeMultiAsync(stream, true);
    };
    Decoder.prototype.decodeStream = function (stream) {
        return this.decodeMultiAsync(stream, false);
    };
    Decoder.prototype.decodeMultiAsync = function (stream, isArray) {
        return __asyncGenerator(this, arguments, function decodeMultiAsync_1() {
            var isArrayHeaderRequired, arrayItemsLeft, stream_2, stream_2_1, buffer, e_2, e_3_1;
            var e_3, _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        isArrayHeaderRequired = isArray;
                        arrayItemsLeft = -1;
                        _b.label = 1;
                    case 1:
                        _b.trys.push([1, 13, 14, 19]);
                        stream_2 = __asyncValues(stream);
                        _b.label = 2;
                    case 2: return [4 /*yield*/, __await(stream_2.next())];
                    case 3:
                        if (!(stream_2_1 = _b.sent(), !stream_2_1.done)) return [3 /*break*/, 12];
                        buffer = stream_2_1.value;
                        if (isArray && arrayItemsLeft === 0) {
                            throw this.createExtraByteError(this.totalPos);
                        }
                        this.appendBuffer(buffer);
                        if (isArrayHeaderRequired) {
                            arrayItemsLeft = this.readArraySize();
                            isArrayHeaderRequired = false;
                            this.complete();
                        }
                        _b.label = 4;
                    case 4:
                        _b.trys.push([4, 9, , 10]);
                        _b.label = 5;
                    case 5:
                        if (false) {}
                        return [4 /*yield*/, __await(this.doDecodeSync())];
                    case 6: return [4 /*yield*/, _b.sent()];
                    case 7:
                        _b.sent();
                        if (--arrayItemsLeft === 0) {
                            return [3 /*break*/, 8];
                        }
                        return [3 /*break*/, 5];
                    case 8: return [3 /*break*/, 10];
                    case 9:
                        e_2 = _b.sent();
                        if (!(e_2 instanceof DataViewIndexOutOfBoundsError)) {
                            throw e_2; // rethrow
                        }
                        return [3 /*break*/, 10];
                    case 10:
                        this.totalPos += this.pos;
                        _b.label = 11;
                    case 11: return [3 /*break*/, 2];
                    case 12: return [3 /*break*/, 19];
                    case 13:
                        e_3_1 = _b.sent();
                        e_3 = { error: e_3_1 };
                        return [3 /*break*/, 19];
                    case 14:
                        _b.trys.push([14, , 17, 18]);
                        if (!(stream_2_1 && !stream_2_1.done && (_a = stream_2.return))) return [3 /*break*/, 16];
                        return [4 /*yield*/, __await(_a.call(stream_2))];
                    case 15:
                        _b.sent();
                        _b.label = 16;
                    case 16: return [3 /*break*/, 18];
                    case 17:
                        if (e_3) throw e_3.error;
                        return [7 /*endfinally*/];
                    case 18: return [7 /*endfinally*/];
                    case 19: return [2 /*return*/];
                }
            });
        });
    };
    Decoder.prototype.doDecodeSync = function () {
        DECODE: while (true) {
            var headByte = this.readHeadByte();
            var object = void 0;
            if (headByte >= 0xe0) {
                // negative fixint (111x xxxx) 0xe0 - 0xff
                object = headByte - 0x100;
            }
            else if (headByte < 0xc0) {
                if (headByte < 0x80) {
                    // positive fixint (0xxx xxxx) 0x00 - 0x7f
                    object = headByte;
                }
                else if (headByte < 0x90) {
                    // fixmap (1000 xxxx) 0x80 - 0x8f
                    var size = headByte - 0x80;
                    if (size !== 0) {
                        this.pushMapState(size);
                        this.complete();
                        continue DECODE;
                    }
                    else {
                        object = {};
                    }
                }
                else if (headByte < 0xa0) {
                    // fixarray (1001 xxxx) 0x90 - 0x9f
                    var size = headByte - 0x90;
                    if (size !== 0) {
                        this.pushArrayState(size);
                        this.complete();
                        continue DECODE;
                    }
                    else {
                        object = [];
                    }
                }
                else {
                    // fixstr (101x xxxx) 0xa0 - 0xbf
                    var byteLength = headByte - 0xa0;
                    object = this.decodeUtf8String(byteLength, 0);
                }
            }
            else if (headByte === 0xc0) {
                // nil
                object = null;
            }
            else if (headByte === 0xc2) {
                // false
                object = false;
            }
            else if (headByte === 0xc3) {
                // true
                object = true;
            }
            else if (headByte === 0xca) {
                // float 32
                object = this.readF32();
            }
            else if (headByte === 0xcb) {
                // float 64
                object = this.readF64();
            }
            else if (headByte === 0xcc) {
                // uint 8
                object = this.readU8();
            }
            else if (headByte === 0xcd) {
                // uint 16
                object = this.readU16();
            }
            else if (headByte === 0xce) {
                // uint 32
                object = this.readU32();
            }
            else if (headByte === 0xcf) {
                // uint 64
                object = this.readU64();
            }
            else if (headByte === 0xd0) {
                // int 8
                object = this.readI8();
            }
            else if (headByte === 0xd1) {
                // int 16
                object = this.readI16();
            }
            else if (headByte === 0xd2) {
                // int 32
                object = this.readI32();
            }
            else if (headByte === 0xd3) {
                // int 64
                object = this.readI64();
            }
            else if (headByte === 0xd9) {
                // str 8
                var byteLength = this.lookU8();
                object = this.decodeUtf8String(byteLength, 1);
            }
            else if (headByte === 0xda) {
                // str 16
                var byteLength = this.lookU16();
                object = this.decodeUtf8String(byteLength, 2);
            }
            else if (headByte === 0xdb) {
                // str 32
                var byteLength = this.lookU32();
                object = this.decodeUtf8String(byteLength, 4);
            }
            else if (headByte === 0xdc) {
                // array 16
                var size = this.readU16();
                if (size !== 0) {
                    this.pushArrayState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = [];
                }
            }
            else if (headByte === 0xdd) {
                // array 32
                var size = this.readU32();
                if (size !== 0) {
                    this.pushArrayState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = [];
                }
            }
            else if (headByte === 0xde) {
                // map 16
                var size = this.readU16();
                if (size !== 0) {
                    this.pushMapState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = {};
                }
            }
            else if (headByte === 0xdf) {
                // map 32
                var size = this.readU32();
                if (size !== 0) {
                    this.pushMapState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = {};
                }
            }
            else if (headByte === 0xc4) {
                // bin 8
                var size = this.lookU8();
                object = this.decodeBinary(size, 1);
            }
            else if (headByte === 0xc5) {
                // bin 16
                var size = this.lookU16();
                object = this.decodeBinary(size, 2);
            }
            else if (headByte === 0xc6) {
                // bin 32
                var size = this.lookU32();
                object = this.decodeBinary(size, 4);
            }
            else if (headByte === 0xd4) {
                // fixext 1
                object = this.decodeExtension(1, 0);
            }
            else if (headByte === 0xd5) {
                // fixext 2
                object = this.decodeExtension(2, 0);
            }
            else if (headByte === 0xd6) {
                // fixext 4
                object = this.decodeExtension(4, 0);
            }
            else if (headByte === 0xd7) {
                // fixext 8
                object = this.decodeExtension(8, 0);
            }
            else if (headByte === 0xd8) {
                // fixext 16
                object = this.decodeExtension(16, 0);
            }
            else if (headByte === 0xc7) {
                // ext 8
                var size = this.lookU8();
                object = this.decodeExtension(size, 1);
            }
            else if (headByte === 0xc8) {
                // ext 16
                var size = this.lookU16();
                object = this.decodeExtension(size, 2);
            }
            else if (headByte === 0xc9) {
                // ext 32
                var size = this.lookU32();
                object = this.decodeExtension(size, 4);
            }
            else {
                throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Unrecognized type byte: ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte)));
            }
            this.complete();
            var stack = this.stack;
            while (stack.length > 0) {
                // arrays and maps
                var state = stack[stack.length - 1];
                if (state.type === 0 /* State.ARRAY */) {
                    state.array[state.position] = object;
                    state.position++;
                    if (state.position === state.size) {
                        stack.pop();
                        object = state.array;
                    }
                    else {
                        continue DECODE;
                    }
                }
                else if (state.type === 1 /* State.MAP_KEY */) {
                    if (!isValidMapKeyType(object)) {
                        throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("The type of key must be string or number but " + typeof object);
                    }
                    if (object === "__proto__") {
                        throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("The key __proto__ is not allowed");
                    }
                    state.key = object;
                    state.type = 2 /* State.MAP_VALUE */;
                    continue DECODE;
                }
                else {
                    // it must be `state.type === State.MAP_VALUE` here
                    state.map[state.key] = object;
                    state.readCount++;
                    if (state.readCount === state.size) {
                        stack.pop();
                        object = state.map;
                    }
                    else {
                        state.key = null;
                        state.type = 1 /* State.MAP_KEY */;
                        continue DECODE;
                    }
                }
            }
            return object;
        }
    };
    Decoder.prototype.readHeadByte = function () {
        if (this.headByte === HEAD_BYTE_REQUIRED) {
            this.headByte = this.readU8();
            // console.log("headByte", prettyByte(this.headByte));
        }
        return this.headByte;
    };
    Decoder.prototype.complete = function () {
        this.headByte = HEAD_BYTE_REQUIRED;
    };
    Decoder.prototype.readArraySize = function () {
        var headByte = this.readHeadByte();
        switch (headByte) {
            case 0xdc:
                return this.readU16();
            case 0xdd:
                return this.readU32();
            default: {
                if (headByte < 0xa0) {
                    return headByte - 0x90;
                }
                else {
                    throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Unrecognized array type byte: ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte)));
                }
            }
        }
    };
    Decoder.prototype.pushMapState = function (size) {
        if (size > this.maxMapLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: map length (".concat(size, ") > maxMapLengthLength (").concat(this.maxMapLength, ")"));
        }
        this.stack.push({
            type: 1 /* State.MAP_KEY */,
            size: size,
            key: null,
            readCount: 0,
            map: {},
        });
    };
    Decoder.prototype.pushArrayState = function (size) {
        if (size > this.maxArrayLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: array length (".concat(size, ") > maxArrayLength (").concat(this.maxArrayLength, ")"));
        }
        this.stack.push({
            type: 0 /* State.ARRAY */,
            size: size,
            array: new Array(size),
            position: 0,
        });
    };
    Decoder.prototype.decodeUtf8String = function (byteLength, headerOffset) {
        var _a;
        if (byteLength > this.maxStrLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: UTF-8 byte length (".concat(byteLength, ") > maxStrLength (").concat(this.maxStrLength, ")"));
        }
        if (this.bytes.byteLength < this.pos + headerOffset + byteLength) {
            throw MORE_DATA;
        }
        var offset = this.pos + headerOffset;
        var object;
        if (this.stateIsMapKey() && ((_a = this.keyDecoder) === null || _a === void 0 ? void 0 : _a.canBeCached(byteLength))) {
            object = this.keyDecoder.decode(this.bytes, offset, byteLength);
        }
        else if (byteLength > _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.TEXT_DECODER_THRESHOLD) {
            object = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.utf8DecodeTD)(this.bytes, offset, byteLength);
        }
        else {
            object = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.utf8DecodeJs)(this.bytes, offset, byteLength);
        }
        this.pos += headerOffset + byteLength;
        return object;
    };
    Decoder.prototype.stateIsMapKey = function () {
        if (this.stack.length > 0) {
            var state = this.stack[this.stack.length - 1];
            return state.type === 1 /* State.MAP_KEY */;
        }
        return false;
    };
    Decoder.prototype.decodeBinary = function (byteLength, headOffset) {
        if (byteLength > this.maxBinLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: bin length (".concat(byteLength, ") > maxBinLength (").concat(this.maxBinLength, ")"));
        }
        if (!this.hasRemaining(byteLength + headOffset)) {
            throw MORE_DATA;
        }
        var offset = this.pos + headOffset;
        var object = this.bytes.subarray(offset, offset + byteLength);
        this.pos += headOffset + byteLength;
        return object;
    };
    Decoder.prototype.decodeExtension = function (size, headOffset) {
        if (size > this.maxExtLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: ext length (".concat(size, ") > maxExtLength (").concat(this.maxExtLength, ")"));
        }
        var extType = this.view.getInt8(this.pos + headOffset);
        var data = this.decodeBinary(size, headOffset + 1 /* extType */);
        return this.extensionCodec.decode(data, extType, this.context);
    };
    Decoder.prototype.lookU8 = function () {
        return this.view.getUint8(this.pos);
    };
    Decoder.prototype.lookU16 = function () {
        return this.view.getUint16(this.pos);
    };
    Decoder.prototype.lookU32 = function () {
        return this.view.getUint32(this.pos);
    };
    Decoder.prototype.readU8 = function () {
        var value = this.view.getUint8(this.pos);
        this.pos++;
        return value;
    };
    Decoder.prototype.readI8 = function () {
        var value = this.view.getInt8(this.pos);
        this.pos++;
        return value;
    };
    Decoder.prototype.readU16 = function () {
        var value = this.view.getUint16(this.pos);
        this.pos += 2;
        return value;
    };
    Decoder.prototype.readI16 = function () {
        var value = this.view.getInt16(this.pos);
        this.pos += 2;
        return value;
    };
    Decoder.prototype.readU32 = function () {
        var value = this.view.getUint32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readI32 = function () {
        var value = this.view.getInt32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readU64 = function () {
        var value = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.getUint64)(this.view, this.pos);
        this.pos += 8;
        return value;
    };
    Decoder.prototype.readI64 = function () {
        var value = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.getInt64)(this.view, this.pos);
        this.pos += 8;
        return value;
    };
    Decoder.prototype.readF32 = function () {
        var value = this.view.getFloat32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readF64 = function () {
        var value = this.view.getFloat64(this.pos);
        this.pos += 8;
        return value;
    };
    return Decoder;
}());

//# sourceMappingURL=Decoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DEFAULT_INITIAL_BUFFER_SIZE: () => (/* binding */ DEFAULT_INITIAL_BUFFER_SIZE),
/* harmony export */   DEFAULT_MAX_DEPTH: () => (/* binding */ DEFAULT_MAX_DEPTH),
/* harmony export */   Encoder: () => (/* binding */ Encoder)
/* harmony export */ });
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");
/* harmony import */ var _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./ExtensionCodec.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
/* harmony import */ var _utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/typedArrays.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs");




var DEFAULT_MAX_DEPTH = 100;
var DEFAULT_INITIAL_BUFFER_SIZE = 2048;
var Encoder = /** @class */ (function () {
    function Encoder(extensionCodec, context, maxDepth, initialBufferSize, sortKeys, forceFloat32, ignoreUndefined, forceIntegerToFloat) {
        if (extensionCodec === void 0) { extensionCodec = _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_0__.ExtensionCodec.defaultCodec; }
        if (context === void 0) { context = undefined; }
        if (maxDepth === void 0) { maxDepth = DEFAULT_MAX_DEPTH; }
        if (initialBufferSize === void 0) { initialBufferSize = DEFAULT_INITIAL_BUFFER_SIZE; }
        if (sortKeys === void 0) { sortKeys = false; }
        if (forceFloat32 === void 0) { forceFloat32 = false; }
        if (ignoreUndefined === void 0) { ignoreUndefined = false; }
        if (forceIntegerToFloat === void 0) { forceIntegerToFloat = false; }
        this.extensionCodec = extensionCodec;
        this.context = context;
        this.maxDepth = maxDepth;
        this.initialBufferSize = initialBufferSize;
        this.sortKeys = sortKeys;
        this.forceFloat32 = forceFloat32;
        this.ignoreUndefined = ignoreUndefined;
        this.forceIntegerToFloat = forceIntegerToFloat;
        this.pos = 0;
        this.view = new DataView(new ArrayBuffer(this.initialBufferSize));
        this.bytes = new Uint8Array(this.view.buffer);
    }
    Encoder.prototype.reinitializeState = function () {
        this.pos = 0;
    };
    /**
     * This is almost equivalent to {@link Encoder#encode}, but it returns an reference of the encoder's internal buffer and thus much faster than {@link Encoder#encode}.
     *
     * @returns Encodes the object and returns a shared reference the encoder's internal buffer.
     */
    Encoder.prototype.encodeSharedRef = function (object) {
        this.reinitializeState();
        this.doEncode(object, 1);
        return this.bytes.subarray(0, this.pos);
    };
    /**
     * @returns Encodes the object and returns a copy of the encoder's internal buffer.
     */
    Encoder.prototype.encode = function (object) {
        this.reinitializeState();
        this.doEncode(object, 1);
        return this.bytes.slice(0, this.pos);
    };
    Encoder.prototype.doEncode = function (object, depth) {
        if (depth > this.maxDepth) {
            throw new Error("Too deep objects in depth ".concat(depth));
        }
        if (object == null) {
            this.encodeNil();
        }
        else if (typeof object === "boolean") {
            this.encodeBoolean(object);
        }
        else if (typeof object === "number") {
            this.encodeNumber(object);
        }
        else if (typeof object === "string") {
            this.encodeString(object);
        }
        else {
            this.encodeObject(object, depth);
        }
    };
    Encoder.prototype.ensureBufferSizeToWrite = function (sizeToWrite) {
        var requiredSize = this.pos + sizeToWrite;
        if (this.view.byteLength < requiredSize) {
            this.resizeBuffer(requiredSize * 2);
        }
    };
    Encoder.prototype.resizeBuffer = function (newSize) {
        var newBuffer = new ArrayBuffer(newSize);
        var newBytes = new Uint8Array(newBuffer);
        var newView = new DataView(newBuffer);
        newBytes.set(this.bytes);
        this.view = newView;
        this.bytes = newBytes;
    };
    Encoder.prototype.encodeNil = function () {
        this.writeU8(0xc0);
    };
    Encoder.prototype.encodeBoolean = function (object) {
        if (object === false) {
            this.writeU8(0xc2);
        }
        else {
            this.writeU8(0xc3);
        }
    };
    Encoder.prototype.encodeNumber = function (object) {
        if (Number.isSafeInteger(object) && !this.forceIntegerToFloat) {
            if (object >= 0) {
                if (object < 0x80) {
                    // positive fixint
                    this.writeU8(object);
                }
                else if (object < 0x100) {
                    // uint 8
                    this.writeU8(0xcc);
                    this.writeU8(object);
                }
                else if (object < 0x10000) {
                    // uint 16
                    this.writeU8(0xcd);
                    this.writeU16(object);
                }
                else if (object < 0x100000000) {
                    // uint 32
                    this.writeU8(0xce);
                    this.writeU32(object);
                }
                else {
                    // uint 64
                    this.writeU8(0xcf);
                    this.writeU64(object);
                }
            }
            else {
                if (object >= -0x20) {
                    // negative fixint
                    this.writeU8(0xe0 | (object + 0x20));
                }
                else if (object >= -0x80) {
                    // int 8
                    this.writeU8(0xd0);
                    this.writeI8(object);
                }
                else if (object >= -0x8000) {
                    // int 16
                    this.writeU8(0xd1);
                    this.writeI16(object);
                }
                else if (object >= -0x80000000) {
                    // int 32
                    this.writeU8(0xd2);
                    this.writeI32(object);
                }
                else {
                    // int 64
                    this.writeU8(0xd3);
                    this.writeI64(object);
                }
            }
        }
        else {
            // non-integer numbers
            if (this.forceFloat32) {
                // float 32
                this.writeU8(0xca);
                this.writeF32(object);
            }
            else {
                // float 64
                this.writeU8(0xcb);
                this.writeF64(object);
            }
        }
    };
    Encoder.prototype.writeStringHeader = function (byteLength) {
        if (byteLength < 32) {
            // fixstr
            this.writeU8(0xa0 + byteLength);
        }
        else if (byteLength < 0x100) {
            // str 8
            this.writeU8(0xd9);
            this.writeU8(byteLength);
        }
        else if (byteLength < 0x10000) {
            // str 16
            this.writeU8(0xda);
            this.writeU16(byteLength);
        }
        else if (byteLength < 0x100000000) {
            // str 32
            this.writeU8(0xdb);
            this.writeU32(byteLength);
        }
        else {
            throw new Error("Too long string: ".concat(byteLength, " bytes in UTF-8"));
        }
    };
    Encoder.prototype.encodeString = function (object) {
        var maxHeaderSize = 1 + 4;
        var strLength = object.length;
        if (strLength > _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.TEXT_ENCODER_THRESHOLD) {
            var byteLength = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8Count)(object);
            this.ensureBufferSizeToWrite(maxHeaderSize + byteLength);
            this.writeStringHeader(byteLength);
            (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8EncodeTE)(object, this.bytes, this.pos);
            this.pos += byteLength;
        }
        else {
            var byteLength = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8Count)(object);
            this.ensureBufferSizeToWrite(maxHeaderSize + byteLength);
            this.writeStringHeader(byteLength);
            (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8EncodeJs)(object, this.bytes, this.pos);
            this.pos += byteLength;
        }
    };
    Encoder.prototype.encodeObject = function (object, depth) {
        // try to encode objects with custom codec first of non-primitives
        var ext = this.extensionCodec.tryToEncode(object, this.context);
        if (ext != null) {
            this.encodeExtension(ext);
        }
        else if (Array.isArray(object)) {
            this.encodeArray(object, depth);
        }
        else if (ArrayBuffer.isView(object)) {
            this.encodeBinary(object);
        }
        else if (typeof object === "object") {
            this.encodeMap(object, depth);
        }
        else {
            // symbol, function and other special object come here unless extensionCodec handles them.
            throw new Error("Unrecognized object: ".concat(Object.prototype.toString.apply(object)));
        }
    };
    Encoder.prototype.encodeBinary = function (object) {
        var size = object.byteLength;
        if (size < 0x100) {
            // bin 8
            this.writeU8(0xc4);
            this.writeU8(size);
        }
        else if (size < 0x10000) {
            // bin 16
            this.writeU8(0xc5);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // bin 32
            this.writeU8(0xc6);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large binary: ".concat(size));
        }
        var bytes = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_2__.ensureUint8Array)(object);
        this.writeU8a(bytes);
    };
    Encoder.prototype.encodeArray = function (object, depth) {
        var size = object.length;
        if (size < 16) {
            // fixarray
            this.writeU8(0x90 + size);
        }
        else if (size < 0x10000) {
            // array 16
            this.writeU8(0xdc);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // array 32
            this.writeU8(0xdd);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large array: ".concat(size));
        }
        for (var _i = 0, object_1 = object; _i < object_1.length; _i++) {
            var item = object_1[_i];
            this.doEncode(item, depth + 1);
        }
    };
    Encoder.prototype.countWithoutUndefined = function (object, keys) {
        var count = 0;
        for (var _i = 0, keys_1 = keys; _i < keys_1.length; _i++) {
            var key = keys_1[_i];
            if (object[key] !== undefined) {
                count++;
            }
        }
        return count;
    };
    Encoder.prototype.encodeMap = function (object, depth) {
        var keys = Object.keys(object);
        if (this.sortKeys) {
            keys.sort();
        }
        var size = this.ignoreUndefined ? this.countWithoutUndefined(object, keys) : keys.length;
        if (size < 16) {
            // fixmap
            this.writeU8(0x80 + size);
        }
        else if (size < 0x10000) {
            // map 16
            this.writeU8(0xde);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // map 32
            this.writeU8(0xdf);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large map object: ".concat(size));
        }
        for (var _i = 0, keys_2 = keys; _i < keys_2.length; _i++) {
            var key = keys_2[_i];
            var value = object[key];
            if (!(this.ignoreUndefined && value === undefined)) {
                this.encodeString(key);
                this.doEncode(value, depth + 1);
            }
        }
    };
    Encoder.prototype.encodeExtension = function (ext) {
        var size = ext.data.length;
        if (size === 1) {
            // fixext 1
            this.writeU8(0xd4);
        }
        else if (size === 2) {
            // fixext 2
            this.writeU8(0xd5);
        }
        else if (size === 4) {
            // fixext 4
            this.writeU8(0xd6);
        }
        else if (size === 8) {
            // fixext 8
            this.writeU8(0xd7);
        }
        else if (size === 16) {
            // fixext 16
            this.writeU8(0xd8);
        }
        else if (size < 0x100) {
            // ext 8
            this.writeU8(0xc7);
            this.writeU8(size);
        }
        else if (size < 0x10000) {
            // ext 16
            this.writeU8(0xc8);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // ext 32
            this.writeU8(0xc9);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large extension object: ".concat(size));
        }
        this.writeI8(ext.type);
        this.writeU8a(ext.data);
    };
    Encoder.prototype.writeU8 = function (value) {
        this.ensureBufferSizeToWrite(1);
        this.view.setUint8(this.pos, value);
        this.pos++;
    };
    Encoder.prototype.writeU8a = function (values) {
        var size = values.length;
        this.ensureBufferSizeToWrite(size);
        this.bytes.set(values, this.pos);
        this.pos += size;
    };
    Encoder.prototype.writeI8 = function (value) {
        this.ensureBufferSizeToWrite(1);
        this.view.setInt8(this.pos, value);
        this.pos++;
    };
    Encoder.prototype.writeU16 = function (value) {
        this.ensureBufferSizeToWrite(2);
        this.view.setUint16(this.pos, value);
        this.pos += 2;
    };
    Encoder.prototype.writeI16 = function (value) {
        this.ensureBufferSizeToWrite(2);
        this.view.setInt16(this.pos, value);
        this.pos += 2;
    };
    Encoder.prototype.writeU32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setUint32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeI32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setInt32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeF32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setFloat32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeF64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        this.view.setFloat64(this.pos, value);
        this.pos += 8;
    };
    Encoder.prototype.writeU64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__.setUint64)(this.view, this.pos, value);
        this.pos += 8;
    };
    Encoder.prototype.writeI64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__.setInt64)(this.view, this.pos, value);
        this.pos += 8;
    };
    return Encoder;
}());

//# sourceMappingURL=Encoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   ExtData: () => (/* binding */ ExtData)
/* harmony export */ });
/**
 * ExtData is used to handle Extension Types that are not registered to ExtensionCodec.
 */
var ExtData = /** @class */ (function () {
    function ExtData(type, data) {
        this.type = type;
        this.data = data;
    }
    return ExtData;
}());

//# sourceMappingURL=ExtData.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs":
/*!***********************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs ***!
  \***********************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   ExtensionCodec: () => (/* binding */ ExtensionCodec)
/* harmony export */ });
/* harmony import */ var _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./ExtData.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs");
/* harmony import */ var _timestamp_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./timestamp.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs");
// ExtensionCodec to handle MessagePack extensions


var ExtensionCodec = /** @class */ (function () {
    function ExtensionCodec() {
        // built-in extensions
        this.builtInEncoders = [];
        this.builtInDecoders = [];
        // custom extensions
        this.encoders = [];
        this.decoders = [];
        this.register(_timestamp_mjs__WEBPACK_IMPORTED_MODULE_0__.timestampExtension);
    }
    ExtensionCodec.prototype.register = function (_a) {
        var type = _a.type, encode = _a.encode, decode = _a.decode;
        if (type >= 0) {
            // custom extensions
            this.encoders[type] = encode;
            this.decoders[type] = decode;
        }
        else {
            // built-in extensions
            var index = 1 + type;
            this.builtInEncoders[index] = encode;
            this.builtInDecoders[index] = decode;
        }
    };
    ExtensionCodec.prototype.tryToEncode = function (object, context) {
        // built-in extensions
        for (var i = 0; i < this.builtInEncoders.length; i++) {
            var encodeExt = this.builtInEncoders[i];
            if (encodeExt != null) {
                var data = encodeExt(object, context);
                if (data != null) {
                    var type = -1 - i;
                    return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
                }
            }
        }
        // custom extensions
        for (var i = 0; i < this.encoders.length; i++) {
            var encodeExt = this.encoders[i];
            if (encodeExt != null) {
                var data = encodeExt(object, context);
                if (data != null) {
                    var type = i;
                    return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
                }
            }
        }
        if (object instanceof _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData) {
            // to keep ExtData as is
            return object;
        }
        return null;
    };
    ExtensionCodec.prototype.decode = function (data, type, context) {
        var decodeExt = type < 0 ? this.builtInDecoders[-1 - type] : this.decoders[type];
        if (decodeExt) {
            return decodeExt(data, type, context);
        }
        else {
            // decode() does not fail, returns ExtData instead.
            return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
        }
    };
    ExtensionCodec.defaultCodec = new ExtensionCodec();
    return ExtensionCodec;
}());

//# sourceMappingURL=ExtensionCodec.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs":
/*!***************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs ***!
  \***************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   decode: () => (/* binding */ decode),
/* harmony export */   decodeMulti: () => (/* binding */ decodeMulti),
/* harmony export */   defaultDecodeOptions: () => (/* binding */ defaultDecodeOptions)
/* harmony export */ });
/* harmony import */ var _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Decoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs");

var defaultDecodeOptions = {};
/**
 * It decodes a single MessagePack object in a buffer.
 *
 * This is a synchronous decoding function.
 * See other variants for asynchronous decoding: {@link decodeAsync()}, {@link decodeStream()}, or {@link decodeArrayStream()}.
 *
 * @throws {@link RangeError} if the buffer is incomplete, including the case where the buffer is empty.
 * @throws {@link DecodeError} if the buffer contains invalid data.
 */
function decode(buffer, options) {
    if (options === void 0) { options = defaultDecodeOptions; }
    var decoder = new _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Decoder(options.extensionCodec, options.context, options.maxStrLength, options.maxBinLength, options.maxArrayLength, options.maxMapLength, options.maxExtLength);
    return decoder.decode(buffer);
}
/**
 * It decodes multiple MessagePack objects in a buffer.
 * This is corresponding to {@link decodeMultiStream()}.
 *
 * @throws {@link RangeError} if the buffer is incomplete, including the case where the buffer is empty.
 * @throws {@link DecodeError} if the buffer contains invalid data.
 */
function decodeMulti(buffer, options) {
    if (options === void 0) { options = defaultDecodeOptions; }
    var decoder = new _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Decoder(options.extensionCodec, options.context, options.maxStrLength, options.maxBinLength, options.maxArrayLength, options.maxMapLength, options.maxExtLength);
    return decoder.decodeMulti(buffer);
}
//# sourceMappingURL=decode.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs":
/*!***************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs ***!
  \***************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   encode: () => (/* binding */ encode)
/* harmony export */ });
/* harmony import */ var _Encoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Encoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs");

var defaultEncodeOptions = {};
/**
 * It encodes `value` in the MessagePack format and
 * returns a byte buffer.
 *
 * The returned buffer is a slice of a larger `ArrayBuffer`, so you have to use its `#byteOffset` and `#byteLength` in order to convert it to another typed arrays including NodeJS `Buffer`.
 */
function encode(value, options) {
    if (options === void 0) { options = defaultEncodeOptions; }
    var encoder = new _Encoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Encoder(options.extensionCodec, options.context, options.maxDepth, options.initialBufferSize, options.sortKeys, options.forceFloat32, options.ignoreUndefined, options.forceIntegerToFloat);
    return encoder.encodeSharedRef(value);
}
//# sourceMappingURL=encode.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs":
/*!******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs ***!
  \******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   EXT_TIMESTAMP: () => (/* binding */ EXT_TIMESTAMP),
/* harmony export */   decodeTimestampExtension: () => (/* binding */ decodeTimestampExtension),
/* harmony export */   decodeTimestampToTimeSpec: () => (/* binding */ decodeTimestampToTimeSpec),
/* harmony export */   encodeDateToTimeSpec: () => (/* binding */ encodeDateToTimeSpec),
/* harmony export */   encodeTimeSpecToTimestamp: () => (/* binding */ encodeTimeSpecToTimestamp),
/* harmony export */   encodeTimestampExtension: () => (/* binding */ encodeTimestampExtension),
/* harmony export */   timestampExtension: () => (/* binding */ timestampExtension)
/* harmony export */ });
/* harmony import */ var _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./DecodeError.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
// https://github.com/msgpack/msgpack/blob/master/spec.md#timestamp-extension-type


var EXT_TIMESTAMP = -1;
var TIMESTAMP32_MAX_SEC = 0x100000000 - 1; // 32-bit unsigned int
var TIMESTAMP64_MAX_SEC = 0x400000000 - 1; // 34-bit unsigned int
function encodeTimeSpecToTimestamp(_a) {
    var sec = _a.sec, nsec = _a.nsec;
    if (sec >= 0 && nsec >= 0 && sec <= TIMESTAMP64_MAX_SEC) {
        // Here sec >= 0 && nsec >= 0
        if (nsec === 0 && sec <= TIMESTAMP32_MAX_SEC) {
            // timestamp 32 = { sec32 (unsigned) }
            var rv = new Uint8Array(4);
            var view = new DataView(rv.buffer);
            view.setUint32(0, sec);
            return rv;
        }
        else {
            // timestamp 64 = { nsec30 (unsigned), sec34 (unsigned) }
            var secHigh = sec / 0x100000000;
            var secLow = sec & 0xffffffff;
            var rv = new Uint8Array(8);
            var view = new DataView(rv.buffer);
            // nsec30 | secHigh2
            view.setUint32(0, (nsec << 2) | (secHigh & 0x3));
            // secLow32
            view.setUint32(4, secLow);
            return rv;
        }
    }
    else {
        // timestamp 96 = { nsec32 (unsigned), sec64 (signed) }
        var rv = new Uint8Array(12);
        var view = new DataView(rv.buffer);
        view.setUint32(0, nsec);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__.setInt64)(view, 4, sec);
        return rv;
    }
}
function encodeDateToTimeSpec(date) {
    var msec = date.getTime();
    var sec = Math.floor(msec / 1e3);
    var nsec = (msec - sec * 1e3) * 1e6;
    // Normalizes { sec, nsec } to ensure nsec is unsigned.
    var nsecInSec = Math.floor(nsec / 1e9);
    return {
        sec: sec + nsecInSec,
        nsec: nsec - nsecInSec * 1e9,
    };
}
function encodeTimestampExtension(object) {
    if (object instanceof Date) {
        var timeSpec = encodeDateToTimeSpec(object);
        return encodeTimeSpecToTimestamp(timeSpec);
    }
    else {
        return null;
    }
}
function decodeTimestampToTimeSpec(data) {
    var view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    // data may be 32, 64, or 96 bits
    switch (data.byteLength) {
        case 4: {
            // timestamp 32 = { sec32 }
            var sec = view.getUint32(0);
            var nsec = 0;
            return { sec: sec, nsec: nsec };
        }
        case 8: {
            // timestamp 64 = { nsec30, sec34 }
            var nsec30AndSecHigh2 = view.getUint32(0);
            var secLow32 = view.getUint32(4);
            var sec = (nsec30AndSecHigh2 & 0x3) * 0x100000000 + secLow32;
            var nsec = nsec30AndSecHigh2 >>> 2;
            return { sec: sec, nsec: nsec };
        }
        case 12: {
            // timestamp 96 = { nsec32 (unsigned), sec64 (signed) }
            var sec = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__.getInt64)(view, 4);
            var nsec = view.getUint32(0);
            return { sec: sec, nsec: nsec };
        }
        default:
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_1__.DecodeError("Unrecognized data size for timestamp (expected 4, 8, or 12): ".concat(data.length));
    }
}
function decodeTimestampExtension(data) {
    var timeSpec = decodeTimestampToTimeSpec(data);
    return new Date(timeSpec.sec * 1e3 + timeSpec.nsec / 1e6);
}
var timestampExtension = {
    type: EXT_TIMESTAMP,
    encode: encodeTimestampExtension,
    decode: decodeTimestampExtension,
};
//# sourceMappingURL=timestamp.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs":
/*!******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs ***!
  \******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   UINT32_MAX: () => (/* binding */ UINT32_MAX),
/* harmony export */   getInt64: () => (/* binding */ getInt64),
/* harmony export */   getUint64: () => (/* binding */ getUint64),
/* harmony export */   setInt64: () => (/* binding */ setInt64),
/* harmony export */   setUint64: () => (/* binding */ setUint64)
/* harmony export */ });
// Integer Utility
var UINT32_MAX = 4294967295;
// DataView extension to handle int64 / uint64,
// where the actual range is 53-bits integer (a.k.a. safe integer)
function setUint64(view, offset, value) {
    var high = value / 4294967296;
    var low = value; // high bits are truncated by DataView
    view.setUint32(offset, high);
    view.setUint32(offset + 4, low);
}
function setInt64(view, offset, value) {
    var high = Math.floor(value / 4294967296);
    var low = value; // high bits are truncated by DataView
    view.setUint32(offset, high);
    view.setUint32(offset + 4, low);
}
function getInt64(view, offset) {
    var high = view.getInt32(offset);
    var low = view.getUint32(offset + 4);
    return high * 4294967296 + low;
}
function getUint64(view, offset) {
    var high = view.getUint32(offset);
    var low = view.getUint32(offset + 4);
    return high * 4294967296 + low;
}
//# sourceMappingURL=int.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs":
/*!*************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs ***!
  \*************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   prettyByte: () => (/* binding */ prettyByte)
/* harmony export */ });
function prettyByte(byte) {
    return "".concat(byte < 0 ? "-" : "", "0x").concat(Math.abs(byte).toString(16).padStart(2, "0"));
}
//# sourceMappingURL=prettyByte.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs":
/*!**************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs ***!
  \**************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   createDataView: () => (/* binding */ createDataView),
/* harmony export */   ensureUint8Array: () => (/* binding */ ensureUint8Array)
/* harmony export */ });
function ensureUint8Array(buffer) {
    if (buffer instanceof Uint8Array) {
        return buffer;
    }
    else if (ArrayBuffer.isView(buffer)) {
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    else if (buffer instanceof ArrayBuffer) {
        return new Uint8Array(buffer);
    }
    else {
        // ArrayLike<number>
        return Uint8Array.from(buffer);
    }
}
function createDataView(buffer) {
    if (buffer instanceof ArrayBuffer) {
        return new DataView(buffer);
    }
    var bufferView = ensureUint8Array(buffer);
    return new DataView(bufferView.buffer, bufferView.byteOffset, bufferView.byteLength);
}
//# sourceMappingURL=typedArrays.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs":
/*!*******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs ***!
  \*******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   TEXT_DECODER_THRESHOLD: () => (/* binding */ TEXT_DECODER_THRESHOLD),
/* harmony export */   TEXT_ENCODER_THRESHOLD: () => (/* binding */ TEXT_ENCODER_THRESHOLD),
/* harmony export */   utf8Count: () => (/* binding */ utf8Count),
/* harmony export */   utf8DecodeJs: () => (/* binding */ utf8DecodeJs),
/* harmony export */   utf8DecodeTD: () => (/* binding */ utf8DecodeTD),
/* harmony export */   utf8EncodeJs: () => (/* binding */ utf8EncodeJs),
/* harmony export */   utf8EncodeTE: () => (/* binding */ utf8EncodeTE)
/* harmony export */ });
/* harmony import */ var _int_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
var _a, _b, _c;
/* eslint-disable @typescript-eslint/no-unnecessary-condition */

var TEXT_ENCODING_AVAILABLE = (typeof process === "undefined" || ((_a = process === null || process === void 0 ? void 0 : process.env) === null || _a === void 0 ? void 0 : _a["TEXT_ENCODING"]) !== "never") &&
    typeof TextEncoder !== "undefined" &&
    typeof TextDecoder !== "undefined";
function utf8Count(str) {
    var strLength = str.length;
    var byteLength = 0;
    var pos = 0;
    while (pos < strLength) {
        var value = str.charCodeAt(pos++);
        if ((value & 0xffffff80) === 0) {
            // 1-byte
            byteLength++;
            continue;
        }
        else if ((value & 0xfffff800) === 0) {
            // 2-bytes
            byteLength += 2;
        }
        else {
            // handle surrogate pair
            if (value >= 0xd800 && value <= 0xdbff) {
                // high surrogate
                if (pos < strLength) {
                    var extra = str.charCodeAt(pos);
                    if ((extra & 0xfc00) === 0xdc00) {
                        ++pos;
                        value = ((value & 0x3ff) << 10) + (extra & 0x3ff) + 0x10000;
                    }
                }
            }
            if ((value & 0xffff0000) === 0) {
                // 3-byte
                byteLength += 3;
            }
            else {
                // 4-byte
                byteLength += 4;
            }
        }
    }
    return byteLength;
}
function utf8EncodeJs(str, output, outputOffset) {
    var strLength = str.length;
    var offset = outputOffset;
    var pos = 0;
    while (pos < strLength) {
        var value = str.charCodeAt(pos++);
        if ((value & 0xffffff80) === 0) {
            // 1-byte
            output[offset++] = value;
            continue;
        }
        else if ((value & 0xfffff800) === 0) {
            // 2-bytes
            output[offset++] = ((value >> 6) & 0x1f) | 0xc0;
        }
        else {
            // handle surrogate pair
            if (value >= 0xd800 && value <= 0xdbff) {
                // high surrogate
                if (pos < strLength) {
                    var extra = str.charCodeAt(pos);
                    if ((extra & 0xfc00) === 0xdc00) {
                        ++pos;
                        value = ((value & 0x3ff) << 10) + (extra & 0x3ff) + 0x10000;
                    }
                }
            }
            if ((value & 0xffff0000) === 0) {
                // 3-byte
                output[offset++] = ((value >> 12) & 0x0f) | 0xe0;
                output[offset++] = ((value >> 6) & 0x3f) | 0x80;
            }
            else {
                // 4-byte
                output[offset++] = ((value >> 18) & 0x07) | 0xf0;
                output[offset++] = ((value >> 12) & 0x3f) | 0x80;
                output[offset++] = ((value >> 6) & 0x3f) | 0x80;
            }
        }
        output[offset++] = (value & 0x3f) | 0x80;
    }
}
var sharedTextEncoder = TEXT_ENCODING_AVAILABLE ? new TextEncoder() : undefined;
var TEXT_ENCODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE
    ? _int_mjs__WEBPACK_IMPORTED_MODULE_0__.UINT32_MAX
    : typeof process !== "undefined" && ((_b = process === null || process === void 0 ? void 0 : process.env) === null || _b === void 0 ? void 0 : _b["TEXT_ENCODING"]) !== "force"
        ? 200
        : 0;
function utf8EncodeTEencode(str, output, outputOffset) {
    output.set(sharedTextEncoder.encode(str), outputOffset);
}
function utf8EncodeTEencodeInto(str, output, outputOffset) {
    sharedTextEncoder.encodeInto(str, output.subarray(outputOffset));
}
var utf8EncodeTE = (sharedTextEncoder === null || sharedTextEncoder === void 0 ? void 0 : sharedTextEncoder.encodeInto) ? utf8EncodeTEencodeInto : utf8EncodeTEencode;
var CHUNK_SIZE = 4096;
function utf8DecodeJs(bytes, inputOffset, byteLength) {
    var offset = inputOffset;
    var end = offset + byteLength;
    var units = [];
    var result = "";
    while (offset < end) {
        var byte1 = bytes[offset++];
        if ((byte1 & 0x80) === 0) {
            // 1 byte
            units.push(byte1);
        }
        else if ((byte1 & 0xe0) === 0xc0) {
            // 2 bytes
            var byte2 = bytes[offset++] & 0x3f;
            units.push(((byte1 & 0x1f) << 6) | byte2);
        }
        else if ((byte1 & 0xf0) === 0xe0) {
            // 3 bytes
            var byte2 = bytes[offset++] & 0x3f;
            var byte3 = bytes[offset++] & 0x3f;
            units.push(((byte1 & 0x1f) << 12) | (byte2 << 6) | byte3);
        }
        else if ((byte1 & 0xf8) === 0xf0) {
            // 4 bytes
            var byte2 = bytes[offset++] & 0x3f;
            var byte3 = bytes[offset++] & 0x3f;
            var byte4 = bytes[offset++] & 0x3f;
            var unit = ((byte1 & 0x07) << 0x12) | (byte2 << 0x0c) | (byte3 << 0x06) | byte4;
            if (unit > 0xffff) {
                unit -= 0x10000;
                units.push(((unit >>> 10) & 0x3ff) | 0xd800);
                unit = 0xdc00 | (unit & 0x3ff);
            }
            units.push(unit);
        }
        else {
            units.push(byte1);
        }
        if (units.length >= CHUNK_SIZE) {
            result += String.fromCharCode.apply(String, units);
            units.length = 0;
        }
    }
    if (units.length > 0) {
        result += String.fromCharCode.apply(String, units);
    }
    return result;
}
var sharedTextDecoder = TEXT_ENCODING_AVAILABLE ? new TextDecoder() : null;
var TEXT_DECODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE
    ? _int_mjs__WEBPACK_IMPORTED_MODULE_0__.UINT32_MAX
    : typeof process !== "undefined" && ((_c = process === null || process === void 0 ? void 0 : process.env) === null || _c === void 0 ? void 0 : _c["TEXT_DECODER"]) !== "force"
        ? 200
        : 0;
function utf8DecodeTD(bytes, inputOffset, byteLength) {
    var stringBytes = bytes.subarray(inputOffset, inputOffset + byteLength);
    return sharedTextDecoder.decode(stringBytes);
}
//# sourceMappingURL=utf8.mjs.map

/***/ })

/******/ });
/************************************************************************/
/******/ // The module cache
/******/ var __webpack_module_cache__ = {};
/******/ 
/******/ // The require function
/******/ function __webpack_require__(moduleId) {
/******/ 	// Check if module is in cache
/******/ 	var cachedModule = __webpack_module_cache__[moduleId];
/******/ 	if (cachedModule !== undefined) {
/******/ 		return cachedModule.exports;
/******/ 	}
/******/ 	// Create a new module (and put it into the cache)
/******/ 	var module = __webpack_module_cache__[moduleId] = {
/******/ 		// no module.id needed
/******/ 		// no module.loaded needed
/******/ 		exports: {}
/******/ 	};
/******/ 
/******/ 	// Execute the module function
/******/ 	__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 
/******/ 	// Return the exports of the module
/******/ 	return module.exports;
/******/ }
/******/ 
/************************************************************************/
/******/ /* webpack/runtime/define property getters */
/******/ (() => {
/******/ 	// define getter functions for harmony exports
/******/ 	__webpack_require__.d = (exports, definition) => {
/******/ 		for(var key in definition) {
/******/ 			if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 				Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 			}
/******/ 		}
/******/ 	};
/******/ })();
/******/ 
/******/ /* webpack/runtime/hasOwnProperty shorthand */
/******/ (() => {
/******/ 	__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ })();
/******/ 
/******/ /* webpack/runtime/make namespace object */
/******/ (() => {
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = (exports) => {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/ })();
/******/ 
/************************************************************************/
var __webpack_exports__ = {};
/*!*********************************!*\
  !*** ./src/websocket-client.js ***!
  \*********************************/
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   API_VERSION: () => (/* reexport safe */ _rpc_js__WEBPACK_IMPORTED_MODULE_0__.API_VERSION),
/* harmony export */   HTTPStreamingRPCConnection: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_4__.HTTPStreamingRPCConnection),
/* harmony export */   LocalWebSocket: () => (/* binding */ LocalWebSocket),
/* harmony export */   RPC: () => (/* reexport safe */ _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC),
/* harmony export */   connectToServer: () => (/* binding */ connectToServer),
/* harmony export */   connectToServerHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_4__.connectToServerHTTP),
/* harmony export */   getRTCService: () => (/* reexport safe */ _webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__.getRTCService),
/* harmony export */   getRemoteService: () => (/* binding */ getRemoteService),
/* harmony export */   getRemoteServiceHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_4__.getRemoteServiceHTTP),
/* harmony export */   loadRequirements: () => (/* reexport safe */ _utils_index_js__WEBPACK_IMPORTED_MODULE_1__.loadRequirements),
/* harmony export */   login: () => (/* binding */ login),
/* harmony export */   logout: () => (/* binding */ logout),
/* harmony export */   normalizeServerUrlHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_4__.normalizeServerUrl),
/* harmony export */   registerRTCService: () => (/* reexport safe */ _webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__.registerRTCService),
/* harmony export */   schemaFunction: () => (/* reexport safe */ _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction),
/* harmony export */   setupLocalClient: () => (/* binding */ setupLocalClient)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./webrtc-client.js */ "./src/webrtc-client.js");
/* harmony import */ var _http_client_js__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./http-client.js */ "./src/http-client.js");





// Import HTTP client for internal use and re-export


// Re-export HTTP client classes and functions






const MAX_RETRY = 1000000;

class WebsocketRPCConnection {
  constructor(
    server_url,
    client_id,
    workspace,
    token,
    reconnection_token = null,
    timeout = 60,
    WebSocketClass = null,
    token_refresh_interval = 2 * 60 * 60,
    additional_headers = null,
  ) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(server_url && client_id, "server_url and client_id are required");
    this._server_url = server_url;
    this._client_id = client_id;
    this._workspace = workspace;
    this._token = token;
    this._reconnection_token = reconnection_token;
    this._websocket = null;
    this._handle_message = null;
    this._handle_connected = null; // Connection open event handler
    this._handle_disconnected = null; // Disconnection event handler
    this._timeout = timeout;
    this._WebSocketClass = WebSocketClass || WebSocket; // Allow overriding the WebSocket class
    this._closed = false;
    this._legacy_auth = null;
    this.connection_info = null;
    this._enable_reconnect = false;
    this._token_refresh_interval = token_refresh_interval;
    this.manager_id = null;
    this._refresh_token_task = null;
    this._reconnect_timeouts = new Set(); // Track reconnection timeouts
    this._additional_headers = additional_headers;
    this._reconnecting = false; // Mutex to prevent overlapping reconnection attempts
    this._disconnectedNotified = false;
  }

  /**
   * Centralized cleanup method to clear all timers and prevent resource leaks
   */
  _cleanup() {
    // Clear token refresh delay timeout
    if (this._refresh_token_delay) {
      clearTimeout(this._refresh_token_delay);
      this._refresh_token_delay = null;
    }

    // Clear token refresh interval
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
      this._refresh_token_task = null;
    }

    // Clear all reconnection timeouts
    for (const timeoutId of this._reconnect_timeouts) {
      clearTimeout(timeoutId);
    }
    this._reconnect_timeouts.clear();
  }

  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  on_connected(handler) {
    this._handle_connected = handler;
  }

  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  async _attempt_connection(server_url, attempt_fallback = true) {
    return new Promise((resolve, reject) => {
      this._legacy_auth = false;
      const websocket = new this._WebSocketClass(server_url);
      websocket.binaryType = "arraybuffer";

      websocket.onopen = () => {
        console.info("WebSocket connection established");
        resolve(websocket);
      };

      websocket.onerror = (event) => {
        console.error("WebSocket connection error:", event);
        reject(new Error(`WebSocket connection error: ${event}`));
      };

      websocket.onclose = (event) => {
        if (event.code === 1003 && attempt_fallback) {
          console.info(
            "Received 1003 error, attempting connection with query parameters.",
          );
          this._legacy_auth = true;
          this._attempt_connection_with_query_params(server_url)
            .then(resolve)
            .catch(reject);
        } else {
          this._notifyDisconnected(event.reason);
        }
      };
    });
  }

  async _attempt_connection_with_query_params(server_url) {
    // Initialize an array to hold parts of the query string
    const queryParamsParts = [];

    // Conditionally add each parameter if it has a non-empty value
    if (this._client_id)
      queryParamsParts.push(`client_id=${encodeURIComponent(this._client_id)}`);
    if (this._workspace)
      queryParamsParts.push(`workspace=${encodeURIComponent(this._workspace)}`);
    if (this._token)
      queryParamsParts.push(`token=${encodeURIComponent(this._token)}`);
    if (this._reconnection_token)
      queryParamsParts.push(
        `reconnection_token=${encodeURIComponent(this._reconnection_token)}`,
      );

    // Join the parts with '&' to form the final query string, prepend '?' if there are any parameters
    const queryString =
      queryParamsParts.length > 0 ? `?${queryParamsParts.join("&")}` : "";

    // Construct the full URL by appending the query string if it exists
    const full_url = server_url + queryString;

    return await this._attempt_connection(full_url, false);
  }

  _establish_connection() {
    return new Promise((resolve, reject) => {
      this._websocket.onmessage = (event) => {
        const data = event.data;
        const first_message = JSON.parse(data);
        if (first_message.type == "connection_info") {
          this.connection_info = first_message;
          if (this._workspace) {
            (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
              this.connection_info.workspace === this._workspace,
              `Connected to the wrong workspace: ${this.connection_info.workspace}, expected: ${this._workspace}`,
            );
          }
          if (this.connection_info.reconnection_token) {
            this._reconnection_token = this.connection_info.reconnection_token;
          }
          if (this.connection_info.reconnection_token_life_time) {
            // make sure the token refresh interval is less than the token life time
            if (
              this._token_refresh_interval >
              this.connection_info.reconnection_token_life_time / 1.5
            ) {
              console.warn(
                `Token refresh interval is too long (${this._token_refresh_interval}), setting it to 1.5 times of the token life time(${this.connection_info.reconnection_token_life_time}).`,
              );
              this._token_refresh_interval =
                this.connection_info.reconnection_token_life_time / 1.5;
            }
          }
          this.manager_id = this.connection_info.manager_id || null;
          console.log(
            `Successfully connected to the server, workspace: ${this.connection_info.workspace}, manager_id: ${this.manager_id}`,
          );
          if (this.connection_info.announcement) {
            console.log(`${this.connection_info.announcement}`);
          }
          resolve(this.connection_info);
        } else if (first_message.type == "error") {
          const error = "ConnectionAbortedError: " + first_message.message;
          console.error("Failed to connect, " + error);
          reject(new Error(error));
          return;
        } else {
          console.error(
            "ConnectionAbortedError: Unexpected message received from the server:",
            data,
          );
          reject(
            new Error(
              "ConnectionAbortedError: Unexpected message received from the server",
            ),
          );
          return;
        }
      };
    });
  }

  async open() {
    console.log(
      "Creating a new websocket connection to",
      this._server_url.split("?")[0],
    );
    try {
      this._websocket = await this._attempt_connection(this._server_url);
      if (this._legacy_auth) {
        throw new Error(
          "NotImplementedError: Legacy authentication is not supported",
        );
      }
      // Send authentication info as the first message if connected without query params
      const authInfo = JSON.stringify({
        client_id: this._client_id,
        workspace: this._workspace,
        token: this._token,
        reconnection_token: this._reconnection_token,
      });
      this._websocket.send(authInfo);
      // Wait for the first message from the server
      await (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.waitFor)(
        this._establish_connection(),
        this._timeout,
        "Failed to receive the first message from the server",
      );
      if (this._token_refresh_interval > 0) {
        this._refresh_token_delay = setTimeout(() => {
          this._refresh_token_delay = null;
          if (this._closed) return;
          this._send_refresh_token();
          this._refresh_token_task = setInterval(() => {
            this._send_refresh_token();
          }, this._token_refresh_interval * 1000);
        }, 2000);
      }
      // Listen to messages from the server
      this._enable_reconnect = true;
      this._closed = false;
      this._disconnectedNotified = false;
      this._websocket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const parsedData = JSON.parse(event.data);
          // Check if the message is a reconnection token
          if (parsedData.type === "reconnection_token") {
            this._reconnection_token = parsedData.reconnection_token;
            // console.log("Reconnection token received");
          } else {
            console.log("Received message from the server:", parsedData);
          }
        } else {
          this._handle_message(event.data);
        }
      };

      this._websocket.onerror = (event) => {
        console.error("WebSocket connection error:", event);
        // Clean up timers on error
        this._cleanup();
      };

      this._websocket.onclose = this._handle_close.bind(this);

      if (this._handle_connected) {
        this._handle_connected(this.connection_info);
      }
      return this.connection_info;
    } catch (error) {
      // Clean up any timers that might have been set up before the error
      this._cleanup();
      console.error(
        "Failed to connect to",
        this._server_url.split("?")[0],
        error,
      );
      throw error;
    }
  }

  _send_refresh_token() {
    if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
      const refreshMessage = JSON.stringify({ type: "refresh_token" });
      this._websocket.send(refreshMessage);
      // console.log("Requested refresh token");
    }
  }

  _notifyDisconnected(reason) {
    if (this._disconnectedNotified) return;
    this._disconnectedNotified = true;
    if (this._handle_disconnected) {
      this._handle_disconnected(reason);
    }
  }

  _handle_close(event) {
    if (
      !this._closed &&
      this._websocket &&
      this._websocket.readyState === WebSocket.CLOSED
    ) {
      // Clean up timers when connection closes
      this._cleanup();
      // Reset the guard so reconnection can re-notify on next disconnect
      this._disconnectedNotified = false;

      // Even if it's a graceful closure (codes 1000, 1001), if it wasn't user-initiated,
      // we should attempt to reconnect (e.g., server restart, k8s upgrade)
      if (this._enable_reconnect) {
        if ([1000, 1001].includes(event.code)) {
          console.warn(
            `Websocket connection closed gracefully by server (code: ${event.code}): ${event.reason} - attempting reconnect`,
          );
        } else {
          console.warn(
            "Websocket connection closed unexpectedly (code: %s): %s",
            event.code,
            event.reason,
          );
        }

        // Notify the RPC layer immediately so it can reject pending calls
        this._notifyDisconnected(event.reason);

        // Prevent overlapping reconnection attempts
        if (this._reconnecting) {
          console.debug("Reconnection already in progress, skipping");
          return;
        }
        this._reconnecting = true;

        let retry = 0;
        const baseDelay = 1000; // Start with 1 second
        const maxDelay = 60000; // Maximum delay of 60 seconds
        const maxJitter = 0.1; // Maximum jitter factor

        const reconnect = async () => {
          // Check if we were explicitly closed
          if (this._closed) {
            console.info("Connection was closed, stopping reconnection");
            this._reconnecting = false;
            return;
          }

          try {
            console.warn(
              `Reconnecting to ${this._server_url.split("?")[0]} (attempt #${retry})`,
            );
            // Open the connection, this will trigger the on_connected callback
            await this.open();

            // Wait a short time for services to be registered
            // This gives time for the on_connected callback to complete
            // which includes re-registering all services to the server
            await new Promise((resolve) => setTimeout(resolve, 500));

            console.warn(
              `Successfully reconnected to server ${this._server_url} (services re-registered)`,
            );
            this._reconnecting = false;
          } catch (e) {
            if (`${e}`.includes("ConnectionAbortedError:")) {
              console.warn("Server refused to reconnect:", e);
              this._closed = true;
              this._reconnecting = false;
              this._notifyDisconnected(`Server refused reconnection: ${e}`);
              return;
            } else if (`${e}`.includes("NotImplementedError:")) {
              console.error(
                `${e}\nIt appears that you are trying to connect to a hypha server that is older than 0.20.0, please upgrade the hypha server or use the websocket client in imjoy-rpc(https://www.npmjs.com/package/imjoy-rpc) instead`,
              );
              this._closed = true;
              this._reconnecting = false;
              this._notifyDisconnected(`Server too old: ${e}`);
              return;
            }

            // Log specific error types for better debugging
            if (e.name === "NetworkError" || e.message.includes("network")) {
              console.error(`Network error during reconnection: ${e.message}`);
            } else if (
              e.name === "TimeoutError" ||
              e.message.includes("timeout")
            ) {
              console.error(
                `Connection timeout during reconnection: ${e.message}`,
              );
            } else {
              console.error(
                `Unexpected error during reconnection: ${e.message}`,
              );
            }

            // Calculate exponential backoff with jitter
            const delay = Math.min(baseDelay * Math.pow(2, retry), maxDelay);
            // Add jitter to prevent thundering herd
            const jitter = (Math.random() * 2 - 1) * maxJitter * delay;
            const finalDelay = Math.max(100, delay + jitter);

            console.debug(
              `Waiting ${(finalDelay / 1000).toFixed(2)}s before next reconnection attempt`,
            );

            // Track the reconnection timeout to prevent leaks
            const timeoutId = setTimeout(async () => {
              this._reconnect_timeouts.delete(timeoutId);

              // Check if connection was restored externally
              if (
                this._websocket &&
                this._websocket.readyState === WebSocket.OPEN
              ) {
                console.info("Connection restored externally");
                this._reconnecting = false;
                return;
              }

              // Check if we were explicitly closed
              if (this._closed) {
                console.info("Connection was closed, stopping reconnection");
                this._reconnecting = false;
                return;
              }

              retry += 1;
              if (retry < MAX_RETRY) {
                await reconnect();
              } else {
                console.error(
                  `Failed to reconnect after ${MAX_RETRY} attempts, giving up.`,
                );
                this._closed = true;
                this._reconnecting = false;
                this._notifyDisconnected("Max reconnection attempts exceeded");
              }
            }, finalDelay);
            this._reconnect_timeouts.add(timeoutId);
          }
        };
        reconnect();
      }
    } else {
      // Clean up timers in all cases
      this._cleanup();
      this._notifyDisconnected(event.reason);
    }
  }

  async emit_message(data) {
    if (this._closed) {
      throw new Error("Connection is closed");
    }
    if (!this._websocket || this._websocket.readyState !== WebSocket.OPEN) {
      await this.open();
    }
    try {
      this._websocket.send(data);
    } catch (exp) {
      console.error(`Failed to send data, error: ${exp}`);
      throw exp;
    }
  }

  disconnect(reason) {
    this._closed = true;
    this._reconnecting = false;
    // Ensure websocket is closed if it exists and is not already closed or closing
    if (
      this._websocket &&
      this._websocket.readyState !== WebSocket.CLOSED &&
      this._websocket.readyState !== WebSocket.CLOSING
    ) {
      this._websocket.close(1000, reason);
    }
    // Use centralized cleanup to clear all timers
    this._cleanup();
    console.info(`WebSocket connection disconnected (${reason})`);
  }
}

function normalizeServerUrl(server_url) {
  if (!server_url) throw new Error("server_url is required");
  if (server_url.startsWith("http://")) {
    server_url =
      server_url.replace("http://", "ws://").replace(/\/$/, "") + "/ws";
  } else if (server_url.startsWith("https://")) {
    server_url =
      server_url.replace("https://", "wss://").replace(/\/$/, "") + "/ws";
  }
  return server_url;
}

/**
 * Login to the hypha server.
 *
 * Configuration options:
 *   server_url: The server URL (required)
 *   workspace: Target workspace (optional)
 *   login_service_id: Login service ID (default: "public/hypha-login")
 *   expires_in: Token expiration time (optional)
 *   login_timeout: Timeout for login process (default: 60)
 *   login_callback: Callback function for login URL (optional)
 *   profile: Whether to return user profile (optional)
 *   additional_headers: Additional HTTP headers (optional)
 *   transport: Transport type - "websocket" (default) or "http"
 */
async function login(config) {
  const service_id = config.login_service_id || "public/hypha-login";
  const workspace = config.workspace;
  const expires_in = config.expires_in;
  const timeout = config.login_timeout || 60;
  const callback = config.login_callback;
  const profile = config.profile;
  const additional_headers = config.additional_headers;
  const transport = config.transport || "websocket";

  const server = await connectToServer({
    name: "initial login client",
    server_url: config.server_url,
    additional_headers: additional_headers,
    transport: transport,
  });
  try {
    const svc = await server.getService(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(svc, `Failed to get the login service: ${service_id}`);
    let context;
    if (workspace) {
      context = await svc.start({ workspace, expires_in, _rkwargs: true });
    } else {
      context = await svc.start();
    }
    if (callback) {
      await callback(context);
    } else {
      console.log(`Please open your browser and login at ${context.login_url}`);
    }
    return await svc.check(context.key, { timeout, profile, _rkwargs: true });
  } catch (error) {
    throw error;
  } finally {
    await server.disconnect();
  }
}

/**
 * Logout from the hypha server.
 *
 * Configuration options:
 *   server_url: The server URL (required)
 *   login_service_id: Login service ID (default: "public/hypha-login")
 *   logout_callback: Callback function for logout URL (optional)
 *   additional_headers: Additional HTTP headers (optional)
 *   transport: Transport type - "websocket" (default) or "http"
 */
async function logout(config) {
  const service_id = config.login_service_id || "public/hypha-login";
  const callback = config.logout_callback;
  const additional_headers = config.additional_headers;
  const transport = config.transport || "websocket";

  const server = await connectToServer({
    name: "initial logout client",
    server_url: config.server_url,
    additional_headers: additional_headers,
    transport: transport,
  });
  try {
    const svc = await server.getService(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(svc, `Failed to get the login service: ${service_id}`);

    // Check if logout function exists for backward compatibility
    if (!svc.logout) {
      throw new Error(
        "Logout is not supported by this server. " +
          "Please upgrade the Hypha server to a version that supports logout.",
      );
    }

    const context = await svc.logout({});
    if (callback) {
      await callback(context);
    } else {
      console.log(
        `Please open your browser to logout at ${context.logout_url}`,
      );
    }
    return context;
  } catch (error) {
    throw error;
  } finally {
    await server.disconnect();
  }
}

async function webrtcGetService(wm, query, config) {
  config = config || {};
  // Default to "auto" since this wrapper is only used when connection was
  // established with webrtc: true
  const webrtc = config.webrtc !== undefined ? config.webrtc : "auto";
  const webrtc_config = config.webrtc_config;
  if (config.webrtc !== undefined) delete config.webrtc;
  if (config.webrtc_config !== undefined) delete config.webrtc_config;
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
    [undefined, true, false, "auto"].includes(webrtc),
    "webrtc must be true, false or 'auto'",
  );

  const svc = await wm.getService(query, config);
  if (webrtc === true || webrtc === "auto") {
    if (svc.id.includes(":") && svc.id.includes("/")) {
      try {
        // Extract remote client_id from service id
        // svc.id format: "workspace/client_id:service_id"
        const wsAndClient = svc.id.split(":")[0]; // "workspace/client_id"
        const parts = wsAndClient.split("/");
        const remoteClientId = parts[parts.length - 1]; // "client_id"
        const remoteWorkspace = parts.slice(0, -1).join("/"); // "workspace"
        const remoteRtcServiceId = `${remoteWorkspace}/${remoteClientId}-rtc`;
        const peer = await (0,_webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__.getRTCService)(wm, remoteRtcServiceId, webrtc_config);
        const rtcSvc = await peer.getService(svc.id.split(":")[1], config);
        rtcSvc._webrtc = true;
        rtcSvc._peer = peer;
        rtcSvc._service = svc;
        return rtcSvc;
      } catch (e) {
        console.warn(
          "Failed to get webrtc service, using websocket connection",
          e,
        );
      }
    }
    if (webrtc === true) {
      throw new Error("Failed to get the service via webrtc");
    }
  }
  return svc;
}

async function connectToServer(config) {
  // Support HTTP transport via transport option
  const transport = config.transport || "websocket";
  if (transport === "http") {
    return await (0,_http_client_js__WEBPACK_IMPORTED_MODULE_4__.connectToServerHTTP)(config);
  }

  if (config.server) {
    config.server_url = config.server_url || config.server.url;
    config.WebSocketClass =
      config.WebSocketClass || config.server.WebSocketClass;
  }
  let clientId = config.client_id;
  if (!clientId) {
    clientId = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();
    config.client_id = clientId;
  }
  if (Object.keys(config).length === 0) {
    if (typeof process !== "undefined" && process.env) {
      // Node.js
      config.server_url = process.env.HYPHA_SERVER_URL;
      config.token = process.env.HYPHA_TOKEN;
      config.client_id = process.env.HYPHA_CLIENT_ID;
      config.workspace = process.env.HYPHA_WORKSPACE;
    } else if (typeof self !== "undefined" && self.env) {
      // WebWorker (only if you inject self.env manually)
      config.server_url = self.env.HYPHA_SERVER_URL;
      config.token = self.env.HYPHA_TOKEN;
      config.client_id = self.env.HYPHA_CLIENT_ID;
      config.workspace = self.env.HYPHA_WORKSPACE;
    } else if (typeof globalThis !== "undefined" && globalThis.env) {
      // Browser (only if you define globalThis.env beforehand)
      config.server_url = globalThis.env.HYPHA_SERVER_URL;
      config.token = globalThis.env.HYPHA_TOKEN;
      config.client_id = globalThis.env.HYPHA_CLIENT_ID;
      config.workspace = globalThis.env.HYPHA_WORKSPACE;
    }
  }

  let server_url = normalizeServerUrl(config.server_url);

  let connection = new WebsocketRPCConnection(
    server_url,
    clientId,
    config.workspace,
    config.token,
    config.reconnection_token,
    config.method_timeout || 60,
    config.WebSocketClass,
    config.token_refresh_interval,
    config.additional_headers,
  );
  const connection_info = await connection.open();
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
    connection_info,
    "Failed to connect to the server, no connection info obtained. This issue is most likely due to an outdated Hypha server version. Please use `imjoy-rpc` for compatibility, or upgrade the Hypha server to the latest version.",
  );
  // wait for 0.5 seconds
  await new Promise((resolve) => setTimeout(resolve, 100));
  // Ensure manager_id is set before proceeding
  if (!connection.manager_id) {
    console.warn("Manager ID not set immediately, waiting...");

    // Wait for manager_id to be set with timeout
    const maxWaitTime = 5000; // 5 seconds
    const checkInterval = 100; // 100ms
    const startTime = Date.now();

    while (!connection.manager_id && Date.now() - startTime < maxWaitTime) {
      await new Promise((resolve) => setTimeout(resolve, checkInterval));
    }

    if (!connection.manager_id) {
      console.error("Manager ID still not set after waiting");
      throw new Error("Failed to get manager ID from server");
    } else {
      console.info(`Manager ID set after waiting: ${connection.manager_id}`);
    }
  }
  if (config.workspace && connection_info.workspace !== config.workspace) {
    throw new Error(
      `Connected to the wrong workspace: ${connection_info.workspace}, expected: ${config.workspace}`,
    );
  }

  const workspace = connection_info.workspace;
  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    workspace,
    default_context: { connection_type: "websocket" },
    name: config.name,
    method_timeout: config.method_timeout,
    app_id: config.app_id,
    server_base_url: connection_info.public_base_url,
    long_message_chunk_size: config.long_message_chunk_size,
  });
  await rpc.waitFor("services_registered", config.method_timeout || 120);
  const wm = await rpc.get_manager_service({
    timeout: config.method_timeout,
    case_conversion: "camel",
    kwargs_expansion: config.kwargs_expansion || false,
  });
  wm.rpc = rpc;

  async function _export(api) {
    api.id = "default";
    api.name = api.name || config.name || api.id;
    api.description = api.description || config.description;
    await rpc.register_service(api, { overwrite: true });
  }

  async function getApp(clientId) {
    clientId = clientId || "*";
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(!clientId.includes(":"), "clientId should not contain ':'");
    if (!clientId.includes("/")) {
      clientId = connection_info.workspace + "/" + clientId;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
      clientId.split("/").length === 2,
      "clientId should match pattern workspace/clientId",
    );
    return await wm.getService(`${clientId}:default`);
  }

  async function listApps(ws) {
    ws = ws || workspace;
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(!ws.includes(":"), "workspace should not contain ':'");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(!ws.includes("/"), "workspace should not contain '/'");
    const query = { workspace: ws, service_id: "default" };
    return await wm.listServices(query);
  }

  if (connection_info) {
    wm.config = Object.assign(wm.config, connection_info);
  }
  wm.export = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(_export, {
    name: "export",
    description: "Export the api.",
    parameters: {
      properties: { api: { description: "The api to export", type: "object" } },
      required: ["api"],
      type: "object",
    },
  });
  wm.getApp = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(getApp, {
    name: "getApp",
    description: "Get the app.",
    parameters: {
      properties: {
        clientId: { default: "*", description: "The clientId", type: "string" },
      },
      type: "object",
    },
  });
  wm.listApps = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(listApps, {
    name: "listApps",
    description: "List the apps.",
    parameters: {
      properties: {
        workspace: {
          default: workspace,
          description: "The workspace",
          type: "string",
        },
      },
      type: "object",
    },
  });
  wm.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.disconnect.bind(rpc), {
    name: "disconnect",
    description: "Disconnect from the server.",
    parameters: { type: "object", properties: {}, required: [] },
  });
  wm.registerCodec = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_codec.bind(rpc), {
    name: "registerCodec",
    description: "Register a codec for the webrtc connection",
    parameters: {
      type: "object",
      properties: {
        codec: {
          type: "object",
          description: "Codec to register",
          properties: {
            name: { type: "string" },
            type: {},
            encoder: { type: "function" },
            decoder: { type: "function" },
          },
        },
      },
    },
  });

  wm.emit = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.emit.bind(rpc), {
    name: "emit",
    description: "Emit a message.",
    parameters: {
      properties: { data: { description: "The data to emit", type: "object" } },
      required: ["data"],
      type: "object",
    },
  });

  wm.on = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.on.bind(rpc), {
    name: "on",
    description: "Register a message handler.",
    parameters: {
      properties: {
        event: { description: "The event to listen to", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.off = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.off.bind(rpc), {
    name: "off",
    description: "Remove a message handler.",
    parameters: {
      properties: {
        event: { description: "The event to remove", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.once = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.once.bind(rpc), {
    name: "once",
    description: "Register a one-time message handler.",
    parameters: {
      properties: {
        event: { description: "The event to listen to", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.getServiceSchema = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.get_service_schema, {
    name: "getServiceSchema",
    description: "Get the service schema.",
    parameters: {
      properties: {
        service: {
          description: "The service to extract schema",
          type: "object",
        },
      },
      required: ["service"],
      type: "object",
    },
  });

  wm.registerService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_service.bind(rpc), {
    name: "registerService",
    description: "Register a service.",
    parameters: {
      properties: {
        service: { description: "The service to register", type: "object" },
        force: {
          default: false,
          description: "Force to register the service",
          type: "boolean",
        },
      },
      required: ["service"],
      type: "object",
    },
  });
  wm.unregisterService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.unregister_service.bind(rpc), {
    name: "unregisterService",
    description: "Unregister a service.",
    parameters: {
      properties: {
        service: {
          description: "The service id to unregister",
          type: "string",
        },
        notify: {
          default: true,
          description: "Notify the workspace manager",
          type: "boolean",
        },
      },
      required: ["service"],
      type: "object",
    },
  });
  if (connection.manager_id) {
    rpc.on("force-exit", async (message) => {
      if (message.from === "*/" + connection.manager_id) {
        console.log("Disconnecting from server, reason:", message.reason);
        await rpc.disconnect();
      }
    });
  }
  if (config.webrtc) {
    await (0,_webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__.registerRTCService)(wm, `${clientId}-rtc`, config.webrtc_config);
    // make a copy of wm, so webrtc can use the original wm.getService
    const _wm = Object.assign({}, wm);
    const description = _wm.getService.__schema__.description;
    // TODO: Fix the schema for adding options for webrtc
    const parameters = _wm.getService.__schema__.parameters;
    wm.getService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(webrtcGetService.bind(null, _wm), {
      name: "getService",
      description,
      parameters,
    });

    wm.getRTCService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(_webrtc_client_js__WEBPACK_IMPORTED_MODULE_3__.getRTCService.bind(null, wm), {
      name: "getRTCService",
      description: "Get the webrtc connection, returns a peer connection.",
      parameters: {
        properties: {
          config: {
            description: "The config for the webrtc service",
            type: "object",
          },
        },
        required: ["config"],
        type: "object",
      },
    });
  } else {
    const _getService = wm.getService;
    wm.getService = (query, config) => {
      config = config || {};
      return _getService(query, config);
    };
    wm.getService.__schema__ = _getService.__schema__;
  }

  async function registerProbes(probes) {
    probes.id = "probes";
    probes.name = "Probes";
    probes.config = { visibility: "public" };
    probes.type = "probes";
    probes.description = `Probes Service, visit ${server_url}/${workspace}services/probes for the available probes.`;
    return await wm.registerService(probes, { overwrite: true });
  }

  wm.registerProbes = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(registerProbes, {
    name: "registerProbes",
    description: "Register probes service",
    parameters: {
      properties: {
        probes: {
          description:
            "The probes to register, e.g. {'liveness': {'type': 'function', 'description': 'Check the liveness of the service'}}",
          type: "object",
        },
      },
      required: ["probes"],
      type: "object",
    },
  });
  return wm;
}

async function getRemoteService(serviceUri, config = {}) {
  const { serverUrl, workspace, clientId, serviceId, appId } =
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.parseServiceUrl)(serviceUri);
  const fullServiceId = `${workspace}/${clientId}:${serviceId}@${appId}`;

  if (config.serverUrl) {
    if (config.serverUrl !== serverUrl) {
      throw new Error(
        "server_url in config does not match the server_url in the url",
      );
    }
  }
  config.serverUrl = serverUrl;
  const server = await connectToServer(config);
  return await server.getService(fullServiceId);
}

class LocalWebSocket {
  constructor(url, client_id, workspace) {
    this.url = url;
    this.onopen = () => {};
    this.onmessage = () => {};
    this.onclose = () => {};
    this.onerror = () => {};
    this.client_id = client_id;
    this.workspace = workspace;
    const context = typeof window !== "undefined" ? window : self;
    const isWindow = typeof window !== "undefined";
    this.postMessage = (message) => {
      if (isWindow) {
        window.parent.postMessage(message, "*");
      } else {
        self.postMessage(message);
      }
    };

    this.readyState = WebSocket.CONNECTING;
    this._context = context;
    this._messageListener = (event) => {
      const { type, data, to } = event.data;
      if (to !== this.client_id) {
        return;
      }
      switch (type) {
        case "message":
          if (this.readyState === WebSocket.OPEN && this.onmessage) {
            this.onmessage({ data: data });
          }
          break;
        case "connected":
          this.readyState = WebSocket.OPEN;
          this.onopen(event);
          break;
        case "closed":
          this.readyState = WebSocket.CLOSED;
          this.onclose(event);
          break;
        default:
          break;
      }
    };
    context.addEventListener("message", this._messageListener, false);

    if (!this.client_id) throw new Error("client_id is required");
    if (!this.workspace) throw new Error("workspace is required");
    this.postMessage({
      type: "connect",
      url: this.url,
      from: this.client_id,
      workspace: this.workspace,
    });
  }

  send(data) {
    if (this.readyState === WebSocket.OPEN) {
      this.postMessage({
        type: "message",
        data: data,
        from: this.client_id,
        workspace: this.workspace,
      });
    }
  }

  close() {
    this.readyState = WebSocket.CLOSING;
    this.postMessage({
      type: "close",
      from: this.client_id,
      workspace: this.workspace,
    });
    if (this._context && this._messageListener) {
      this._context.removeEventListener(
        "message",
        this._messageListener,
        false,
      );
      this._messageListener = null;
    }
    this.onclose();
  }

  addEventListener(type, listener) {
    if (type === "message") {
      this.onmessage = listener;
    }
    if (type === "open") {
      this.onopen = listener;
    }
    if (type === "close") {
      this.onclose = listener;
    }
    if (type === "error") {
      this.onerror = listener;
    }
  }
}

function setupLocalClient({
  enable_execution = false,
  on_ready = null,
}) {
  return new Promise((resolve, reject) => {
    const context = typeof window !== "undefined" ? window : self;
    const isWindow = typeof window !== "undefined";
    context.addEventListener(
      "message",
      (event) => {
        const {
          type,
          server_url,
          workspace,
          client_id,
          token,
          method_timeout,
          name,
          config,
        } = event.data;

        if (type === "initializeHyphaClient") {
          if (!server_url || !workspace || !client_id) {
            console.error("server_url, workspace, and client_id are required.");
            return;
          }

          if (!server_url.startsWith("https://local-hypha-server:")) {
            console.error(
              "server_url should start with https://local-hypha-server:",
            );
            return;
          }

          class FixedLocalWebSocket extends LocalWebSocket {
            constructor(url) {
              // Call the parent class's constructor with fixed values
              super(url, client_id, workspace);
            }
          }
          connectToServer({
            server_url,
            workspace,
            client_id,
            token,
            method_timeout,
            name,
            WebSocketClass: FixedLocalWebSocket,
          }).then(async (server) => {
            globalThis.api = server;
            try {
              // for iframe
              if (isWindow && enable_execution) {
                function loadScript(script) {
                  return new Promise((resolve, reject) => {
                    const scriptElement = document.createElement("script");
                    scriptElement.innerHTML = script.content;
                    scriptElement.lang = script.lang;

                    scriptElement.onload = () => resolve();
                    scriptElement.onerror = (e) => reject(e);

                    document.head.appendChild(scriptElement);
                  });
                }
                if (config.styles && config.styles.length > 0) {
                  for (const style of config.styles) {
                    const styleElement = document.createElement("style");
                    styleElement.innerHTML = style.content;
                    styleElement.lang = style.lang;
                    document.head.appendChild(styleElement);
                  }
                }
                if (config.links && config.links.length > 0) {
                  for (const link of config.links) {
                    const linkElement = document.createElement("a");
                    linkElement.href = link.url;
                    linkElement.innerText = link.text;
                    document.body.appendChild(linkElement);
                  }
                }
                if (config.windows && config.windows.length > 0) {
                  for (const w of config.windows) {
                    document.body.innerHTML = w.content;
                    break;
                  }
                }
                if (config.scripts && config.scripts.length > 0) {
                  for (const script of config.scripts) {
                    if (script.lang !== "javascript")
                      throw new Error("Only javascript scripts are supported");
                    await loadScript(script); // Await the loading of each script
                  }
                }
              }
              // for web worker
              else if (
                !isWindow &&
                enable_execution &&
                config.scripts &&
                config.scripts.length > 0
              ) {
                for (const script of config.scripts) {
                  if (script.lang !== "javascript")
                    throw new Error("Only javascript scripts are supported");
                  eval(script.content);
                }
              }

              if (on_ready) {
                await on_ready(server, config);
              }
              resolve(server);
            } catch (e) {
              reject(e);
            }
          });
        }
      },
      false,
    );
    if (isWindow) {
      window.parent.postMessage({ type: "hyphaClientReady" }, "*");
    } else {
      self.postMessage({ type: "hyphaClientReady" });
    }
  });
}

var __webpack_exports__API_VERSION = __webpack_exports__.API_VERSION;
var __webpack_exports__HTTPStreamingRPCConnection = __webpack_exports__.HTTPStreamingRPCConnection;
var __webpack_exports__LocalWebSocket = __webpack_exports__.LocalWebSocket;
var __webpack_exports__RPC = __webpack_exports__.RPC;
var __webpack_exports__connectToServer = __webpack_exports__.connectToServer;
var __webpack_exports__connectToServerHTTP = __webpack_exports__.connectToServerHTTP;
var __webpack_exports__getRTCService = __webpack_exports__.getRTCService;
var __webpack_exports__getRemoteService = __webpack_exports__.getRemoteService;
var __webpack_exports__getRemoteServiceHTTP = __webpack_exports__.getRemoteServiceHTTP;
var __webpack_exports__loadRequirements = __webpack_exports__.loadRequirements;
var __webpack_exports__login = __webpack_exports__.login;
var __webpack_exports__logout = __webpack_exports__.logout;
var __webpack_exports__normalizeServerUrlHTTP = __webpack_exports__.normalizeServerUrlHTTP;
var __webpack_exports__registerRTCService = __webpack_exports__.registerRTCService;
var __webpack_exports__schemaFunction = __webpack_exports__.schemaFunction;
var __webpack_exports__setupLocalClient = __webpack_exports__.setupLocalClient;
export { __webpack_exports__API_VERSION as API_VERSION, __webpack_exports__HTTPStreamingRPCConnection as HTTPStreamingRPCConnection, __webpack_exports__LocalWebSocket as LocalWebSocket, __webpack_exports__RPC as RPC, __webpack_exports__connectToServer as connectToServer, __webpack_exports__connectToServerHTTP as connectToServerHTTP, __webpack_exports__getRTCService as getRTCService, __webpack_exports__getRemoteService as getRemoteService, __webpack_exports__getRemoteServiceHTTP as getRemoteServiceHTTP, __webpack_exports__loadRequirements as loadRequirements, __webpack_exports__login as login, __webpack_exports__logout as logout, __webpack_exports__normalizeServerUrlHTTP as normalizeServerUrlHTTP, __webpack_exports__registerRTCService as registerRTCService, __webpack_exports__schemaFunction as schemaFunction, __webpack_exports__setupLocalClient as setupLocalClient };

//# sourceMappingURL=hypha-rpc-websocket.mjs.map