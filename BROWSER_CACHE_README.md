# Browser Worker Smart Caching System

This document describes the smart caching system implemented for the Hypha browser worker to speed up loading of web applications, particularly those that require loading many remote resources like Pyodide, web containers, ImageJ.js, etc.

## Overview

The browser worker now includes intelligent caching capabilities that can significantly reduce load times for applications by caching frequently accessed remote resources. The system uses Redis for persistent storage and Playwright's route interception for transparent caching.

## Features

### 1. Smart Route Interception
- Uses Playwright's `page.route()` to intercept HTTP requests
- Automatically serves cached responses when available
- Transparently fetches and caches new resources
- Supports pattern-based URL matching with wildcards

### 2. Application-Specific Caching
- Each app gets its own isolated cache namespace (`acache:workspace/app_id`)
- Cache entries are automatically cleaned up when apps are uninstalled
- Recording can be enabled/disabled per app session

### 3. Default Cache Routes for Web-Python Apps
Web-python applications automatically get default cache routes for common dependencies:
- `https://cdn.jsdelivr.net/pyodide/*` - Pyodide runtime files
- `https://pypi.org/*` - PyPI package index
- `https://files.pythonhosted.org/*` - Python package files
- `https://pyodide.org/*` - Additional Pyodide resources
- `*.whl` - Python wheel files
- `*.tar.gz` - Python source distributions

### 4. New "web-app" Application Type
A new application type that allows wrapping external web applications:

```json
{
  "name": "My External App",
  "type": "web-app", 
  "version": "1.0.0",
  "url": "https://example.com/app",
  "detached": true,
  "cache_routes": [
    "https://example.com/*",
    "https://cdn.jsdelivr.net/*"
  ],
  "cookies": {"session": "value"},
  "localStorage": {"theme": "dark"},
  "authorization_token": "Bearer token"
}
```

## Configuration

### Cache Routes in Manifest
Applications can specify which URLs to cache using the `cache_routes` field:

```json
{
  "type": "web-python",
  "cache_routes": [
    "https://cdn.jsdelivr.net/pyodide/*",
    "https://example.com/api/*",
    "*.js",
    "*.wasm"
  ]
}
```

### Pattern Matching
The cache system supports glob-style patterns:
- `*` matches any sequence of characters
- `?` matches any single character
- Patterns are converted to regular expressions for matching

Examples:
- `https://cdn.jsdelivr.net/*` matches all jsdelivr CDN resources
- `*.whl` matches all Python wheel files
- `https://api.example.com/v*/data` matches versioned API endpoints

## API Methods

The browser worker exposes new cache management methods:

### `get_app_cache_stats(workspace: str, app_id: str)`
Returns cache statistics for an application:
```python
{
  "entry_count": 42,
  "total_size_bytes": 1048576,
  "oldest_entry": 1640995200.0,
  "newest_entry": 1640995800.0,
  "recording": true
}
```

### `clear_app_cache(workspace: str, app_id: str)`
Clears all cached entries for an application:
```python
{
  "deleted_entries": 42
}
```

## Implementation Details

### Cache Storage
- **Backend**: Redis with configurable TTL (default 24 hours)
- **Key Format**: `acache:workspace/app_id:url_hash`
- **Serialization**: JSON with hex-encoded binary data
- **Isolation**: Complete separation between workspaces and applications

### Cache Entry Structure
```python
{
  "url": "https://example.com/file.js",
  "status": 200,
  "headers": {"Content-Type": "application/javascript"},
  "body": "636f6e736f6c652e6c6f672827746573742729",  # hex-encoded
  "timestamp": 1640995200.0
}
```

### Recording Sessions
- Caching is only active during "recording" sessions
- Recording starts when an app is installed (during the `start` phase)
- Recording stops when the app session ends
- This prevents unnecessary caching of non-cacheable requests

## Usage Examples

### 1. Web-Python App with Default Caching
```python
# Install a web-python app - automatically gets default cache routes
app_info = await controller.install(
    source="""
import numpy as np
api.export({"compute": lambda x: np.sqrt(x)})
""",
    manifest={
        "name": "NumPy Calculator",
        "type": "web-python",
        "version": "1.0.0"
    }
)
```

### 2. Web-Python App with Custom Cache Routes
```python
app_info = await controller.install(
    source=python_code,
    manifest={
        "name": "Custom Cached App",
        "type": "web-python",
        "version": "1.0.0",
        "cache_routes": [
            "https://cdn.jsdelivr.net/pyodide/*",
            "https://my-cdn.example.com/*",
            "*.wasm"
        ]
    }
)
```

### 3. Web-App Type for External Applications
```python
app_info = await controller.install(
    manifest={
        "name": "ImageJ.js",
        "type": "web-app",
        "version": "1.0.0",
        "url": "https://ij.imjoy.io",
        "cache_routes": [
            "https://ij.imjoy.io/*",
            "https://cdn.jsdelivr.net/*",
            "*.wasm",
            "*.js"
        ]
    },
    files=[]  # No source files needed
)
```

### 4. Managing Cache
```python
# Get browser worker service
browser_worker = await api.get_service("public/browser-worker")

# Check cache statistics
stats = await browser_worker.get_app_cache_stats("default", app_id)
print(f"Cache has {stats['entry_count']} entries using {stats['total_size_bytes']} bytes")

# Clear cache if needed
result = await browser_worker.clear_app_cache("default", app_id)
print(f"Cleared {result['deleted_entries']} cache entries")
```

## Performance Benefits

### Expected Improvements
1. **First Load**: Normal speed (resources cached during load)
2. **Subsequent Loads**: Significantly faster due to cached resources
3. **Network Usage**: Reduced bandwidth consumption
4. **Reliability**: Better offline/poor connection performance

### Benchmarking
The system includes performance comparison tests that measure:
- Initial app startup time (cache population)
- Subsequent startup time (cache utilization)
- Cache hit/miss ratios
- Network request reduction

## Testing

### Unit Tests
- `tests/test_browser_cache.py` - Core caching functionality
- Pattern matching, serialization, Redis operations
- Cache isolation and TTL handling

### Integration Tests  
- `tests/test_browser_cache_integration.py` - Full system testing
- Performance comparisons with/without caching
- Route interception and cache management APIs
- Multi-user and workspace isolation

### Server App Tests
- `tests/test_server_apps.py` - New web-app type testing
- Template generation and compilation
- Cache route configuration

## Configuration Options

### Environment Variables
- `HYPHA_CACHE_TTL`: Cache time-to-live in seconds (default: 86400)
- `HYPHA_REDIS_*`: Redis connection settings

### Manifest Options
- `cache_routes`: Array of URL patterns to cache
- `detached`: For web-app type, runs in isolated context
- `cookies`: Key-value pairs for web-app authentication
- `localStorage`: Key-value pairs for web-app state
- `authorization_token`: Bearer token for web-app requests

## Limitations and Considerations

### Current Limitations
1. **CORS Restrictions**: Some external resources may not be cacheable due to CORS policies
2. **Dynamic Content**: Only static resources benefit from caching
3. **Storage Usage**: Cache entries consume Redis memory
4. **TTL Management**: Cached entries expire after configured time

### Security Considerations
1. **Isolation**: Cache entries are isolated per workspace/app
2. **Authorization**: Cached responses don't include authentication headers
3. **Content Validation**: No integrity checking of cached content

### Performance Considerations
1. **Memory Usage**: Large applications may generate significant cache data
2. **Redis Performance**: High cache hit rates improve overall system performance
3. **Network Patterns**: Most effective for applications with many small, repeated requests

## Future Enhancements

### Planned Features
1. **Cache Compression**: Reduce memory usage with compressed storage
2. **Selective Cache Clearing**: Clear specific URL patterns
3. **Cache Statistics Dashboard**: Visual monitoring of cache performance
4. **Intelligent TTL**: Dynamic TTL based on resource type and usage patterns
5. **Cache Warming**: Pre-populate cache for known resource patterns

### Monitoring and Analytics
1. **Hit/Miss Ratios**: Track cache effectiveness
2. **Performance Metrics**: Measure load time improvements
3. **Storage Usage**: Monitor Redis memory consumption
4. **Popular Resources**: Identify most frequently cached resources

## Troubleshooting

### Common Issues

#### Cache Not Working
1. Check if `cache_routes` are specified correctly
2. Verify Redis connection is working
3. Ensure recording session is active
4. Check browser worker logs for errors

#### Performance Not Improved
1. Verify cache is being populated (check stats)
2. Ensure cached resources are actually being requested
3. Check for CORS restrictions preventing caching
4. Review cache hit/miss ratios

#### Memory Usage High
1. Monitor cache statistics for entry counts
2. Consider reducing TTL for less critical resources
3. Implement selective cache clearing
4. Review cache patterns for overly broad matching

### Debugging
```python
# Enable debug logging
import logging
logging.getLogger("browser").setLevel(logging.DEBUG)

# Check cache stats regularly
stats = await browser_worker.get_app_cache_stats(workspace, app_id)
print(f"Cache stats: {stats}")

# Monitor Redis directly
# redis-cli KEYS "acache:*"
# redis-cli INFO memory
```

## Conclusion

The browser worker smart caching system provides significant performance improvements for web applications that load many remote resources. By transparently caching frequently accessed files and providing fine-grained control over caching behavior, the system reduces load times and improves user experience while maintaining security and isolation between applications.