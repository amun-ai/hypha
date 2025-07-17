# Kafka Error Handling - No Skipping, Let Errors Explode!

## Summary of Changes

I have removed all pytest skipif statements and ensured that Kafka errors will explode properly instead of being hidden or skipped.

## Changes Made

### 1. Removed pytest.mark.skipif Decorators

**Files Modified:**
- `tests/test_kafka_event_bus.py`
- `tests/test_kafka_servers.py`

**Before:**
```python
@pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka not available")
async def test_kafka_server_communication(...):
```

**After:**
```python
async def test_kafka_server_communication(...):
```

### 2. Removed pytest.skip() Calls

**Files Modified:**
- `tests/test_kafka_event_bus.py`

**Before:**
```python
if not KAFKA_AVAILABLE:
    pytest.skip("Kafka not available")
```

**After:**
```python
# Removed entirely - let errors explode!
```

### 3. Enhanced Import Error Handling

**File Modified:** `hypha/core/__init__.py`

**Before:**
```python
try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
```

**After:**
```python
try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError as e:
    # For tests, we want to fail hard if Kafka is not available
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise ImportError(f"aiokafka is required for tests: {e}")
    KAFKA_AVAILABLE = False
```

### 4. Removed Import Guards in KafkaEventBus

**File Modified:** `hypha/core/__init__.py`

**Before:**
```python
def __init__(self, kafka_uri: str, server_id: str = None) -> None:
    """Initialize the Kafka event bus."""
    if not KAFKA_AVAILABLE:
        raise ImportError("aiokafka is not installed. Please install it with: pip install aiokafka")
    
    self._kafka_uri = kafka_uri
```

**After:**
```python
def __init__(self, kafka_uri: str, server_id: str = None) -> None:
    """Initialize the Kafka event bus."""
    self._kafka_uri = kafka_uri
```

### 5. Enhanced Error Handling in Test Fixtures

**File Modified:** `tests/conftest.py`

**Enhanced Docker Compose Error Handling:**
```python
# Start Kafka cluster
try:
    result = subprocess.run([
        "docker-compose", "-f", "/tmp/kafka-test-compose.yml", "up", "-d"
    ], check=True, capture_output=True, text=True)
    print(f"Docker Compose output: {result.stdout}")
except subprocess.CalledProcessError as e:
    print(f"Docker Compose failed: {e.stderr}")
    raise
```

**Enhanced Kafka Readiness Check:**
```python
# Wait for Kafka to be ready
timeout = 60
last_error = None
while timeout > 0:
    try:
        # ... test code ...
        break
    except ImportError:
        # If aiokafka is not available, fail hard
        raise
    except Exception as e:
        last_error = e
        print(f"Kafka not ready yet (timeout={timeout}): {e}")
    timeout -= 1
    time.sleep(1)

if timeout <= 0:
    raise RuntimeError(f"Kafka server failed to start within timeout. Last error: {last_error}")
```

### 6. Added aiokafka to Test Requirements

**File Modified:** `requirements_test.txt`

**Added:**
```
aiokafka==0.10.0
```

### 7. Removed KAFKA_AVAILABLE Imports from Tests

**Files Modified:**
- `tests/test_kafka_event_bus.py`
- `tests/test_kafka_servers.py`

**Before:**
```python
from hypha.core import KafkaEventBus, KAFKA_AVAILABLE
```

**After:**
```python
from hypha.core import KafkaEventBus
```

## Expected Behavior

### ✅ What Will Happen Now:

1. **Import Errors Will Explode**: If `aiokafka` is not installed, tests will fail with clear ImportError messages
2. **Kafka Connection Errors Will Explode**: If Kafka server is not available, tests will fail with connection errors
3. **No Silent Skipping**: Tests will not be silently skipped - they will fail loudly
4. **Clear Error Messages**: All errors will be properly propagated with detailed messages

### ❌ What Will NOT Happen:

1. **No pytest.skip()**: Tests will not be skipped silently
2. **No pytest.mark.skipif**: Tests will not be conditionally skipped
3. **No Graceful Degradation**: No fallback to "fake" or "mock" implementations
4. **No Hidden Errors**: All errors will be visible and must be fixed

## Testing the Error Handling

### Test Import Errors:
```bash
# Remove aiokafka temporarily
pip uninstall aiokafka -y

# Run tests - they should fail with ImportError
pytest tests/test_kafka_event_bus.py -v
```

### Test Kafka Connection Errors:
```bash
# Install aiokafka but don't start Kafka server
pip install aiokafka

# Run tests - they should fail with connection errors
pytest tests/test_kafka_event_bus.py -v
```

### Test with Proper Setup:
```bash
# Start Kafka server and install dependencies
docker-compose -f /tmp/kafka-test-compose.yml up -d
pip install aiokafka

# Run tests - they should pass
pytest tests/test_kafka_event_bus.py -v
```

## CI/CD Integration

The GitHub Actions workflow has been updated to:

1. **Start Kafka and Zookeeper services** as Docker containers
2. **Install Docker Compose** for test fixtures
3. **Install aiokafka** via tox/requirements
4. **Let all errors explode** during test execution

If any part of the Kafka setup fails, the CI will fail loudly with clear error messages.

## Verification

Run this command to verify all skip statements have been removed:

```bash
# Check for any remaining skip statements
grep -r "pytest.skip\|mark.skipif" tests/test_kafka_*.py || echo "No skip statements found ✓"

# Check for KAFKA_AVAILABLE guards
grep -r "KAFKA_AVAILABLE" tests/test_kafka_*.py || echo "No KAFKA_AVAILABLE guards found ✓"

# Check that aiokafka is in requirements
grep "aiokafka" requirements*.txt && echo "aiokafka found in requirements ✓"
```

## Result

🎯 **Mission Accomplished**: All pytest skip statements have been removed, and Kafka errors will now explode properly instead of being hidden or skipped. The implementation follows the "fail fast, fail loud" principle as requested.