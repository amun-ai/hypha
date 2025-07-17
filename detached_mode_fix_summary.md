# Fix for Detached Mode App Timeout in test_detached_mode_apps

## Problem

The `test_detached_mode_apps` test in PR #763 was failing with a timeout error:

```
Exception: Failed to start the app: stupid-weasel-possess-gleefully during installation, error: Failed to start the app: ws-user-user-1/stupid-weasel-possess-gleefully, timeout reached (60s)
```

## Root Cause Analysis

The issue was caused by two missing `detached=True` parameters in the call chain:

1. **In `launch` method** (`hypha/apps.py`): The `launch` method was not passing the `detached` parameter to the `install` method, causing apps to be installed in non-detached mode even when `detached=True` was specified.

2. **In `test_detached_mode_apps` test** (`tests/test_server_apps.py`): Test 2 was calling `controller.install()` without the `detached=True` parameter, causing the app installation to timeout when the app doesn't register any services.

## Call Flow Analysis

When `detached=False` (default):
```
test calls controller.launch(detached=True)
  → launch calls install(stage=True, detached=❌ missing)
    → install with stage=True doesn't call commit_app (✓ correct)
  → launch calls start(detached=True, stage=True) (✓ correct)

test calls controller.install(detached=❌ missing)
  → install with stage=False calls commit_app(detached=False)
    → commit_app calls start(detached=False)
      → start waits for services that never get registered
      → TIMEOUT after 60 seconds
```

When `detached=True` (fixed):
```
test calls controller.launch(detached=True)
  → launch calls install(stage=True, detached=✓ True)
    → install with stage=True doesn't call commit_app (✓ correct)
  → launch calls start(detached=True, stage=True) (✓ correct)

test calls controller.install(detached=✓ True)
  → install with stage=False calls commit_app(detached=True)
    → commit_app calls start(detached=True)
      → start doesn't wait for services
      → SUCCESS immediately
```

## Solution

### Fix 1: Update `launch` method in `hypha/apps.py`

```python
# Install app in stage mode
app_info = await self.install(
    source=source,
    config=config,
    timeout=timeout,
    overwrite=overwrite,
    stage=True,
+   detached=detached,  # Pass detached parameter to install
    context=context,
)
```

### Fix 2: Update `test_detached_mode_apps` in `tests/test_server_apps.py`

```python
# Test 2: Install and start with detached=True
print("Testing detached mode with install + start...")
app_info = await controller.install(
    source=detached_script,
    config={"type": "window", "name": "Detached Script Install"},
    overwrite=True,
+   detached=True,  # Add detached=True to prevent timeout
)
```

## Impact

- **Fixes the failing test**: `test_detached_mode_apps` now passes without timeout errors
- **Improves detached mode functionality**: Ensures detached mode works correctly in all code paths
- **No breaking changes**: The fix only adds missing parameters that should have been there
- **Better error handling**: Apps that don't register services can now be installed without timeouts when using detached mode

## Testing

The fix addresses the specific timeout issue in the test while maintaining backward compatibility. The detached mode functionality now works correctly for:

1. `launch()` method with `detached=True`
2. `install()` method with `detached=True` followed by `start()` with `detached=True`
3. Apps that don't register any services (like simple scripts)

## PR Information

- **Branch**: `fix-detached-mode-test-timeout`
- **Base Branch**: `enhance-artifact-file-listing`
- **Files Changed**: 
  - `hypha/apps.py` (1 line added)
  - `tests/test_server_apps.py` (1 line added)
- **Commit**: `dae0fe0` - "Fix detached mode app timeout in test_detached_mode_apps"