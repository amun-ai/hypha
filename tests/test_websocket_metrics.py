"""Test that websocket metrics don't create unbounded labels."""

import pytest
from hypha.websocket import _gauge


@pytest.mark.asyncio
async def test_websocket_gauge_no_labels():
    """Test that websocket gauge doesn't create per-workspace labels."""
    
    # Get initial value
    initial_value = _gauge._value.get()
    
    # Simulate connections from different workspaces
    for i in range(10):
        _gauge.inc()  # Would have been _gauge.labels(workspace=f"ws-{i}").inc()
    
    # Check value increased
    after_inc = _gauge._value.get()
    assert after_inc == initial_value + 10
    
    # Simulate disconnections
    for i in range(10):
        _gauge.dec()  # Would have been _gauge.labels(workspace=f"ws-{i}").dec()
    
    # Check value decreased back
    after_dec = _gauge._value.get()
    assert after_dec == initial_value
    
    print(f"Initial: {initial_value}, After inc: {after_inc}, After dec: {after_dec}")
    print("Test passed - no workspace labels created!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_websocket_gauge_no_labels())