#!/usr/bin/env python
"""Simple test to verify LiteLLM integration works."""
import asyncio
import sys

async def test_litellm_integration():
    """Test that LiteLLM is properly integrated into Hypha."""
    
    print("Testing LiteLLM integration...")
    
    # Test 1: Import the main modules
    try:
        from hypha.litellm.proxy import proxy_server
        from hypha.litellm.router import Router
        print("✓ Successfully imported proxy_server and Router from hypha.litellm")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Check proxy_server has expected attributes
    if hasattr(proxy_server, 'app'):
        print("✓ proxy_server.app exists")
    else:
        print("✗ proxy_server.app not found")
        return False
    
    # Test 3: Create a Router instance
    try:
        router = Router(model_list=[])
        print("✓ Successfully created Router instance")
    except Exception as e:
        print(f"✗ Failed to create Router instance: {e}")
        return False
    
    # Test 4: Import the LLM proxy worker
    try:
        from hypha.workers.llm_proxy import LLMProxyWorker
        print("✓ Successfully imported LLMProxyWorker")
    except ImportError as e:
        print(f"✗ Failed to import LLMProxyWorker: {e}")
        return False
    
    # Test 5: Create LLMProxyWorker instance
    try:
        worker = LLMProxyWorker(store=None, workspace_manager=None, worker_id="test-llm-proxy")
        print("✓ Successfully created LLMProxyWorker instance")
        
        # Check worker properties
        assert worker.name == "LLM Proxy Worker"
        assert "llm-proxy" in worker.supported_types
        print("✓ Worker has correct name and supported types")
        
    except Exception as e:
        print(f"✗ Failed to create LLMProxyWorker: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test compile method signature
    try:
        import inspect
        compile_sig = inspect.signature(worker.compile)
        expected_params = ['manifest', 'files', 'config', 'context']
        actual_params = list(compile_sig.parameters.keys())
        
        if all(p in actual_params for p in expected_params):
            print("✓ Worker compile method has correct signature")
        else:
            print(f"✗ Worker compile method has unexpected signature: {actual_params}")
            return False
    except Exception as e:
        print(f"✗ Failed to check compile method: {e}")
        return False
    
    print("\n✅ All LiteLLM integration tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_litellm_integration())
    sys.exit(0 if success else 1)