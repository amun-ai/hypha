"""Test script to reproduce duplicate worker issue without Docker dependencies."""
import asyncio
import sys
import traceback
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_litellm_imports_and_boto3():
    """Test for litellm imports and boto3 compatibility."""
    
    # Test importing litellm modules first
    try:
        logger.info("Testing litellm imports...")
        
        from hypha.litellm.proxy import proxy_server
        logger.info("✓ proxy_server imported successfully")
        
        from hypha.litellm.router import Router
        logger.info("✓ Router imported successfully")
        
        # Test boto3 and aioboto3 imports
        logger.info("\nTesting boto3/aioboto3 imports...")
        import aioboto3
        import boto3
        from botocore.client import Config
        logger.info(f"✓ aioboto3 version: {aioboto3.__version__}")
        logger.info(f"✓ boto3 version: {boto3.__version__}")
        
        # Test S3 client creation
        logger.info("\nTesting S3 client creation...")
        
        # Test sync boto3 client
        s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        logger.info("✓ boto3 S3 client created successfully")
        
        # Test async aioboto3 client
        async with aioboto3.Session().client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        ) as async_s3_client:
            logger.info("✓ aioboto3 S3 client created successfully")
        
        # Test artifact module which uses S3
        logger.info("\nTesting artifact module...")
        from hypha.artifact import ArtifactManager
        logger.info("✓ ArtifactManager imported successfully")
        
        # Test S3 controller
        from hypha.s3 import S3Controller
        logger.info("✓ S3Controller imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    try:
        success = await test_litellm_imports_and_boto3()
        if success:
            logger.info("\n✅ All tests passed!")
            sys.exit(0)
        else:
            logger.error("\n❌ Test failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())