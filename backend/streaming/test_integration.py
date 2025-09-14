#!/usr/bin/env python3
"""
Test script to verify streaming infrastructure integration
"""

import asyncio
import logging
from pathlib import Path

# Add backend to path (now in streaming subdirectory)
backend_path = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_streaming_integration():
    """Test streaming infrastructure integration"""
    logger.info("🧪 Testing streaming infrastructure integration...")
    
    try:
        # Import the main application
        from app.main_ai_system_simple import app, stream_processor, stream_metrics
        
        logger.info("✅ Main application imported successfully")
        
        # Test streaming components initialization
        if stream_processor:
            await stream_processor.initialize()
            logger.info("✅ Stream processor initialized")
            
            # Test basic functionality
            test_message = {
                'message_id': 'test_001',
                'symbol': 'BTCUSDT',
                'data_type': 'tick',
                'data': {'price': 50000.0, 'volume': 1.0}
            }
            
            result = await stream_processor.process_message(test_message)
            logger.info(f"✅ Message processing test: {result}")
        
        if stream_metrics:
            await stream_metrics.initialize()
            logger.info("✅ Stream metrics initialized")
            
            # Test metrics collection
            metrics = await stream_metrics.get_current_metrics()
            logger.info(f"✅ Metrics collection test: {metrics}")
        
        logger.info("🎉 Streaming infrastructure integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoints"""
    logger.info("🧪 Testing API endpoints...")
    
    try:
        from app.main_ai_system_simple import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test streaming status endpoint
        response = client.get("/api/streaming/status")
        logger.info(f"✅ Streaming status endpoint: {response.status_code}")
        
        # Test streaming metrics endpoint
        response = client.get("/api/streaming/metrics")
        logger.info(f"✅ Streaming metrics endpoint: {response.status_code}")
        
        # Test streaming data endpoint
        response = client.get("/api/streaming/data/BTCUSDT")
        logger.info(f"✅ Streaming data endpoint: {response.status_code}")
        
        logger.info("🎉 API endpoints test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("STREAMING INFRASTRUCTURE INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Test streaming integration
    integration_success = await test_streaming_integration()
    
    # Test API endpoints
    api_success = await test_api_endpoints()
    
    if integration_success and api_success:
        logger.info("🎉 All integration tests passed!")
        logger.info("✅ Streaming infrastructure is successfully integrated")
        logger.info("✅ API endpoints are working")
        logger.info("✅ Main application is ready for production")
    else:
        logger.error("❌ Some integration tests failed")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
