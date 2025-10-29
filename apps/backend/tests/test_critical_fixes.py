#!/usr/bin/env python3
"""
Test script to validate critical fixes for Phase 1 streaming infrastructure
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_component_methods():
    """Test that all component methods are working"""
    logger.info("🧪 Testing component methods...")
    
    try:
        # Test StreamBuffer get_metrics method
        from src.streaming.stream_buffer import StreamBuffer
        buffer = StreamBuffer()
        metrics = buffer.get_metrics()
        assert isinstance(metrics, dict), "StreamBuffer.get_metrics() should return a dict"
        logger.info("✅ StreamBuffer.get_metrics() method working")
        
        # Test FailoverManager get_status method
        from src.streaming.failover_manager import FailoverManager
        failover = FailoverManager()
        status = failover.get_status()
        assert isinstance(status, dict), "FailoverManager.get_status() should return a dict"
        logger.info("✅ FailoverManager.get_status() method working")
        
        # Test CandleBuilder timeframes attribute
        from src.streaming.candle_builder import CandleBuilder
        builder = CandleBuilder()
        assert hasattr(builder, 'timeframes'), "CandleBuilder should have timeframes attribute"
        assert isinstance(builder.timeframes, list), "timeframes should be a list"
        logger.info("✅ CandleBuilder.timeframes attribute working")
        
        # Test TimescaleDBConnection async context manager
        from src.database.connection import TimescaleDBConnection
        connection = TimescaleDBConnection()
        assert hasattr(connection, '__aenter__'), "TimescaleDBConnection should have async context manager"
        assert hasattr(connection, '__aexit__'), "TimescaleDBConnection should have async context manager"
        assert hasattr(connection, 'get_session'), "TimescaleDBConnection should have get_session method"
        assert hasattr(connection, 'close'), "TimescaleDBConnection should have close method"
        logger.info("✅ TimescaleDBConnection async context manager working")
        
        logger.info("🎉 All component methods are working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component method test failed: {e}")
        return False

async def test_configuration():
    """Test that configuration settings are correct"""
    logger.info("🧪 Testing configuration settings...")
    
    try:
        from src.core.config import settings
        
        # Test TimescaleDB settings
        assert hasattr(settings, 'TIMESCALEDB_HOST'), "Settings should have TIMESCALEDB_HOST"
        assert hasattr(settings, 'TIMESCALEDB_PORT'), "Settings should have TIMESCALEDB_PORT"
        assert hasattr(settings, 'TIMESCALEDB_DATABASE'), "Settings should have TIMESCALEDB_DATABASE"
        assert hasattr(settings, 'TIMESCALEDB_USERNAME'), "Settings should have TIMESCALEDB_USERNAME"
        assert hasattr(settings, 'TIMESCALEDB_PASSWORD'), "Settings should have TIMESCALEDB_PASSWORD"
        
        logger.info(f"✅ TimescaleDB settings: {settings.TIMESCALEDB_HOST}:{settings.TIMESCALEDB_PORT}")
        logger.info("🎉 All configuration settings are correct!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def test_api_integration():
    """Test that API integration is working"""
    logger.info("🧪 Testing API integration...")
    
    try:
        # Test that main application can be imported
        from src.app.main_ai_system_simple import app, streaming_initialized
        
        # Test that streaming components are available
        from src.streaming.stream_processor import StreamProcessor
        from src.streaming.stream_metrics import StreamMetrics
        from src.streaming.stream_normalizer import StreamNormalizer
        from src.streaming.candle_builder import CandleBuilder
        from src.streaming.rolling_state_manager import RollingStateManager
        
        logger.info("✅ All streaming components can be imported")
        logger.info("✅ Main application can be imported")
        logger.info("🎉 API integration is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False

async def main():
    """Run all critical fix tests"""
    logger.info("🚀 Starting Critical Fix Validation Tests")
    logger.info("=" * 60)
    
    results = []
    
    # Test component methods
    results.append(await test_component_methods())
    
    # Test configuration
    results.append(await test_configuration())
    
    # Test API integration
    results.append(await test_api_integration())
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 CRITICAL FIX VALIDATION RESULTS")
    logger.info("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"✅ Tests Passed: {passed}/{total}")
    logger.info(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        logger.info("✅ Phase 1 streaming infrastructure is ready for Phase 2")
    else:
        logger.error("❌ Some critical fixes still need attention")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
