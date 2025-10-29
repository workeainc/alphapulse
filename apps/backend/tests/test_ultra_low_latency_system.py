#!/usr/bin/env python3
"""
Test Ultra-Low Latency System for AlphaPlus
Tests the integration of WebSocket client, pattern detection, and database storage
"""

import asyncio
import logging
import time
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_vectorized_pattern_detector():
    """Test the vectorized pattern detector"""
    try:
        logger.info("🧪 Testing Vectorized Pattern Detector...")
        
        from src.strategies.vectorized_pattern_detector import VectorizedPatternDetector
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        base_price = 50000.0
        prices = []
        for i in range(100):
            if i == 0:
                price = base_price
            else:
                change = np.random.normal(0, 0.001) * price
                price = price + change
            prices.append(price)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Create realistic OHLCV from base price
            volatility = 0.002
            open_price = price * (1 + np.random.normal(0, volatility))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility)))
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Initialize pattern detector
        detector = VectorizedPatternDetector(max_workers=2)
        
        # Test pattern detection
        start_time = time.time()
        patterns = await detector.detect_patterns_vectorized(
            df, 
            use_talib=True, 
            use_incremental=True
        )
        detection_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Pattern detection completed in {detection_time:.2f}ms")
        logger.info(f"📊 Detected {len(patterns)} patterns")
        
        # Log some pattern details
        for i, pattern in enumerate(patterns[:5]):  # Show first 5 patterns
            logger.info(f"   Pattern {i+1}: {pattern.pattern_name} ({pattern.pattern_type}) - Confidence: {pattern.confidence:.3f}")
        
        await detector.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Vectorized pattern detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_connection():
    """Test database connection and basic operations"""
    try:
        logger.info("🗄️ Testing Database Connection...")
        
        from src.database.connection import TimescaleDBConnection
        
        # Initialize database connection
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        
        db_connection = TimescaleDBConnection(db_config)
        await db_connection.initialize()
        
        # Test basic query
        async with db_connection.get_async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT COUNT(*) FROM ultra_low_latency_patterns"))
            count = result.scalar()
            logger.info(f"✅ Database connection successful. Patterns table has {count} records")
        
        await db_connection.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_service():
    """Test the integration service with mock data"""
    try:
        logger.info("🔗 Testing Integration Service...")
        
        from src.services.ultra_low_latency_integration import UltraLowLatencyIntegrationService, IntegrationConfig
        
        # Create configuration
        config = IntegrationConfig(
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            redis_url="redis://localhost:6379",
            db_url="postgresql://alpha_emon:Emon_@17711@localhost:5432/alphapulse",
            max_workers=2,
            confidence_threshold=0.6
        )
        
        # Initialize service
        service = UltraLowLatencyIntegrationService(config)
        await service.initialize()
        
        # Test with mock candlestick data
        mock_candlestick = {
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'timestamp': int(time.time() * 1000),
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 100.0
        }
        
        # Process mock data
        await service._process_message_ultra_fast(mock_candlestick)
        
        # Get performance stats
        stats = await service.get_performance_stats()
        logger.info(f"✅ Integration service test successful")
        logger.info(f"📊 Performance stats: {stats}")
        
        await service.stop()
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_redis_connection():
    """Test Redis connection for shared memory buffers"""
    try:
        logger.info("🔴 Testing Redis Connection...")
        
        import redis
        
        # Test Redis connection
        r = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        
        # Test ping
        response = r.ping()
        if response:
            logger.info("✅ Redis connection successful")
            
            # Test basic operations
            r.set("test_key", "test_value")
            value = r.get("test_key")
            if value == "test_value":
                logger.info("✅ Redis read/write operations successful")
                r.delete("test_key")
                return True
            else:
                logger.error("❌ Redis read/write test failed")
                return False
        else:
            logger.error("❌ Redis ping failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Redis connection test failed: {e}")
        logger.info("💡 Make sure Redis is running: redis-server")
        return False

async def test_websocket_client():
    """Test WebSocket client initialization"""
    try:
        logger.info("🔌 Testing WebSocket Client...")
        
        from src.core.ultra_low_latency_websocket import UltraLowLatencyWebSocketClient
        
        # Initialize WebSocket client
        client = UltraLowLatencyWebSocketClient("redis://localhost:6379")
        await client.initialize()
        
        logger.info("✅ WebSocket client initialization successful")
        
        # Test performance stats
        stats = await client.get_performance_stats()
        logger.info(f"📊 WebSocket stats: {stats}")
        
        await client.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Ultra-Low Latency System Tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Redis Connection", test_redis_connection),
        ("Vectorized Pattern Detector", test_vectorized_pattern_detector),
        ("WebSocket Client", test_websocket_client),
        ("Integration Service", test_integration_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ultra-low latency system is ready!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
