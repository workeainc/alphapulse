#!/usr/bin/env python3
"""
Test script to verify database connection and service initialization fixes
"""

import asyncio
import logging
import sys
import os
from sqlalchemy import text

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection with fallback support"""
    try:
        logger.info("🔍 Testing database connection...")
        
        # Import the database connection
        from src.database.connection import TimescaleDBConnection
        
        # Test configuration
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'fallback_db_path': 'data/test_fallback.db'
        }
        
        # Initialize connection
        db_conn = TimescaleDBConnection(config)
        await db_conn.initialize(create_tables=False)
        
        if db_conn.use_fallback:
            logger.info("✅ SQLite fallback connection successful")
        else:
            logger.info("✅ TimescaleDB connection successful")
        
        # Test basic query
        if db_conn.use_fallback:
            # Test SQLite query
            cursor = await db_conn.sqlite_connection.execute("SELECT 1 as test")
            result = await cursor.fetchone()
            logger.info(f"✅ SQLite test query result: {result}")
        else:
            # Test TimescaleDB query
            async with db_conn.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                logger.info(f"✅ TimescaleDB test query result: {result.fetchone()}")
        
        await db_conn.close()
        logger.info("✅ Database connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

async def test_service_imports():
    """Test service imports to verify no import errors"""
    try:
        logger.info("🔍 Testing service imports...")
        
        # Test ML Pattern Detector import
        from src.app.strategies.ml_pattern_detector import MLPatternDetector
        logger.info("✅ ML Pattern Detector import successful")
        
        # Test Strategy Manager import
        from src.app.strategies.strategy_manager import StrategyManager
        logger.info("✅ Strategy Manager import successful")
        
        # Test Pattern Integration Service import
        from src.app.services.pattern_integration_service import PatternIntegrationService
        logger.info("✅ Pattern Integration Service import successful")
        
        # Test Market Data Service import
        from src.app.services.market_data_service import MarketDataService
        logger.info("✅ Market Data Service import successful")
        
        logger.info("✅ All service imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service import test failed: {e}")
        return False

async def test_service_initialization():
    """Test service initialization"""
    try:
        logger.info("🔍 Testing service initialization...")
        
        # Test ML Pattern Detector initialization
        from src.app.strategies.ml_pattern_detector import MLPatternDetector
        ml_detector = MLPatternDetector()
        logger.info("✅ ML Pattern Detector initialized")
        
        # Test Strategy Manager initialization
        from src.app.strategies.strategy_manager import StrategyManager
        strategy_manager = StrategyManager()
        logger.info("✅ Strategy Manager initialized")
        
        # Test Market Data Service initialization
        from src.app.services.market_data_service import MarketDataService
        market_data_service = MarketDataService()
        logger.info("✅ Market Data Service initialized")
        
        logger.info("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service initialization test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting barrier fixes verification tests...")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Service Imports", test_service_imports),
        ("Service Initialization", test_service_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        result = await test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Tests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Barrier fixes are working correctly.")
    else:
        logger.info("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    asyncio.run(main())
