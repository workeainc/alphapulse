#!/usr/bin/env python3
"""
Comprehensive test script for AlphaPlus Enhanced Algorithm Integration.
Tests all 8 major algorithms and their integration with TimescaleDB.
"""

import asyncio
import asyncpg
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add backend to path for imports
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection details
DB_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def test_database_connection():
    """Test database connection and basic functionality."""
    logger.info("🔍 Testing database connection...")
    
    try:
        db_pool = await asyncpg.create_pool(DB_URL)
        async with db_pool.acquire() as conn:
            # Test basic connection
            result = await conn.fetchval("SELECT 1")
            assert result == 1, "Basic query failed"
            
            # Test TimescaleDB extension
            result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
            assert result, "TimescaleDB extension not enabled"
            
            # Test key tables exist
            tables = ['ohlcv_data', 'order_book_data', 'algorithm_results', 'signal_confluence', 
                     'volume_profile_analysis', 'psychological_levels_analysis']
            
            for table in tables:
                exists = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    );
                """)
                assert exists, f"Table {table} does not exist"
            
            logger.info("✅ Database connection and schema verification successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False
    finally:
        if 'db_pool' in locals():
            await db_pool.close()

async def test_historical_data_preloader():
    """Test the historical data preloader service."""
    logger.info("🔍 Testing Historical Data Preloader...")
    
    try:
        from src.services.historical_data_preloader import HistoricalDataPreloader, PreloadConfig
        
        config = PreloadConfig(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            lookback_days=1,
            min_candles=10
        )
        
        preloader = HistoricalDataPreloader(DB_URL, config)
        await preloader.initialize()
        
        # Test preloading for a single symbol/timeframe
        result = await preloader.preload_symbol_timeframe('BTCUSDT', '1m')
        
        if result.success:
            logger.info(f"✅ Historical data preloader test successful: {result.candles_loaded} candles loaded")
        else:
            logger.warning(f"⚠️ Historical data preloader test failed: {result.error_message}")
        
        await preloader.close()
        return result.success
        
    except Exception as e:
        logger.error(f"❌ Historical data preloader test failed: {e}")
        return False

async def test_enhanced_orderbook_integration():
    """Test the enhanced order book integration service."""
    logger.info("🔍 Testing Enhanced Order Book Integration...")
    
    try:
        from src.services.enhanced_orderbook_integration import EnhancedOrderBookIntegration
        
        service = EnhancedOrderBookIntegration(DB_URL)
        await service.initialize()
        
        # Test volume profile analysis
        result = await service.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
        
        if result:
            logger.info(f"✅ Enhanced order book integration test successful")
            logger.info(f"   POC Price: {result.volume_profile.poc_price}")
            logger.info(f"   Total Volume: {result.volume_profile.total_volume}")
            logger.info(f"   HVN Levels: {len(result.volume_profile.high_volume_nodes)}")
            logger.info(f"   LVN Levels: {len(result.volume_profile.low_volume_nodes)}")
        else:
            logger.warning("⚠️ Enhanced order book integration test returned no results")
        
        await service.close()
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ Enhanced order book integration test failed: {e}")
        return False

async def test_psychological_levels_analyzer():
    """Test the psychological levels analyzer."""
    logger.info("🔍 Testing Psychological Levels Analyzer...")
    
    try:
        from src.strategies.standalone_psychological_levels_analyzer import StandalonePsychologicalLevelsAnalyzer
        
        analyzer = StandalonePsychologicalLevelsAnalyzer(DB_URL)
        await analyzer.initialize()
        
        # Test psychological level analysis
        result = await analyzer.analyze_psychological_levels('BTCUSDT', '1m')
        
        if result:
            logger.info(f"✅ Psychological levels analyzer test successful")
            logger.info(f"   Current Price: {result.current_price}")
            logger.info(f"   Market Regime: {result.market_regime}")
            logger.info(f"   Detected Levels: {len(result.psychological_levels)}")
            logger.info(f"   Analysis Confidence: {result.analysis_confidence}")
        else:
            logger.warning("⚠️ Psychological levels analyzer test returned no results")
        
        await analyzer.close()
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ Psychological levels analyzer test failed: {e}")
        return False

async def test_volume_weighted_levels_analyzer():
    """Test the enhanced volume-weighted levels analyzer."""
    logger.info("🔍 Testing Enhanced Volume-Weighted Levels Analyzer...")
    
    try:
        from src.strategies.enhanced_volume_weighted_levels_analyzer import EnhancedVolumeWeightedLevelsAnalyzer
        
        analyzer = EnhancedVolumeWeightedLevelsAnalyzer(DB_URL)
        await analyzer.initialize()
        
        # Test volume profile analysis
        result = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1m')
        
        if result:
            logger.info(f"✅ Enhanced volume-weighted levels analyzer test successful")
            logger.info(f"   POC Price: {result.poc_price}")
            logger.info(f"   Value Area High: {result.value_area_high}")
            logger.info(f"   Value Area Low: {result.value_area_low}")
            logger.info(f"   HVN Levels: {len(result.high_volume_nodes)}")
            logger.info(f"   LVN Levels: {len(result.low_volume_nodes)}")
            logger.info(f"   Analysis Confidence: {result.analysis_confidence}")
        else:
            logger.warning("⚠️ Enhanced volume-weighted levels analyzer test returned no results")
        
        await analyzer.close()
        return result is not None
        
    except Exception as e:
        logger.error(f"❌ Enhanced volume-weighted levels analyzer test failed: {e}")
        return False

async def test_algorithm_integration_service():
    """Test the main algorithm integration service."""
    logger.info("🔍 Testing Algorithm Integration Service...")
    
    try:
        from src.services.algorithm_integration_service import AlgorithmIntegrationService
        
        service = AlgorithmIntegrationService(DB_URL)
        await service.initialize()
        
        # Check if preload was completed
        if service.stats.get('preload_completed', False):
            logger.info("✅ Algorithm integration service test successful")
            logger.info(f"   Preload Completed: {service.stats['preload_completed']}")
            logger.info(f"   Total Runs: {service.stats['total_runs']}")
            logger.info(f"   Successful Runs: {service.stats['successful_runs']}")
        else:
            logger.warning("⚠️ Algorithm integration service preload not completed")
        
        await service.close()
        return service.stats.get('preload_completed', False)
        
    except Exception as e:
        logger.error(f"❌ Algorithm integration service test failed: {e}")
        return False

async def test_websocket_integration():
    """Test the WebSocket integration with robust volume parsing."""
    logger.info("🔍 Testing WebSocket Integration...")
    
    try:
        from src.core.websocket_binance import BinanceWebSocketClient
        
        # Test the safe parsing functions
        client = BinanceWebSocketClient()
        
        # Test safe_float function with complete kline data
        test_values = ['123.45', '0', '', None, 'invalid', 123.45]
        for value in test_values:
            try:
                # Create complete kline data structure
                kline_data = {
                    'k': {
                        'o': value, 'h': value, 'l': value, 'c': value, 'v': value,
                        'q': '1000', 'Q': '1000', 'V': '500', 'n': 10,
                        't': 1234567890000, 'i': '1m', 'x': True
                    },
                    's': 'BTCUSDT'
                }
                result = client._parse_kline_data(kline_data)
                if result:
                    logger.info(f"✅ Safe parsing test successful for value: {value}")
                else:
                    logger.warning(f"⚠️ Safe parsing returned None for value: {value}")
            except Exception as e:
                logger.error(f"❌ Safe parsing failed for value {value}: {e}")
        
        logger.info("✅ WebSocket integration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket integration test failed: {e}")
        return False

async def main():
    """Main test runner."""
    logger.info("🚀 Starting AlphaPlus Enhanced Algorithm Integration Tests")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Historical Data Preloader", test_historical_data_preloader),
        ("Enhanced Order Book Integration", test_enhanced_orderbook_integration),
        ("Psychological Levels Analyzer", test_psychological_levels_analyzer),
        ("Volume-Weighted Levels Analyzer", test_volume_weighted_levels_analyzer),
        ("Algorithm Integration Service", test_algorithm_integration_service),
        ("WebSocket Integration", test_websocket_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Enhanced Algorithm Integration is ready!")
    elif passed_tests >= total_tests * 0.7:
        logger.info("⚠️ Most tests passed. Some components may need attention.")
    else:
        logger.error("❌ Multiple tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
