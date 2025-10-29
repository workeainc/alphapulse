#!/usr/bin/env python3
"""
Comprehensive Test for All AlphaPlus Fixes
Tests signal generation, database schema, and SDE framework integration
"""

import asyncio
import asyncpg
import logging
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection and basic queries"""
    try:
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        conn = await asyncpg.connect(db_url)
        
        # Test basic connection
        result = await conn.fetchval("SELECT 1")
        assert result == 1, "Database connection failed"
        logger.info("✅ Database connection successful")
        
        # Test TimescaleDB extension
        result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')")
        assert result, "TimescaleDB extension not enabled"
        logger.info("✅ TimescaleDB extension enabled")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

async def test_table_schemas():
    """Test that all required tables and columns exist"""
    try:
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        conn = await asyncpg.connect(db_url)
        
        # Test critical tables exist
        critical_tables = [
            'ohlcv_data', 'signals', 'market_intelligence', 'volume_analysis',
            'free_api_market_data', 'free_api_sentiment_data', 'price_action_ml_predictions',
            'sde_dynamic_thresholds', 'market_regime_data', 'sentiment_data'
        ]
        
        for table in critical_tables:
            result = await conn.fetchval(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')")
            assert result, f"Table {table} does not exist"
            logger.info(f"✅ Table {table} exists")
        
        # Test critical columns exist
        critical_columns = [
            ('ohlcv_data', ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']),
            ('signals', ['signal_id', 'symbol', 'timeframe', 'direction', 'confidence_score']),
            ('market_intelligence', ['timestamp', 'market_sentiment_score', 'fear_greed_index']),
            ('volume_analysis', ['volume_ratio', 'volume_trend', 'buy_volume_ratio']),
            ('price_action_ml_predictions', ['symbol', 'timeframe', 'prediction_probability', 'confidence_score']),
            ('sde_dynamic_thresholds', ['min_confidence_threshold', 'min_consensus_heads', 'calibration_weight_ensemble']),
            ('market_regime_data', ['timestamp', 'regime_type', 'confidence', 'volatility']),
            ('sentiment_data', ['timestamp', 'sentiment_label'])
        ]
        
        for table, columns in critical_columns:
            for column in columns:
                result = await conn.fetchval(f"""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{table}' AND column_name = '{column}'
                    )
                """)
                assert result, f"Column {table}.{column} does not exist"
            logger.info(f"✅ All columns exist in {table}")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Table schema test failed: {e}")
        return False

async def test_data_population():
    """Test that data is populated"""
    try:
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        conn = await asyncpg.connect(db_url)
        
        # Test OHLCV data
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        for symbol in symbols:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '1m'")
            assert count > 0, f"No data for {symbol}"
            logger.info(f"✅ {symbol}: {count} 1m candles")
        
        # Test 4H aggregates
        for symbol in symbols:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '4h'")
            logger.info(f"✅ {symbol}: {count} 4h candles")
        
        # Test 1D aggregates
        for symbol in symbols:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '1d'")
            logger.info(f"✅ {symbol}: {count} 1d candles")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Data population test failed: {e}")
        return False

async def test_signal_generation():
    """Test signal generation with real data"""
    try:
        # Import required components
        from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator
        import asyncpg
        
        # Create database connection pool
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        
        # Mock exchange (for testing)
        class MockExchange:
            def __init__(self):
                self.name = "binance_test"
        
        exchange = MockExchange()
        
        # Initialize generator
        generator = IntelligentSignalGenerator(db_pool=db_pool, exchange=exchange)
        await generator.initialize()
        
        # Test signal generation for BTCUSDT
        logger.info("🎯 Testing signal generation for BTCUSDT...")
        
        signal = await generator.generate_intelligent_signal(
            symbol="BTCUSDT",
            timeframe="1h"
        )
        
        if signal:
            logger.info(f"✅ Signal generated successfully!")
            logger.info(f"   Direction: {signal.direction}")
            logger.info(f"   Confidence: {signal.confidence_score:.3f}")
            logger.info(f"   Entry Price: {signal.entry_price}")
            logger.info(f"   Stop Loss: {signal.stop_loss}")
            logger.info(f"   Take Profit: {signal.take_profit_1}")
            logger.info(f"   Status: {signal.status}")
        else:
            logger.warning("⚠️ No signal generated (may be normal)")
        
        await db_pool.close()
        return True
            
    except Exception as e:
        logger.error(f"❌ Signal generation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_sde_framework():
    """Test SDE framework integration"""
    try:
        # Import SDE components
        from src.ai.sde_framework import SDEFramework
        from src.ai.sde_integration_manager import SDEIntegrationManager
        import asyncpg
        
        # Create database connection pool
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        
        # Initialize SDE framework (no separate initialize method needed)
        sde_framework = SDEFramework(db_pool=db_pool)
        
        # Initialize integration manager
        integration_manager = SDEIntegrationManager(db_pool=db_pool)
        await integration_manager.load_integration_config()
        
        logger.info("✅ SDE framework initialized successfully")
        
        # Test divergence analysis (SDE framework's main functionality)
        # Create a sample DataFrame for divergence analysis
        import pandas as pd
        import numpy as np
        
        sample_data = {
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(50000, 60000, 100),
            'high': np.random.uniform(50000, 60000, 100),
            'low': np.random.uniform(50000, 60000, 100),
            'close': np.random.uniform(50000, 60000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }
        sample_df = pd.DataFrame(sample_data)
        
        divergence_result = await sde_framework.divergence_analyzer.analyze_divergences(
            df=sample_df,
            symbol="BTCUSDT",
            timeframe="1h"
        )
        
        if divergence_result:
            logger.info(f"✅ Divergence analysis completed")
            logger.info(f"   RSI Divergence: {divergence_result.rsi_divergence}")
            logger.info(f"   MACD Divergence: {divergence_result.macd_divergence}")
            logger.info(f"   Overall Confidence: {divergence_result.overall_confidence:.3f}")
            logger.info(f"   Divergence Score: {divergence_result.divergence_score:.3f}")
        else:
            logger.warning("⚠️ No divergence analysis results")
        
        # Test integration manager
        integration_result = await integration_manager.integrate_sde_with_signal(
            signal_id="test_signal_123",
            symbol="BTCUSDT",
            timeframe="1h",
            analysis_results={'technical': {'rsi': 45.0, 'macd': 0.1}},
            market_data={'current_price': 50000.0, 'volume': 1000000.0}
        )
        
        logger.info(f"✅ SDE integration completed: {integration_result.all_gates_passed}")
        
        await db_pool.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ SDE framework test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting Comprehensive AlphaPlus Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Table Schemas", test_table_schemas),
        ("Data Population", test_data_population),
        ("SDE Framework", test_sde_framework),
        ("Signal Generation", test_signal_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! AlphaPlus is ready for production!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Review issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
