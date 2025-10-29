#!/usr/bin/env python3
"""
Test script for Data Versioning Schema Implementation
Phase 1: Database Schema Implementation
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection and basic operations"""
    try:
        logger.info("🧪 Testing database connection...")
        
        # Import database components
        from ..src.database.connection import TimescaleDBConnection
        from ..src.database.data_versioning_dao import DataVersioningDAO
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        # Get async session
        async_session = db_connection.get_async_session()
        if not async_session:
            logger.error("❌ Async session not available")
            return False
        
        async with async_session() as session:
            # Test DAO initialization
            dao = DataVersioningDAO(session)
            logger.info("✅ DataVersioningDAO initialized successfully")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

async def test_signals_table():
    """Test signals table operations"""
    try:
        logger.info("🧪 Testing signals table operations...")
        
        from ..src.database.connection import TimescaleDBConnection
        from ..src.database.data_versioning_dao import DataVersioningDAO
        
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async_session = db_connection.get_async_session()
        if not async_session:
            logger.error("❌ Async session not available")
            return False
        
        async with async_session() as session:
            dao = DataVersioningDAO(session)
            
            # Test creating a signal
            test_signal = {
                'label': 'BUY',
                'pred': 'BUY',
                'proba': 0.85,
                'ts': datetime.now(),
                'symbol': 'BTCUSDT',
                'tf': '1h',
                'features': {
                    'rsi': 25.5,
                    'macd': 0.0023,
                    'bb_position': 0.1,
                    'volume_ratio': 1.2
                },
                'model_id': 'test_model_v1',
                'outcome': None,
                'realized_rr': None,
                'latency_ms': 45
            }
            
            result = await dao.create_signal(test_signal)
            logger.info(f"✅ Signal created: {result}")
            
            # Test retrieving signals
            signals = await dao.get_signals(
                symbol='BTCUSDT',
                timeframe='1h',
                limit=10
            )
            logger.info(f"✅ Retrieved {len(signals)} signals")
            
            # Test updating signal outcome
            if signals:
                signal_id = signals[0]['id']
                updated = await dao.update_signal_outcome(
                    signal_id, 
                    'win', 
                    2.5
                )
                logger.info(f"✅ Signal outcome updated: {updated}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Signals table test failed: {e}")
        return False

async def test_candles_table():
    """Test candles table operations"""
    try:
        logger.info("🧪 Testing candles table operations...")
        
        from ..src.database.connection import TimescaleDBConnection
        from ..src.database.data_versioning_dao import DataVersioningDAO
        
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async_session = db_connection.get_async_session()
        if not async_session:
            logger.error("❌ Async session not available")
            return False
        
        async with async_session() as session:
            dao = DataVersioningDAO(session)
            
            # Test creating a candle
            test_candle = {
                'symbol': 'BTCUSDT',
                'tf': '1h',
                'ts': datetime.now(),
                'o': 45000.0,
                'h': 45100.0,
                'l': 44900.0,
                'c': 45050.0,
                'v': 1250.5,
                'vwap': 45025.0,
                'taker_buy_vol': 800.3,
                'features': {
                    'ema_20': 44900.0,
                    'ema_50': 44800.0,
                    'rsi': 55.2,
                    'atr': 150.0
                }
            }
            
            result = await dao.create_candle(test_candle)
            logger.info(f"✅ Candle created: {result}")
            
            # Test retrieving candles
            candles = await dao.get_candles(
                symbol='BTCUSDT',
                timeframe='1h',
                limit=10
            )
            logger.info(f"✅ Retrieved {len(candles)} candles")
            
            # Test updating candle features
            if candles:
                candle_id = candles[0]['id']
                new_features = {
                    'ema_20': 44900.0,
                    'ema_50': 44800.0,
                    'rsi': 55.2,
                    'atr': 150.0,
                    'macd': 0.0015,
                    'bb_upper': 45200.0,
                    'bb_lower': 44800.0
                }
                updated = await dao.update_candle_features(candle_id, new_features)
                logger.info(f"✅ Candle features updated: {updated}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Candles table test failed: {e}")
        return False

async def test_retrain_queue():
    """Test retrain queue operations"""
    try:
        logger.info("🧪 Testing retrain queue operations...")
        
        from ..src.database.connection import TimescaleDBConnection
        from ..src.database.data_versioning_dao import DataVersioningDAO
        
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async_session = db_connection.get_async_session()
        if not async_session:
            logger.error("❌ Async session not available")
            return False
        
        async with async_session() as session:
            dao = DataVersioningDAO(session)
            
            # First create a signal to reference
            test_signal = {
                'label': 'SELL',
                'pred': 'SELL',
                'proba': 0.78,
                'ts': datetime.now(),
                'symbol': 'ETHUSDT',
                'tf': '4h',
                'features': {
                    'rsi': 75.5,
                    'macd': -0.0018,
                    'bb_position': 0.9
                },
                'model_id': 'test_model_v1',
                'outcome': None,
                'realized_rr': None,
                'latency_ms': 32
            }
            
            signal_result = await dao.create_signal(test_signal)
            signal_id = signal_result['id']
            
            # Test adding to retrain queue
            queue_result = await dao.add_to_retrain_queue(
                signal_id=signal_id,
                reason="Model performance degradation detected",
                priority=3
            )
            logger.info(f"✅ Added to retrain queue: {queue_result}")
            
            # Test retrieving retrain queue
            queue_items = await dao.get_retrain_queue(limit=10)
            logger.info(f"✅ Retrieved {len(queue_items)} queue items")
            
            # Test updating retrain status
            if queue_items:
                queue_id = queue_items[0]['id']
                
                # Update to processing
                updated = await dao.update_retrain_status(queue_id, 'processing')
                logger.info(f"✅ Status updated to processing: {updated}")
                
                # Update to completed
                updated = await dao.update_retrain_status(queue_id, 'completed')
                logger.info(f"✅ Status updated to completed: {updated}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Retrain queue test failed: {e}")
        return False

async def test_analytics_queries():
    """Test analytics queries"""
    try:
        logger.info("🧪 Testing analytics queries...")
        
        from ..src.database.connection import TimescaleDBConnection
        from ..src.database.data_versioning_dao import DataVersioningDAO
        
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async_session = db_connection.get_async_session()
        if not async_session:
            logger.error("❌ Async session not available")
            return False
        
        async with async_session() as session:
            dao = DataVersioningDAO(session)
            
            # Test signal performance summary
            performance = await dao.get_signal_performance_summary(
                model_id='test_model_v1',
                days=30
            )
            logger.info(f"✅ Signal performance summary: {performance}")
            
            # Test feature importance analysis
            feature_importance = await dao.get_feature_importance_analysis(
                model_id='test_model_v1',
                days=30
            )
            logger.info(f"✅ Feature importance analysis: {feature_importance}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Analytics queries test failed: {e}")
        return False

async def run_all_tests():
    """Run all data versioning schema tests"""
    logger.info("🚀 Starting Data Versioning Schema Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Database Connection
    test_results['database_connection'] = await test_database_connection()
    
    # Test 2: Signals Table
    test_results['signals_table'] = await test_signals_table()
    
    # Test 3: Candles Table
    test_results['candles_table'] = await test_candles_table()
    
    # Test 4: Retrain Queue
    test_results['retrain_queue'] = await test_retrain_queue()
    
    # Test 5: Analytics Queries
    test_results['analytics_queries'] = await test_analytics_queries()
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info("\n" + "="*60)
    logger.info("📊 DATA VERSIONING SCHEMA TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:30} {status}")
    
    logger.info("="*60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Data versioning schema is working correctly.")
        logger.info("🚀 Phase 1: Database Schema Implementation is complete!")
    else:
        logger.warning("⚠️ Some tests failed. Please review the implementation.")
    
    return test_results

if __name__ == "__main__":
    try:
        # Run tests
        test_results = asyncio.run(run_all_tests())
        
        # Exit with appropriate code
        if all(test_results.values()):
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)
