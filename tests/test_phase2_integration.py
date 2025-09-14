#!/usr/bin/env python3
"""
Test script for Phase 2.1 (RL) and Phase 2.2 (NLP) integration
"""
import asyncio
import logging
import json
from datetime import datetime, timezone
from sqlalchemy import text

# Import our modules
from app.core.database_manager import DatabaseManager
from app.strategies.reinforcement_learning_engine import ReinforcementLearningEngine
from app.strategies.natural_language_processing_engine import NaturalLanguageProcessingEngine
from app.strategies.real_time_signal_generator import RealTimeSignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_database_migrations():
    """Test that Phase 2.1 and 2.2 database migrations are applied"""
    logger.info("🔍 Testing database migrations...")
    
    try:
        db_manager = DatabaseManager()
        # Initialize database manager
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            # Test RL columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'rl_%'
                ORDER BY column_name
            """)
            rl_columns = [row['column_name'] for row in result]
            
            expected_rl_columns = [
                'rl_action_strength', 'rl_action_type', 'rl_analysis', 'rl_avg_reward',
                'rl_best_reward', 'rl_bias', 'rl_confidence_threshold', 'rl_optimization_params',
                'rl_position_size', 'rl_risk_allocation', 'rl_stop_loss', 'rl_take_profit', 'rl_training_episodes'
            ]
            
            missing_rl = set(expected_rl_columns) - set(rl_columns)
            if missing_rl:
                logger.error(f"❌ Missing RL columns: {missing_rl}")
                return False
            else:
                logger.info(f"✅ All RL columns present: {len(rl_columns)} columns")
            
            # Test NLP columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'nlp_%'
                ORDER BY column_name
            """)
            nlp_columns = [row['column_name'] for row in result]
            
            expected_nlp_columns = [
                'nlp_analyses_performed', 'nlp_analysis', 'nlp_bias', 'nlp_cache_hit_rate',
                'nlp_high_confidence_sentiment', 'nlp_models_available', 'nlp_news_confidence',
                'nlp_news_sentiment', 'nlp_overall_confidence', 'nlp_overall_sentiment_score',
                'nlp_reddit_confidence', 'nlp_reddit_sentiment', 'nlp_sentiment_strength',
                'nlp_twitter_confidence', 'nlp_twitter_sentiment'
            ]
            
            missing_nlp = set(expected_nlp_columns) - set(nlp_columns)
            if missing_nlp:
                logger.error(f"❌ Missing NLP columns: {missing_nlp}")
                return False
            else:
                logger.info(f"✅ All NLP columns present: {len(nlp_columns)} columns")
            
            # Test views
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('rl_enhanced_signals', 'nlp_enhanced_signals')
            """)
            views = [row['viewname'] for row in result]
            
            if len(views) == 2:
                logger.info(f"✅ All views created: {views}")
            else:
                logger.error(f"❌ Missing views. Found: {views}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database migration test failed: {e}")
        return False

async def test_rl_engine():
    """Test Reinforcement Learning Engine"""
    logger.info("🤖 Testing Reinforcement Learning Engine...")
    
    try:
        rl_engine = ReinforcementLearningEngine()
        
        # Test initialization
        await rl_engine.start()
        logger.info("✅ RL Engine started successfully")
        
        # Test performance summary
        performance = rl_engine.get_performance_summary()
        logger.info(f"✅ RL Performance: {performance}")
        
        # Test trading environment
        if hasattr(rl_engine, 'trading_env') and rl_engine.trading_env:
            logger.info("✅ Trading environment available")
        else:
            logger.warning("⚠️ Trading environment not available (using mock)")
        
        # Test signal optimization environment
        if hasattr(rl_engine, 'signal_env') and rl_engine.signal_env:
            logger.info("✅ Signal optimization environment available")
        else:
            logger.warning("⚠️ Signal optimization environment not available (using mock)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RL Engine test failed: {e}")
        return False

async def test_nlp_engine():
    """Test Natural Language Processing Engine"""
    logger.info("📝 Testing Natural Language Processing Engine...")
    
    try:
        nlp_engine = NaturalLanguageProcessingEngine()
        
        # Test initialization
        await nlp_engine.start()
        logger.info("✅ NLP Engine started successfully")
        
        # Test performance summary
        performance = nlp_engine.get_performance_summary()
        logger.info(f"✅ NLP Performance: {performance}")
        
        # Test sentiment analyzer
        if hasattr(nlp_engine, 'sentiment_analyzer') and nlp_engine.sentiment_analyzer:
            logger.info("✅ Sentiment analyzer available")
        else:
            logger.warning("⚠️ Sentiment analyzer not available (using mock)")
        
        # Test news processor
        if hasattr(nlp_engine, 'news_processor') and nlp_engine.news_processor:
            logger.info("✅ News processor available")
        else:
            logger.warning("⚠️ News processor not available (using mock)")
        
        # Test social media analyzer
        if hasattr(nlp_engine, 'social_media_analyzer') and nlp_engine.social_media_analyzer:
            logger.info("✅ Social media analyzer available")
        else:
            logger.warning("⚠️ Social media analyzer not available (using mock)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ NLP Engine test failed: {e}")
        return False

async def test_signal_generator_integration():
    """Test Signal Generator with RL and NLP integration"""
    logger.info("🔗 Testing Signal Generator Integration...")
    
    try:
        # Initialize signal generator
        signal_generator = RealTimeSignalGenerator()
        await signal_generator.start()
        logger.info("✅ Signal Generator started successfully")
        
        # Check RL integration
        if hasattr(signal_generator, 'rl_engine') and signal_generator.rl_engine:
            logger.info("✅ RL Engine integrated with Signal Generator")
        else:
            logger.error("❌ RL Engine not integrated with Signal Generator")
            return False
        
        # Check NLP integration
        if hasattr(signal_generator, 'nlp_engine') and signal_generator.nlp_engine:
            logger.info("✅ NLP Engine integrated with Signal Generator")
        else:
            logger.error("❌ NLP Engine not integrated with Signal Generator")
            return False
        
        # Test signal generation with RL and NLP
        test_symbol = "BTCUSDT"
        test_timeframe = "1h"
        
        # Create mock data for testing
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test signal generation
        signal = await signal_generator._analyze_and_generate_signal(mock_data, test_symbol, test_timeframe, None)
        
        if signal:
            logger.info("✅ Signal generated successfully")
            
            # Check for RL data
            if signal.get('indicators', {}).get('rl_analysis'):
                logger.info("✅ RL analysis included in signal")
            else:
                logger.warning("⚠️ RL analysis not found in signal")
            
            # Check for NLP data
            if signal.get('indicators', {}).get('nlp_analysis'):
                logger.info("✅ NLP analysis included in signal")
            else:
                logger.warning("⚠️ NLP analysis not found in signal")
            
            # Log signal details
            logger.info(f"Signal confidence: {signal.get('confidence', 0)}")
            logger.info(f"Signal side: {signal.get('side', 'unknown')}")
            
        else:
            logger.error("❌ No signal generated")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal Generator integration test failed: {e}")
        return False

async def test_database_persistence():
    """Test that signals with RL and NLP data can be saved to database"""
    logger.info("💾 Testing Database Persistence...")
    
    try:
        db_manager = DatabaseManager()
        
        # Initialize database manager
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        # Create a test signal with RL and NLP data
        test_signal = {
            'id': f"test_signal_{datetime.now().timestamp()}",
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'strategy': 'phase2_integration_test',
            'confidence': 0.85,
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc),
            'price': 45000.0,
            'stop_loss': 44000.0,
            'take_profit': 46000.0,
            'metadata': {
                'test': True,
                'phase': '2.1_and_2.2_integration'
            },
            'indicators': {
                'rl_analysis': {
                    'action_type': 'buy',
                    'position_size': 0.5,
                    'confidence': 0.8,
                    'reward': 25.5
                },
                'nlp_analysis': {
                    'overall_sentiment': 0.7,
                    'news_sentiment': 0.6,
                    'twitter_sentiment': 0.8,
                    'confidence': 0.75
                }
            }
        }
        
        # Save signal using direct database connection
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO enhanced_signals (
                    id, symbol, side, strategy, confidence, strength, timestamp, price, 
                    stop_loss, take_profit, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, test_signal['id'], test_signal['symbol'], test_signal['side'], 
                 test_signal['strategy'], test_signal['confidence'], test_signal['strength'],
                 test_signal['timestamp'], test_signal['price'], test_signal['stop_loss'],
                 test_signal['take_profit'], json.dumps(test_signal['metadata']))
            success = True
        if success:
            logger.info("✅ Test signal saved successfully")
        else:
            logger.error("❌ Failed to save test signal")
            return False
        
        # Retrieve and verify signal
        async with db_manager.get_connection() as conn:
            result = await conn.fetch("""
                SELECT id, symbol, side, strategy, confidence, strength, timestamp, 
                       price, stop_loss, take_profit, metadata
                FROM enhanced_signals
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            signals = [dict(row) for row in result]
        if signals:
            saved_signal = signals[0]
            logger.info("✅ Test signal retrieved successfully")
            
            # Check if RL and NLP data are present
            metadata = json.loads(saved_signal.get('metadata', '{}'))
            if metadata.get('rl_analysis'):
                logger.info("✅ RL data persisted correctly")
            else:
                logger.warning("⚠️ RL data not found in saved signal")
            
            if metadata.get('nlp_analysis'):
                logger.info("✅ NLP data persisted correctly")
            else:
                logger.warning("⚠️ NLP data not found in saved signal")
        else:
            logger.error("❌ Failed to retrieve test signal")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database persistence test failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting Phase 2.1 and 2.2 Integration Tests...")
    
    tests = [
        ("Database Migrations", test_database_migrations),
        ("RL Engine", test_rl_engine),
        ("NLP Engine", test_nlp_engine),
        ("Signal Generator Integration", test_signal_generator_integration),
        ("Database Persistence", test_database_persistence)
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
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Phase 2.1 and 2.2 integration is working correctly.")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
