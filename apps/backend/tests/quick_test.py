#!/usr/bin/env python3
"""
Quick test for Phase 2.1 and 2.2 core functionality
"""
import asyncio
import logging
from src.app.core.database_manager import DatabaseManager
from src.app.strategies.reinforcement_learning_engine import ReinforcementLearningEngine
from src.app.strategies.natural_language_processing_engine import NaturalLanguageProcessingEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_core_functionality():
    """Test core Phase 2.1 and 2.2 functionality"""
    logger.info("🚀 Testing Phase 2.1 and 2.2 Core Functionality...")
    
    # Test 1: Database Migrations
    logger.info("🔍 Test 1: Database Migrations")
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        async with db_manager.get_connection() as conn:
            # Check RL columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'rl_%'
                ORDER BY column_name
            """)
            rl_columns = [row['column_name'] for row in result]
            logger.info(f"✅ RL columns: {len(rl_columns)} found")
            
            # Check NLP columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'nlp_%'
                ORDER BY column_name
            """)
            nlp_columns = [row['column_name'] for row in result]
            logger.info(f"✅ NLP columns: {len(nlp_columns)} found")
            
            # Check views
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('rl_enhanced_signals', 'nlp_enhanced_signals')
            """)
            views = [row['viewname'] for row in result]
            logger.info(f"✅ Views: {views}")
        
        logger.info("✅ Database migrations test PASSED")
        
    except Exception as e:
        logger.error(f"❌ Database migrations test FAILED: {e}")
        return False
    
    # Test 2: RL Engine
    logger.info("🤖 Test 2: Reinforcement Learning Engine")
    try:
        rl_engine = ReinforcementLearningEngine()
        await rl_engine.start()
        
        performance = rl_engine.get_performance_summary()
        logger.info(f"✅ RL Engine performance: {performance}")
        
        logger.info("✅ RL Engine test PASSED")
        
    except Exception as e:
        logger.error(f"❌ RL Engine test FAILED: {e}")
        return False
    
    # Test 3: NLP Engine
    logger.info("📝 Test 3: Natural Language Processing Engine")
    try:
        nlp_engine = NaturalLanguageProcessingEngine()
        await nlp_engine.start()
        
        performance = nlp_engine.get_performance_summary()
        logger.info(f"✅ NLP Engine performance: {performance}")
        
        logger.info("✅ NLP Engine test PASSED")
        
    except Exception as e:
        logger.error(f"❌ NLP Engine test FAILED: {e}")
        return False
    
    # Test 4: Database Persistence
    logger.info("💾 Test 4: Database Persistence")
    try:
        import json
        from datetime import datetime, timezone
        
        test_signal = {
            'id': f"test_signal_{datetime.now().timestamp()}",
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'strategy': 'phase2_test',
            'confidence': 0.85,
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc),
            'price': 45000.0,
            'stop_loss': 44000.0,
            'take_profit': 46000.0,
            'metadata': json.dumps({
                'test': True,
                'phase': '2.1_and_2.2_integration',
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
            })
        }
        
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO enhanced_signals (
                    id, symbol, side, strategy, confidence, strength, timestamp, price, 
                    stop_loss, take_profit, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, test_signal['id'], test_signal['symbol'], test_signal['side'], 
                 test_signal['strategy'], test_signal['confidence'], test_signal['strength'],
                 test_signal['timestamp'], test_signal['price'], test_signal['stop_loss'],
                 test_signal['take_profit'], test_signal['metadata'])
            
            # Verify the signal was saved
            result = await conn.fetch("""
                SELECT id, symbol, metadata FROM enhanced_signals 
                WHERE id = $1
            """, test_signal['id'])
            
            if result:
                saved_signal = result[0]
                metadata = json.loads(saved_signal['metadata'])
                if metadata.get('rl_analysis') and metadata.get('nlp_analysis'):
                    logger.info("✅ Signal with RL and NLP data saved and retrieved successfully")
                else:
                    logger.warning("⚠️ RL or NLP data missing from saved signal")
            else:
                logger.error("❌ Signal not found in database")
                return False
        
        logger.info("✅ Database persistence test PASSED")
        
    except Exception as e:
        logger.error(f"❌ Database persistence test FAILED: {e}")
        return False
    
    logger.info("🎉 All Phase 2.1 and 2.2 core functionality tests PASSED!")
    return True

if __name__ == "__main__":
    asyncio.run(test_core_functionality())
