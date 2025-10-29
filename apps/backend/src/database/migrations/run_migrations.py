#!/usr/bin/env python3
"""
Standalone migration script for Phase 2.1 (RL) and Phase 2.2 (NLP) database changes
"""
import asyncio
import logging
from src.app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migrations():
    """Run Phase 2.1 and 2.2 migrations"""
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize database connection
        logger.info("Initializing database connection...")
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 5,
            'max_size': 20,
            'command_timeout': 60
        }
        await db_manager.initialize(config)
        
        # Test connection
        logger.info("Testing database connection...")
        async with db_manager.get_connection() as conn:
            await conn.execute("SELECT 1")
            logger.info("✅ Database connection successful")
        
        # Create enhanced_signals table if it doesn't exist
        logger.info("Creating enhanced_signals table...")
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    confidence DECIMAL(5, 4) NOT NULL,
                    strength DECIMAL(5, 4) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    stop_loss DECIMAL(20, 8),
                    take_profit DECIMAL(20, 8),
                    metadata JSONB,
                    ichimoku_data JSONB,
                    fibonacci_data JSONB,
                    volume_analysis JSONB,
                    advanced_indicators JSONB,
                    smc_analysis JSONB,
                    order_blocks_data JSONB,
                    fair_value_gaps_data JSONB,
                    liquidity_sweeps_data JSONB,
                    market_structures_data JSONB,
                    smc_confidence DECIMAL(5, 4),
                    smc_bias VARCHAR(20),
                    dl_analysis JSONB,
                    lstm_prediction DECIMAL(20, 8),
                    cnn_prediction DECIMAL(20, 8),
                    lstm_cnn_prediction DECIMAL(20, 8),
                    ensemble_prediction DECIMAL(20, 8),
                    dl_confidence DECIMAL(5, 4),
                    dl_bias VARCHAR(20),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Enhanced signals table created")
        
        # Phase 2.1: Reinforcement Learning columns
        logger.info("Applying Phase 2.1: Reinforcement Learning migrations...")
        async with db_manager.get_connection() as conn:
            # Add RL columns to enhanced_signals table
            await conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS rl_analysis JSONB,
                ADD COLUMN IF NOT EXISTS rl_action_type VARCHAR(20),
                ADD COLUMN IF NOT EXISTS rl_position_size FLOAT,
                ADD COLUMN IF NOT EXISTS rl_stop_loss FLOAT,
                ADD COLUMN IF NOT EXISTS rl_take_profit FLOAT,
                ADD COLUMN IF NOT EXISTS rl_confidence_threshold FLOAT,
                ADD COLUMN IF NOT EXISTS rl_risk_allocation FLOAT,
                ADD COLUMN IF NOT EXISTS rl_optimization_params JSONB,
                ADD COLUMN IF NOT EXISTS rl_bias VARCHAR(20),
                ADD COLUMN IF NOT EXISTS rl_action_strength FLOAT,
                ADD COLUMN IF NOT EXISTS rl_training_episodes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS rl_avg_reward FLOAT,
                ADD COLUMN IF NOT EXISTS rl_best_reward FLOAT
            """)
            
            # Create RL indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_bias 
                ON enhanced_signals (rl_bias)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_action_strength 
                ON enhanced_signals (rl_action_strength)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_training_episodes 
                ON enhanced_signals (rl_training_episodes)
            """)
            
            # Create RL view
            await conn.execute("""
                CREATE OR REPLACE VIEW rl_enhanced_signals AS
                SELECT * FROM enhanced_signals
                WHERE rl_analysis IS NOT NULL
                  AND rl_action_strength >= 0.5
                  AND rl_training_episodes >= 10
                  AND confidence >= 0.6
                ORDER BY rl_action_strength DESC, rl_avg_reward DESC
            """)
            
            # Create RL quality function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_rl_enhanced_quality(
                    p_confidence FLOAT,
                    p_rl_action_strength FLOAT,
                    p_rl_avg_reward FLOAT,
                    p_rl_training_episodes INTEGER
                ) RETURNS FLOAT AS $$
                BEGIN
                    RETURN (
                        p_confidence * 0.4 +
                        p_rl_action_strength * 0.3 +
                        LEAST(p_rl_avg_reward / 100.0, 1.0) * 0.2 +
                        LEAST(p_rl_training_episodes / 1000.0, 1.0) * 0.1
                    );
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Phase 2.1 (RL) migrations completed")
        
        # Phase 2.2: Natural Language Processing columns
        logger.info("Applying Phase 2.2: Natural Language Processing migrations...")
        async with db_manager.get_connection() as conn:
            # Add NLP columns to enhanced_signals table
            await conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS nlp_analysis JSONB,
                ADD COLUMN IF NOT EXISTS nlp_overall_sentiment_score FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_overall_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_news_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_news_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_twitter_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_twitter_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_reddit_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_reddit_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_bias VARCHAR(20),
                ADD COLUMN IF NOT EXISTS nlp_sentiment_strength FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_high_confidence_sentiment BOOLEAN,
                ADD COLUMN IF NOT EXISTS nlp_analyses_performed INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS nlp_cache_hit_rate FLOAT,
                ADD COLUMN IF NOT EXISTS nlp_models_available JSONB
            """)
            
            # Create NLP indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_bias 
                ON enhanced_signals (nlp_bias)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_sentiment_strength 
                ON enhanced_signals (nlp_sentiment_strength)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_overall_confidence 
                ON enhanced_signals (nlp_overall_confidence)
            """)
            
            # Create NLP view
            await conn.execute("""
                CREATE OR REPLACE VIEW nlp_enhanced_signals AS
                SELECT * FROM enhanced_signals
                WHERE nlp_analysis IS NOT NULL
                  AND nlp_overall_confidence >= 0.5
                  AND nlp_high_confidence_sentiment = TRUE
                  AND confidence >= 0.6
                ORDER BY nlp_overall_confidence DESC, nlp_sentiment_strength DESC
            """)
            
            # Create NLP quality function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_nlp_enhanced_quality(
                    p_confidence FLOAT,
                    p_nlp_overall_confidence FLOAT,
                    p_nlp_sentiment_strength FLOAT,
                    p_nlp_cache_hit_rate FLOAT
                ) RETURNS FLOAT AS $$
                BEGIN
                    RETURN (
                        p_confidence * 0.4 +
                        p_nlp_overall_confidence * 0.3 +
                        p_nlp_sentiment_strength * 0.2 +
                        p_nlp_cache_hit_rate * 0.1
                    );
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Phase 2.2 (NLP) migrations completed")
        
        # Verify migrations
        logger.info("Verifying migrations...")
        async with db_manager.get_connection() as conn:
            # Check RL columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'rl_%'
                ORDER BY column_name
            """)
            rl_columns = [row['column_name'] for row in result]
            logger.info(f"RL columns found: {len(rl_columns)} - {rl_columns}")
            
            # Check NLP columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'nlp_%'
                ORDER BY column_name
            """)
            nlp_columns = [row['column_name'] for row in result]
            logger.info(f"NLP columns found: {len(nlp_columns)} - {nlp_columns}")
            
            # Check views
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('rl_enhanced_signals', 'nlp_enhanced_signals')
            """)
            views = [row['viewname'] for row in result]
            logger.info(f"Views created: {views}")
        
        logger.info("🎉 All migrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migrations())
