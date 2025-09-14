#!/usr/bin/env python3
"""
Migration script to add deep learning support to AlphaPlus
Phase 10A: Deep Learning Foundation - Simple Version
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def create_deep_learning_tables():
    """Create deep learning tables and extend existing tables"""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("🚀 Connected to database successfully")
        
        logger.info("🚀 Starting Phase 10A: Deep Learning Foundation migration...")
        
        # 1. Extend model_predictions table with deep learning support
        logger.info("📊 Extending model_predictions table...")
        await conn.execute("""
            ALTER TABLE model_predictions 
            ADD COLUMN IF NOT EXISTS deep_learning_model_version VARCHAR(50),
            ADD COLUMN IF NOT EXISTS model_architecture JSONB,
            ADD COLUMN IF NOT EXISTS training_parameters JSONB,
            ADD COLUMN IF NOT EXISTS inference_latency_ms INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS gpu_used BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS batch_size INTEGER DEFAULT 1
        """)
        
        # 2. Extend feature_importance table with deep learning features
        logger.info("🔍 Extending feature_importance table...")
        await conn.execute("""
            ALTER TABLE feature_importance 
            ADD COLUMN IF NOT EXISTS deep_learning_importance FLOAT,
            ADD COLUMN IF NOT EXISTS attention_weights JSONB,
            ADD COLUMN IF NOT EXISTS layer_importance JSONB
        """)
        
        # 3. Extend volume_analysis table with sentiment scores
        logger.info("📈 Extending volume_analysis table...")
        await conn.execute("""
            ALTER TABLE volume_analysis 
            ADD COLUMN IF NOT EXISTS sentiment_score FLOAT,
            ADD COLUMN IF NOT EXISTS news_sentiment FLOAT,
            ADD COLUMN IF NOT EXISTS social_sentiment FLOAT,
            ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT
        """)
        
        # 4. Create deep_learning_predictions table for detailed DL results
        logger.info("🧠 Creating deep_learning_predictions table...")
        
        # Drop table if exists
        await conn.execute("DROP TABLE IF EXISTS deep_learning_predictions CASCADE")
        
        await conn.execute("""
            CREATE TABLE deep_learning_predictions (
                id SERIAL,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                prediction_type VARCHAR(50) NOT NULL,
                prediction_value FLOAT NOT NULL,
                confidence_score FLOAT NOT NULL,
                input_sequence_length INTEGER NOT NULL,
                output_horizon INTEGER NOT NULL,
                model_architecture JSONB,
                training_parameters JSONB,
                inference_metadata JSONB,
                attention_weights JSONB,
                feature_contributions JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, id)
            )
        """)
        
        # 5. Create hypertable for deep_learning_predictions
        await conn.execute("""
            SELECT create_hypertable('deep_learning_predictions', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour'
            )
        """)
        
        # 6. Create sentiment_analysis table for news and social media
        logger.info("📰 Creating sentiment_analysis table...")
        
        # Drop table if exists
        await conn.execute("DROP TABLE IF EXISTS sentiment_analysis CASCADE")
        
        await conn.execute("""
            CREATE TABLE sentiment_analysis (
                id SERIAL,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                source VARCHAR(50) NOT NULL,
                sentiment_score FLOAT NOT NULL,
                sentiment_label VARCHAR(20) NOT NULL,
                confidence_score FLOAT NOT NULL,
                text_content TEXT,
                source_url VARCHAR(500),
                author VARCHAR(100),
                engagement_metrics JSONB,
                keywords JSONB,
                language VARCHAR(10) DEFAULT 'en',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, id)
            )
        """)
        
        # 7. Create hypertable for sentiment_analysis
        await conn.execute("""
            SELECT create_hypertable('sentiment_analysis', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour'
            )
        """)
        
        # 8. Create multi_agent_states table for reinforcement learning
        logger.info("🤖 Creating multi_agent_states table...")
        
        # Drop table if exists
        await conn.execute("DROP TABLE IF EXISTS multi_agent_states CASCADE")
        
        await conn.execute("""
            CREATE TABLE multi_agent_states (
                id SERIAL,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                agent_id VARCHAR(50) NOT NULL,
                agent_type VARCHAR(50) NOT NULL,
                state_vector JSONB NOT NULL,
                action_taken VARCHAR(50),
                reward_received FLOAT,
                next_state JSONB,
                episode_id VARCHAR(50),
                training_step INTEGER,
                epsilon FLOAT,
                learning_rate FLOAT,
                model_version VARCHAR(50),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, id)
            )
        """)
        
        # 9. Create hypertable for multi_agent_states
        await conn.execute("""
            SELECT create_hypertable('multi_agent_states', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour'
            )
        """)
        
        # 10. Create market_regime_forecasts table for predictive analytics
        logger.info("🔮 Creating market_regime_forecasts table...")
        
        # Drop table if exists
        await conn.execute("DROP TABLE IF EXISTS market_regime_forecasts CASCADE")
        
        await conn.execute("""
            CREATE TABLE market_regime_forecasts (
                id SERIAL,
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                forecast_horizon INTEGER NOT NULL,
                regime_type VARCHAR(50) NOT NULL,
                confidence_score FLOAT NOT NULL,
                probability_distribution JSONB,
                volatility_forecast FLOAT,
                trend_strength_forecast FLOAT,
                liquidity_forecast FLOAT,
                model_version VARCHAR(50),
                feature_importance JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, id)
            )
        """)
        
        # 11. Create hypertable for market_regime_forecasts
        await conn.execute("""
            SELECT create_hypertable('market_regime_forecasts', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour'
            )
        """)
        
        # 12. Create indexes for optimal performance
        logger.info("🔍 Creating performance indexes...")
        
        # Deep learning predictions indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dl_predictions_symbol_time 
            ON deep_learning_predictions (symbol, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dl_predictions_model_type 
            ON deep_learning_predictions (model_type, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dl_predictions_prediction_type 
            ON deep_learning_predictions (prediction_type, confidence_score DESC)
        """)
        
        # Sentiment analysis indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time 
            ON sentiment_analysis (symbol, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_source 
            ON sentiment_analysis (source, sentiment_score DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_label 
            ON sentiment_analysis (sentiment_label, timestamp DESC)
        """)
        
        # Multi-agent states indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_symbol_time 
            ON multi_agent_states (symbol, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_type 
            ON multi_agent_states (agent_type, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_episode 
            ON multi_agent_states (episode_id, training_step)
        """)
        
        # Market regime forecasts indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_forecast_symbol_time 
            ON market_regime_forecasts (symbol, timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_forecast_type 
            ON market_regime_forecasts (regime_type, confidence_score DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_forecast_horizon 
            ON market_regime_forecasts (forecast_horizon, timestamp DESC)
        """)
        
        logger.info("✅ Phase 10A: Deep Learning Foundation migration completed successfully!")
        
        # Close connection
        await conn.close()
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(create_deep_learning_tables())
