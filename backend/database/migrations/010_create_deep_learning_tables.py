#!/usr/bin/env python3
"""
Migration script to add deep learning support to AlphaPlus
Phase 10A: Deep Learning Foundation
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def create_deep_learning_tables():
    """Create deep learning tables and extend existing tables"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("🚀 Starting Phase 10A: Deep Learning Foundation migration...")
            
            # 1. Extend model_predictions table with deep learning support
            logger.info("📊 Extending model_predictions table...")
            await conn.execute(text("""
                ALTER TABLE model_predictions 
                ADD COLUMN IF NOT EXISTS deep_learning_model_version VARCHAR(50),
                ADD COLUMN IF NOT EXISTS model_architecture JSONB,
                ADD COLUMN IF NOT EXISTS training_parameters JSONB,
                ADD COLUMN IF NOT EXISTS inference_latency_ms INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS gpu_used BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS batch_size INTEGER DEFAULT 1
            """))
            
            # 2. Extend feature_importance table with deep learning features
            logger.info("🔍 Extending feature_importance table...")
            await conn.execute(text("""
                ALTER TABLE feature_importance 
                ADD COLUMN IF NOT EXISTS deep_learning_importance FLOAT,
                ADD COLUMN IF NOT EXISTS attention_weights JSONB,
                ADD COLUMN IF NOT EXISTS layer_importance JSONB
            """))
            
            # 3. Extend volume_analysis table with sentiment scores
            logger.info("📈 Extending volume_analysis table...")
            await conn.execute(text("""
                ALTER TABLE volume_analysis 
                ADD COLUMN IF NOT EXISTS sentiment_score FLOAT,
                ADD COLUMN IF NOT EXISTS news_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS social_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT
            """))
            
            # 4. Create deep_learning_predictions table for detailed DL results
            logger.info("🧠 Creating deep_learning_predictions table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS deep_learning_predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'transformer', 'cnn'
                    model_version VARCHAR(50) NOT NULL,
                    prediction_type VARCHAR(50) NOT NULL, -- 'price', 'direction', 'volatility'
                    prediction_value FLOAT NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    input_sequence_length INTEGER NOT NULL,
                    output_horizon INTEGER NOT NULL,
                    model_architecture JSONB,
                    training_parameters JSONB,
                    inference_metadata JSONB, -- GPU usage, batch size, latency
                    attention_weights JSONB, -- For transformer models
                    feature_contributions JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 5. Create hypertable for deep_learning_predictions
            await conn.execute(text("""
                SELECT create_hypertable('deep_learning_predictions', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """))
            
            # 6. Create sentiment_analysis table for news and social media
            logger.info("📰 Creating sentiment_analysis table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    source VARCHAR(50) NOT NULL, -- 'twitter', 'reddit', 'news', 'combined'
                    sentiment_score FLOAT NOT NULL,
                    sentiment_label VARCHAR(20) NOT NULL, -- 'positive', 'negative', 'neutral'
                    confidence_score FLOAT NOT NULL,
                    text_content TEXT,
                    source_url VARCHAR(500),
                    author VARCHAR(100),
                    engagement_metrics JSONB, -- likes, retweets, comments
                    keywords JSONB,
                    language VARCHAR(10) DEFAULT 'en',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 7. Create hypertable for sentiment_analysis
            await conn.execute(text("""
                SELECT create_hypertable('sentiment_analysis', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """))
            
            # 8. Create multi_agent_states table for reinforcement learning
            logger.info("🤖 Creating multi_agent_states table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS multi_agent_states (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    agent_id VARCHAR(50) NOT NULL,
                    agent_type VARCHAR(50) NOT NULL, -- 'market_maker', 'trend_follower', 'mean_reversion', 'risk_manager'
                    state_vector JSONB NOT NULL,
                    action_taken VARCHAR(50),
                    reward_received FLOAT,
                    next_state JSONB,
                    episode_id VARCHAR(50),
                    training_step INTEGER,
                    epsilon FLOAT, -- Exploration rate
                    learning_rate FLOAT,
                    model_version VARCHAR(50),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 9. Create hypertable for multi_agent_states
            await conn.execute(text("""
                SELECT create_hypertable('multi_agent_states', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """))
            
            # 10. Create market_regime_forecasts table for predictive analytics
            logger.info("🔮 Creating market_regime_forecasts table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_regime_forecasts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    forecast_horizon INTEGER NOT NULL, -- hours ahead
                    regime_type VARCHAR(50) NOT NULL, -- 'trending', 'ranging', 'volatile', 'low_volatility'
                    confidence_score FLOAT NOT NULL,
                    probability_distribution JSONB, -- Probabilities for each regime
                    volatility_forecast FLOAT,
                    trend_strength_forecast FLOAT,
                    liquidity_forecast FLOAT,
                    model_version VARCHAR(50),
                    feature_importance JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 11. Create hypertable for market_regime_forecasts
            await conn.execute(text("""
                SELECT create_hypertable('market_regime_forecasts', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                )
            """))
            
            # 12. Create indexes for optimal performance
            logger.info("🔍 Creating performance indexes...")
            
            # Deep learning predictions indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_dl_predictions_symbol_time 
                ON deep_learning_predictions (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_dl_predictions_model_type 
                ON deep_learning_predictions (model_type, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_dl_predictions_prediction_type 
                ON deep_learning_predictions (prediction_type, confidence_score DESC)
            """))
            
            # Sentiment analysis indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time 
                ON sentiment_analysis (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_source 
                ON sentiment_analysis (source, sentiment_score DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_label 
                ON sentiment_analysis (sentiment_label, timestamp DESC)
            """))
            
            # Multi-agent states indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_symbol_time 
                ON multi_agent_states (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_type 
                ON multi_agent_states (agent_type, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_agent_episode 
                ON multi_agent_states (episode_id, training_step)
            """))
            
            # Market regime forecasts indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_regime_forecast_symbol_time 
                ON market_regime_forecasts (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_regime_forecast_type 
                ON market_regime_forecasts (regime_type, confidence_score DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_regime_forecast_horizon 
                ON market_regime_forecasts (forecast_horizon, timestamp DESC)
            """))
            
            logger.info("✅ Phase 10A: Deep Learning Foundation migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_deep_learning_tables())
