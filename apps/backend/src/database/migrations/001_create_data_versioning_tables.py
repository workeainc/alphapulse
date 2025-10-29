#!/usr/bin/env python3
"""
Migration script to create data versioning tables
Phase 1: Database Schema Implementation
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string (update this with your actual connection)
DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def create_data_versioning_tables():
    """Create the required data versioning tables with TimescaleDB optimizations"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("Starting data versioning tables creation...")
            
            # 1. Create signals table with composite primary key for TimescaleDB
            logger.info("Creating signals table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL,
                    label VARCHAR(10),
                    pred VARCHAR(10),
                    proba FLOAT,
                    ts TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20),
                    tf VARCHAR(10),
                    features JSONB,
                    model_id VARCHAR(50),
                    outcome VARCHAR(20),
                    realized_rr FLOAT,
                    latency_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (ts, id)
                )
            """))
            
            # 1.1 Create enhanced signals table for real-time signal generator
            logger.info("Creating enhanced signals table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    strength FLOAT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    price FLOAT NOT NULL,
                    stop_loss FLOAT,
                    take_profit FLOAT,
                    metadata JSONB,
                    ichimoku_data JSONB,
                    fibonacci_data JSONB,
                    volume_analysis JSONB,
                    advanced_indicators JSONB,
                    signal_quality_score FLOAT,
                    confirmation_count INTEGER DEFAULT 0,
                    smc_analysis JSONB,
                    order_blocks_data JSONB[],
                    fair_value_gaps_data JSONB[],
                    liquidity_sweeps_data JSONB[],
                    market_structures_data JSONB[],
                    smc_confidence FLOAT,
                    smc_bias VARCHAR(20),
                    dl_analysis JSONB,
                    lstm_prediction FLOAT,
                    cnn_prediction FLOAT,
                    lstm_cnn_prediction FLOAT,
                    ensemble_prediction FLOAT,
                    dl_confidence FLOAT,
                    dl_bias VARCHAR(20),
                    rl_analysis JSONB,
                    rl_action_type VARCHAR(20),
                    rl_position_size FLOAT,
                    rl_stop_loss FLOAT,
                    rl_take_profit FLOAT,
                    rl_confidence_threshold FLOAT,
                    rl_risk_allocation FLOAT,
                    rl_optimization_params JSONB,
                    rl_bias VARCHAR(20),
                    rl_action_strength FLOAT,
                    rl_training_episodes INTEGER DEFAULT 0,
                    rl_avg_reward FLOAT,
                    rl_best_reward FLOAT,
                    nlp_analysis JSONB,
                    nlp_overall_sentiment_score FLOAT,
                    nlp_overall_confidence FLOAT,
                    nlp_news_sentiment FLOAT,
                    nlp_news_confidence FLOAT,
                    nlp_twitter_sentiment FLOAT,
                    nlp_twitter_confidence FLOAT,
                    nlp_reddit_sentiment FLOAT,
                    nlp_reddit_confidence FLOAT,
                    nlp_bias VARCHAR(20),
                    nlp_sentiment_strength FLOAT,
                    nlp_high_confidence_sentiment BOOLEAN,
                    nlp_analyses_performed INTEGER DEFAULT 0,
                    nlp_cache_hit_rate FLOAT,
                    nlp_models_available JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # Create index on enhanced_signals for performance
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_symbol_timestamp 
                ON enhanced_signals (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_confidence 
                ON enhanced_signals (confidence DESC)
            """))
            
            # Create indexes for RL data
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_bias 
                ON enhanced_signals (rl_bias)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_action_strength 
                ON enhanced_signals (rl_action_strength DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_rl_training_episodes 
                ON enhanced_signals (rl_training_episodes DESC)
            """))
            
            # Create indexes for NLP data
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_bias 
                ON enhanced_signals (nlp_bias)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_sentiment_strength 
                ON enhanced_signals (nlp_sentiment_strength DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_nlp_overall_confidence 
                ON enhanced_signals (nlp_overall_confidence DESC)
            """))
            
            # Create view for RL-enhanced signals
            await conn.execute(text("""
                CREATE OR REPLACE VIEW rl_enhanced_signals AS
                SELECT 
                    id, symbol, side, strategy, confidence, strength, timestamp, price,
                    stop_loss, take_profit, rl_action_type, rl_position_size, rl_bias,
                    rl_action_strength, rl_training_episodes, rl_avg_reward, rl_best_reward
                FROM enhanced_signals 
                WHERE rl_analysis IS NOT NULL 
                AND rl_action_strength > 0.5
                ORDER BY timestamp DESC
            """))
            
            # Create view for NLP-enhanced signals
            await conn.execute(text("""
                CREATE OR REPLACE VIEW nlp_enhanced_signals AS
                SELECT 
                    id, symbol, side, strategy, confidence, strength, timestamp, price,
                    stop_loss, take_profit, nlp_overall_sentiment_score, nlp_overall_confidence,
                    nlp_bias, nlp_sentiment_strength, nlp_high_confidence_sentiment,
                    nlp_analyses_performed, nlp_cache_hit_rate
                FROM enhanced_signals 
                WHERE nlp_analysis IS NOT NULL 
                AND nlp_sentiment_strength > 0.5
                ORDER BY timestamp DESC
            """))
            
            # Create function to calculate RL-enhanced quality
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION calculate_rl_enhanced_quality(
                    base_confidence FLOAT,
                    rl_action_strength FLOAT,
                    rl_avg_reward FLOAT,
                    rl_training_episodes INTEGER
                ) RETURNS FLOAT AS $$
                BEGIN
                    -- Base quality from confidence
                    DECLARE quality FLOAT := base_confidence;
                    
                    -- Boost from RL action strength
                    IF rl_action_strength > 0.7 THEN
                        quality := quality + 0.1;
                    ELSIF rl_action_strength > 0.5 THEN
                        quality := quality + 0.05;
                    END IF;
                    
                    -- Boost from RL performance
                    IF rl_avg_reward > 0.1 THEN
                        quality := quality + 0.05;
                    END IF;
                    
                    -- Boost from training episodes (more training = better model)
                    IF rl_training_episodes > 100 THEN
                        quality := quality + 0.05;
                    ELSIF rl_training_episodes > 50 THEN
                        quality := quality + 0.02;
                    END IF;
                    
                    -- Cap at 1.0
                    RETURN LEAST(quality, 1.0);
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # Create function to calculate NLP-enhanced quality
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION calculate_nlp_enhanced_quality(
                    base_confidence FLOAT,
                    nlp_sentiment_strength FLOAT,
                    nlp_overall_confidence FLOAT,
                    nlp_high_confidence_sentiment BOOLEAN
                ) RETURNS FLOAT AS $$
                BEGIN
                    -- Base quality from confidence
                    DECLARE quality FLOAT := base_confidence;
                    
                    -- Boost from NLP sentiment strength
                    IF nlp_sentiment_strength > 0.7 THEN
                        quality := quality + 0.1;
                    ELSIF nlp_sentiment_strength > 0.5 THEN
                        quality := quality + 0.05;
                    END IF;
                    
                    -- Boost from NLP confidence
                    IF nlp_overall_confidence > 0.8 THEN
                        quality := quality + 0.05;
                    ELSIF nlp_overall_confidence > 0.6 THEN
                        quality := quality + 0.02;
                    END IF;
                    
                    -- Boost from high confidence sentiment
                    IF nlp_high_confidence_sentiment THEN
                        quality := quality + 0.05;
                    END IF;
                    
                    -- Cap at 1.0
                    RETURN LEAST(quality, 1.0);
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # 2. Create candles table with composite primary key for TimescaleDB
            logger.info("Creating candles table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS candles (
                    id SERIAL,
                    symbol VARCHAR(20),
                    tf VARCHAR(10),
                    ts TIMESTAMPTZ NOT NULL,
                    o FLOAT,
                    h FLOAT,
                    l FLOAT,
                    c FLOAT,
                    v FLOAT,
                    vwap FLOAT,
                    taker_buy_vol FLOAT,
                    features JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (ts, id)
                )
            """))
            
            # 3. Create retrain_queue table (regular table, not hypertable)
            logger.info("Creating retrain_queue table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS retrain_queue (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER,
                    reason TEXT,
                    inserted_at TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(20) DEFAULT 'pending',
                    priority INTEGER DEFAULT 1,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 4. Create TimescaleDB hypertables
            logger.info("Creating TimescaleDB hypertables...")
            
            # Convert signals table to hypertable
            await conn.execute(text("""
                SELECT create_hypertable('signals', 'ts', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE)
            """))
            
            # Convert candles table to hypertable
            await conn.execute(text("""
                SELECT create_hypertable('candles', 'ts',
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE)
            """))
            
            # 5. Enable columnstore compression for hypertables
            logger.info("Enabling columnstore compression...")
            
            # Enable columnstore for signals table
            await conn.execute(text("""
                ALTER TABLE signals SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, tf')
            """))
            
            # Enable columnstore for candles table
            await conn.execute(text("""
                ALTER TABLE candles SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, tf')
            """))
            
            # 6. Create indexes for optimal performance
            logger.info("Creating performance indexes...")
            
            # Signals table indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts 
                ON signals (symbol, ts DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_signals_model_id_ts 
                ON signals (model_id, ts DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_signals_features 
                ON signals USING GIN (features)
            """))
            
            # Candles table indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts 
                ON candles (symbol, tf, ts DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_candles_features 
                ON candles USING GIN (features)
            """))
            
            # Retrain queue indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_retrain_queue_status_priority 
                ON retrain_queue (status, priority DESC, inserted_at)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_retrain_queue_signal_id 
                ON retrain_queue (signal_id)
            """))
            
            # 7. Set up compression policies (with if_not_exists to avoid duplicates)
            logger.info("Setting up compression policies...")
            
            # Compress signals older than 7 days
            try:
                await conn.execute(text("""
                    SELECT add_compression_policy('signals', INTERVAL '7 days', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Compression policy for signals already exists")
            
            # Compress candles older than 3 days
            try:
                await conn.execute(text("""
                    SELECT add_compression_policy('candles', INTERVAL '3 days', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Compression policy for candles already exists")
            
            # 8. Set up retention policies (with if_not_exists to avoid duplicates)
            logger.info("Setting up retention policies...")
            
            # Keep signals for 1 year
            try:
                await conn.execute(text("""
                    SELECT add_retention_policy('signals', INTERVAL '1 year', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Retention policy for signals already exists")
            
            # Keep candles for 6 months
            try:
                await conn.execute(text("""
                    SELECT add_retention_policy('candles', INTERVAL '6 months', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Retention policy for candles already exists")
            
            # Note: retrain_queue is not a hypertable, so no retention policy
            # Manual cleanup can be implemented if needed
            logger.info("Retrain queue retention: manual cleanup (not a hypertable)")
            
            logger.info("Data versioning tables created successfully!")
            
            # 9. Verify table creation
            logger.info("Verifying table creation...")
            result = await conn.execute(text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_name IN ('signals', 'candles', 'retrain_queue')
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            for table in tables:
                logger.info(f"   - {table[0]} ({table[1]})")
            
            # 10. Verify hypertables
            result = await conn.execute(text("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_name IN ('signals', 'candles')
                ORDER BY hypertable_name
            """))
            
            hypertables = result.fetchall()
            for ht in hypertables:
                logger.info(f"   - Hypertable: {ht[0]}")
            
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main migration function"""
    try:
        await create_data_versioning_tables()
        logger.info("Data versioning tables migration completed!")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
