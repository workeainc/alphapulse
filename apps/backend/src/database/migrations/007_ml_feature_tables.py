#!/usr/bin/env python3
"""
Migration: Create ML Feature Tables for Phase 4A
Create TimescaleDB tables for ML-ready features, labels, model metadata, and predictions
"""
import asyncio
import logging
import os
import asyncpg
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['PGPASSWORD'] = 'Emon_@17711'
DB_CONFIG = {
    'host': 'localhost', 'port': 5432, 'database': 'alphapulse', 'user': 'alpha_emon', 'password': 'Emon_@17711'
}

async def create_ml_feature_tables():
    """Create ML feature tables and metadata storage"""
    
    # ML Features OHLCV Table
    create_ml_features_ohlcv = """
    CREATE TABLE IF NOT EXISTS ml_features_ohlcv (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        open_price NUMERIC(20,8),
        high_price NUMERIC(20,8),
        low_price NUMERIC(20,8),
        close_price NUMERIC(20,8),
        volume NUMERIC(20,8),
        vwap NUMERIC(20,8),
        atr NUMERIC(20,8),
        rsi NUMERIC(6,4),
        macd NUMERIC(20,8),
        macd_signal NUMERIC(20,8),
        macd_histogram NUMERIC(20,8),
        bollinger_upper NUMERIC(20,8),
        bollinger_middle NUMERIC(20,8),
        bollinger_lower NUMERIC(20,8),
        stoch_k NUMERIC(6,4),
        stoch_d NUMERIC(6,4),
        williams_r NUMERIC(6,4),
        cci NUMERIC(10,4),
        adx NUMERIC(6,4),
        obv NUMERIC(20,8),
        mfi NUMERIC(6,4),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    

    
    # ML Features Sentiment Table
    create_ml_features_sentiment = """
    CREATE TABLE IF NOT EXISTS ml_features_sentiment (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        fear_greed_index INTEGER,
        social_sentiment_score NUMERIC(6,4),
        news_sentiment_score NUMERIC(6,4),
        weighted_coin_sentiment NUMERIC(6,4),
        whale_sentiment_proxy NUMERIC(6,4),
        sentiment_divergence_score NUMERIC(6,4),
        multi_timeframe_sentiment JSONB,
        sentiment_momentum NUMERIC(6,4),
        sentiment_volatility NUMERIC(6,4),
        bullish_sentiment_ratio NUMERIC(6,4),
        bearish_sentiment_ratio NUMERIC(6,4),
        neutral_sentiment_ratio NUMERIC(6,4),
        sentiment_trend_strength NUMERIC(6,4),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # ML Labels Table
    create_ml_labels = """
    CREATE TABLE IF NOT EXISTS ml_labels (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        label_type VARCHAR(50) NOT NULL, -- 'regime_change', 'sector_rotation', 'price_direction'
        label_value VARCHAR(50), -- 'bullish', 'bearish', 'sideways', 'btc_dominance', 'altcoin_rotation'
        label_confidence NUMERIC(6,4),
        future_timestamp TIMESTAMPTZ, -- when this label will be realized
        realized_value VARCHAR(50), -- actual outcome when known
        realized_confidence NUMERIC(6,4),
        is_realized BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # ML Models Metadata Table
    create_ml_models_metadata = """
    CREATE TABLE IF NOT EXISTS ml_models_metadata (
        model_id SERIAL PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        model_version VARCHAR(20) NOT NULL,
        model_type VARCHAR(50) NOT NULL, -- 'xgboost', 'catboost', 'lstm', 'ensemble'
        model_path TEXT NOT NULL, -- file path to saved model
        training_timestamp TIMESTAMPTZ NOT NULL,
        is_active BOOLEAN DEFAULT FALSE,
        performance_metrics JSONB, -- F1, Brier score, Sharpe ratio, etc.
        feature_importance JSONB,
        hyperparameters JSONB,
        training_data_size INTEGER,
        validation_accuracy NUMERIC(6,4),
        test_accuracy NUMERIC(6,4),
        drift_metrics JSONB,
        last_prediction_timestamp TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # ML Predictions Table
    create_ml_predictions = """
    CREATE TABLE IF NOT EXISTS ml_predictions (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        model_version VARCHAR(20) NOT NULL,
        predicted_regime VARCHAR(50),
        predicted_probability NUMERIC(6,4),
        confidence_score NUMERIC(6,4),
        prediction_horizon VARCHAR(20), -- '1h', '4h', '12h', '24h'
        features_used JSONB,
        prediction_latency_ms INTEGER,
        actual_outcome VARCHAR(50), -- filled in when realized
        accuracy_score NUMERIC(6,4), -- filled in when realized
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    # Create TimescaleDB hypertables
    create_hypertables = """
    SELECT create_hypertable('ml_features_ohlcv', 'timestamp', if_not_exists => TRUE);
    SELECT create_hypertable('ml_features_sentiment', 'timestamp', if_not_exists => TRUE);
    SELECT create_hypertable('ml_labels', 'timestamp', if_not_exists => TRUE);
    SELECT create_hypertable('ml_predictions', 'timestamp', if_not_exists => TRUE);
    """
    
    # Create indexes for performance
    create_indexes = """
    -- ML Features OHLCV indexes
    CREATE INDEX IF NOT EXISTS idx_ml_features_ohlcv_symbol_timeframe ON ml_features_ohlcv (symbol, timeframe, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_features_ohlcv_timestamp ON ml_features_ohlcv (timestamp DESC);
    
    -- ML Features Sentiment indexes
    CREATE INDEX IF NOT EXISTS idx_ml_features_sentiment_symbol ON ml_features_sentiment (symbol, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_features_sentiment_timestamp ON ml_features_sentiment (timestamp DESC);
    
    -- ML Labels indexes
    CREATE INDEX IF NOT EXISTS idx_ml_labels_symbol_type ON ml_labels (symbol, label_type, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_labels_timestamp ON ml_labels (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_labels_realized ON ml_labels (is_realized, timestamp DESC);
    
    -- ML Predictions indexes
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_version ON ml_predictions (symbol, model_version, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence ON ml_predictions (confidence_score DESC, timestamp DESC);
    
    -- ML Models Metadata indexes
    CREATE INDEX IF NOT EXISTS idx_ml_models_metadata_active ON ml_models_metadata (is_active, model_type);
    CREATE INDEX IF NOT EXISTS idx_ml_models_metadata_version ON ml_models_metadata (model_version, model_name);
    """
    
    # Create continuous aggregates for ML features
    create_continuous_aggregates = """
    -- Hourly aggregated ML features (simplified to avoid JOIN issues)
    CREATE MATERIALIZED VIEW IF NOT EXISTS ml_features_ohlcv_hourly_agg
    WITH (timescaledb.continuous) AS
    SELECT 
        time_bucket('1 hour', timestamp) AS bucket,
        symbol,
        timeframe,
        AVG(close_price) as avg_close_price,
        MAX(high_price) as max_high_price,
        MIN(low_price) as min_low_price,
        SUM(volume) as total_volume,
        AVG(rsi) as avg_rsi,
        AVG(macd) as avg_macd,
        AVG(vwap) as avg_vwap,
        AVG(atr) as avg_atr
    FROM ml_features_ohlcv
    GROUP BY bucket, symbol, timeframe
    WITH NO DATA;
    
    -- Daily aggregated ML features (simplified to avoid JOIN issues)
    CREATE MATERIALIZED VIEW IF NOT EXISTS ml_features_ohlcv_daily_agg
    WITH (timescaledb.continuous) AS
    SELECT 
        time_bucket('1 day', timestamp) AS bucket,
        symbol,
        timeframe,
        AVG(close_price) as avg_close_price,
        MAX(high_price) as max_high_price,
        MIN(low_price) as min_low_price,
        SUM(volume) as total_volume,
        AVG(rsi) as avg_rsi,
        AVG(macd) as avg_macd,
        AVG(vwap) as avg_vwap,
        AVG(atr) as avg_atr
    FROM ml_features_ohlcv
    GROUP BY bucket, symbol, timeframe
    WITH NO DATA;
    """
    
    # Set up compression and retention policies
    setup_policies = """
    -- Enable compression on ML feature tables
    ALTER TABLE ml_features_ohlcv SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol,timeframe');
    ALTER TABLE ml_features_sentiment SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
    ALTER TABLE ml_labels SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol,label_type');
    ALTER TABLE ml_predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol,model_version');
    
    -- Set retention policies (keep ML data for 90 days)
    SELECT add_retention_policy('ml_features_ohlcv', INTERVAL '90 days');
    SELECT add_retention_policy('ml_features_sentiment', INTERVAL '90 days');
    SELECT add_retention_policy('ml_labels', INTERVAL '90 days');
    SELECT add_retention_policy('ml_predictions', INTERVAL '90 days');
    
    -- Set compression policies (compress after 1 day)
    SELECT add_compression_policy('ml_features_ohlcv', INTERVAL '1 day');
    SELECT add_compression_policy('ml_features_sentiment', INTERVAL '1 day');
    SELECT add_compression_policy('ml_labels', INTERVAL '1 day');
    SELECT add_compression_policy('ml_predictions', INTERVAL '1 day');
    """
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Create tables
        logger.info("Creating ML Features OHLCV table...")
        await conn.execute(create_ml_features_ohlcv)
        logger.info("✅ ML Features OHLCV table created")
        
        logger.info("Creating ML Features Sentiment table...")
        await conn.execute(create_ml_features_sentiment)
        logger.info("✅ ML Features Sentiment table created")
        
        logger.info("Creating ML Labels table...")
        await conn.execute(create_ml_labels)
        logger.info("✅ ML Labels table created")
        
        logger.info("Creating ML Models Metadata table...")
        await conn.execute(create_ml_models_metadata)
        logger.info("✅ ML Models Metadata table created")
        
        logger.info("Creating ML Predictions table...")
        await conn.execute(create_ml_predictions)
        logger.info("✅ ML Predictions table created")
        
        # Create hypertables
        logger.info("Creating TimescaleDB hypertables...")
        await conn.execute(create_hypertables)
        logger.info("✅ TimescaleDB hypertables created")
        
        # Create indexes
        logger.info("Creating indexes for performance...")
        await conn.execute(create_indexes)
        logger.info("✅ Indexes created")
        
        # Create continuous aggregates
        logger.info("Creating continuous aggregates...")
        await conn.execute(create_continuous_aggregates)
        logger.info("✅ Continuous aggregates created")
        
        # Set up compression and retention policies
        logger.info("Setting up compression and retention policies...")
        await conn.execute(setup_policies)
        logger.info("✅ Compression and retention policies configured")
        
        await conn.close()
        logger.info("✅ Phase 4A ML Feature Tables migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating ML feature tables: {e}")
        raise

async def main():
    logger.info("🚀 Starting Phase 4A ML Feature Tables Migration...")
    await create_ml_feature_tables()
    logger.info("✅ Phase 4A ML Feature Tables Migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
