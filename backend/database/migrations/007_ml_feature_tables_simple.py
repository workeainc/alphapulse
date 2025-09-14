#!/usr/bin/env python3
"""
Migration: Create ML Feature Tables for Phase 4A (Simplified)
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
        label_type VARCHAR(50) NOT NULL,
        label_value VARCHAR(50),
        label_confidence NUMERIC(6,4),
        future_timestamp TIMESTAMPTZ,
        realized_value VARCHAR(50),
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
        model_type VARCHAR(50) NOT NULL,
        model_path TEXT NOT NULL,
        training_timestamp TIMESTAMPTZ NOT NULL,
        is_active BOOLEAN DEFAULT FALSE,
        performance_metrics JSONB,
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
        prediction_horizon VARCHAR(20),
        features_used JSONB,
        prediction_latency_ms INTEGER,
        actual_outcome VARCHAR(50),
        accuracy_score NUMERIC(6,4),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
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
        
        # Create TimescaleDB hypertables
        logger.info("Creating TimescaleDB hypertables...")
        await conn.execute("SELECT create_hypertable('ml_features_ohlcv', 'timestamp', if_not_exists => TRUE);")
        await conn.execute("SELECT create_hypertable('ml_features_sentiment', 'timestamp', if_not_exists => TRUE);")
        await conn.execute("SELECT create_hypertable('ml_labels', 'timestamp', if_not_exists => TRUE);")
        await conn.execute("SELECT create_hypertable('ml_predictions', 'timestamp', if_not_exists => TRUE);")
        logger.info("✅ TimescaleDB hypertables created")
        
        await conn.close()
        logger.info("✅ Phase 4A ML Feature Tables migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating ML feature tables: {e}")
        raise

async def main():
    logger.info("🚀 Starting Phase 4A ML Feature Tables Migration (Simplified)...")
    await create_ml_feature_tables()
    logger.info("✅ Phase 4A ML Feature Tables Migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
