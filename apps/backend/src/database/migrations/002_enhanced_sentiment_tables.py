#!/usr/bin/env python3
"""
Migration script to create enhanced sentiment analysis tables
Phase 2: Enhanced Sentiment Analysis Implementation
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

async def create_enhanced_sentiment_tables():
    """Create enhanced sentiment analysis tables with TimescaleDB optimizations"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("Starting enhanced sentiment tables creation...")
            
            # 1. Enhanced Sentiment Data Table (main sentiment storage)
            logger.info("Creating enhanced_sentiment_data table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enhanced_sentiment_data (
                    id SERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    source VARCHAR(50) NOT NULL, -- 'twitter', 'reddit', 'news', 'telegram', 'discord', 'onchain'
                    sentiment_score FLOAT NOT NULL, -- -1.0 to +1.0
                    sentiment_label VARCHAR(20) NOT NULL, -- 'positive', 'negative', 'neutral'
                    confidence FLOAT NOT NULL, -- 0.0 to 1.0
                    volume INTEGER, -- post/tweet volume
                    keywords TEXT[], -- relevant keywords and hashtags
                    raw_text TEXT, -- original text content
                    processed_text TEXT, -- cleaned and processed text
                    language VARCHAR(10) DEFAULT 'en',
                    user_id VARCHAR(100), -- social media user ID
                    user_followers INTEGER, -- user influence metric
                    user_verified BOOLEAN DEFAULT FALSE,
                    engagement_metrics JSONB, -- likes, retweets, comments, etc.
                    topic_classification VARCHAR(100), -- 'price_moving', 'noise', 'news', 'opinion'
                    sarcasm_detected BOOLEAN DEFAULT FALSE,
                    context_score FLOAT, -- context relevance score
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for sentiment data
            await conn.execute(text("""
                SELECT create_hypertable('enhanced_sentiment_data', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 2. Real-time Sentiment Aggregation Table
            logger.info("Creating real_time_sentiment_aggregation table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS real_time_sentiment_aggregation (
                    id SERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    window_size VARCHAR(10) NOT NULL, -- '1min', '5min', '15min', '1hour'
                    
                    -- Aggregated Sentiment Scores
                    overall_sentiment_score FLOAT NOT NULL,
                    positive_sentiment_score FLOAT NOT NULL,
                    negative_sentiment_score FLOAT NOT NULL,
                    neutral_sentiment_score FLOAT NOT NULL,
                    
                    -- Source Breakdown
                    twitter_sentiment FLOAT,
                    reddit_sentiment FLOAT,
                    news_sentiment FLOAT,
                    telegram_sentiment FLOAT,
                    discord_sentiment FLOAT,
                    onchain_sentiment FLOAT,
                    
                    -- Volume Metrics
                    total_volume INTEGER NOT NULL,
                    twitter_volume INTEGER,
                    reddit_volume INTEGER,
                    news_volume INTEGER,
                    telegram_volume INTEGER,
                    discord_volume INTEGER,
                    onchain_volume INTEGER,
                    
                    -- Quality Metrics
                    confidence_weighted_score FLOAT NOT NULL,
                    source_diversity_score FLOAT NOT NULL,
                    outlier_filtered BOOLEAN DEFAULT FALSE,
                    
                    -- Trend Analysis
                    sentiment_trend VARCHAR(20), -- 'increasing', 'decreasing', 'stable'
                    trend_strength FLOAT,
                    momentum_score FLOAT,
                    
                    -- Market Context
                    fear_greed_index INTEGER,
                    market_regime VARCHAR(20),
                    volatility_level VARCHAR(20),
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for aggregation
            await conn.execute(text("""
                SELECT create_hypertable('real_time_sentiment_aggregation', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 3. Sentiment Correlation Table (price-sentiment correlation)
            logger.info("Creating sentiment_correlation table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_correlation (
                    id SERIAL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    timeframe VARCHAR(10) NOT NULL, -- '1min', '5min', '15min', '1hour'
                    
                    -- Price Data
                    price_change_percent FLOAT NOT NULL,
                    volume_change_percent FLOAT NOT NULL,
                    volatility FLOAT NOT NULL,
                    
                    -- Sentiment Correlation
                    sentiment_price_correlation FLOAT NOT NULL,
                    sentiment_volume_correlation FLOAT NOT NULL,
                    sentiment_volatility_correlation FLOAT NOT NULL,
                    
                    -- Lag Analysis
                    sentiment_lag_1min FLOAT,
                    sentiment_lag_5min FLOAT,
                    sentiment_lag_15min FLOAT,
                    sentiment_lag_1hour FLOAT,
                    
                    -- Predictive Power
                    sentiment_predictive_power FLOAT,
                    price_prediction_accuracy FLOAT,
                    
                    -- Cross-Asset Correlation
                    btc_correlation FLOAT,
                    eth_correlation FLOAT,
                    market_correlation FLOAT,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for correlation
            await conn.execute(text("""
                SELECT create_hypertable('sentiment_correlation', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 4. Sentiment Alerts Table
            logger.info("Creating sentiment_alerts table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_alerts (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    alert_type VARCHAR(50) NOT NULL, -- 'sentiment_spike', 'trend_reversal', 'anomaly'
                    alert_severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
                    
                    -- Alert Details
                    sentiment_change FLOAT NOT NULL,
                    volume_change FLOAT NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    
                    -- Trigger Conditions
                    trigger_threshold FLOAT NOT NULL,
                    trigger_source VARCHAR(50) NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    
                    -- Alert Status
                    status VARCHAR(20) DEFAULT 'active', -- 'active', 'acknowledged', 'resolved'
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    acknowledged_by VARCHAR(100),
                    
                    -- Alert Actions
                    action_taken VARCHAR(100),
                    action_result TEXT,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 5. Sentiment Model Performance Table
            logger.info("Creating sentiment_model_performance table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    
                    -- Performance Metrics
                    accuracy FLOAT NOT NULL,
                    precision FLOAT NOT NULL,
                    recall FLOAT NOT NULL,
                    f1_score FLOAT NOT NULL,
                    
                    -- Model Details
                    training_data_size INTEGER NOT NULL,
                    validation_data_size INTEGER NOT NULL,
                    training_duration_seconds FLOAT,
                    
                    -- Feature Importance
                    feature_importance JSONB,
                    
                    -- Model Status
                    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
                    is_current BOOLEAN DEFAULT FALSE,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 6. Create indexes for performance
            logger.info("Creating performance indexes...")
            
            # Enhanced sentiment data indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_sentiment_symbol_timestamp 
                ON enhanced_sentiment_data (symbol, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_sentiment_source_timestamp 
                ON enhanced_sentiment_data (source, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_sentiment_score_timestamp 
                ON enhanced_sentiment_data (sentiment_score, timestamp DESC);
            """))
            
            # Real-time aggregation indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_real_time_sentiment_symbol_window 
                ON real_time_sentiment_aggregation (symbol, window_size, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_real_time_sentiment_overall_score 
                ON real_time_sentiment_aggregation (overall_sentiment_score, timestamp DESC);
            """))
            
            # Correlation indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_correlation_symbol_timeframe 
                ON sentiment_correlation (symbol, timeframe, timestamp DESC);
            """))
            
            # Alerts indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_alerts_symbol_status 
                ON sentiment_alerts (symbol, status, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_alerts_severity 
                ON sentiment_alerts (alert_severity, timestamp DESC);
            """))
            
            logger.info("✅ Enhanced sentiment tables created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating enhanced sentiment tables: {e}")
        raise
    finally:
        await engine.dispose()

async def create_sentiment_views():
    """Create useful views for sentiment analysis"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("Creating sentiment analysis views...")
            
            # 1. Latest Sentiment Summary View
            await conn.execute(text("""
                CREATE OR REPLACE VIEW latest_sentiment_summary AS
                SELECT 
                    symbol,
                    timestamp,
                    overall_sentiment_score,
                    positive_sentiment_score,
                    negative_sentiment_score,
                    neutral_sentiment_score,
                    total_volume,
                    confidence_weighted_score,
                    sentiment_trend,
                    trend_strength,
                    fear_greed_index,
                    market_regime
                FROM real_time_sentiment_aggregation
                WHERE window_size = '5min'
                AND timestamp >= NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC;
            """))
            
            # 2. Sentiment Trend Analysis View
            await conn.execute(text("""
                CREATE OR REPLACE VIEW sentiment_trend_analysis AS
                SELECT 
                    symbol,
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(overall_sentiment_score) as avg_sentiment,
                    STDDEV(overall_sentiment_score) as sentiment_volatility,
                    COUNT(*) as data_points,
                    MAX(overall_sentiment_score) as max_sentiment,
                    MIN(overall_sentiment_score) as min_sentiment
                FROM real_time_sentiment_aggregation
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY symbol, DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC;
            """))
            
            # 3. High Confidence Sentiment Signals View
            await conn.execute(text("""
                CREATE OR REPLACE VIEW high_confidence_sentiment_signals AS
                SELECT 
                    symbol,
                    timestamp,
                    overall_sentiment_score,
                    confidence_weighted_score,
                    sentiment_trend,
                    trend_strength,
                    total_volume
                FROM real_time_sentiment_aggregation
                WHERE confidence_weighted_score >= 0.8
                AND ABS(overall_sentiment_score) >= 0.3
                AND total_volume >= 100
                AND timestamp >= NOW() - INTERVAL '1 hour'
                ORDER BY confidence_weighted_score DESC, timestamp DESC;
            """))
            
            logger.info("✅ Sentiment analysis views created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating sentiment views: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Enhanced Sentiment Analysis Migration...")
    
    try:
        await create_enhanced_sentiment_tables()
        await create_sentiment_views()
        logger.info("✅ Enhanced Sentiment Analysis Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
