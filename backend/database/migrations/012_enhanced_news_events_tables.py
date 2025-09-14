#!/usr/bin/env python3
"""
Migration script to create enhanced news and event processing tables
Phase 3: Advanced News and Event Processing Implementation
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

async def create_enhanced_news_events_tables():
    """Create enhanced news and event processing tables with TimescaleDB optimizations"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("Starting enhanced news and events tables creation...")
            
            # 1. Raw News Content Table (store full news articles)
            logger.info("Creating raw_news_content table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw_news_content (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    content TEXT,
                    url TEXT,
                    source VARCHAR(100) NOT NULL,
                    author VARCHAR(200),
                    published_at TIMESTAMPTZ,
                    language VARCHAR(10) DEFAULT 'en',
                    category VARCHAR(50),
                    tags TEXT[],
                    relevance_score FLOAT DEFAULT 0.0,
                    impact_score FLOAT DEFAULT 0.0,
                    breaking_news BOOLEAN DEFAULT FALSE,
                    verified_source BOOLEAN DEFAULT FALSE,
                    sentiment_score FLOAT,
                    sentiment_label VARCHAR(20),
                    confidence FLOAT DEFAULT 0.0,
                    keywords TEXT[],
                    entities JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for news content
            await conn.execute(text("""
                SELECT create_hypertable('raw_news_content', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 2. Economic Events Calendar Table
            logger.info("Creating economic_events_calendar table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS economic_events_calendar (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_name VARCHAR(200) NOT NULL,
                    event_type VARCHAR(50) NOT NULL, -- 'fomc', 'cpi', 'nfp', 'gdp', 'fed_rate', 'ecb_rate', 'boe_rate'
                    country VARCHAR(50),
                    currency VARCHAR(10),
                    importance VARCHAR(20), -- 'low', 'medium', 'high', 'very_high'
                    previous_value VARCHAR(100),
                    forecast_value VARCHAR(100),
                    actual_value VARCHAR(100),
                    impact_score FLOAT DEFAULT 0.0,
                    market_impact VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
                    affected_assets TEXT[],
                    description TEXT,
                    source VARCHAR(100),
                    event_id VARCHAR(100) UNIQUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for economic events
            await conn.execute(text("""
                SELECT create_hypertable('economic_events_calendar', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """))
            
            # 3. Crypto Events Table
            logger.info("Creating crypto_events table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS crypto_events (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_name VARCHAR(200) NOT NULL,
                    event_type VARCHAR(50) NOT NULL, -- 'halving', 'upgrade', 'fork', 'airdrop', 'listing', 'delisting', 'regulation'
                    symbol VARCHAR(20),
                    blockchain VARCHAR(50),
                    importance VARCHAR(20), -- 'low', 'medium', 'high', 'very_high'
                    impact_score FLOAT DEFAULT 0.0,
                    market_impact VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
                    affected_assets TEXT[],
                    description TEXT,
                    source VARCHAR(100),
                    event_id VARCHAR(100) UNIQUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for crypto events
            await conn.execute(text("""
                SELECT create_hypertable('crypto_events', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """))
            
            # 4. News-Event Correlation Table
            logger.info("Creating news_event_correlation table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS news_event_correlation (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    news_id INTEGER,
                    event_id VARCHAR(100),
                    event_type VARCHAR(50), -- 'economic', 'crypto', 'regulatory'
                    correlation_score FLOAT NOT NULL,
                    correlation_type VARCHAR(50), -- 'direct', 'indirect', 'sentiment', 'timing'
                    impact_prediction FLOAT,
                    confidence FLOAT DEFAULT 0.0,
                    affected_symbols TEXT[],
                    analysis_notes TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for correlations
            await conn.execute(text("""
                SELECT create_hypertable('news_event_correlation', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 5. Breaking News Alerts Table
            logger.info("Creating breaking_news_alerts table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS breaking_news_alerts (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    alert_id VARCHAR(100) UNIQUE,
                    news_id INTEGER,
                    alert_type VARCHAR(50), -- 'breaking_news', 'market_moving', 'regulatory', 'whale_movement'
                    priority VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
                    title TEXT NOT NULL,
                    summary TEXT,
                    affected_symbols TEXT[],
                    impact_prediction FLOAT,
                    confidence FLOAT DEFAULT 0.0,
                    sent_to_users BOOLEAN DEFAULT FALSE,
                    sent_to_websocket BOOLEAN DEFAULT FALSE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for alerts
            await conn.execute(text("""
                SELECT create_hypertable('breaking_news_alerts', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 6. News Impact Analysis Table
            logger.info("Creating news_impact_analysis table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS news_impact_analysis (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    news_id INTEGER,
                    symbol VARCHAR(20),
                    impact_type VARCHAR(50), -- 'price', 'volume', 'volatility', 'sentiment'
                    pre_news_value FLOAT,
                    post_news_value FLOAT,
                    impact_magnitude FLOAT,
                    impact_direction VARCHAR(20), -- 'positive', 'negative', 'neutral'
                    time_to_impact_minutes INTEGER,
                    impact_duration_minutes INTEGER,
                    confidence FLOAT DEFAULT 0.0,
                    analysis_notes TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for impact analysis
            await conn.execute(text("""
                SELECT create_hypertable('news_impact_analysis', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # 7. Multi-language News Processing Table
            logger.info("Creating multi_language_news table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS multi_language_news (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    original_news_id INTEGER,
                    language VARCHAR(10) NOT NULL,
                    translated_title TEXT,
                    translated_description TEXT,
                    translated_content TEXT,
                    translation_confidence FLOAT DEFAULT 0.0,
                    sentiment_score FLOAT,
                    sentiment_label VARCHAR(20),
                    regional_impact_score FLOAT DEFAULT 0.0,
                    affected_regions TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # Create TimescaleDB hypertable for multi-language news
            await conn.execute(text("""
                SELECT create_hypertable('multi_language_news', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """))
            
            # Create indexes for performance optimization
            logger.info("Creating performance indexes...")
            
            # Indexes for raw_news_content
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_source_timestamp 
                ON raw_news_content (source, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_breaking_timestamp 
                ON raw_news_content (breaking_news, timestamp DESC) WHERE breaking_news = TRUE;
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_sentiment_timestamp 
                ON raw_news_content (sentiment_score, timestamp DESC);
            """))
            
            # Indexes for economic_events_calendar
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_type_timestamp 
                ON economic_events_calendar (event_type, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_importance_timestamp 
                ON economic_events_calendar (importance, timestamp DESC);
            """))
            
            # Indexes for crypto_events
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_symbol_timestamp 
                ON crypto_events (symbol, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_type_timestamp 
                ON crypto_events (event_type, timestamp DESC);
            """))
            
            # Indexes for breaking_news_alerts
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_priority_timestamp 
                ON breaking_news_alerts (priority, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_sent_timestamp 
                ON breaking_news_alerts (sent_to_websocket, timestamp DESC);
            """))
            
            # GIN indexes for JSONB fields
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_entities_gin 
                ON raw_news_content USING GIN (entities);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_metadata_gin 
                ON raw_news_content USING GIN (metadata);
            """))
            
            # Create compression policies for older data
            logger.info("Setting up compression policies...")
            
            # Compress news content older than 7 days
            await conn.execute(text("""
                SELECT add_compression_policy('raw_news_content', INTERVAL '7 days');
            """))
            
            # Compress economic events older than 30 days
            await conn.execute(text("""
                SELECT add_compression_policy('economic_events_calendar', INTERVAL '30 days');
            """))
            
            # Compress crypto events older than 30 days
            await conn.execute(text("""
                SELECT add_compression_policy('crypto_events', INTERVAL '30 days');
            """))
            
            # Compress correlations older than 7 days
            await conn.execute(text("""
                SELECT add_compression_policy('news_event_correlation', INTERVAL '7 days');
            """))
            
            # Compress alerts older than 3 days
            await conn.execute(text("""
                SELECT add_compression_policy('breaking_news_alerts', INTERVAL '3 days');
            """))
            
            # Compress impact analysis older than 7 days
            await conn.execute(text("""
                SELECT add_compression_policy('news_impact_analysis', INTERVAL '7 days');
            """))
            
            # Compress multi-language news older than 7 days
            await conn.execute(text("""
                SELECT add_compression_policy('multi_language_news', INTERVAL '7 days');
            """))
            
            # Create retention policies
            logger.info("Setting up retention policies...")
            
            # Keep news content for 90 days
            await conn.execute(text("""
                SELECT add_retention_policy('raw_news_content', INTERVAL '90 days');
            """))
            
            # Keep economic events for 1 year
            await conn.execute(text("""
                SELECT add_retention_policy('economic_events_calendar', INTERVAL '1 year');
            """))
            
            # Keep crypto events for 1 year
            await conn.execute(text("""
                SELECT add_retention_policy('crypto_events', INTERVAL '1 year');
            """))
            
            # Keep correlations for 30 days
            await conn.execute(text("""
                SELECT add_retention_policy('news_event_correlation', INTERVAL '30 days');
            """))
            
            # Keep alerts for 7 days
            await conn.execute(text("""
                SELECT add_retention_policy('breaking_news_alerts', INTERVAL '7 days');
            """))
            
            # Keep impact analysis for 30 days
            await conn.execute(text("""
                SELECT add_retention_policy('news_impact_analysis', INTERVAL '30 days');
            """))
            
            # Keep multi-language news for 30 days
            await conn.execute(text("""
                SELECT add_retention_policy('multi_language_news', INTERVAL '30 days');
            """))
            
            logger.info("✅ Enhanced news and events tables created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating enhanced news and events tables: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main function to run the migration"""
    try:
        await create_enhanced_news_events_tables()
        logger.info("🎉 Enhanced news and events migration completed successfully!")
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
