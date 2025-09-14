#!/usr/bin/env python3
"""
Migration script to create enhanced news and event processing tables
Simplified version using asyncpg directly
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def create_enhanced_news_events_tables():
    """Create enhanced news and event processing tables with TimescaleDB optimizations"""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("✅ Connected to database successfully")
        
        try:
            logger.info("Starting enhanced news and events tables creation...")
            
            # 1. Raw News Content Table (store full news articles)
            logger.info("Creating raw_news_content table...")
            await conn.execute("""
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
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for news content
            await conn.execute("""
                SELECT create_hypertable('raw_news_content', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE raw_news_content ADD CONSTRAINT raw_news_content_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 2. Economic Events Calendar Table
            logger.info("Creating economic_events_calendar table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_events_calendar (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_name VARCHAR(200) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    country VARCHAR(50),
                    currency VARCHAR(10),
                    importance VARCHAR(20),
                    previous_value VARCHAR(100),
                    forecast_value VARCHAR(100),
                    actual_value VARCHAR(100),
                    impact_score FLOAT DEFAULT 0.0,
                    market_impact VARCHAR(20),
                    affected_assets TEXT[],
                    description TEXT,
                    source VARCHAR(100),
                    event_id VARCHAR(100) UNIQUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for economic events
            await conn.execute("""
                SELECT create_hypertable('economic_events_calendar', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE economic_events_calendar ADD CONSTRAINT economic_events_calendar_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 3. Crypto Events Table
            logger.info("Creating crypto_events table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS crypto_events (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    event_name VARCHAR(200) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20),
                    blockchain VARCHAR(50),
                    importance VARCHAR(20),
                    impact_score FLOAT DEFAULT 0.0,
                    market_impact VARCHAR(20),
                    affected_assets TEXT[],
                    description TEXT,
                    source VARCHAR(100),
                    event_id VARCHAR(100) UNIQUE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for crypto events
            await conn.execute("""
                SELECT create_hypertable('crypto_events', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE crypto_events ADD CONSTRAINT crypto_events_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 4. News-Event Correlation Table
            logger.info("Creating news_event_correlation table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_event_correlation (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    news_id INTEGER,
                    event_id VARCHAR(100),
                    event_type VARCHAR(50),
                    correlation_score FLOAT NOT NULL,
                    correlation_type VARCHAR(50),
                    impact_prediction FLOAT,
                    confidence FLOAT DEFAULT 0.0,
                    affected_symbols TEXT[],
                    analysis_notes TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for correlations
            await conn.execute("""
                SELECT create_hypertable('news_event_correlation', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE news_event_correlation ADD CONSTRAINT news_event_correlation_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 5. Breaking News Alerts Table
            logger.info("Creating breaking_news_alerts table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS breaking_news_alerts (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    alert_id VARCHAR(100) UNIQUE,
                    news_id INTEGER,
                    alert_type VARCHAR(50),
                    priority VARCHAR(20),
                    title TEXT NOT NULL,
                    summary TEXT,
                    affected_symbols TEXT[],
                    impact_prediction FLOAT,
                    confidence FLOAT DEFAULT 0.0,
                    sent_to_users BOOLEAN DEFAULT FALSE,
                    sent_to_websocket BOOLEAN DEFAULT FALSE,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for alerts
            await conn.execute("""
                SELECT create_hypertable('breaking_news_alerts', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE breaking_news_alerts ADD CONSTRAINT breaking_news_alerts_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 6. News Impact Analysis Table
            logger.info("Creating news_impact_analysis table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_impact_analysis (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    news_id INTEGER,
                    symbol VARCHAR(20),
                    impact_type VARCHAR(50),
                    pre_news_value FLOAT,
                    post_news_value FLOAT,
                    impact_magnitude FLOAT,
                    impact_direction VARCHAR(20),
                    time_to_impact_minutes INTEGER,
                    impact_duration_minutes INTEGER,
                    confidence FLOAT DEFAULT 0.0,
                    analysis_notes TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for impact analysis
            await conn.execute("""
                SELECT create_hypertable('news_impact_analysis', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE news_impact_analysis ADD CONSTRAINT news_impact_analysis_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # 7. Multi-language News Processing Table
            logger.info("Creating multi_language_news table...")
            await conn.execute("""
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
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create TimescaleDB hypertable for multi-language news
            await conn.execute("""
                SELECT create_hypertable('multi_language_news', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)
            
            # Add primary key after hypertable creation
            await conn.execute("""
                ALTER TABLE multi_language_news ADD CONSTRAINT multi_language_news_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # Create indexes for performance optimization
            logger.info("Creating performance indexes...")
            
            # Indexes for raw_news_content
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_source_timestamp 
                ON raw_news_content (source, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_breaking_timestamp 
                ON raw_news_content (breaking_news, timestamp DESC) WHERE breaking_news = TRUE;
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_sentiment_timestamp 
                ON raw_news_content (sentiment_score, timestamp DESC);
            """)
            
            # Indexes for economic_events_calendar
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_type_timestamp 
                ON economic_events_calendar (event_type, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_importance_timestamp 
                ON economic_events_calendar (importance, timestamp DESC);
            """)
            
            # Indexes for crypto_events
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_symbol_timestamp 
                ON crypto_events (symbol, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_type_timestamp 
                ON crypto_events (event_type, timestamp DESC);
            """)
            
            # Indexes for breaking_news_alerts
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_priority_timestamp 
                ON breaking_news_alerts (priority, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_sent_timestamp 
                ON breaking_news_alerts (sent_to_websocket, timestamp DESC);
            """)
            
            # GIN indexes for JSONB fields
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_entities_gin 
                ON raw_news_content USING GIN (entities);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_metadata_gin 
                ON raw_news_content USING GIN (metadata);
            """)
            
            # Create compression policies for older data
            logger.info("Setting up compression policies...")
            
            # Compress news content older than 7 days
            await conn.execute("""
                SELECT add_compression_policy('raw_news_content', INTERVAL '7 days');
            """)
            
            # Compress economic events older than 30 days
            await conn.execute("""
                SELECT add_compression_policy('economic_events_calendar', INTERVAL '30 days');
            """)
            
            # Compress crypto events older than 30 days
            await conn.execute("""
                SELECT add_compression_policy('crypto_events', INTERVAL '30 days');
            """)
            
            # Compress correlations older than 7 days
            await conn.execute("""
                SELECT add_compression_policy('news_event_correlation', INTERVAL '7 days');
            """)
            
            # Compress alerts older than 3 days
            await conn.execute("""
                SELECT add_compression_policy('breaking_news_alerts', INTERVAL '3 days');
            """)
            
            # Compress impact analysis older than 7 days
            await conn.execute("""
                SELECT add_compression_policy('news_impact_analysis', INTERVAL '7 days');
            """)
            
            # Compress multi-language news older than 7 days
            await conn.execute("""
                SELECT add_compression_policy('multi_language_news', INTERVAL '7 days');
            """)
            
            # Create retention policies
            logger.info("Setting up retention policies...")
            
            # Keep news content for 90 days
            await conn.execute("""
                SELECT add_retention_policy('raw_news_content', INTERVAL '90 days');
            """)
            
            # Keep economic events for 1 year
            await conn.execute("""
                SELECT add_retention_policy('economic_events_calendar', INTERVAL '1 year');
            """)
            
            # Keep crypto events for 1 year
            await conn.execute("""
                SELECT add_retention_policy('crypto_events', INTERVAL '1 year');
            """)
            
            # Keep correlations for 30 days
            await conn.execute("""
                SELECT add_retention_policy('news_event_correlation', INTERVAL '30 days');
            """)
            
            # Keep alerts for 7 days
            await conn.execute("""
                SELECT add_retention_policy('breaking_news_alerts', INTERVAL '7 days');
            """)
            
            # Keep impact analysis for 30 days
            await conn.execute("""
                SELECT add_retention_policy('news_impact_analysis', INTERVAL '30 days');
            """)
            
            # Keep multi-language news for 30 days
            await conn.execute("""
                SELECT add_retention_policy('multi_language_news', INTERVAL '30 days');
            """)
            
            logger.info("✅ Enhanced news and events tables created successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error creating enhanced news and events tables: {e}")
            raise
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

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
