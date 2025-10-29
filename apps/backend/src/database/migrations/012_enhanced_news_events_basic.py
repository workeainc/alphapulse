#!/usr/bin/env python3
"""
Basic migration script to create enhanced news and event processing tables
Simple version without TimescaleDB features for testing
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

async def create_basic_news_events_tables():
    """Create basic news and event processing tables"""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("✅ Connected to database successfully")
        
        try:
            logger.info("Starting basic news and events tables creation...")
            
            # 1. Raw News Content Table
            logger.info("Creating raw_news_content table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_news_content (
                    id SERIAL PRIMARY KEY,
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
            
            # 2. Economic Events Calendar Table
            logger.info("Creating economic_events_calendar table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_events_calendar (
                    id SERIAL PRIMARY KEY,
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
            
            # 3. Crypto Events Table
            logger.info("Creating crypto_events table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS crypto_events (
                    id SERIAL PRIMARY KEY,
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
            
            # 4. News-Event Correlation Table
            logger.info("Creating news_event_correlation table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_event_correlation (
                    id SERIAL PRIMARY KEY,
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
            
            # 5. Breaking News Alerts Table
            logger.info("Creating breaking_news_alerts table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS breaking_news_alerts (
                    id SERIAL PRIMARY KEY,
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
            
            # 6. News Impact Analysis Table
            logger.info("Creating news_impact_analysis table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_impact_analysis (
                    id SERIAL PRIMARY KEY,
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
            
            # 7. Multi-language News Processing Table
            logger.info("Creating multi_language_news table...")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS multi_language_news (
                    id SERIAL PRIMARY KEY,
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
            
            # Create basic indexes
            logger.info("Creating basic indexes...")
            
            # Indexes for raw_news_content
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_timestamp 
                ON raw_news_content (timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_source 
                ON raw_news_content (source);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_breaking 
                ON raw_news_content (breaking_news) WHERE breaking_news = TRUE;
            """)
            
            # Indexes for economic_events_calendar
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_timestamp 
                ON economic_events_calendar (timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_type 
                ON economic_events_calendar (event_type);
            """)
            
            # Indexes for crypto_events
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_timestamp 
                ON crypto_events (timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_symbol 
                ON crypto_events (symbol);
            """)
            
            # Indexes for breaking_news_alerts
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_timestamp 
                ON breaking_news_alerts (timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_priority 
                ON breaking_news_alerts (priority);
            """)
            
            logger.info("✅ Basic news and events tables created successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error creating basic news and events tables: {e}")
            raise
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

async def main():
    """Main function to run the migration"""
    try:
        await create_basic_news_events_tables()
        logger.info("🎉 Basic news and events migration completed successfully!")
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
