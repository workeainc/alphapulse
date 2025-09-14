#!/usr/bin/env python3
"""
Enhanced News and Events Migration for TimescaleDB
Proper implementation with hypertables, compression, and retention policies
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

async def create_enhanced_news_events_timescaledb():
    """Create enhanced news and event processing tables with proper TimescaleDB optimization"""
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("✅ Connected to TimescaleDB successfully")
        
        try:
            logger.info("🚀 Starting TimescaleDB enhanced news and events tables creation...")
            
            # Ensure TimescaleDB extension is enabled
            logger.info("📦 Ensuring TimescaleDB extension is enabled...")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # 1. Raw News Content Table (Primary news storage with TimescaleDB optimization)
            logger.info("📰 Creating raw_news_content hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS raw_news_content CASCADE;
                CREATE TABLE raw_news_content (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
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
                    entities JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for raw_news_content
            await conn.execute("""
                SELECT create_hypertable('raw_news_content', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint for TimescaleDB
            await conn.execute("""
                ALTER TABLE raw_news_content 
                ADD CONSTRAINT raw_news_content_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 2. Economic Events Calendar Table
            logger.info("📊 Creating economic_events_calendar hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS economic_events_calendar CASCADE;
                CREATE TABLE economic_events_calendar (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
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
                    event_id VARCHAR(100),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for economic events
            await conn.execute("""
                SELECT create_hypertable('economic_events_calendar', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE economic_events_calendar 
                ADD CONSTRAINT economic_events_calendar_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 3. Crypto Events Table
            logger.info("🪙 Creating crypto_events hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS crypto_events CASCADE;
                CREATE TABLE crypto_events (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
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
                    event_id VARCHAR(100),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for crypto events
            await conn.execute("""
                SELECT create_hypertable('crypto_events', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE crypto_events 
                ADD CONSTRAINT crypto_events_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 4. News-Event Correlation Table
            logger.info("🔗 Creating news_event_correlation hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS news_event_correlation CASCADE;
                CREATE TABLE news_event_correlation (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
                    news_id INTEGER,
                    event_id VARCHAR(100),
                    event_type VARCHAR(50),
                    correlation_score FLOAT NOT NULL,
                    correlation_type VARCHAR(50),
                    impact_prediction FLOAT,
                    confidence FLOAT DEFAULT 0.0,
                    affected_symbols TEXT[],
                    analysis_notes TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for correlations
            await conn.execute("""
                SELECT create_hypertable('news_event_correlation', 'timestamp', 
                    chunk_time_interval => INTERVAL '2 hours',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE news_event_correlation 
                ADD CONSTRAINT news_event_correlation_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 5. Breaking News Alerts Table
            logger.info("🚨 Creating breaking_news_alerts hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS breaking_news_alerts CASCADE;
                CREATE TABLE breaking_news_alerts (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
                    alert_id VARCHAR(100),
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
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for breaking news alerts
            await conn.execute("""
                SELECT create_hypertable('breaking_news_alerts', 'timestamp', 
                    chunk_time_interval => INTERVAL '30 minutes',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE breaking_news_alerts 
                ADD CONSTRAINT breaking_news_alerts_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 6. News Impact Analysis Table
            logger.info("📈 Creating news_impact_analysis hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS news_impact_analysis CASCADE;
                CREATE TABLE news_impact_analysis (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
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
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for impact analysis
            await conn.execute("""
                SELECT create_hypertable('news_impact_analysis', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE news_impact_analysis 
                ADD CONSTRAINT news_impact_analysis_pkey PRIMARY KEY (timestamp, id);
            """)
            
            # 7. Multi-language News Processing Table
            logger.info("🌍 Creating multi_language_news hypertable...")
            await conn.execute("""
                DROP TABLE IF EXISTS multi_language_news CASCADE;
                CREATE TABLE multi_language_news (
                    timestamp TIMESTAMPTZ NOT NULL,
                    id SERIAL,
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
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for multi-language news
            await conn.execute("""
                SELECT create_hypertable('multi_language_news', 'timestamp', 
                    chunk_time_interval => INTERVAL '2 hours',
                    if_not_exists => TRUE
                );
            """)
            
            # Add primary key constraint
            await conn.execute("""
                ALTER TABLE multi_language_news 
                ADD CONSTRAINT multi_language_news_pkey PRIMARY KEY (timestamp, id);
            """)
            
            logger.info("📊 Creating TimescaleDB optimized indexes...")
            
            # Create optimized indexes for raw_news_content
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_source_time 
                ON raw_news_content (source, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_breaking_time 
                ON raw_news_content (timestamp DESC, breaking_news) 
                WHERE breaking_news = TRUE;
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_sentiment_time 
                ON raw_news_content (timestamp DESC, sentiment_score);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_news_content_impact_time 
                ON raw_news_content (timestamp DESC, impact_score);
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
            
            # Create indexes for economic events
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_type_time 
                ON economic_events_calendar (event_type, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_importance_time 
                ON economic_events_calendar (importance, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_economic_events_country_time 
                ON economic_events_calendar (country, timestamp DESC);
            """)
            
            # Create indexes for crypto events
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_symbol_time 
                ON crypto_events (symbol, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_events_type_time 
                ON crypto_events (event_type, timestamp DESC);
            """)
            
            # Create indexes for breaking news alerts
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_priority_time 
                ON breaking_news_alerts (priority, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_breaking_news_sent_time 
                ON breaking_news_alerts (timestamp DESC, sent_to_websocket);
            """)
            
            # Create indexes for news correlations
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_correlation_score_time 
                ON news_event_correlation (correlation_score, timestamp DESC);
            """)
            
            logger.info("🗜️ Setting up TimescaleDB compression policies...")
            
            # Add compression policies - compress data older than specified intervals
            compression_policies = [
                ("raw_news_content", "3 days"),
                ("economic_events_calendar", "7 days"),
                ("crypto_events", "7 days"),
                ("news_event_correlation", "1 day"),
                ("breaking_news_alerts", "1 day"),
                ("news_impact_analysis", "2 days"),
                ("multi_language_news", "3 days")
            ]
            
            for table, interval in compression_policies:
                try:
                    await conn.execute(f"""
                        SELECT add_compression_policy('{table}', INTERVAL '{interval}');
                    """)
                    logger.info(f"✅ Added compression policy for {table} (compress after {interval})")
                except Exception as e:
                    logger.warning(f"⚠️ Compression policy for {table} may already exist: {e}")
            
            logger.info("🗂️ Setting up TimescaleDB retention policies...")
            
            # Add retention policies - automatically drop old data
            retention_policies = [
                ("raw_news_content", "90 days"),
                ("economic_events_calendar", "2 years"),
                ("crypto_events", "2 years"),
                ("news_event_correlation", "30 days"),
                ("breaking_news_alerts", "14 days"),
                ("news_impact_analysis", "60 days"),
                ("multi_language_news", "60 days")
            ]
            
            for table, interval in retention_policies:
                try:
                    await conn.execute(f"""
                        SELECT add_retention_policy('{table}', INTERVAL '{interval}');
                    """)
                    logger.info(f"✅ Added retention policy for {table} (drop after {interval})")
                except Exception as e:
                    logger.warning(f"⚠️ Retention policy for {table} may already exist: {e}")
            
            # Create unique indexes for certain fields after hypertable creation
            logger.info("🔑 Creating unique constraints...")
            
            # Create unique index for alert_id in breaking_news_alerts
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_breaking_news_alerts_alert_id_time 
                ON breaking_news_alerts (alert_id, timestamp);
            """)
            
            # Create unique index for event_id in economic_events_calendar
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_economic_events_event_id_time 
                ON economic_events_calendar (event_id, timestamp) 
                WHERE event_id IS NOT NULL;
            """)
            
            # Create unique index for event_id in crypto_events
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_crypto_events_event_id_time 
                ON crypto_events (event_id, timestamp) 
                WHERE event_id IS NOT NULL;
            """)
            
            logger.info("📊 Verifying hypertables...")
            
            # Verify hypertables were created correctly
            hypertables = await conn.fetch("""
                SELECT hypertable_name, num_dimensions, num_chunks 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name IN (
                    'raw_news_content', 'economic_events_calendar', 'crypto_events',
                    'news_event_correlation', 'breaking_news_alerts', 'news_impact_analysis',
                    'multi_language_news'
                );
            """)
            
            for hypertable in hypertables:
                logger.info(f"✅ Hypertable: {hypertable['hypertable_name']} - Dimensions: {hypertable['num_dimensions']} - Chunks: {hypertable['num_chunks']}")
            
            logger.info("✅ TimescaleDB enhanced news and events tables created successfully!")
            logger.info(f"📊 Created {len(hypertables)} hypertables with optimized chunk intervals")
            logger.info("🗜️ Compression policies configured for automatic data optimization")
            logger.info("🗂️ Retention policies configured for automatic data lifecycle management")
            
        except Exception as e:
            logger.error(f"❌ Error creating TimescaleDB enhanced news and events tables: {e}")
            raise
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ TimescaleDB connection failed: {e}")
        raise

async def main():
    """Main function to run the TimescaleDB migration"""
    try:
        await create_enhanced_news_events_timescaledb()
        logger.info("🎉 TimescaleDB enhanced news and events migration completed successfully!")
        logger.info("🚀 Your AlphaPlus system now has professional-grade time-series optimized news processing!")
    except Exception as e:
        logger.error(f"💥 TimescaleDB migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
