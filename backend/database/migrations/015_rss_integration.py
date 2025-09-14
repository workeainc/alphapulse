#!/usr/bin/env python3
"""
RSS Integration Migration
Extends existing tables with RSS feed tracking and optimization
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_rss_integration_tables():
    """Extend existing tables with RSS feed integration"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Extend raw_news_content table with RSS-specific columns
        logger.info("📰 Extending raw_news_content table with RSS columns...")
        await conn.execute("""
            ALTER TABLE raw_news_content 
            ADD COLUMN IF NOT EXISTS rss_feed_url TEXT,
            ADD COLUMN IF NOT EXISTS rss_feed_name TEXT,
            ADD COLUMN IF NOT EXISTS rss_category TEXT,
            ADD COLUMN IF NOT EXISTS rss_published_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS rss_guid TEXT,
            ADD COLUMN IF NOT EXISTS feed_credibility FLOAT DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS rss_priority_level TEXT DEFAULT 'medium',
            ADD COLUMN IF NOT EXISTS rss_backfill BOOLEAN DEFAULT FALSE;
        """)
        
        # Create rss_feed_status table for feed health monitoring
        logger.info("📊 Creating rss_feed_status hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS rss_feed_status CASCADE;
            CREATE TABLE rss_feed_status (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                feed_name TEXT NOT NULL,
                feed_url TEXT NOT NULL,
                status TEXT NOT NULL,
                articles_collected INTEGER DEFAULT 0,
                articles_processed INTEGER DEFAULT 0,
                articles_duplicates INTEGER DEFAULT 0,
                response_time_ms FLOAT DEFAULT 0.0,
                error_message TEXT,
                last_successful_fetch TIMESTAMPTZ,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for rss_feed_status
        await conn.execute("""
            SELECT create_hypertable('rss_feed_status', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
            ALTER TABLE rss_feed_status 
            ADD CONSTRAINT rss_feed_status_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating RSS performance indexes...")
        
        # Indexes for raw_news_content RSS extensions
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_rss_feed 
            ON raw_news_content (rss_feed_name, timestamp DESC) 
            WHERE rss_feed_name IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_rss_category 
            ON raw_news_content (rss_category, timestamp DESC) 
            WHERE rss_category IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_rss_guid 
            ON raw_news_content (rss_guid) WHERE rss_guid IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_rss_published 
            ON raw_news_content (rss_published_at DESC) 
            WHERE rss_published_at IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_feed_credibility 
            ON raw_news_content (timestamp DESC, feed_credibility) 
            WHERE feed_credibility IS NOT NULL;
        """)
        
        # Indexes for rss_feed_status
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rss_feed_status_name_time 
            ON rss_feed_status (feed_name, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_rss_feed_status_status_time 
            ON rss_feed_status (status, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_rss_feed_status_performance 
            ON rss_feed_status (timestamp DESC, articles_collected, response_time_ms);
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up RSS compression policies...")
        await conn.execute("""
            ALTER TABLE rss_feed_status SET (timescaledb.compress, timescaledb.compress_segmentby = 'feed_name');
            SELECT add_compression_policy('rss_feed_status', INTERVAL '2 days', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up RSS retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('rss_feed_status', INTERVAL '30 days', if_not_exists => TRUE);
        """)
        
        # Verify tables were created/extended
        logger.info("✅ Verifying RSS integration...")
        
        # Check raw_news_content extensions
        rss_columns = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'raw_news_content' 
            AND column_name IN ('rss_feed_url', 'rss_category', 'rss_guid', 'feed_credibility');
        """)
        
        # Check rss_feed_status table
        feed_status_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'rss_feed_status'
            );
        """)
        
        # Check if rss_feed_status is a hypertable
        is_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'rss_feed_status'
            );
        """)
        
        logger.info(f"✅ Extended raw_news_content with {len(rss_columns)} RSS columns: {[c['column_name'] for c in rss_columns]}")
        logger.info(f"✅ Created rss_feed_status table: {feed_status_table}")
        logger.info(f"✅ rss_feed_status is hypertable: {is_hypertable}")
        
        # Create initial RSS feed status entries
        logger.info("📊 Initializing RSS feed status tracking...")
        await conn.execute("""
            INSERT INTO rss_feed_status (
                timestamp, feed_name, feed_url, status, articles_collected, 
                articles_processed, articles_duplicates, response_time_ms
            ) VALUES 
            (NOW(), 'CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/', 'initialized', 0, 0, 0, 0.0),
            (NOW(), 'CoinTelegraph', 'https://cointelegraph.com/rss', 'initialized', 0, 0, 0, 0.0),
            (NOW(), 'Binance Blog', 'https://www.binance.com/en/feed/blogs/rss', 'initialized', 0, 0, 0, 0.0),
            (NOW(), 'Ethereum Blog', 'https://blog.ethereum.org/feed.xml', 'initialized', 0, 0, 0, 0.0)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        logger.info("🎉 RSS integration migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ RSS integration migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_rss_integration_tables())
