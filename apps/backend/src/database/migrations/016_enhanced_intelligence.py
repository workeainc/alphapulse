#!/usr/bin/env python3
"""
Enhanced Intelligence Migration
Adds entity recognition, latency tracking, and cross-source validation features
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_enhanced_intelligence_tables():
    """Add enhanced intelligence features to existing tables"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Extend raw_news_content with enhanced intelligence columns
        logger.info("🧠 Extending raw_news_content with intelligence features...")
        await conn.execute("""
            ALTER TABLE raw_news_content 
            ADD COLUMN IF NOT EXISTS entities JSONB DEFAULT '[]',
            ADD COLUMN IF NOT EXISTS event_types JSONB DEFAULT '[]',
            ADD COLUMN IF NOT EXISTS entity_confidence FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS publish_latency_ms FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS cross_source_validation BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS validation_sources JSONB DEFAULT '[]',
            ADD COLUMN IF NOT EXISTS similarity_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS dup_group_id TEXT,
            ADD COLUMN IF NOT EXISTS market_impact_prediction FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS feed_performance_score FLOAT DEFAULT 0.5;
        """)
        
        # Create feed_performance_metrics hypertable for advanced monitoring
        logger.info("📊 Creating feed_performance_metrics hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS feed_performance_metrics CASCADE;
            CREATE TABLE feed_performance_metrics (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                feed_name TEXT NOT NULL,
                feed_url TEXT NOT NULL,
                avg_latency_ms FLOAT DEFAULT 0.0,
                success_rate FLOAT DEFAULT 0.0,
                articles_per_hour FLOAT DEFAULT 0.0,
                avg_entity_count FLOAT DEFAULT 0.0,
                avg_impact_score FLOAT DEFAULT 0.0,
                reliability_score FLOAT DEFAULT 0.5,
                last_24h_articles INTEGER DEFAULT 0,
                last_24h_errors INTEGER DEFAULT 0,
                performance_trend TEXT DEFAULT 'stable',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for feed_performance_metrics
        await conn.execute("""
            SELECT create_hypertable('feed_performance_metrics', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
            ALTER TABLE feed_performance_metrics 
            ADD CONSTRAINT feed_performance_metrics_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create news_entity_correlation hypertable for entity analysis
        logger.info("🔗 Creating news_entity_correlation hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS news_entity_correlation CASCADE;
            CREATE TABLE news_entity_correlation (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                news_count INTEGER DEFAULT 0,
                avg_sentiment FLOAT DEFAULT 0.0,
                avg_impact_score FLOAT DEFAULT 0.0,
                price_correlation_5m FLOAT DEFAULT 0.0,
                price_correlation_15m FLOAT DEFAULT 0.0,
                price_correlation_1h FLOAT DEFAULT 0.0,
                volume_correlation_5m FLOAT DEFAULT 0.0,
                volume_correlation_15m FLOAT DEFAULT 0.0,
                volume_correlation_1h FLOAT DEFAULT 0.0,
                market_moving_events INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for news_entity_correlation
        await conn.execute("""
            SELECT create_hypertable('news_entity_correlation', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE news_entity_correlation 
            ADD CONSTRAINT news_entity_correlation_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating enhanced intelligence indexes...")
        
        # Indexes for raw_news_content enhancements
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_entities 
            ON raw_news_content USING GIN (entities) WHERE entities IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_event_types 
            ON raw_news_content USING GIN (event_types) WHERE event_types IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_latency 
            ON raw_news_content (publish_latency_ms DESC) WHERE publish_latency_ms > 0;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_validation 
            ON raw_news_content (cross_source_validation, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_dup_group 
            ON raw_news_content (dup_group_id) WHERE dup_group_id IS NOT NULL;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_performance 
            ON raw_news_content (feed_performance_score DESC, timestamp DESC);
        """)
        
        # Indexes for feed_performance_metrics
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feed_performance_name_time 
            ON feed_performance_metrics (feed_name, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feed_performance_reliability 
            ON feed_performance_metrics (reliability_score DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feed_performance_latency 
            ON feed_performance_metrics (avg_latency_ms, timestamp DESC);
        """)
        
        # Indexes for news_entity_correlation
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_correlation_name 
            ON news_entity_correlation (entity_name, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_entity_correlation_type 
            ON news_entity_correlation (entity_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_entity_correlation_impact 
            ON news_entity_correlation (avg_impact_score DESC, timestamp DESC);
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up enhanced intelligence compression policies...")
        await conn.execute("""
            ALTER TABLE feed_performance_metrics SET (timescaledb.compress, timescaledb.compress_segmentby = 'feed_name');
            SELECT add_compression_policy('feed_performance_metrics', INTERVAL '3 days', if_not_exists => TRUE);
            
            ALTER TABLE news_entity_correlation SET (timescaledb.compress, timescaledb.compress_segmentby = 'entity_name');
            SELECT add_compression_policy('news_entity_correlation', INTERVAL '1 day', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up enhanced intelligence retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('feed_performance_metrics', INTERVAL '60 days', if_not_exists => TRUE);
            SELECT add_retention_policy('news_entity_correlation', INTERVAL '30 days', if_not_exists => TRUE);
        """)
        
        # Verify tables were created/extended
        logger.info("✅ Verifying enhanced intelligence features...")
        
        # Check raw_news_content extensions
        enhanced_columns = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'raw_news_content' 
            AND column_name IN ('entities', 'event_types', 'publish_latency_ms', 'cross_source_validation');
        """)
        
        # Check new tables
        feed_performance_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'feed_performance_metrics'
            );
        """)
        
        entity_correlation_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'news_entity_correlation'
            );
        """)
        
        # Check if new tables are hypertables
        feed_performance_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'feed_performance_metrics'
            );
        """)
        
        entity_correlation_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'news_entity_correlation'
            );
        """)
        
        logger.info(f"✅ Extended raw_news_content with {len(enhanced_columns)} intelligence columns: {[c['column_name'] for c in enhanced_columns]}")
        logger.info(f"✅ Created feed_performance_metrics table: {feed_performance_table}")
        logger.info(f"✅ Created news_entity_correlation table: {entity_correlation_table}")
        logger.info(f"✅ feed_performance_metrics is hypertable: {feed_performance_hypertable}")
        logger.info(f"✅ news_entity_correlation is hypertable: {entity_correlation_hypertable}")
        
        # Initialize feed performance tracking
        logger.info("📊 Initializing feed performance tracking...")
        await conn.execute("""
            INSERT INTO feed_performance_metrics (
                timestamp, feed_name, feed_url, avg_latency_ms, success_rate, 
                articles_per_hour, avg_entity_count, avg_impact_score, reliability_score
            ) VALUES 
            (NOW(), 'CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/', 430.0, 0.95, 25.0, 3.2, 0.7, 0.9),
            (NOW(), 'CoinTelegraph', 'https://cointelegraph.com/rss', 110.0, 0.98, 30.0, 4.1, 0.6, 0.8),
            (NOW(), 'Decrypt', 'https://decrypt.co/feed', 690.0, 0.92, 27.0, 2.8, 0.5, 0.7),
            (NOW(), 'Ethereum Blog', 'https://blog.ethereum.org/feed.xml', 630.0, 0.99, 1.0, 5.5, 0.9, 0.95)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        logger.info("🎉 Enhanced intelligence migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced intelligence migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_enhanced_intelligence_tables())
