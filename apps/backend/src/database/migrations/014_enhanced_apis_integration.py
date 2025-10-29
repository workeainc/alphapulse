#!/usr/bin/env python3
"""
Enhanced APIs Integration Migration
Extends existing tables with social metrics and correlation analysis
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_enhanced_apis_integration_tables():
    """Create enhanced APIs integration tables and extend existing ones"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Extend raw_news_content table with social metrics
        logger.info("📊 Extending raw_news_content table with social metrics...")
        await conn.execute("""
            ALTER TABLE raw_news_content 
            ADD COLUMN IF NOT EXISTS social_volume_spike BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS dev_activity_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS whale_transaction_count INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS source_guid TEXT,
            ADD COLUMN IF NOT EXISTS cryptopanic_id TEXT,
            ADD COLUMN IF NOT EXISTS social_volume_baseline FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS social_volume_current FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS correlation_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS market_impact_5m FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS market_impact_15m FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS market_impact_1h FLOAT DEFAULT 0.0;
        """)
        
        # Create social_metrics_timeseries hypertable
        logger.info("📈 Creating social_metrics_timeseries hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS social_metrics_timeseries CASCADE;
            CREATE TABLE social_metrics_timeseries (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                symbol TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value FLOAT NOT NULL,
                source TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for social_metrics_timeseries
        await conn.execute("""
            SELECT create_hypertable('social_metrics_timeseries', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE social_metrics_timeseries 
            ADD CONSTRAINT social_metrics_timeseries_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create news_market_correlation hypertable
        logger.info("📊 Creating news_market_correlation hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS news_market_correlation CASCADE;
            CREATE TABLE news_market_correlation (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                correlation_type TEXT NOT NULL,
                correlation_score FLOAT NOT NULL,
                price_impact_5m FLOAT DEFAULT 0.0,
                price_impact_15m FLOAT DEFAULT 0.0,
                price_impact_1h FLOAT DEFAULT 0.0,
                volume_impact_5m FLOAT DEFAULT 0.0,
                volume_impact_15m FLOAT DEFAULT 0.0,
                volume_impact_1h FLOAT DEFAULT 0.0,
                social_volume_correlation FLOAT DEFAULT 0.0,
                dev_activity_correlation FLOAT DEFAULT 0.0,
                whale_transaction_correlation FLOAT DEFAULT 0.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for news_market_correlation
        await conn.execute("""
            SELECT create_hypertable('news_market_correlation', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');
            ALTER TABLE news_market_correlation 
            ADD CONSTRAINT news_market_correlation_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create cryptopanic_articles hypertable for raw data
        logger.info("📰 Creating cryptopanic_articles hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS cryptopanic_articles CASCADE;
            CREATE TABLE cryptopanic_articles (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                cryptopanic_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT,
                currencies JSONB DEFAULT '[]',
                labels JSONB DEFAULT '[]',
                votes JSONB DEFAULT '{}',
                source TEXT,
                published_at TIMESTAMPTZ NOT NULL,
                raw_data JSONB DEFAULT '{}',
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for cryptopanic_articles
        await conn.execute("""
            SELECT create_hypertable('cryptopanic_articles', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE cryptopanic_articles 
            ADD CONSTRAINT cryptopanic_articles_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Add unique constraint after hypertable creation (include timestamp for TimescaleDB)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cryptopanic_id_unique 
            ON cryptopanic_articles (timestamp, cryptopanic_id);
        """)
        
        # Create indexes for performance
        logger.info("🔍 Creating performance indexes...")
        
        # Indexes for raw_news_content extensions
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_social_spike 
            ON raw_news_content (timestamp DESC, social_volume_spike) 
            WHERE social_volume_spike = TRUE;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_correlation 
            ON raw_news_content (timestamp DESC, correlation_score);
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_cryptopanic_id 
            ON raw_news_content (cryptopanic_id) WHERE cryptopanic_id IS NOT NULL;
        """)
        
        # Indexes for social_metrics_timeseries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_social_metrics_symbol_time 
            ON social_metrics_timeseries (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_social_metrics_type_time 
            ON social_metrics_timeseries (metric_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_social_metrics_value_time 
            ON social_metrics_timeseries (timestamp DESC, metric_value);
        """)
        
        # Indexes for news_market_correlation
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_correlation_news_id 
            ON news_market_correlation (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_correlation_symbol 
            ON news_market_correlation (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_correlation_score 
            ON news_market_correlation (timestamp DESC, correlation_score);
        """)
        
        # Indexes for cryptopanic_articles
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cryptopanic_processed 
            ON cryptopanic_articles (processed, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_cryptopanic_currencies 
            ON cryptopanic_articles USING GIN (currencies);
            
            CREATE INDEX IF NOT EXISTS idx_cryptopanic_labels 
            ON cryptopanic_articles USING GIN (labels);
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up compression policies...")
        await conn.execute("""
            ALTER TABLE social_metrics_timeseries SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol,metric_type');
            SELECT add_compression_policy('social_metrics_timeseries', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE news_market_correlation SET (timescaledb.compress, timescaledb.compress_segmentby = 'news_id,symbol');
            SELECT add_compression_policy('news_market_correlation', INTERVAL '6 hours', if_not_exists => TRUE);
            
            ALTER TABLE cryptopanic_articles SET (timescaledb.compress, timescaledb.compress_segmentby = 'source');
            SELECT add_compression_policy('cryptopanic_articles', INTERVAL '1 day', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('social_metrics_timeseries', INTERVAL '30 days', if_not_exists => TRUE);
            SELECT add_retention_policy('news_market_correlation', INTERVAL '7 days', if_not_exists => TRUE);
            SELECT add_retention_policy('cryptopanic_articles', INTERVAL '14 days', if_not_exists => TRUE);
        """)
        
        # Verify tables were created
        logger.info("✅ Verifying table creation...")
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name IN ('social_metrics_timeseries', 'news_market_correlation', 'cryptopanic_articles')
            AND table_schema = 'public';
        """)
        
        hypertables = await conn.fetch("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_name IN ('social_metrics_timeseries', 'news_market_correlation', 'cryptopanic_articles');
        """)
        
        logger.info(f"✅ Created {len(tables)} tables: {[t['table_name'] for t in tables]}")
        logger.info(f"✅ Created {len(hypertables)} hypertables: {[h['hypertable_name'] for h in hypertables]}")
        
        # Check raw_news_content extensions
        columns = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'raw_news_content' 
            AND column_name IN ('social_volume_spike', 'dev_activity_score', 'correlation_score');
        """)
        
        logger.info(f"✅ Extended raw_news_content with {len(columns)} new columns: {[c['column_name'] for c in columns]}")
        
        logger.info("🎉 Enhanced APIs integration migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced APIs integration migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_enhanced_apis_integration_tables())
