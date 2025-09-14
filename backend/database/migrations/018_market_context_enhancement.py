#!/usr/bin/env python3
"""
Market Context Enhancement Migration
Adds market context, normalized sentiment, and enhanced correlation features
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def enhance_market_context_tables():
    """Enhance existing tables with market context and correlation features"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Enhance raw_news_content with market context
        logger.info("📊 Enhancing raw_news_content with market context...")
        await conn.execute("""
            ALTER TABLE raw_news_content 
            ADD COLUMN IF NOT EXISTS market_regime TEXT DEFAULT 'neutral',
            ADD COLUMN IF NOT EXISTS btc_dominance DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS market_volatility DECIMAL(8,6) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS normalized_sentiment DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS sentiment_confidence DECIMAL(6,4) DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS market_cap_total DECIMAL(20,2) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS fear_greed_index INTEGER DEFAULT 50,
            ADD COLUMN IF NOT EXISTS correlation_30m DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS correlation_2h DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS correlation_24h DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_30m DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_2h DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_24h DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS regime_aware_score DECIMAL(6,4) DEFAULT 0.0;
        """)
        
        # Enhance news_market_impact with real-time correlation
        logger.info("📈 Enhancing news_market_impact with real-time correlation...")
        await conn.execute("""
            ALTER TABLE news_market_impact 
            ADD COLUMN IF NOT EXISTS real_time_correlation DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_30m DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_2h DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS impact_24h DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS regime_aware_score DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS volume_impact_percent DECIMAL(8,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS volatility_impact DECIMAL(8,6) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS dominance_impact DECIMAL(6,4) DEFAULT 0.0;
        """)
        
        # Enhance price_data with technical indicators
        logger.info("📊 Enhancing price_data with technical indicators...")
        await conn.execute("""
            ALTER TABLE price_data 
            ADD COLUMN IF NOT EXISTS rsi DECIMAL(6,4) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS macd DECIMAL(10,6) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS macd_signal DECIMAL(10,6) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS bollinger_upper DECIMAL(20,8) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS bollinger_lower DECIMAL(20,8) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS volume_sma DECIMAL(20,8) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS atr DECIMAL(10,6) DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS market_regime TEXT DEFAULT 'neutral',
            ADD COLUMN IF NOT EXISTS volatility_index DECIMAL(8,6) DEFAULT 0.0;
        """)
        
        # Create market_regime_data hypertable
        logger.info("📊 Creating market_regime_data hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS market_regime_data CASCADE;
            CREATE TABLE market_regime_data (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                market_regime TEXT NOT NULL,
                btc_dominance DECIMAL(6,4) DEFAULT 0.0,
                total_market_cap DECIMAL(20,2) DEFAULT 0.0,
                fear_greed_index INTEGER DEFAULT 50,
                volatility_index DECIMAL(8,6) DEFAULT 0.0,
                bull_bear_ratio DECIMAL(6,4) DEFAULT 1.0,
                altcoin_dominance DECIMAL(6,4) DEFAULT 0.0,
                defi_dominance DECIMAL(6,4) DEFAULT 0.0,
                institutional_flow DECIMAL(12,2) DEFAULT 0.0,
                retail_sentiment DECIMAL(6,4) DEFAULT 0.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for market_regime_data
        await conn.execute("""
            SELECT create_hypertable('market_regime_data', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE market_regime_data 
            ADD CONSTRAINT market_regime_data_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create sentiment_analysis_enhanced hypertable
        logger.info("🧠 Creating sentiment_analysis_enhanced hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS sentiment_analysis_enhanced CASCADE;
            CREATE TABLE sentiment_analysis_enhanced (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                source_name TEXT NOT NULL,
                raw_sentiment DECIMAL(6,4) DEFAULT 0.0,
                normalized_sentiment DECIMAL(6,4) DEFAULT 0.0,
                sentiment_confidence DECIMAL(6,4) DEFAULT 0.5,
                sentiment_label TEXT DEFAULT 'neutral',
                source_weight DECIMAL(6,4) DEFAULT 0.5,
                weighted_sentiment DECIMAL(6,4) DEFAULT 0.0,
                market_context_sentiment DECIMAL(6,4) DEFAULT 0.0,
                correlation_impact DECIMAL(6,4) DEFAULT 0.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for sentiment_analysis_enhanced
        await conn.execute("""
            SELECT create_hypertable('sentiment_analysis_enhanced', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');
            ALTER TABLE sentiment_analysis_enhanced 
            ADD CONSTRAINT sentiment_analysis_enhanced_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating enhanced market context indexes...")
        
        # Indexes for raw_news_content enhancements
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_market_regime 
            ON raw_news_content (market_regime, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_normalized_sentiment 
            ON raw_news_content (normalized_sentiment DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_correlation_30m 
            ON raw_news_content (correlation_30m DESC, timestamp DESC) WHERE correlation_30m > 0;
            
            CREATE INDEX IF NOT EXISTS idx_raw_news_content_impact_30m 
            ON raw_news_content (impact_30m DESC, timestamp DESC) WHERE impact_30m > 0;
        """)
        
        # Indexes for news_market_impact enhancements
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_real_time 
            ON news_market_impact (real_time_correlation DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_regime 
            ON news_market_impact (regime_aware_score DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_30m 
            ON news_market_impact (impact_30m DESC, timestamp DESC) WHERE impact_30m > 0;
        """)
        
        # Indexes for price_data enhancements
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_data_rsi 
            ON price_data (rsi, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_price_data_macd 
            ON price_data (macd DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_price_data_regime 
            ON price_data (market_regime, timestamp DESC);
        """)
        
        # Indexes for market_regime_data
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_regime_data_regime 
            ON market_regime_data (market_regime, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_market_regime_data_dominance 
            ON market_regime_data (btc_dominance DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_market_regime_data_fear_greed 
            ON market_regime_data (fear_greed_index, timestamp DESC);
        """)
        
        # Indexes for sentiment_analysis_enhanced
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_enhanced_news_id 
            ON sentiment_analysis_enhanced (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_sentiment_enhanced_normalized 
            ON sentiment_analysis_enhanced (normalized_sentiment DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_sentiment_enhanced_source 
            ON sentiment_analysis_enhanced (source_name, timestamp DESC);
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up enhanced compression policies...")
        await conn.execute("""
            ALTER TABLE market_regime_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'market_regime');
            SELECT add_compression_policy('market_regime_data', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE sentiment_analysis_enhanced SET (timescaledb.compress, timescaledb.compress_segmentby = 'source_name');
            SELECT add_compression_policy('sentiment_analysis_enhanced', INTERVAL '1 day', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up enhanced retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('market_regime_data', INTERVAL '90 days', if_not_exists => TRUE);
            SELECT add_retention_policy('sentiment_analysis_enhanced', INTERVAL '60 days', if_not_exists => TRUE);
        """)
        
        # Verify enhancements
        logger.info("✅ Verifying market context enhancements...")
        
        # Check raw_news_content enhancements
        enhanced_columns = await conn.fetch("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'raw_news_content' 
            AND column_name IN ('market_regime', 'normalized_sentiment', 'correlation_30m', 'impact_30m');
        """)
        
        # Check new tables
        market_regime_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'market_regime_data'
            );
        """)
        
        sentiment_enhanced_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'sentiment_analysis_enhanced'
            );
        """)
        
        # Check if new tables are hypertables
        market_regime_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'market_regime_data'
            );
        """)
        
        sentiment_enhanced_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'sentiment_analysis_enhanced'
            );
        """)
        
        logger.info(f"✅ Enhanced raw_news_content with {len(enhanced_columns)} market context columns: {[c['column_name'] for c in enhanced_columns]}")
        logger.info(f"✅ Created market_regime_data table: {market_regime_table}")
        logger.info(f"✅ Created sentiment_analysis_enhanced table: {sentiment_enhanced_table}")
        logger.info(f"✅ market_regime_data is hypertable: {market_regime_hypertable}")
        logger.info(f"✅ sentiment_analysis_enhanced is hypertable: {sentiment_enhanced_hypertable}")
        
        # Initialize sample market regime data
        logger.info("📊 Initializing sample market regime data...")
        await conn.execute("""
            INSERT INTO market_regime_data (
                timestamp, market_regime, btc_dominance, total_market_cap, 
                fear_greed_index, volatility_index, bull_bear_ratio
            ) VALUES 
            (NOW(), 'bull', 48.5, 2500000000000.00, 65, 0.025, 1.2),
            (NOW() - INTERVAL '1 hour', 'neutral', 49.2, 2480000000000.00, 55, 0.020, 1.0),
            (NOW() - INTERVAL '2 hours', 'bear', 50.1, 2450000000000.00, 35, 0.035, 0.8)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        logger.info("🎉 Market context enhancement migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Market context enhancement migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(enhance_market_context_tables())
