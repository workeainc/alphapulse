#!/usr/bin/env python3
"""
Market Correlation Migration
Adds market correlation analysis, feed reliability tracking, and automated classification
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_market_correlation_tables():
    """Create market correlation and analytics tables"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Create price_data hypertable for market correlation
        logger.info("💹 Creating price_data hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS price_data CASCADE;
            CREATE TABLE price_data (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                symbol TEXT NOT NULL,
                open_price DECIMAL(20,8) NOT NULL,
                high_price DECIMAL(20,8) NOT NULL,
                low_price DECIMAL(20,8) NOT NULL,
                close_price DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                price_change DECIMAL(10,6) DEFAULT 0.0,
                price_change_percent DECIMAL(8,4) DEFAULT 0.0,
                volume_change_percent DECIMAL(8,4) DEFAULT 0.0,
                data_source TEXT DEFAULT 'binance',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for price_data
        await conn.execute("""
            SELECT create_hypertable('price_data', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE price_data 
            ADD CONSTRAINT price_data_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create news_market_impact hypertable for correlation analysis
        logger.info("📊 Creating news_market_impact hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS news_market_impact CASCADE;
            CREATE TABLE news_market_impact (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                impact_window_seconds INTEGER NOT NULL,
                price_impact_percent DECIMAL(8,4) DEFAULT 0.0,
                volume_impact_percent DECIMAL(8,4) DEFAULT 0.0,
                correlation_coefficient DECIMAL(6,4) DEFAULT 0.0,
                prediction_accuracy DECIMAL(6,4) DEFAULT 0.0,
                actual_movement DECIMAL(8,4) DEFAULT 0.0,
                predicted_movement DECIMAL(8,4) DEFAULT 0.0,
                confidence_score DECIMAL(6,4) DEFAULT 0.0,
                impact_classification TEXT DEFAULT 'minor',
                market_moving BOOLEAN DEFAULT FALSE,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for news_market_impact
        await conn.execute("""
            SELECT create_hypertable('news_market_impact', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');
            ALTER TABLE news_market_impact 
            ADD CONSTRAINT news_market_impact_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create feed_reliability_scores hypertable
        logger.info("🎯 Creating feed_reliability_scores hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS feed_reliability_scores CASCADE;
            CREATE TABLE feed_reliability_scores (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                feed_name TEXT NOT NULL,
                reliability_score DECIMAL(6,4) DEFAULT 0.5,
                response_time_score DECIMAL(6,4) DEFAULT 0.5,
                success_rate_score DECIMAL(6,4) DEFAULT 0.5,
                prediction_accuracy_score DECIMAL(6,4) DEFAULT 0.5,
                article_quality_score DECIMAL(6,4) DEFAULT 0.5,
                market_moving_articles INTEGER DEFAULT 0,
                total_articles INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                total_predictions INTEGER DEFAULT 0,
                avg_response_time_ms DECIMAL(10,2) DEFAULT 0.0,
                reliability_trend TEXT DEFAULT 'stable',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for feed_reliability_scores
        await conn.execute("""
            SELECT create_hypertable('feed_reliability_scores', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE feed_reliability_scores 
            ADD CONSTRAINT feed_reliability_scores_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create news_classification_predictions hypertable
        logger.info("🤖 Creating news_classification_predictions hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS news_classification_predictions CASCADE;
            CREATE TABLE news_classification_predictions (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                predicted_impact_class TEXT NOT NULL,
                predicted_price_movement DECIMAL(8,4) DEFAULT 0.0,
                confidence_score DECIMAL(6,4) DEFAULT 0.0,
                feature_vector JSONB DEFAULT '{}',
                model_version TEXT DEFAULT 'v1.0',
                prediction_horizon_seconds INTEGER DEFAULT 3600,
                actual_impact_class TEXT,
                actual_price_movement DECIMAL(8,4),
                prediction_accuracy DECIMAL(6,4),
                learning_feedback JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for news_classification_predictions
        await conn.execute("""
            SELECT create_hypertable('news_classification_predictions', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE news_classification_predictions 
            ADD CONSTRAINT news_classification_predictions_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating market correlation indexes...")
        
        # Indexes for price_data
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_data_symbol_time 
            ON price_data (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_price_data_price_change 
            ON price_data (price_change_percent DESC, timestamp DESC);
        """)
        
        # Indexes for news_market_impact
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_news_id 
            ON news_market_impact (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_symbol 
            ON news_market_impact (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_market_impact_moving 
            ON news_market_impact (market_moving, timestamp DESC) WHERE market_moving = TRUE;
        """)
        
        # Indexes for feed_reliability_scores
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feed_reliability_feed_time 
            ON feed_reliability_scores (feed_name, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feed_reliability_score 
            ON feed_reliability_scores (reliability_score DESC, timestamp DESC);
        """)
        
        # Indexes for news_classification_predictions
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_classification_news_id 
            ON news_classification_predictions (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_classification_accuracy 
            ON news_classification_predictions (prediction_accuracy DESC, timestamp DESC) 
            WHERE prediction_accuracy IS NOT NULL;
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up market correlation compression policies...")
        await conn.execute("""
            ALTER TABLE price_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
            SELECT add_compression_policy('price_data', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE news_market_impact SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
            SELECT add_compression_policy('news_market_impact', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE feed_reliability_scores SET (timescaledb.compress, timescaledb.compress_segmentby = 'feed_name');
            SELECT add_compression_policy('feed_reliability_scores', INTERVAL '3 days', if_not_exists => TRUE);
            
            ALTER TABLE news_classification_predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'model_version');
            SELECT add_compression_policy('news_classification_predictions', INTERVAL '2 days', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up market correlation retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('price_data', INTERVAL '90 days', if_not_exists => TRUE);
            SELECT add_retention_policy('news_market_impact', INTERVAL '60 days', if_not_exists => TRUE);
            SELECT add_retention_policy('feed_reliability_scores', INTERVAL '180 days', if_not_exists => TRUE);
            SELECT add_retention_policy('news_classification_predictions', INTERVAL '90 days', if_not_exists => TRUE);
        """)
        
        # Verify tables were created
        logger.info("✅ Verifying market correlation tables...")
        
        tables_to_check = ['price_data', 'news_market_impact', 'feed_reliability_scores', 'news_classification_predictions']
        
        for table_name in tables_to_check:
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                );
            """, table_name)
            
            is_hypertable = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = $1
                );
            """, table_name)
            
            logger.info(f"✅ {table_name}: table={table_exists}, hypertable={is_hypertable}")
        
        # Initialize sample data
        logger.info("📊 Initializing sample correlation data...")
        await conn.execute("""
            INSERT INTO feed_reliability_scores (
                timestamp, feed_name, reliability_score, response_time_score, 
                success_rate_score, prediction_accuracy_score, article_quality_score
            ) VALUES 
            (NOW(), 'CoinDesk', 0.92, 0.85, 0.95, 0.88, 0.90),
            (NOW(), 'CoinTelegraph', 0.85, 0.95, 0.88, 0.82, 0.83),
            (NOW(), 'Decrypt', 0.78, 0.72, 0.92, 0.75, 0.80),
            (NOW(), 'Ethereum Blog', 0.96, 0.75, 0.99, 0.95, 0.98)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        logger.info("🎉 Market correlation migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Market correlation migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_market_correlation_tables())
