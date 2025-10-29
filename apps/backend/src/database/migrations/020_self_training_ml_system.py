#!/usr/bin/env python3
"""
Self-Training ML System Migration
Implements auto-labeling, feature engineering, and model training infrastructure
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_self_training_ml_tables():
    """Create tables for self-training ML system"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # 1. Create labels_news_market table for auto-labeling
        logger.info("🏷️ Creating labels_news_market hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS labels_news_market CASCADE;
            CREATE TABLE labels_news_market (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                publish_time TIMESTAMPTZ NOT NULL,
                
                -- Target variables (binary classification)
                y_30m BOOLEAN DEFAULT FALSE,
                y_2h BOOLEAN DEFAULT FALSE,
                y_24h BOOLEAN DEFAULT FALSE,
                
                -- Actual returns (for analysis)
                ret_30m DECIMAL(10,6),
                ret_2h DECIMAL(10,6),
                ret_24h DECIMAL(10,6),
                
                -- Price data at publish time
                price_at_publish DECIMAL(20,8),
                volume_at_publish DECIMAL(20,8),
                
                -- Labeling metadata
                labeling_method TEXT DEFAULT 'auto',
                confidence_score DECIMAL(6,4) DEFAULT 0.0,
                labeling_metadata JSONB DEFAULT '{}',
                
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for labels
        await conn.execute("""
            SELECT create_hypertable('labels_news_market', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE labels_news_market 
            ADD CONSTRAINT labels_news_market_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # 2. Create news_embeddings table for cached embeddings
        logger.info("🧠 Creating news_embeddings hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS news_embeddings CASCADE;
            CREATE TABLE news_embeddings (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_vector REAL[] NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                embedding_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for embeddings
        await conn.execute("""
            SELECT create_hypertable('news_embeddings', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '2 hours');
            ALTER TABLE news_embeddings 
            ADD CONSTRAINT news_embeddings_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # 3. Create feature_engineering_pipeline table
        logger.info("🔧 Creating feature_engineering_pipeline hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS feature_engineering_pipeline CASCADE;
            CREATE TABLE feature_engineering_pipeline (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                
                -- Text features
                title_tfidf_ngrams JSONB DEFAULT '{}',
                embedding_384d REAL[],
                entities JSONB DEFAULT '{}',
                event_tags TEXT[] DEFAULT '{}',
                
                -- Meta features
                source_trust DECIMAL(6,4) DEFAULT 0.0,
                is_breaking BOOLEAN DEFAULT FALSE,
                is_important BOOLEAN DEFAULT FALSE,
                is_hot BOOLEAN DEFAULT FALSE,
                publish_hour INTEGER,
                day_of_week INTEGER,
                dedup_cluster_size INTEGER DEFAULT 1,
                
                -- Social & on-chain context
                social_volume_zscore_30m DECIMAL(10,6),
                social_volume_zscore_neg30m DECIMAL(10,6),
                dev_activity_7d_change DECIMAL(10,6),
                whale_tx_usd_1m_plus_24h_change DECIMAL(10,6),
                
                -- Market regime controls
                btc_dominance DECIMAL(10,6),
                total_mc_zscore DECIMAL(10,6),
                asset_vol_10d DECIMAL(10,6),
                atr_14 DECIMAL(10,6),
                funding_rate DECIMAL(10,6),
                
                -- Feature engineering metadata
                feature_version TEXT DEFAULT 'v1.0',
                feature_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for feature engineering
        await conn.execute("""
            SELECT create_hypertable('feature_engineering_pipeline', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE feature_engineering_pipeline 
            ADD CONSTRAINT feature_engineering_pipeline_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # 4. Create model_training_history table
        logger.info("📊 Creating model_training_history hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS model_training_history CASCADE;
            CREATE TABLE model_training_history (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                target_variable TEXT NOT NULL,
                
                -- Training metrics
                training_samples INTEGER DEFAULT 0,
                validation_samples INTEGER DEFAULT 0,
                test_samples INTEGER DEFAULT 0,
                
                -- Performance metrics
                accuracy_score DECIMAL(6,4),
                precision_score DECIMAL(6,4),
                recall_score DECIMAL(6,4),
                f1_score DECIMAL(6,4),
                auc_score DECIMAL(6,4),
                
                -- Model details
                model_type TEXT NOT NULL,
                hyperparameters JSONB DEFAULT '{}',
                feature_importance JSONB DEFAULT '{}',
                training_duration_seconds INTEGER,
                
                -- Training metadata
                training_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for training history
        await conn.execute("""
            SELECT create_hypertable('model_training_history', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
            ALTER TABLE model_training_history 
            ADD CONSTRAINT model_training_history_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # 5. Create online_learning_buffer table
        logger.info("🔄 Creating online_learning_buffer hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS online_learning_buffer CASCADE;
            CREATE TABLE online_learning_buffer (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                
                -- Features and labels
                features JSONB NOT NULL,
                label_30m BOOLEAN,
                label_2h BOOLEAN,
                label_24h BOOLEAN,
                
                -- Prediction results
                prediction_30m DECIMAL(6,4),
                prediction_2h DECIMAL(6,4),
                prediction_24h DECIMAL(6,4),
                
                -- Online learning metadata
                buffer_status TEXT DEFAULT 'pending',
                learning_priority DECIMAL(6,4) DEFAULT 0.0,
                buffer_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for online learning buffer
        await conn.execute("""
            SELECT create_hypertable('online_learning_buffer', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');
            ALTER TABLE online_learning_buffer 
            ADD CONSTRAINT online_learning_buffer_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating self-training ML indexes...")
        
        # Indexes for labels_news_market
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_labels_news_market_news_id 
            ON labels_news_market (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_labels_news_market_symbol 
            ON labels_news_market (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_labels_news_market_targets 
            ON labels_news_market (y_30m, y_2h, y_24h, timestamp DESC);
        """)
        
        # Indexes for news_embeddings
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_embeddings_news_id 
            ON news_embeddings (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_news_embeddings_model 
            ON news_embeddings (embedding_model, timestamp DESC);
        """)
        
        # Indexes for feature_engineering_pipeline
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_news_id 
            ON feature_engineering_pipeline (news_id, symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_symbol 
            ON feature_engineering_pipeline (symbol, timestamp DESC);
        """)
        
        # Indexes for model_training_history
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_training_model_name 
            ON model_training_history (model_name, model_version, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_model_training_target 
            ON model_training_history (target_variable, timestamp DESC);
        """)
        
        # Indexes for online_learning_buffer
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_online_learning_status 
            ON online_learning_buffer (buffer_status, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_online_learning_priority 
            ON online_learning_buffer (learning_priority DESC, timestamp DESC);
        """)
        
        # Add compression policies (with error handling)
        logger.info("🗜️ Adding compression policies...")
        try:
            await conn.execute("""
                SELECT add_compression_policy('labels_news_market', INTERVAL '7 days');
            """)
        except Exception as e:
            logger.warning(f"⚠️ Could not add compression policy for labels_news_market: {e}")
        
        try:
            await conn.execute("""
                SELECT add_compression_policy('news_embeddings', INTERVAL '14 days');
            """)
        except Exception as e:
            logger.warning(f"⚠️ Could not add compression policy for news_embeddings: {e}")
        
        try:
            await conn.execute("""
                SELECT add_compression_policy('feature_engineering_pipeline', INTERVAL '7 days');
            """)
        except Exception as e:
            logger.warning(f"⚠️ Could not add compression policy for feature_engineering_pipeline: {e}")
        
        try:
            await conn.execute("""
                SELECT add_compression_policy('model_training_history', INTERVAL '30 days');
            """)
        except Exception as e:
            logger.warning(f"⚠️ Could not add compression policy for model_training_history: {e}")
        
        try:
            await conn.execute("""
                SELECT add_compression_policy('online_learning_buffer', INTERVAL '3 days');
            """)
        except Exception as e:
            logger.warning(f"⚠️ Could not add compression policy for online_learning_buffer: {e}")
        
        # Add retention policies
        logger.info("🗑️ Adding retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('labels_news_market', INTERVAL '180 days');
            SELECT add_retention_policy('news_embeddings', INTERVAL '365 days');
            SELECT add_retention_policy('feature_engineering_pipeline', INTERVAL '180 days');
            SELECT add_retention_policy('model_training_history', INTERVAL '730 days');
            SELECT add_retention_policy('online_learning_buffer', INTERVAL '30 days');
        """)
        
        logger.info("✅ Self-training ML system tables created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating self-training ML tables: {e}")
        raise
    finally:
        await conn.close()

async def main():
    """Main migration function"""
    logger.info("🚀 Starting self-training ML system migration...")
    await create_self_training_ml_tables()
    logger.info("🎉 Self-training ML system migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
