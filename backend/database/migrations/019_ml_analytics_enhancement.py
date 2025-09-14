#!/usr/bin/env python3
"""
Machine Learning & Advanced Analytics Enhancement Migration
Adds ML prediction models, advanced correlation, and real-time alerts
"""

import asyncio
import asyncpg
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def enhance_ml_analytics_tables():
    """Enhance existing tables with ML and advanced analytics features"""
    
    # Database connection
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("🔌 Connected to database")
        
        # Create ml_predictions hypertable
        logger.info("🤖 Creating ml_predictions hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS ml_predictions CASCADE;
            CREATE TABLE ml_predictions (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_value DECIMAL(10,6) DEFAULT 0.0,
                confidence_score DECIMAL(6,4) DEFAULT 0.0,
                feature_importance JSONB DEFAULT '{}',
                model_version TEXT DEFAULT 'v1.0',
                prediction_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for ml_predictions
        await conn.execute("""
            SELECT create_hypertable('ml_predictions', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE ml_predictions 
            ADD CONSTRAINT ml_predictions_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create advanced_correlation_analysis hypertable
        logger.info("📊 Creating advanced_correlation_analysis hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS advanced_correlation_analysis CASCADE;
            CREATE TABLE advanced_correlation_analysis (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                correlation_method TEXT NOT NULL,
                time_window_seconds INTEGER NOT NULL,
                correlation_coefficient DECIMAL(8,6) DEFAULT 0.0,
                p_value DECIMAL(10,8) DEFAULT 1.0,
                significance_level DECIMAL(6,4) DEFAULT 0.05,
                price_impact_percent DECIMAL(8,4) DEFAULT 0.0,
                volume_impact_percent DECIMAL(8,4) DEFAULT 0.0,
                volatility_impact DECIMAL(8,6) DEFAULT 0.0,
                granger_causality_score DECIMAL(8,6) DEFAULT 0.0,
                analysis_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for advanced_correlation_analysis
        await conn.execute("""
            SELECT create_hypertable('advanced_correlation_analysis', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 minutes');
            ALTER TABLE advanced_correlation_analysis 
            ADD CONSTRAINT advanced_correlation_analysis_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create real_time_alerts hypertable
        logger.info("🚨 Creating real_time_alerts hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS real_time_alerts CASCADE;
            CREATE TABLE real_time_alerts (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                alert_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                news_id INTEGER,
                symbol TEXT,
                alert_message TEXT NOT NULL,
                trigger_value DECIMAL(10,6) DEFAULT 0.0,
                threshold_value DECIMAL(10,6) DEFAULT 0.0,
                confidence_score DECIMAL(6,4) DEFAULT 0.0,
                sent_to_websocket BOOLEAN DEFAULT FALSE,
                sent_to_database BOOLEAN DEFAULT TRUE,
                alert_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for real_time_alerts
        await conn.execute("""
            SELECT create_hypertable('real_time_alerts', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '15 minutes');
            ALTER TABLE real_time_alerts 
            ADD CONSTRAINT real_time_alerts_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create model_performance_tracking hypertable
        logger.info("📈 Creating model_performance_tracking hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS model_performance_tracking CASCADE;
            CREATE TABLE model_performance_tracking (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                model_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                accuracy_score DECIMAL(6,4) DEFAULT 0.0,
                precision_score DECIMAL(6,4) DEFAULT 0.0,
                recall_score DECIMAL(6,4) DEFAULT 0.0,
                f1_score DECIMAL(6,4) DEFAULT 0.0,
                auc_score DECIMAL(6,4) DEFAULT 0.0,
                training_samples INTEGER DEFAULT 0,
                validation_samples INTEGER DEFAULT 0,
                feature_importance_summary JSONB DEFAULT '{}',
                performance_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for model_performance_tracking
        await conn.execute("""
            SELECT create_hypertable('model_performance_tracking', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
            ALTER TABLE model_performance_tracking 
            ADD CONSTRAINT model_performance_tracking_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create feature_engineering_data hypertable
        logger.info("🔧 Creating feature_engineering_data hypertable...")
        await conn.execute("""
            DROP TABLE IF EXISTS feature_engineering_data CASCADE;
            CREATE TABLE feature_engineering_data (
                timestamp TIMESTAMPTZ NOT NULL,
                id SERIAL,
                news_id INTEGER NOT NULL,
                feature_category TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value DECIMAL(15,8) DEFAULT 0.0,
                feature_type TEXT NOT NULL,
                feature_metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Create hypertable for feature_engineering_data
        await conn.execute("""
            SELECT create_hypertable('feature_engineering_data', 'timestamp', 
                if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');
            ALTER TABLE feature_engineering_data 
            ADD CONSTRAINT feature_engineering_data_pkey PRIMARY KEY (timestamp, id);
        """)
        
        # Create performance indexes
        logger.info("🔍 Creating ML analytics indexes...")
        
        # Indexes for ml_predictions
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_predictions_news_id 
            ON ml_predictions (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_type 
            ON ml_predictions (model_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence 
            ON ml_predictions (confidence_score DESC, timestamp DESC) WHERE confidence_score > 0.7;
        """)
        
        # Indexes for advanced_correlation_analysis
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_advanced_correlation_news_id 
            ON advanced_correlation_analysis (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_advanced_correlation_symbol 
            ON advanced_correlation_analysis (symbol, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_advanced_correlation_coefficient 
            ON advanced_correlation_analysis (correlation_coefficient DESC, timestamp DESC) WHERE correlation_coefficient > 0.5;
        """)
        
        # Indexes for real_time_alerts
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_real_time_alerts_type 
            ON real_time_alerts (alert_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_real_time_alerts_priority 
            ON real_time_alerts (priority, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_real_time_alerts_news_id 
            ON real_time_alerts (news_id, timestamp DESC) WHERE news_id IS NOT NULL;
        """)
        
        # Indexes for model_performance_tracking
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_performance_model_type 
            ON model_performance_tracking (model_type, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_model_performance_accuracy 
            ON model_performance_tracking (accuracy_score DESC, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_model_performance_f1 
            ON model_performance_tracking (f1_score DESC, timestamp DESC);
        """)
        
        # Indexes for feature_engineering_data
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_news_id 
            ON feature_engineering_data (news_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_category 
            ON feature_engineering_data (feature_category, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_name 
            ON feature_engineering_data (feature_name, timestamp DESC);
        """)
        
        # Set up compression policies
        logger.info("🗜️ Setting up ML analytics compression policies...")
        await conn.execute("""
            ALTER TABLE ml_predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'model_type');
            SELECT add_compression_policy('ml_predictions', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE advanced_correlation_analysis SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
            SELECT add_compression_policy('advanced_correlation_analysis', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE real_time_alerts SET (timescaledb.compress, timescaledb.compress_segmentby = 'alert_type');
            SELECT add_compression_policy('real_time_alerts', INTERVAL '1 day', if_not_exists => TRUE);
            
            ALTER TABLE model_performance_tracking SET (timescaledb.compress, timescaledb.compress_segmentby = 'model_type');
            SELECT add_compression_policy('model_performance_tracking', INTERVAL '7 days', if_not_exists => TRUE);
            
            ALTER TABLE feature_engineering_data SET (timescaledb.compress, timescaledb.compress_segmentby = 'feature_category');
            SELECT add_compression_policy('feature_engineering_data', INTERVAL '1 day', if_not_exists => TRUE);
        """)
        
        # Set up retention policies
        logger.info("🗑️ Setting up ML analytics retention policies...")
        await conn.execute("""
            SELECT add_retention_policy('ml_predictions', INTERVAL '90 days', if_not_exists => TRUE);
            SELECT add_retention_policy('advanced_correlation_analysis', INTERVAL '60 days', if_not_exists => TRUE);
            SELECT add_retention_policy('real_time_alerts', INTERVAL '30 days', if_not_exists => TRUE);
            SELECT add_retention_policy('model_performance_tracking', INTERVAL '365 days', if_not_exists => TRUE);
            SELECT add_retention_policy('feature_engineering_data', INTERVAL '60 days', if_not_exists => TRUE);
        """)
        
        # Verify enhancements
        logger.info("✅ Verifying ML analytics enhancements...")
        
        # Check new tables
        ml_predictions_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'ml_predictions'
            );
        """)
        
        advanced_correlation_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'advanced_correlation_analysis'
            );
        """)
        
        real_time_alerts_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'real_time_alerts'
            );
        """)
        
        model_performance_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_performance_tracking'
            );
        """)
        
        feature_engineering_table = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'feature_engineering_data'
            );
        """)
        
        # Check if new tables are hypertables
        ml_predictions_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'ml_predictions'
            );
        """)
        
        advanced_correlation_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'advanced_correlation_analysis'
            );
        """)
        
        real_time_alerts_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'real_time_alerts'
            );
        """)
        
        model_performance_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'model_performance_tracking'
            );
        """)
        
        feature_engineering_hypertable = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'feature_engineering_data'
            );
        """)
        
        logger.info(f"✅ Created ml_predictions table: {ml_predictions_table}")
        logger.info(f"✅ Created advanced_correlation_analysis table: {advanced_correlation_table}")
        logger.info(f"✅ Created real_time_alerts table: {real_time_alerts_table}")
        logger.info(f"✅ Created model_performance_tracking table: {model_performance_table}")
        logger.info(f"✅ Created feature_engineering_data table: {feature_engineering_table}")
        
        logger.info(f"✅ ml_predictions is hypertable: {ml_predictions_hypertable}")
        logger.info(f"✅ advanced_correlation_analysis is hypertable: {advanced_correlation_hypertable}")
        logger.info(f"✅ real_time_alerts is hypertable: {real_time_alerts_hypertable}")
        logger.info(f"✅ model_performance_tracking is hypertable: {model_performance_hypertable}")
        logger.info(f"✅ feature_engineering_data is hypertable: {feature_engineering_hypertable}")
        
        # Initialize sample ML performance data
        logger.info("🤖 Initializing sample ML performance data...")
        await conn.execute("""
            INSERT INTO model_performance_tracking (
                timestamp, model_type, model_version, accuracy_score, precision_score, 
                recall_score, f1_score, auc_score, training_samples, validation_samples
            ) VALUES 
            (NOW(), 'impact_prediction', 'v1.0', 0.75, 0.72, 0.68, 0.70, 0.78, 10000, 2500),
            (NOW(), 'sentiment_enhancement', 'v1.0', 0.82, 0.79, 0.75, 0.77, 0.85, 8000, 2000),
            (NOW(), 'timing_optimization', 'v1.0', 0.68, 0.65, 0.62, 0.63, 0.71, 12000, 3000)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        # Initialize sample real-time alerts
        logger.info("🚨 Initializing sample real-time alerts...")
        await conn.execute("""
            INSERT INTO real_time_alerts (
                timestamp, alert_id, alert_type, priority, news_id, symbol, 
                alert_message, trigger_value, threshold_value, confidence_score
            ) VALUES 
            (NOW(), 'alert_001', 'high_correlation', 'high', 1, 'BTCUSDT', 
             'High correlation detected for BTC news', 0.85, 0.80, 0.90),
            (NOW(), 'alert_002', 'impact_prediction', 'critical', 2, 'ETHUSDT', 
             'High impact prediction for ETH news', 0.92, 0.75, 0.88),
            (NOW(), 'alert_003', 'regime_change', 'medium', NULL, 'BTCUSDT', 
             'Market regime change detected', 0.75, 0.70, 0.82)
            ON CONFLICT (timestamp, id) DO NOTHING;
        """)
        
        logger.info("🎉 ML analytics enhancement migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ ML analytics enhancement migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(enhance_ml_analytics_tables())
