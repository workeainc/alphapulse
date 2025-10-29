#!/usr/bin/env python3
"""
Migration: Phase 7 ML Enhancements
Extend existing tables and add ML-specific infrastructure
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_phase7_ml_enhancements():
    """Create Phase 7 ML enhancements"""
    
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("🔗 Connected to database for Phase 7 ML enhancements")
        
        # Extend existing volume_analysis_ml_dataset table with comprehensive features
        extend_ml_dataset = """
        ALTER TABLE volume_analysis_ml_dataset 
        ADD COLUMN IF NOT EXISTS technical_features JSONB,
        ADD COLUMN IF NOT EXISTS order_book_features JSONB,
        ADD COLUMN IF NOT EXISTS time_features JSONB,
        ADD COLUMN IF NOT EXISTS multi_timeframe_features JSONB,
        ADD COLUMN IF NOT EXISTS market_regime VARCHAR(20),
        ADD COLUMN IF NOT EXISTS volatility_regime VARCHAR(20),
        ADD COLUMN IF NOT EXISTS feature_importance JSONB;
        """
        
        # Create model predictions table
        model_predictions_table = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            prediction_type VARCHAR(30) NOT NULL, -- 'breakout', 'return', 'anomaly'
            prediction_value DECIMAL(8,6) NOT NULL,
            confidence_score DECIMAL(3,2) NOT NULL,
            feature_contributions JSONB,
            shap_values JSONB,
            model_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Create model performance tracking table
        model_performance_table = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL,
            model_version VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            metric_name VARCHAR(30) NOT NULL, -- 'auc', 'precision', 'recall', 'sharpe'
            metric_value DECIMAL(8,6) NOT NULL,
            sample_size INTEGER,
            evaluation_window VARCHAR(20), -- '1h', '1d', '1w'
            performance_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Create model versions registry
        model_versions_table = """
        CREATE TABLE IF NOT EXISTS model_versions (
            id SERIAL,
            model_name VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            model_type VARCHAR(30) NOT NULL, -- 'lightgbm', 'xgboost', 'lstm'
            training_start TIMESTAMPTZ NOT NULL,
            training_end TIMESTAMPTZ NOT NULL,
            training_samples INTEGER NOT NULL,
            validation_auc DECIMAL(6,4),
            validation_precision DECIMAL(6,4),
            validation_recall DECIMAL(6,4),
            model_path VARCHAR(255),
            feature_list JSONB,
            hyperparameters JSONB,
            is_active BOOLEAN DEFAULT FALSE,
            is_production BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (model_name, version)
        );
        """
        
        # Create feature importance tracking
        feature_importance_table = """
        CREATE TABLE IF NOT EXISTS feature_importance (
            id SERIAL,
            model_version VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            feature_name VARCHAR(50) NOT NULL,
            importance_score DECIMAL(8,6) NOT NULL,
            shap_mean_value DECIMAL(8,6),
            shap_std_value DECIMAL(8,6),
            feature_category VARCHAR(30), -- 'volume', 'price', 'technical', 'orderbook'
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (model_version, feature_name)
        );
        """
        
        # Create labels table for supervised learning
        labels_table = """
        CREATE TABLE IF NOT EXISTS ml_labels (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            label_type VARCHAR(30) NOT NULL, -- 'binary_breakout', 'regression_return', 'multiclass_direction'
            label_value DECIMAL(8,6) NOT NULL,
            label_metadata JSONB, -- contains TP/SL levels, time windows, etc.
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Execute table creation
        commands = [
            ("Extend ML dataset", extend_ml_dataset),
            ("Model predictions", model_predictions_table),
            ("Model performance", model_performance_table),
            ("Model versions", model_versions_table),
            ("Feature importance", feature_importance_table),
            ("ML labels", labels_table)
        ]
        
        for name, command in commands:
            try:
                logger.info(f"Creating {name}...")
                await conn.execute(command)
                logger.info(f"✅ {name} created successfully")
            except Exception as e:
                logger.error(f"❌ Error creating {name}: {e}")
                continue
        
        # Create TimescaleDB hypertables
        hypertable_commands = [
            "SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('model_performance', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('ml_labels', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
        ]
        
        for i, command in enumerate(hypertable_commands, 1):
            try:
                logger.info(f"Creating ML hypertable {i}/{len(hypertable_commands)}...")
                await conn.execute(command)
                logger.info(f"✅ ML hypertable {i} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ ML hypertable {i} creation warning: {e}")
                continue
        
        # Create indexes
        index_commands = [
            "CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_model ON model_predictions (symbol, model_version, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_model_predictions_type ON model_predictions (prediction_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_model_performance_version ON model_performance (model_version, metric_name, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions (is_active, is_production, created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_feature_importance_version ON feature_importance (model_version, importance_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_ml_labels_symbol_type ON ml_labels (symbol, label_type, timestamp DESC);"
        ]
        
        for i, command in enumerate(index_commands, 1):
            try:
                logger.info(f"Creating ML index {i}/{len(index_commands)}...")
                await conn.execute(command)
                logger.info(f"✅ ML index {i} created successfully")
            except Exception as e:
                logger.warning(f"⚠️ ML index {i} creation warning: {e}")
                continue
        
        # Create materialized views for ML monitoring
        await create_ml_materialized_views(conn)
        
        logger.info("✅ Phase 7 ML enhancements completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 7 ML enhancements failed: {e}")
        return False
    finally:
        if conn:
            await conn.close()

async def create_ml_materialized_views(conn):
    """Create materialized views for ML monitoring"""
    try:
        logger.info("🔍 Creating ML materialized views...")
        
        # Recent model predictions view
        recent_predictions_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS recent_model_predictions AS
        SELECT 
            symbol,
            timeframe,
            model_version,
            prediction_type,
            prediction_value,
            confidence_score,
            timestamp
        FROM model_predictions
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        ORDER BY timestamp DESC;
        """
        
        # Model performance summary view
        model_performance_summary_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_summary AS
        SELECT 
            model_version,
            symbol,
            metric_name,
            AVG(metric_value) as avg_metric_value,
            COUNT(*) as evaluation_count,
            MAX(timestamp) as last_evaluation
        FROM model_performance
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY model_version, symbol, metric_name
        ORDER BY model_version, metric_name;
        """
        
        # Active model versions view
        active_models_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS active_model_versions AS
        SELECT 
            model_name,
            version,
            model_type,
            validation_auc,
            validation_precision,
            validation_recall,
            is_active,
            is_production,
            created_at
        FROM model_versions
        WHERE is_active = TRUE
        ORDER BY created_at DESC;
        """
        
        # Feature importance summary view
        feature_importance_summary_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS feature_importance_summary AS
        SELECT 
            model_version,
            feature_category,
            feature_name,
            importance_score,
            shap_mean_value
        FROM feature_importance
        WHERE importance_score > 0.01
        ORDER BY model_version, importance_score DESC;
        """
        
        views = [
            ("Recent model predictions", recent_predictions_view),
            ("Model performance summary", model_performance_summary_view),
            ("Active model versions", active_models_view),
            ("Feature importance summary", feature_importance_summary_view)
        ]
        
        for name, command in views:
            try:
                logger.info(f"Creating {name} view...")
                await conn.execute(command)
                logger.info(f"✅ {name} view created successfully")
            except Exception as e:
                logger.warning(f"⚠️ {name} view creation warning: {e}")
                continue
        
        logger.info("✅ ML materialized views created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating ML materialized views: {e}")

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Phase 7 ML Enhancements Migration...")
    
    try:
        success = await create_phase7_ml_enhancements()
        if success:
            logger.info("✅ Phase 7 ML Enhancements Migration completed successfully!")
        else:
            logger.error("❌ Migration failed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
