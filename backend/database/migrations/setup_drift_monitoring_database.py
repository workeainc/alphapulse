#!/usr/bin/env python3
"""
Setup Drift Monitoring Database Tables
Extends existing ML auto-retraining system with drift detection tables
"""

import psycopg2
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

def setup_drift_monitoring_database():
    """Setup drift monitoring database tables"""
    logger.info("🔧 Setting up drift monitoring database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        logger.info("✅ Database connection established")
        
        # 1. Create ml_drift_alerts table
        logger.info("📝 Creating ml_drift_alerts table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_drift_alerts (
                alert_timestamp TIMESTAMPTZ NOT NULL,
                alert_id SERIAL,
                symbol VARCHAR(20) NOT NULL,
                regime VARCHAR(50) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                drift_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL CHECK (severity IN ('low','medium','high','critical')),
                features_affected JSONB NOT NULL,
                overall_drift_score DECIMAL(5,4) NOT NULL,
                action_required VARCHAR(50) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (alert_timestamp, alert_id)
            )
        """)
        
        # Convert to TimescaleDB hypertable
        try:
            cursor.execute("SELECT create_hypertable('ml_drift_alerts', 'alert_timestamp', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_drift_alerts to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_drift_alerts hypertable creation: {e}")
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_alerts_symbol_regime
            ON ml_drift_alerts(symbol, regime, alert_timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_alerts_severity
            ON ml_drift_alerts(severity, alert_timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_alerts_model_name
            ON ml_drift_alerts(model_name, alert_timestamp DESC)
        """)
        
        # 2. Create ml_drift_details table
        logger.info("📝 Creating ml_drift_details table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_drift_details (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                detail_id SERIAL,
                symbol VARCHAR(20) NOT NULL,
                regime VARCHAR(50) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                drift_type VARCHAR(50) NOT NULL,
                drift_score DECIMAL(5,4) NOT NULL,
                threshold DECIMAL(5,4) NOT NULL,
                is_drift BOOLEAN NOT NULL,
                p_value DECIMAL(10,8),
                reference_stats JSONB NOT NULL,
                current_stats JSONB NOT NULL,
                PRIMARY KEY (created_at, detail_id)
            )
        """)
        
        # Convert to TimescaleDB hypertable
        try:
            cursor.execute("SELECT create_hypertable('ml_drift_details', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_drift_details to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_drift_details hypertable creation: {e}")
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_details_symbol_regime
            ON ml_drift_details(symbol, regime, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_details_feature
            ON ml_drift_details(feature_name, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_details_is_drift
            ON ml_drift_details(is_drift, created_at DESC)
        """)
        
        # 3. Create ml_reference_features table (for caching reference distributions)
        logger.info("📝 Creating ml_reference_features table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_reference_features (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reference_id SERIAL,
                model_name VARCHAR(100) NOT NULL,
                regime VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_stats JSONB NOT NULL,
                sample_count INTEGER NOT NULL,
                reference_period_start TIMESTAMPTZ NOT NULL,
                reference_period_end TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (created_at, reference_id)
            )
        """)
        
        # Convert to TimescaleDB hypertable
        try:
            cursor.execute("SELECT create_hypertable('ml_reference_features', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_reference_features to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_reference_features hypertable creation: {e}")
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_reference_features_model
            ON ml_reference_features(model_name, regime, symbol, feature_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_reference_features_period
            ON ml_reference_features(reference_period_start, reference_period_end)
        """)
        
        # 4. Create ml_drift_thresholds table (for configurable thresholds)
        logger.info("📝 Creating ml_drift_thresholds table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_drift_thresholds (
                threshold_id SERIAL PRIMARY KEY,
                threshold_name VARCHAR(50) NOT NULL UNIQUE,
                threshold_value DECIMAL(5,4) NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Insert default thresholds
        default_thresholds = [
            ('ks_test_threshold', 0.15, 'Kolmogorov-Smirnov test threshold for data drift detection'),
            ('psi_threshold', 0.25, 'Population Stability Index threshold for distribution drift'),
            ('chi2_threshold', 0.05, 'Chi-square test p-value threshold for categorical drift'),
            ('mutual_info_threshold', 0.10, 'Mutual information threshold for feature drift'),
            ('critical_drift_threshold', 0.30, 'Critical drift threshold for immediate retraining'),
            ('high_drift_threshold', 0.20, 'High drift threshold for scheduled retraining'),
            ('medium_drift_threshold', 0.15, 'Medium drift threshold for monitoring')
        ]
        
        for threshold_name, threshold_value, description in default_thresholds:
            cursor.execute("""
                INSERT INTO ml_drift_thresholds (threshold_name, threshold_value, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (threshold_name) DO UPDATE SET
                    threshold_value = EXCLUDED.threshold_value,
                    description = EXCLUDED.description,
                    updated_at = NOW()
            """, (threshold_name, threshold_value, description))
        
        # 5. Create ml_drift_actions table (for tracking drift-triggered actions)
        logger.info("📝 Creating ml_drift_actions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_drift_actions (
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                action_id SERIAL,
                alert_id INTEGER,
                action_type VARCHAR(50) NOT NULL,
                action_status VARCHAR(20) NOT NULL CHECK (action_status IN ('pending','executing','completed','failed')),
                action_details JSONB,
                executed_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                PRIMARY KEY (created_at, action_id)
            )
        """)
        
        # Convert to TimescaleDB hypertable
        try:
            cursor.execute("SELECT create_hypertable('ml_drift_actions', 'created_at', if_not_exists => TRUE)")
            logger.info("✅ Converted ml_drift_actions to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ ml_drift_actions hypertable creation: {e}")
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_actions_alert
            ON ml_drift_actions(alert_id, created_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_drift_actions_status
            ON ml_drift_actions(action_status, created_at DESC)
        """)
        
        # Commit all changes
        conn.commit()
        logger.info("✅ All drift monitoring tables created successfully")
        
        # Verify table creation
        logger.info("🔍 Verifying table creation...")
        tables_to_check = [
            'ml_drift_alerts',
            'ml_drift_details', 
            'ml_reference_features',
            'ml_drift_thresholds',
            'ml_drift_actions'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ {table}: {count} records")
        
        # Show table structure
        logger.info("📊 Drift monitoring database structure:")
        for table in tables_to_check:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            logger.info(f"📋 {table}:")
            for col_name, data_type, is_nullable in columns:
                logger.info(f"   - {col_name}: {data_type} ({'NULL' if is_nullable == 'YES' else 'NOT NULL'})")
        
        conn.close()
        logger.info("✅ Drift monitoring database setup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Drift monitoring database setup failed: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_drift_monitoring_database()
