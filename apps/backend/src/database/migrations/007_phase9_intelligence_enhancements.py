#!/usr/bin/env python3
"""
Migration: Phase 9 Intelligence Enhancements
Auto-retraining pipeline, market regime detection, and explainability layer
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

async def create_phase9_intelligence_enhancements():
    """Create Phase 9 intelligence enhancements"""
    
    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("🔗 Connected to database for Phase 9 Intelligence Enhancements")
        
        # 1. Auto-Retraining Pipeline Tables
        
        # Model retraining history
        model_retraining_history = """
        CREATE TABLE IF NOT EXISTS model_retraining_history (
            id SERIAL,
            model_name VARCHAR(50) NOT NULL,
            retraining_trigger VARCHAR(30) NOT NULL, -- 'scheduled', 'drift_detected', 'performance_degradation'
            retraining_start TIMESTAMPTZ NOT NULL,
            retraining_end TIMESTAMPTZ,
            training_samples INTEGER,
            validation_samples INTEGER,
            old_model_version VARCHAR(20),
            new_model_version VARCHAR(20),
            performance_improvement DECIMAL(6,4),
            drift_metrics JSONB,
            retraining_metadata JSONB,
            status VARCHAR(20) DEFAULT 'in_progress', -- 'in_progress', 'completed', 'failed'
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (id)
        );
        """
        
        # Data drift metrics
        data_drift_metrics = """
        CREATE TABLE IF NOT EXISTS data_drift_metrics (
            id SERIAL,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            drift_type VARCHAR(30) NOT NULL, -- 'psi', 'kl_divergence', 'statistical'
            feature_name VARCHAR(50),
            drift_score DECIMAL(8,6) NOT NULL,
            threshold DECIMAL(8,6) NOT NULL,
            is_drift_detected BOOLEAN NOT NULL,
            drift_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # 2. Market Regime Detection Tables
        
        # Market regimes
        market_regimes = """
        CREATE TABLE IF NOT EXISTS market_regimes (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            regime_type VARCHAR(30) NOT NULL, -- 'trending', 'ranging', 'high_volatility', 'low_volatility'
            volatility_regime VARCHAR(30) NOT NULL, -- 'high', 'medium', 'low'
            liquidity_regime VARCHAR(30) NOT NULL, -- 'high', 'medium', 'low'
            regime_confidence DECIMAL(3,2) NOT NULL,
            regime_features JSONB,
            regime_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Regime-specific thresholds
        regime_thresholds = """
        CREATE TABLE IF NOT EXISTS regime_thresholds (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            regime_type VARCHAR(30) NOT NULL,
            threshold_type VARCHAR(50) NOT NULL, -- 'volume_spike', 'breakout', 'anomaly'
            threshold_value DECIMAL(8,6) NOT NULL,
            confidence_level DECIMAL(3,2) NOT NULL,
            sample_size INTEGER,
            last_updated TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (symbol, regime_type, threshold_type)
        );
        """
        
        # 3. Explainability Layer Tables
        
        # Trade explanations
        trade_explanations = """
        CREATE TABLE IF NOT EXISTS trade_explanations (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            decision_type VARCHAR(30) NOT NULL, -- 'volume_signal', 'ml_prediction', 'rl_action', 'anomaly_alert'
            decision_value VARCHAR(50) NOT NULL,
            confidence_score DECIMAL(3,2) NOT NULL,
            explanation_text TEXT,
            feature_contributions JSONB,
            shap_values JSONB,
            contributing_factors JSONB,
            explanation_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Feature importance history
        feature_importance_history = """
        CREATE TABLE IF NOT EXISTS feature_importance_history (
            id SERIAL,
            model_name VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            feature_name VARCHAR(50) NOT NULL,
            importance_score DECIMAL(8,6) NOT NULL,
            importance_rank INTEGER,
            importance_change DECIMAL(8,6), -- change from previous period
            feature_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # 4. Execute all table creations
        tables = [
            ("Model Retraining History", model_retraining_history),
            ("Data Drift Metrics", data_drift_metrics),
            ("Market Regimes", market_regimes),
            ("Regime Thresholds", regime_thresholds),
            ("Trade Explanations", trade_explanations),
            ("Feature Importance History", feature_importance_history)
        ]
        
        for table_name, table_sql in tables:
            try:
                await conn.execute(table_sql)
                logger.info(f"✅ Created {table_name} table")
            except Exception as e:
                logger.warning(f"⚠️ {table_name} table creation warning: {e}")
        
        # 5. Create TimescaleDB hypertables
        hypertables = [
            "SELECT create_hypertable('model_retraining_history', 'retraining_start', if_not_exists => TRUE);",
            "SELECT create_hypertable('data_drift_metrics', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('market_regimes', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('trade_explanations', 'timestamp', if_not_exists => TRUE);",
            "SELECT create_hypertable('feature_importance_history', 'timestamp', if_not_exists => TRUE);"
        ]
        
        for hypertable_sql in hypertables:
            try:
                await conn.execute(hypertable_sql)
            except Exception as e:
                logger.warning(f"⚠️ Hypertable creation warning: {e}")
        
        logger.info("✅ Created TimescaleDB hypertables")
        
        # 6. Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_model_retraining_model_name ON model_retraining_history (model_name, retraining_start DESC);",
            "CREATE INDEX IF NOT EXISTS idx_model_retraining_status ON model_retraining_history (status, retraining_start DESC);",
            "CREATE INDEX IF NOT EXISTS idx_data_drift_model_symbol ON data_drift_metrics (model_name, symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_data_drift_detected ON data_drift_metrics (is_drift_detected, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_regimes_symbol ON market_regimes (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_regimes_type ON market_regimes (regime_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trade_explanations_symbol ON trade_explanations (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trade_explanations_type ON trade_explanations (decision_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_feature_importance_model ON feature_importance_history (model_name, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_feature_importance_rank ON feature_importance_history (importance_rank, timestamp DESC);"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
        
        logger.info("✅ Created performance indexes")
        
        # 7. Create materialized views for real-time monitoring
        materialized_views = [
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS current_market_regime AS
            SELECT 
                symbol,
                timeframe,
                regime_type,
                volatility_regime,
                liquidity_regime,
                regime_confidence,
                timestamp
            FROM market_regimes
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC;
            """,
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS recent_drift_alerts AS
            SELECT 
                model_name,
                symbol,
                timeframe,
                drift_type,
                feature_name,
                drift_score,
                is_drift_detected,
                timestamp
            FROM data_drift_metrics
            WHERE is_drift_detected = TRUE 
            AND timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC;
            """,
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS explainability_summary AS
            SELECT 
                symbol,
                timeframe,
                decision_type,
                COUNT(*) as decision_count,
                AVG(confidence_score) as avg_confidence,
                MAX(timestamp) as latest_decision
            FROM trade_explanations
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            GROUP BY symbol, timeframe, decision_type
            ORDER BY latest_decision DESC;
            """
        ]
        
        for view_sql in materialized_views:
            try:
                await conn.execute(view_sql)
            except Exception as e:
                logger.warning(f"⚠️ Materialized view creation warning: {e}")
        
        logger.info("✅ Created materialized views")
        
        # 8. Enable compression for older data
        compression_commands = [
            "SELECT add_compression_policy('model_retraining_history', INTERVAL '7 days');",
            "SELECT add_compression_policy('data_drift_metrics', INTERVAL '7 days');",
            "SELECT add_compression_policy('market_regimes', INTERVAL '7 days');",
            "SELECT add_compression_policy('trade_explanations', INTERVAL '7 days');",
            "SELECT add_compression_policy('feature_importance_history', INTERVAL '7 days');"
        ]
        
        for compression_sql in compression_commands:
            try:
                await conn.execute(compression_sql)
            except Exception as e:
                logger.warning(f"⚠️ Compression policy warning: {e}")
        
        logger.info("✅ Enabled TimescaleDB compression")
        
        logger.info("🎉 Phase 9 Intelligence Enhancements migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 9 migration: {e}")
        raise
    finally:
        if conn:
            await conn.close()

if __name__ == "__main__":
    asyncio.run(create_phase9_intelligence_enhancements())
