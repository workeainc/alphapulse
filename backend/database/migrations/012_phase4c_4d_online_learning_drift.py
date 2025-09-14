"""Phase 4C & 4D: Online Learning & Advanced Drift Detection

Revision ID: 012_phase4c_4d_online_learning_drift
Revises: 011_phase4b_ml_retraining
Create Date: 2024-01-20 12:00:00.000000

Description:
Phase 4C & 4D implementation for:
Phase 4C: Online & Safe Self-Retraining
- Shadow mode validation tracking
- Mini-batch processing logs
- Auto-rollback decision tracking
- Incremental learning performance

Phase 4D: Robust Drift & Concept-Change Detection
- ADWIN drift detection metrics
- Page-Hinkley test results
- KL-divergence distribution shifts
- Calibration drift (Brier/ECE) tracking
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

logger = logging.getLogger(__name__)

# Database connection configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def run_phase4c_4d_migration():
    """Run Phase 4C & 4D database migration"""
    
    logger.info("🚀 Starting Phase 4C & 4D Database Migration...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("✅ Connected to TimescaleDB")
        
        # Phase 4C: Online Learning & Shadow Mode Tables
        await create_online_learning_tables(conn)
        
        # Phase 4D: Advanced Drift Detection Tables
        await create_advanced_drift_tables(conn)
        
        # Create indexes and functions
        await create_indexes_and_functions(conn)
        
        # Verify migration
        await verify_migration(conn)
        
        await conn.close()
        logger.info("✅ Phase 4C & 4D Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

async def create_online_learning_tables(conn):
    """Create Phase 4C online learning tables"""
    
    logger.info("📊 Creating Phase 4C: Online Learning Tables...")
    
    # 1. Shadow Mode Validation Table
    await conn.execute("""
        DROP TABLE IF EXISTS shadow_mode_validations CASCADE;
        
        CREATE TABLE shadow_mode_validations (
            validation_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            shadow_model_version VARCHAR(50) NOT NULL,
            production_model_version VARCHAR(50) NOT NULL,
            validation_timestamp TIMESTAMPTZ NOT NULL,
            validation_period_hours INTEGER NOT NULL,
            shadow_accuracy NUMERIC(5,4),
            production_accuracy NUMERIC(5,4),
            accuracy_delta NUMERIC(5,4),
            shadow_precision NUMERIC(5,4),
            production_precision NUMERIC(5,4),
            precision_delta NUMERIC(5,4),
            shadow_recall NUMERIC(5,4),
            production_recall NUMERIC(5,4),
            recall_delta NUMERIC(5,4),
            shadow_f1 NUMERIC(5,4),
            production_f1 NUMERIC(5,4),
            f1_delta NUMERIC(5,4),
            shadow_auc NUMERIC(5,4),
            production_auc NUMERIC(5,4),
            auc_delta NUMERIC(5,4),
            shadow_calibration_score NUMERIC(5,4),
            production_calibration_score NUMERIC(5,4),
            calibration_delta NUMERIC(5,4),
            validation_threshold NUMERIC(5,4) NOT NULL,
            promotion_decision VARCHAR(20) NOT NULL, -- promote, reject, rollback
            decision_reason TEXT,
            samples_processed INTEGER NOT NULL,
            validation_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (validation_id, created_at)
        );
    """)
    
    # 2. Mini-Batch Processing Table
    await conn.execute("""
        DROP TABLE IF EXISTS mini_batch_processing CASCADE;
        
        CREATE TABLE mini_batch_processing (
            batch_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            batch_timestamp TIMESTAMPTZ NOT NULL,
            batch_size INTEGER NOT NULL,
            samples_processed INTEGER NOT NULL,
            processing_time_ms INTEGER NOT NULL,
            learning_rate NUMERIC(8,6),
            loss_before NUMERIC(8,6),
            loss_after NUMERIC(8,6),
            loss_improvement NUMERIC(8,6),
            accuracy_before NUMERIC(5,4),
            accuracy_after NUMERIC(5,4),
            accuracy_improvement NUMERIC(5,4),
            warm_start_used BOOLEAN DEFAULT FALSE,
            partial_fit_used BOOLEAN DEFAULT FALSE,
            model_updated BOOLEAN DEFAULT FALSE,
            error_occurred BOOLEAN DEFAULT FALSE,
            error_message TEXT,
            batch_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (batch_id, created_at)
        );
    """)
    
    # 3. Auto-Rollback Tracking Table
    await conn.execute("""
        DROP TABLE IF EXISTS auto_rollback_events CASCADE;
        
        CREATE TABLE auto_rollback_events (
            rollback_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            rollback_timestamp TIMESTAMPTZ NOT NULL,
            promoted_model_version VARCHAR(50) NOT NULL,
            rolled_back_model_version VARCHAR(50) NOT NULL,
            rollback_reason VARCHAR(100) NOT NULL, -- performance_degradation, drift_detected, validation_failed
            performance_metric VARCHAR(50) NOT NULL, -- accuracy, precision, recall, f1, auc
            metric_threshold NUMERIC(5,4) NOT NULL,
            actual_metric_value NUMERIC(5,4) NOT NULL,
            degradation_percentage NUMERIC(5,2),
            samples_since_promotion INTEGER,
            time_since_promotion_hours NUMERIC(8,2),
            rollback_trigger_source VARCHAR(50) NOT NULL, -- shadow_validation, drift_monitor, performance_tracker
            rollback_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (rollback_id, created_at)
        );
    """)
    
    # 4. Incremental Learning Performance Table
    await conn.execute("""
        DROP TABLE IF EXISTS incremental_learning_performance CASCADE;
        
        CREATE TABLE incremental_learning_performance (
            performance_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            performance_timestamp TIMESTAMPTZ NOT NULL,
            learning_window_hours INTEGER NOT NULL,
            samples_processed INTEGER NOT NULL,
            batches_processed INTEGER NOT NULL,
            total_learning_time_ms INTEGER NOT NULL,
            avg_batch_time_ms NUMERIC(8,2),
            initial_accuracy NUMERIC(5,4),
            final_accuracy NUMERIC(5,4),
            accuracy_improvement NUMERIC(5,4),
            initial_loss NUMERIC(8,6),
            final_loss NUMERIC(8,6),
            loss_improvement NUMERIC(8,6),
            drift_detections INTEGER DEFAULT 0,
            retraining_events INTEGER DEFAULT 0,
            shadow_validations INTEGER DEFAULT 0,
            successful_promotions INTEGER DEFAULT 0,
            rollbacks INTEGER DEFAULT 0,
            performance_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (performance_id, created_at)
        );
    """)
    
    logger.info("✅ Phase 4C: Online Learning Tables created")

async def create_advanced_drift_tables(conn):
    """Create Phase 4D advanced drift detection tables"""
    
    logger.info("📊 Creating Phase 4D: Advanced Drift Detection Tables...")
    
    # 1. ADWIN Drift Detection Table
    await conn.execute("""
        DROP TABLE IF EXISTS adwin_drift_detections CASCADE;
        
        CREATE TABLE adwin_drift_detections (
            detection_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            detection_timestamp TIMESTAMPTZ NOT NULL,
            window_size INTEGER NOT NULL,
            delta_threshold NUMERIC(5,4) NOT NULL,
            actual_delta NUMERIC(8,6) NOT NULL,
            drift_detected BOOLEAN NOT NULL,
            change_point_index INTEGER,
            left_window_mean NUMERIC(10,6),
            right_window_mean NUMERIC(10,6),
            left_window_std NUMERIC(10,6),
            right_window_std NUMERIC(10,6),
            confidence_level NUMERIC(5,4),
            p_value NUMERIC(8,6),
            samples_processed INTEGER NOT NULL,
            detection_latency_ms INTEGER,
            detection_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (detection_id, created_at)
        );
    """)
    
    # 2. Page-Hinkley Drift Detection Table
    await conn.execute("""
        DROP TABLE IF EXISTS page_hinkley_drift_detections CASCADE;
        
        CREATE TABLE page_hinkley_drift_detections (
            detection_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            detection_timestamp TIMESTAMPTZ NOT NULL,
            threshold NUMERIC(5,4) NOT NULL,
            actual_statistic NUMERIC(10,6) NOT NULL,
            drift_detected BOOLEAN NOT NULL,
            change_point_index INTEGER,
            cumulative_sum NUMERIC(12,6),
            running_mean NUMERIC(10,6),
            running_variance NUMERIC(10,6),
            confidence_level NUMERIC(5,4),
            p_value NUMERIC(8,6),
            samples_processed INTEGER NOT NULL,
            detection_latency_ms INTEGER,
            detection_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (detection_id, created_at)
        );
    """)
    
    # 3. KL-Divergence Drift Detection Table
    await conn.execute("""
        DROP TABLE IF EXISTS kl_divergence_drift_detections CASCADE;
        
        CREATE TABLE kl_divergence_drift_detections (
            detection_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            detection_timestamp TIMESTAMPTZ NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            reference_distribution JSONB NOT NULL,
            current_distribution JSONB NOT NULL,
            kl_divergence_score NUMERIC(10,6) NOT NULL,
            threshold NUMERIC(5,4) NOT NULL,
            drift_detected BOOLEAN NOT NULL,
            distribution_type VARCHAR(20) NOT NULL, -- continuous, discrete, categorical
            bin_count INTEGER,
            sample_size_reference INTEGER,
            sample_size_current INTEGER,
            confidence_level NUMERIC(5,4),
            p_value NUMERIC(8,6),
            detection_latency_ms INTEGER,
            detection_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (detection_id, created_at)
        );
    """)
    
    # 4. Calibration Drift Detection Table
    await conn.execute("""
        DROP TABLE IF EXISTS calibration_drift_detections CASCADE;
        
        CREATE TABLE calibration_drift_detections (
            detection_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            detection_timestamp TIMESTAMPTZ NOT NULL,
            brier_score_before NUMERIC(8,6),
            brier_score_after NUMERIC(8,6),
            brier_score_delta NUMERIC(8,6),
            ece_score_before NUMERIC(8,6),
            ece_score_after NUMERIC(8,6),
            ece_score_delta NUMERIC(8,6),
            calibration_threshold NUMERIC(5,4) NOT NULL,
            drift_detected BOOLEAN NOT NULL,
            confidence_bins JSONB, -- Store confidence bin statistics
            reliability_diagram JSONB, -- Store reliability diagram data
            calibration_curve JSONB, -- Store calibration curve data
            samples_processed INTEGER NOT NULL,
            detection_latency_ms INTEGER,
            detection_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (detection_id, created_at)
        );
    """)
    
    # 5. Combined Drift Metrics Table
    await conn.execute("""
        DROP TABLE IF EXISTS combined_drift_metrics CASCADE;
        
        CREATE TABLE combined_drift_metrics (
            metric_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            metric_timestamp TIMESTAMPTZ NOT NULL,
            adwin_drift_score NUMERIC(5,4),
            page_hinkley_drift_score NUMERIC(5,4),
            kl_divergence_drift_score NUMERIC(5,4),
            calibration_drift_score NUMERIC(5,4),
            combined_drift_score NUMERIC(5,4) NOT NULL,
            drift_severity VARCHAR(20) NOT NULL, -- low, medium, high, critical
            drift_confidence NUMERIC(5,4),
            features_affected JSONB,
            detection_methods_used JSONB,
            overall_status VARCHAR(20) NOT NULL, -- healthy, warning, critical
            recommendations JSONB,
            metric_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (metric_id, created_at)
        );
    """)
    
    logger.info("✅ Phase 4D: Advanced Drift Detection Tables created")

async def create_indexes_and_functions(conn):
    """Create indexes and functions for Phase 4C & 4D"""
    
    logger.info("🔧 Creating indexes and functions...")
    
    # Convert tables to TimescaleDB hypertables
    tables_to_hypertable = [
        'shadow_mode_validations',
        'mini_batch_processing', 
        'auto_rollback_events',
        'incremental_learning_performance',
        'adwin_drift_detections',
        'page_hinkley_drift_detections',
        'kl_divergence_drift_detections',
        'calibration_drift_detections',
        'combined_drift_metrics'
    ]
    
    for table in tables_to_hypertable:
        try:
            await conn.execute(f"SELECT create_hypertable('{table}', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            logger.info(f"✅ Created hypertable for {table}")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation for {table} failed (may already exist): {e}")
    
    # Create indexes for performance
    indexes = [
        # Phase 4C indexes
        ("CREATE INDEX IF NOT EXISTS idx_shadow_mode_model_type ON shadow_mode_validations (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_shadow_mode_decision ON shadow_mode_validations (promotion_decision);"),
        ("CREATE INDEX IF NOT EXISTS idx_mini_batch_model_type ON mini_batch_processing (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_mini_batch_updated ON mini_batch_processing (model_updated);"),
        ("CREATE INDEX IF NOT EXISTS idx_rollback_model_type ON auto_rollback_events (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_rollback_reason ON auto_rollback_events (rollback_reason);"),
        ("CREATE INDEX IF NOT EXISTS idx_incremental_model_type ON incremental_learning_performance (model_type);"),
        
        # Phase 4D indexes
        ("CREATE INDEX IF NOT EXISTS idx_adwin_model_type ON adwin_drift_detections (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_adwin_drift_detected ON adwin_drift_detections (drift_detected);"),
        ("CREATE INDEX IF NOT EXISTS idx_page_hinkley_model_type ON page_hinkley_drift_detections (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_page_hinkley_drift_detected ON page_hinkley_drift_detections (drift_detected);"),
        ("CREATE INDEX IF NOT EXISTS idx_kl_divergence_model_type ON kl_divergence_drift_detections (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_kl_divergence_feature ON kl_divergence_drift_detections (feature_name);"),
        ("CREATE INDEX IF NOT EXISTS idx_calibration_model_type ON calibration_drift_detections (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_calibration_drift_detected ON calibration_drift_detections (drift_detected);"),
        ("CREATE INDEX IF NOT EXISTS idx_combined_model_type ON combined_drift_metrics (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_combined_severity ON combined_drift_metrics (drift_severity);"),
        ("CREATE INDEX IF NOT EXISTS idx_combined_status ON combined_drift_metrics (overall_status);"),
    ]
    
    for index_sql in indexes:
        try:
            await conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"⚠️ Index creation failed: {e}")
    
    # Create SQL functions for analytics
    functions = [
        # Function to calculate online learning performance metrics
        """
        CREATE OR REPLACE FUNCTION calculate_online_learning_stats(
            p_model_type VARCHAR(50),
            p_hours_back INTEGER DEFAULT 24
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'model_type', p_model_type,
                'time_period_hours', p_hours_back,
                'total_batches', COUNT(*),
                'total_samples', SUM(samples_processed),
                'avg_batch_time_ms', AVG(processing_time_ms),
                'accuracy_improvement', AVG(accuracy_improvement),
                'loss_improvement', AVG(loss_improvement),
                'successful_updates', COUNT(*) FILTER (WHERE model_updated = true),
                'errors', COUNT(*) FILTER (WHERE error_occurred = true),
                'shadow_validations', (
                    SELECT COUNT(*) FROM shadow_mode_validations 
                    WHERE model_type = p_model_type 
                    AND validation_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                ),
                'successful_promotions', (
                    SELECT COUNT(*) FROM shadow_mode_validations 
                    WHERE model_type = p_model_type 
                    AND promotion_decision = 'promote'
                    AND validation_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                ),
                'rollbacks', (
                    SELECT COUNT(*) FROM auto_rollback_events 
                    WHERE model_type = p_model_type 
                    AND rollback_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                )
            ) INTO result
            FROM mini_batch_processing
            WHERE model_type = p_model_type 
            AND batch_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to calculate combined drift metrics
        """
        CREATE OR REPLACE FUNCTION calculate_combined_drift_metrics(
            p_model_type VARCHAR(50),
            p_hours_back INTEGER DEFAULT 24
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'model_type', p_model_type,
                'time_period_hours', p_hours_back,
                'adwin_detections', COUNT(*) FILTER (WHERE adwin_drift_score > 0.5),
                'page_hinkley_detections', COUNT(*) FILTER (WHERE page_hinkley_drift_score > 0.5),
                'kl_divergence_detections', COUNT(*) FILTER (WHERE kl_divergence_drift_score > 0.5),
                'calibration_detections', COUNT(*) FILTER (WHERE calibration_drift_score > 0.5),
                'avg_combined_score', AVG(combined_drift_score),
                'max_combined_score', MAX(combined_drift_score),
                'critical_drift_count', COUNT(*) FILTER (WHERE drift_severity = 'critical'),
                'high_drift_count', COUNT(*) FILTER (WHERE drift_severity = 'high'),
                'warning_drift_count', COUNT(*) FILTER (WHERE drift_severity = 'medium'),
                'healthy_count', COUNT(*) FILTER (WHERE overall_status = 'healthy'),
                'latest_status', (
                    SELECT overall_status FROM combined_drift_metrics 
                    WHERE model_type = p_model_type 
                    ORDER BY metric_timestamp DESC LIMIT 1
                ),
                'latest_severity', (
                    SELECT drift_severity FROM combined_drift_metrics 
                    WHERE model_type = p_model_type 
                    ORDER BY metric_timestamp DESC LIMIT 1
                )
            ) INTO result
            FROM combined_drift_metrics
            WHERE model_type = p_model_type 
            AND metric_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        """
    ]
    
    for function_sql in functions:
        try:
            await conn.execute(function_sql)
            logger.info("✅ Created SQL function")
        except Exception as e:
            logger.warning(f"⚠️ Function creation failed: {e}")
    
    logger.info("✅ Indexes and functions created")

async def verify_migration(conn):
    """Verify the migration was successful"""
    
    logger.info("🔍 Verifying migration...")
    
    # Check if all tables exist
    tables_to_check = [
        'shadow_mode_validations',
        'mini_batch_processing',
        'auto_rollback_events', 
        'incremental_learning_performance',
        'adwin_drift_detections',
        'page_hinkley_drift_detections',
        'kl_divergence_drift_detections',
        'calibration_drift_detections',
        'combined_drift_metrics'
    ]
    
    for table in tables_to_check:
        result = await conn.fetchval(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}');")
        if result:
            logger.info(f"✅ Table {table} exists")
        else:
            logger.error(f"❌ Table {table} missing")
            raise Exception(f"Table {table} was not created")
    
    # Test SQL functions
    try:
        result = await conn.fetchval("SELECT calculate_online_learning_stats('test_model', 1);")
        logger.info("✅ Online learning stats function works")
    except Exception as e:
        logger.warning(f"⚠️ Online learning stats function test failed: {e}")
    
    try:
        result = await conn.fetchval("SELECT calculate_combined_drift_metrics('test_model', 1);")
        logger.info("✅ Combined drift metrics function works")
    except Exception as e:
        logger.warning(f"⚠️ Combined drift metrics function test failed: {e}")
    
    logger.info("✅ Migration verification completed")

if __name__ == "__main__":
    asyncio.run(run_phase4c_4d_migration())
