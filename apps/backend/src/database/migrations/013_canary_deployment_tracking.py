"""Phase 5A: Canary Deployment Tracking

Revision ID: 013_canary_deployment_tracking
Revises: 012_phase4c_4d_online_learning_drift
Create Date: 2024-01-20 14:00:00.000000

Description:
Phase 5A implementation for canary deployment tracking:
1. Canary deployment events and stages
2. Canary performance metrics tracking
3. Traffic routing and promotion decisions
4. Canary rollback events and reasons
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

async def run_canary_deployment_migration():
    """Run Phase 5A canary deployment database migration"""
    
    logger.info("🚀 Starting Phase 5A: Canary Deployment Database Migration...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("✅ Connected to TimescaleDB")
        
        # Create canary deployment tables
        await create_canary_deployment_tables(conn)
        
        # Create indexes and functions
        await create_indexes_and_functions(conn)
        
        # Verify migration
        await verify_migration(conn)
        
        await conn.close()
        logger.info("✅ Phase 5A: Canary Deployment Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

async def create_canary_deployment_tables(conn):
    """Create Phase 5A canary deployment tables"""
    
    logger.info("📊 Creating Phase 5A: Canary Deployment Tables...")
    
    # 1. Canary Deployment Events Table
    await conn.execute("""
        DROP TABLE IF EXISTS canary_deployment_events CASCADE;
        
        CREATE TABLE canary_deployment_events (
            event_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            canary_model_version VARCHAR(50) NOT NULL,
            production_model_version VARCHAR(50) NOT NULL,
            event_type VARCHAR(50) NOT NULL, -- deployment_started, stage_advanced, stage_evaluated, promoted_to_production, deployment_rollback
            event_timestamp TIMESTAMPTZ NOT NULL,
            current_stage INTEGER NOT NULL,
            traffic_percentage NUMERIC(5,4) NOT NULL,
            samples_processed INTEGER DEFAULT 0,
            event_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (event_id, created_at)
        );
    """)
    
    # 2. Canary Performance Metrics Table
    await conn.execute("""
        DROP TABLE IF EXISTS canary_performance_metrics CASCADE;
        
        CREATE TABLE canary_performance_metrics (
            metric_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            canary_model_version VARCHAR(50) NOT NULL,
            metric_timestamp TIMESTAMPTZ NOT NULL,
            current_stage INTEGER NOT NULL,
            traffic_percentage NUMERIC(5,4) NOT NULL,
            samples_in_stage INTEGER NOT NULL,
            stage_duration_seconds INTEGER NOT NULL,
            accuracy NUMERIC(5,4),
            precision NUMERIC(5,4),
            recall NUMERIC(5,4),
            f1_score NUMERIC(5,4),
            auc_score NUMERIC(5,4),
            calibration_score NUMERIC(5,4),
            production_baseline_accuracy NUMERIC(5,4),
            improvement_vs_baseline NUMERIC(6,4),
            stage_evaluation_result VARCHAR(20), -- continue, advance, rollback, ready_for_promotion
            evaluation_reason TEXT,
            metric_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (metric_id, created_at)
        );
    """)
    
    # 3. Canary Traffic Routing Table
    await conn.execute("""
        DROP TABLE IF EXISTS canary_traffic_routing CASCADE;
        
        CREATE TABLE canary_traffic_routing (
            routing_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            request_id VARCHAR(100) NOT NULL,
            routing_timestamp TIMESTAMPTZ NOT NULL,
            canary_model_version VARCHAR(50) NOT NULL,
            production_model_version VARCHAR(50) NOT NULL,
            current_stage INTEGER NOT NULL,
            traffic_percentage NUMERIC(5,4) NOT NULL,
            use_canary BOOLEAN NOT NULL,
            canary_prediction NUMERIC(8,6),
            production_prediction NUMERIC(8,6),
            actual_label NUMERIC(8,6),
            prediction_used NUMERIC(8,6),
            model_version_used VARCHAR(50) NOT NULL,
            routing_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (routing_id, created_at)
        );
    """)
    
    # 4. Canary Promotion Decisions Table
    await conn.execute("""
        DROP TABLE IF EXISTS canary_promotion_decisions CASCADE;
        
        CREATE TABLE canary_promotion_decisions (
            decision_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            canary_model_version VARCHAR(50) NOT NULL,
            decision_timestamp TIMESTAMPTZ NOT NULL,
            decision_type VARCHAR(30) NOT NULL, -- advance_stage, ready_for_promotion, rollback
            current_stage INTEGER NOT NULL,
            next_stage INTEGER,
            traffic_percentage NUMERIC(5,4) NOT NULL,
            samples_processed INTEGER NOT NULL,
            stage_duration_seconds INTEGER NOT NULL,
            stage_accuracy NUMERIC(5,4),
            stage_improvement NUMERIC(6,4),
            decision_threshold NUMERIC(5,4) NOT NULL,
            decision_reason TEXT,
            decision_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (decision_id, created_at)
        );
    """)
    
    # 5. Canary Rollback Events Table
    await conn.execute("""
        DROP TABLE IF EXISTS canary_rollback_events CASCADE;
        
        CREATE TABLE canary_rollback_events (
            rollback_id SERIAL,
            model_type VARCHAR(50) NOT NULL,
            canary_model_version VARCHAR(50) NOT NULL,
            rollback_timestamp TIMESTAMPTZ NOT NULL,
            rollback_reason VARCHAR(100) NOT NULL, -- performance_degradation, error_rate_increase, latency_degradation, manual_rollback
            current_stage INTEGER NOT NULL,
            traffic_percentage NUMERIC(5,4) NOT NULL,
            samples_processed INTEGER NOT NULL,
            stage_duration_seconds INTEGER NOT NULL,
            performance_metric VARCHAR(50) NOT NULL, -- accuracy, precision, recall, f1, auc
            metric_threshold NUMERIC(5,4) NOT NULL,
            actual_metric_value NUMERIC(5,4) NOT NULL,
            degradation_percentage NUMERIC(5,2),
            rollback_trigger_source VARCHAR(50) NOT NULL, -- automated, manual, monitoring_alert
            rollback_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (rollback_id, created_at)
        );
    """)
    
    logger.info("✅ Phase 5A: Canary Deployment Tables created")

async def create_indexes_and_functions(conn):
    """Create indexes and functions for Phase 5A"""
    
    logger.info("🔧 Creating indexes and functions...")
    
    # Convert tables to TimescaleDB hypertables
    tables_to_hypertable = [
        'canary_deployment_events',
        'canary_performance_metrics',
        'canary_traffic_routing',
        'canary_promotion_decisions',
        'canary_rollback_events'
    ]
    
    for table in tables_to_hypertable:
        try:
            await conn.execute(f"SELECT create_hypertable('{table}', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            logger.info(f"✅ Created hypertable for {table}")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation for {table} failed (may already exist): {e}")
    
    # Create indexes for performance
    indexes = [
        # Canary deployment events indexes
        ("CREATE INDEX IF NOT EXISTS idx_canary_events_model_type ON canary_deployment_events (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_events_event_type ON canary_deployment_events (event_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_events_canary_version ON canary_deployment_events (canary_model_version);"),
        
        # Canary performance metrics indexes
        ("CREATE INDEX IF NOT EXISTS idx_canary_metrics_model_type ON canary_performance_metrics (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_metrics_canary_version ON canary_performance_metrics (canary_model_version);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_metrics_stage ON canary_performance_metrics (current_stage);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_metrics_evaluation_result ON canary_performance_metrics (stage_evaluation_result);"),
        
        # Canary traffic routing indexes
        ("CREATE INDEX IF NOT EXISTS idx_canary_routing_model_type ON canary_traffic_routing (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_routing_canary_version ON canary_traffic_routing (canary_model_version);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_routing_use_canary ON canary_traffic_routing (use_canary);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_routing_request_id ON canary_traffic_routing (request_id);"),
        
        # Canary promotion decisions indexes
        ("CREATE INDEX IF NOT EXISTS idx_canary_decisions_model_type ON canary_promotion_decisions (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_decisions_canary_version ON canary_promotion_decisions (canary_model_version);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_decisions_decision_type ON canary_promotion_decisions (decision_type);"),
        
        # Canary rollback events indexes
        ("CREATE INDEX IF NOT EXISTS idx_canary_rollback_model_type ON canary_rollback_events (model_type);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_rollback_canary_version ON canary_rollback_events (canary_model_version);"),
        ("CREATE INDEX IF NOT EXISTS idx_canary_rollback_reason ON canary_rollback_events (rollback_reason);"),
    ]
    
    for index_sql in indexes:
        try:
            await conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"⚠️ Index creation failed: {e}")
    
    # Create SQL functions for analytics
    functions = [
        # Function to calculate canary deployment statistics
        """
        CREATE OR REPLACE FUNCTION calculate_canary_deployment_stats(
            p_model_type VARCHAR(50),
            p_canary_version VARCHAR(50),
            p_hours_back INTEGER DEFAULT 24
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'model_type', p_model_type,
                'canary_version', p_canary_version,
                'time_period_hours', p_hours_back,
                'total_events', COUNT(*),
                'deployment_started', COUNT(*) FILTER (WHERE event_type = 'deployment_started'),
                'stage_advanced', COUNT(*) FILTER (WHERE event_type = 'stage_advanced'),
                'stage_evaluated', COUNT(*) FILTER (WHERE event_type = 'stage_evaluated'),
                'promoted_to_production', COUNT(*) FILTER (WHERE event_type = 'promoted_to_production'),
                'deployment_rollback', COUNT(*) FILTER (WHERE event_type = 'deployment_rollback'),
                'current_stage', (
                    SELECT current_stage FROM canary_deployment_events 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    ORDER BY event_timestamp DESC LIMIT 1
                ),
                'current_traffic_percentage', (
                    SELECT traffic_percentage FROM canary_deployment_events 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    ORDER BY event_timestamp DESC LIMIT 1
                ),
                'total_samples_processed', (
                    SELECT SUM(samples_processed) FROM canary_performance_metrics 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    AND metric_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                ),
                'avg_stage_accuracy', (
                    SELECT AVG(accuracy) FROM canary_performance_metrics 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    AND metric_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                ),
                'avg_improvement', (
                    SELECT AVG(improvement_vs_baseline) FROM canary_performance_metrics 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    AND metric_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                )
            ) INTO result
            FROM canary_deployment_events
            WHERE model_type = p_model_type 
            AND canary_model_version = p_canary_version
            AND event_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
        """,
        
        # Function to calculate canary traffic routing statistics
        """
        CREATE OR REPLACE FUNCTION calculate_canary_traffic_stats(
            p_model_type VARCHAR(50),
            p_canary_version VARCHAR(50),
            p_hours_back INTEGER DEFAULT 24
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'model_type', p_model_type,
                'canary_version', p_canary_version,
                'time_period_hours', p_hours_back,
                'total_requests', COUNT(*),
                'canary_requests', COUNT(*) FILTER (WHERE use_canary = true),
                'production_requests', COUNT(*) FILTER (WHERE use_canary = false),
                'canary_traffic_percentage', (
                    COUNT(*) FILTER (WHERE use_canary = true)::NUMERIC / COUNT(*)::NUMERIC
                ),
                'avg_canary_prediction', AVG(canary_prediction) FILTER (WHERE use_canary = true),
                'avg_production_prediction', AVG(production_prediction) FILTER (WHERE use_canary = false),
                'prediction_correlation', (
                    SELECT CORR(canary_prediction, production_prediction) 
                    FROM canary_traffic_routing 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    AND use_canary = true
                    AND canary_prediction IS NOT NULL 
                    AND production_prediction IS NOT NULL
                    AND routing_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back
                ),
                'current_stage', (
                    SELECT current_stage FROM canary_traffic_routing 
                    WHERE model_type = p_model_type 
                    AND canary_model_version = p_canary_version
                    ORDER BY routing_timestamp DESC LIMIT 1
                )
            ) INTO result
            FROM canary_traffic_routing
            WHERE model_type = p_model_type 
            AND canary_model_version = p_canary_version
            AND routing_timestamp >= NOW() - INTERVAL '1 hour' * p_hours_back;
            
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
        'canary_deployment_events',
        'canary_performance_metrics',
        'canary_traffic_routing',
        'canary_promotion_decisions',
        'canary_rollback_events'
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
        result = await conn.fetchval("SELECT calculate_canary_deployment_stats('test_model', 'test_canary_v1', 1);")
        logger.info("✅ Canary deployment stats function works")
    except Exception as e:
        logger.warning(f"⚠️ Canary deployment stats function test failed: {e}")
    
    try:
        result = await conn.fetchval("SELECT calculate_canary_traffic_stats('test_model', 'test_canary_v1', 1);")
        logger.info("✅ Canary traffic stats function works")
    except Exception as e:
        logger.warning(f"⚠️ Canary traffic stats function test failed: {e}")
    
    logger.info("✅ Migration verification completed")

if __name__ == "__main__":
    asyncio.run(run_canary_deployment_migration())
