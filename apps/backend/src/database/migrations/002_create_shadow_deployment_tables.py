#!/usr/bin/env python3
"""
Migration 002: Create Shadow Deployment Tables
Creates tables for shadow/canary deployment system
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def upgrade(session: AsyncSession):
    """Create shadow deployment tables"""
    
    # Create shadow_deployments table
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS shadow_deployments (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(50) UNIQUE NOT NULL,
            candidate_model_id VARCHAR(100) NOT NULL,
            production_model_id VARCHAR(100) NOT NULL,
            traffic_split DECIMAL(5,4) NOT NULL DEFAULT 0.10,
            promotion_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.70,
            min_trades INTEGER NOT NULL DEFAULT 100,
            max_trades_for_evaluation INTEGER NOT NULL DEFAULT 1000,
            evaluation_window_hours INTEGER NOT NULL DEFAULT 24,
            auto_rollback_threshold DECIMAL(5,4) NOT NULL DEFAULT 0.30,
            max_rollback_trades INTEGER NOT NULL DEFAULT 50,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """))
    
    # Create shadow_predictions table
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS shadow_predictions (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(50) UNIQUE NOT NULL,
            deployment_id VARCHAR(50),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            features JSONB NOT NULL,
            production_prediction DECIMAL(10,6) NOT NULL,
            candidate_prediction DECIMAL(10,6),
            actual_outcome DECIMAL(10,6),
            production_confidence DECIMAL(5,4) DEFAULT 0.0,
            candidate_confidence DECIMAL(5,4) DEFAULT 0.0,
            latency_ms DECIMAL(10,2) DEFAULT 0.0,
            model_versions JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """))
    
    # Create deployment_events table
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS deployment_events (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(50) NOT NULL,
            event_type VARCHAR(20) NOT NULL,
            event_data JSONB DEFAULT '{}',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """))
    
    # Create deployment_metrics table
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS deployment_metrics (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(50) UNIQUE NOT NULL,
            total_requests INTEGER DEFAULT 0,
            production_requests INTEGER DEFAULT 0,
            candidate_requests INTEGER DEFAULT 0,
            production_accuracy DECIMAL(5,4) DEFAULT 0.0,
            candidate_accuracy DECIMAL(5,4) DEFAULT 0.0,
            production_auc DECIMAL(5,4) DEFAULT 0.0,
            candidate_auc DECIMAL(5,4) DEFAULT 0.0,
            production_latency_p95 DECIMAL(10,2) DEFAULT 0.0,
            candidate_latency_p95 DECIMAL(10,2) DEFAULT 0.0,
            accuracy_improvement DECIMAL(5,4) DEFAULT 0.0,
            auc_improvement DECIMAL(5,4) DEFAULT 0.0,
            latency_improvement DECIMAL(5,4) DEFAULT 0.0,
            overall_score DECIMAL(5,4) DEFAULT 0.0,
            last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """))
    
    # Create indexes for performance
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_shadow_deployments_status 
        ON shadow_deployments(status)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_shadow_deployments_created_at 
        ON shadow_deployments(created_at)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_shadow_predictions_deployment_id 
        ON shadow_predictions(deployment_id)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_shadow_predictions_timestamp 
        ON shadow_predictions(timestamp)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_shadow_predictions_request_id 
        ON shadow_predictions(request_id)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_events_deployment_id 
        ON deployment_events(deployment_id)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_events_event_type 
        ON deployment_events(event_type)
    """))
    
    await session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_events_timestamp 
        ON deployment_events(timestamp)
    """))
    
    # If using TimescaleDB, create hypertables for time-series data
    try:
        # Create hypertable for shadow_predictions
        await session.execute(text("""
            SELECT create_hypertable('shadow_predictions', 'timestamp', 
                                   if_not_exists => TRUE)
        """))
        
        # Create hypertable for deployment_events
        await session.execute(text("""
            SELECT create_hypertable('deployment_events', 'timestamp', 
                                   if_not_exists => TRUE)
        """))
        
        # Create hypertable for deployment_metrics
        await session.execute(text("""
            SELECT create_hypertable('deployment_metrics', 'last_updated', 
                                   if_not_exists => TRUE)
        """))
        
        logger.info("✅ TimescaleDB hypertables created for shadow deployment tables")
        
    except Exception as e:
        logger.warning(f"⚠️ TimescaleDB not available, using regular tables: {e}")
    
    # Commit the migration
    await session.commit()
    
    logger.info("✅ Migration 002 completed: Shadow deployment tables created")

async def downgrade(session: AsyncSession):
    """Drop shadow deployment tables"""
    
    # Drop tables in reverse order
    await session.execute(text("DROP TABLE IF EXISTS deployment_metrics CASCADE"))
    await session.execute(text("DROP TABLE IF EXISTS deployment_events CASCADE"))
    await session.execute(text("DROP TABLE IF EXISTS shadow_predictions CASCADE"))
    await session.execute(text("DROP TABLE IF EXISTS shadow_deployments CASCADE"))
    
    # Commit the rollback
    await session.commit()
    
    logger.info("✅ Migration 002 rolled back: Shadow deployment tables dropped")
