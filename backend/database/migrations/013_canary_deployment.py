"""Phase 5A: Canary Deployment Tracking Migration"""

import asyncio
import logging
import asyncpg

logger = logging.getLogger(__name__)

DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def run_canary_migration():
    """Run Phase 5A canary deployment migration"""
    
    logger.info("🚀 Starting Phase 5A: Canary Deployment Migration...")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("✅ Connected to database")
        
        # Create canary deployment events table
        await conn.execute("""
            DROP TABLE IF EXISTS canary_deployment_events CASCADE;
            
            CREATE TABLE canary_deployment_events (
                event_id SERIAL,
                model_type VARCHAR(50) NOT NULL,
                canary_model_version VARCHAR(50) NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                event_timestamp TIMESTAMPTZ NOT NULL,
                current_stage INTEGER NOT NULL,
                traffic_percentage NUMERIC(5,4) NOT NULL,
                event_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (event_id, created_at)
            );
        """)
        
        # Create canary performance metrics table
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
                accuracy NUMERIC(5,4),
                improvement_vs_baseline NUMERIC(6,4),
                stage_evaluation_result VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (metric_id, created_at)
            );
        """)
        
        # Create hypertables
        try:
            await conn.execute("SELECT create_hypertable('canary_deployment_events', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            await conn.execute("SELECT create_hypertable('canary_performance_metrics', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            logger.info("✅ Created hypertables")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation failed: {e}")
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_canary_events_model_type ON canary_deployment_events (model_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_canary_events_event_type ON canary_deployment_events (event_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_canary_metrics_model_type ON canary_performance_metrics (model_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_canary_metrics_stage ON canary_performance_metrics (current_stage);")
        
        logger.info("✅ Created indexes")
        
        await conn.close()
        logger.info("✅ Phase 5A: Canary Deployment Migration completed!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_canary_migration())
