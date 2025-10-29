"""
Migration: Production Monitoring Tables
Adds tables for production monitoring, real-time metrics, and deployment tracking
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the migration"""
    logger.info("🚀 Starting migration: Production Monitoring Tables")
    
    # Database connection
    db_pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="alphapulse",
        user="alpha_emon",
        password="Emon_@17711"
    )
    
    try:
        async with db_pool.acquire() as conn:
            logger.info("✅ Database connection established")
            
            # Create real_time_metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS real_time_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    active_connections INTEGER DEFAULT 0,
                    total_connections INTEGER DEFAULT 0,
                    messages_sent INTEGER DEFAULT 0,
                    messages_received INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created real_time_metrics table")
            
            # Create deployment_history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS deployment_history (
                    id SERIAL PRIMARY KEY,
                    deployment_id VARCHAR(100) UNIQUE NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ,
                    health_checks_passed INTEGER DEFAULT 0,
                    health_checks_failed INTEGER DEFAULT 0,
                    error_message TEXT,
                    rollback_triggered BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created deployment_history table")
            
            # Create alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    alert_id VARCHAR(100) UNIQUE NOT NULL,
                    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
                    service VARCHAR(100) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created alerts table")
            
            # Create system_metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    cpu_percent DECIMAL(5,2),
                    memory_percent DECIMAL(5,2),
                    disk_percent DECIMAL(5,2),
                    network_io JSONB DEFAULT '{}',
                    active_connections INTEGER DEFAULT 0,
                    process_count INTEGER DEFAULT 0,
                    uptime_seconds DECIMAL(10,2),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created system_metrics table")
            
            # Create service_health table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
                    response_time_ms DECIMAL(10,2),
                    error_rate DECIMAL(5,4),
                    last_check TIMESTAMPTZ NOT NULL,
                    error_message TEXT,
                    metrics JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created service_health table")
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_real_time_metrics_timestamp 
                ON real_time_metrics (timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deployment_history_status 
                ON deployment_history (status)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deployment_history_start_time 
                ON deployment_history (start_time)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity_timestamp 
                ON alerts (severity, timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_service_timestamp 
                ON alerts (service, timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                ON system_metrics (timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_service_health_service_status 
                ON service_health (service_name, status)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_service_health_last_check 
                ON service_health (last_check)
            """)
            
            logger.info("✅ Created performance indexes")
            
            # Create TimescaleDB hypertables if available
            try:
                await conn.execute("""
                    SELECT create_hypertable('real_time_metrics', 'timestamp', 
                        if_not_exists => TRUE)
                """)
                logger.info("✅ Created TimescaleDB hypertable for real_time_metrics")
            except Exception as e:
                logger.warning(f"⚠️ TimescaleDB hypertable creation failed: {e}")
            
            try:
                await conn.execute("""
                    SELECT create_hypertable('system_metrics', 'timestamp', 
                        if_not_exists => TRUE)
                """)
                logger.info("✅ Created TimescaleDB hypertable for system_metrics")
            except Exception as e:
                logger.warning(f"⚠️ TimescaleDB hypertable creation failed: {e}")
            
            # Insert sample data for testing
            await conn.execute("""
                INSERT INTO real_time_metrics (timestamp, active_connections, total_connections, messages_sent, messages_received, errors)
                VALUES (NOW(), 0, 0, 0, 0, 0)
                ON CONFLICT DO NOTHING
            """)
            
            await conn.execute("""
                INSERT INTO system_metrics (timestamp, cpu_percent, memory_percent, disk_percent, active_connections, process_count, uptime_seconds)
                VALUES (NOW(), 0.0, 0.0, 0.0, 0, 0, 0.0)
                ON CONFLICT DO NOTHING
            """)
            
            logger.info("✅ Inserted sample data")
            
            logger.info("🎉 Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
