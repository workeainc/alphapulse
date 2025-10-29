"""
Migration: Production Deployment System Tables
Creates tables for comprehensive production deployment management
"""

import asyncio
import asyncpg
from datetime import datetime

async def upgrade(connection):
    """Upgrade database schema"""
    
    # Create deployment_metrics table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS deployment_metrics (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(100) UNIQUE NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            total_services INTEGER NOT NULL DEFAULT 0,
            deployed_services INTEGER NOT NULL DEFAULT 0,
            failed_services INTEGER NOT NULL DEFAULT 0,
            health_checks_passed INTEGER NOT NULL DEFAULT 0,
            health_checks_failed INTEGER NOT NULL DEFAULT 0,
            rollback_triggered BOOLEAN NOT NULL DEFAULT FALSE,
            deployment_duration FLOAT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create service_health table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS service_health (
            id SERIAL PRIMARY KEY,
            service_name VARCHAR(100) NOT NULL,
            status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
            response_time_ms FLOAT NOT NULL,
            status_code INTEGER NOT NULL,
            last_check TIMESTAMP NOT NULL,
            error_count INTEGER NOT NULL DEFAULT 0,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            deployment_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create deployment_configs table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS deployment_configs (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(100) UNIQUE NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(20) NOT NULL CHECK (environment IN ('development', 'staging', 'production')),
            strategy VARCHAR(20) NOT NULL CHECK (strategy IN ('blue_green', 'canary', 'rolling', 'recreate')),
            services JSONB NOT NULL,
            replicas INTEGER NOT NULL DEFAULT 3,
            health_check_endpoints JSONB,
            rollback_version VARCHAR(50),
            deployment_timeout INTEGER NOT NULL DEFAULT 600,
            health_check_timeout INTEGER NOT NULL DEFAULT 60,
            max_retries INTEGER NOT NULL DEFAULT 3,
            auto_rollback BOOLEAN NOT NULL DEFAULT TRUE,
            monitoring_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            alerting_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create deployment_events table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS deployment_events (
            id SERIAL PRIMARY KEY,
            deployment_id VARCHAR(100) NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            event_data JSONB,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            severity VARCHAR(20) NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create system_health_metrics table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS system_health_metrics (
            id SERIAL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            cpu_percent FLOAT NOT NULL,
            memory_percent FLOAT NOT NULL,
            disk_percent FLOAT NOT NULL,
            network_io_in FLOAT,
            network_io_out FLOAT,
            active_connections INTEGER NOT NULL DEFAULT 0,
            total_requests INTEGER NOT NULL DEFAULT 0,
            error_rate FLOAT NOT NULL DEFAULT 0.0,
            response_time_avg FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, id)
        )
    """)
    
    # Create deployment_alerts table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS deployment_alerts (
            id SERIAL PRIMARY KEY,
            alert_type VARCHAR(50) NOT NULL,
            alert_message TEXT NOT NULL,
            deployment_id VARCHAR(100),
            severity VARCHAR(20) NOT NULL DEFAULT 'warning' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
            status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
            acknowledged_by VARCHAR(100),
            acknowledged_at TIMESTAMP,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create performance_metrics table
    await connection.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL,
            deployment_id VARCHAR(100) NOT NULL,
            service_name VARCHAR(100) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_unit VARCHAR(20),
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (timestamp, id)
        )
    """)
    
    # Create indexes for better performance
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_metrics_deployment_id 
        ON deployment_metrics(deployment_id)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_metrics_start_time 
        ON deployment_metrics(start_time)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_service_health_service_name 
        ON service_health(service_name)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_service_health_status 
        ON service_health(status)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_configs_deployment_id 
        ON deployment_configs(deployment_id)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_configs_environment 
        ON deployment_configs(environment)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_events_deployment_id 
        ON deployment_events(deployment_id)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_events_timestamp 
        ON deployment_events(timestamp)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_system_health_metrics_timestamp 
        ON system_health_metrics(timestamp)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_alerts_status 
        ON deployment_alerts(status)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_deployment_alerts_severity 
        ON deployment_alerts(severity)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_deployment_id 
        ON performance_metrics(deployment_id)
    """)
    
    await connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
        ON performance_metrics(timestamp)
    """)
    
    # Create TimescaleDB hypertables for time-series data
    try:
        await connection.execute("""
            SELECT create_hypertable('system_health_metrics', 'timestamp', 
                                   if_not_exists => TRUE)
        """)
        print("✅ TimescaleDB hypertable created for system_health_metrics")
    except Exception as e:
        print(f"⚠️ TimescaleDB hypertable creation failed for system_health_metrics: {e}")
    
    try:
        await connection.execute("""
            SELECT create_hypertable('performance_metrics', 'timestamp', 
                                   if_not_exists => TRUE)
        """)
        print("✅ TimescaleDB hypertable created for performance_metrics")
    except Exception as e:
        print(f"⚠️ TimescaleDB hypertable creation failed for performance_metrics: {e}")
    
    # Create views for common queries (only after tables are created)
    try:
        await connection.execute("""
            CREATE OR REPLACE VIEW deployment_summary AS
            SELECT 
                dm.deployment_id,
                dm.start_time,
                dm.end_time,
                dm.total_services,
                dm.deployed_services,
                dm.failed_services,
                dm.health_checks_passed,
                dm.health_checks_failed,
                dm.rollback_triggered,
                dm.deployment_duration,
                dc.environment,
                dc.strategy,
                dc.version,
                CASE 
                    WHEN dm.health_checks_failed = 0 THEN 'success'
                    WHEN dm.rollback_triggered = TRUE THEN 'rolled_back'
                    ELSE 'failed'
                END as deployment_status
            FROM deployment_metrics dm
            LEFT JOIN deployment_configs dc ON dm.deployment_id = dc.deployment_id
            ORDER BY dm.start_time DESC
        """)
        print("✅ Deployment summary view created")
    except Exception as e:
        print(f"⚠️ Deployment summary view creation failed: {e}")
    
    try:
        await connection.execute("""
            CREATE OR REPLACE VIEW service_health_summary AS
            SELECT 
                service_name,
                status,
                COUNT(*) as check_count,
                AVG(response_time_ms) as avg_response_time,
                MAX(last_check) as last_check,
                SUM(error_count) as total_errors,
                MAX(consecutive_failures) as max_consecutive_failures
            FROM service_health
            WHERE last_check >= NOW() - INTERVAL '1 hour'
            GROUP BY service_name, status
            ORDER BY service_name, last_check DESC
        """)
        print("✅ Service health summary view created")
    except Exception as e:
        print(f"⚠️ Service health summary view creation failed: {e}")
    
    try:
        await connection.execute("""
            CREATE OR REPLACE VIEW active_alerts AS
            SELECT 
                alert_type,
                alert_message,
                deployment_id,
                severity,
                created_at,
                EXTRACT(EPOCH FROM (NOW() - created_at))/3600 as hours_since_created
            FROM deployment_alerts
            WHERE status = 'active'
            ORDER BY created_at DESC
        """)
        print("✅ Active alerts view created")
    except Exception as e:
        print(f"⚠️ Active alerts view creation failed: {e}")
    
    print("✅ Production deployment system tables created successfully")

async def downgrade(connection):
    """Downgrade database schema"""
    
    # Drop views
    await connection.execute("DROP VIEW IF EXISTS deployment_summary")
    await connection.execute("DROP VIEW IF EXISTS service_health_summary")
    await connection.execute("DROP VIEW IF EXISTS active_alerts")
    
    # Drop tables
    await connection.execute("DROP TABLE IF EXISTS performance_metrics")
    await connection.execute("DROP TABLE IF EXISTS deployment_alerts")
    await connection.execute("DROP TABLE IF EXISTS system_health_metrics")
    await connection.execute("DROP TABLE IF EXISTS deployment_events")
    await connection.execute("DROP TABLE IF EXISTS deployment_configs")
    await connection.execute("DROP TABLE IF EXISTS service_health")
    await connection.execute("DROP TABLE IF EXISTS deployment_metrics")
    
    print("✅ Production deployment system tables dropped successfully")

if __name__ == "__main__":
    # Test the migration
    async def test_migration():
        # Connect to database
        connection = await asyncpg.connect(
            host="localhost",
            port=5432,
            database="alphapulse",
            user="postgres",
            password="Emon_@17711"
        )
        
        try:
            print("🔄 Running production deployment system migration...")
            await upgrade(connection)
            print("✅ Migration completed successfully")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            await downgrade(connection)
            
        finally:
            await connection.close()
    
    asyncio.run(test_migration())
