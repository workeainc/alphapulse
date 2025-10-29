"""
Migration 054: Production Monitoring & Deployment System (Phase 10)
Implements comprehensive production monitoring tables and metrics
"""

import asyncio
import asyncpg
import logging
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Execute Phase 10 migration for Production Monitoring & Deployment System"""
    
    # Database connection parameters
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Create tables for Phase 10
        
        # 1. Production Metrics Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS production_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                cpu_percent DECIMAL(5,2) NOT NULL,
                memory_percent DECIMAL(5,2) NOT NULL,
                disk_percent DECIMAL(5,2) NOT NULL,
                network_io JSONB NOT NULL,
                active_connections INTEGER NOT NULL,
                process_count INTEGER NOT NULL,
                uptime_seconds DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created production_metrics table")
        
        # 2. Service Health Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS service_health (
                id SERIAL PRIMARY KEY,
                service_name VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                response_time_ms DECIMAL(8,2) NOT NULL,
                error_rate DECIMAL(5,4) NOT NULL,
                last_check TIMESTAMP WITH TIME ZONE NOT NULL,
                error_message TEXT,
                metrics JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created service_health table")
        
        # 3. System Alerts Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_alerts (
                id SERIAL PRIMARY KEY,
                alert_id VARCHAR(100) UNIQUE NOT NULL,
                severity VARCHAR(20) NOT NULL,
                service VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
                resolved BOOLEAN NOT NULL DEFAULT FALSE,
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created system_alerts table")
        
        # 4. Deployment Status Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deployment_status (
                id SERIAL PRIMARY KEY,
                deployment_id VARCHAR(100) UNIQUE NOT NULL,
                version VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                end_time TIMESTAMP WITH TIME ZONE,
                health_checks_passed INTEGER NOT NULL DEFAULT 0,
                health_checks_failed INTEGER NOT NULL DEFAULT 0,
                services TEXT[] NOT NULL DEFAULT '{}',
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created deployment_status table")
        
        # 5. Performance Analytics Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_analytics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(10,4) NOT NULL,
                metric_unit VARCHAR(20),
                period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                aggregation_type VARCHAR(20) NOT NULL, -- 'avg', 'max', 'min', 'sum', 'count'
                service_name VARCHAR(50),
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created performance_analytics table")
        
        # 6. Monitoring Configuration Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_config (
                id SERIAL PRIMARY KEY,
                config_key VARCHAR(100) UNIQUE NOT NULL,
                config_value JSONB NOT NULL,
                description TEXT,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        logger.info("✅ Created monitoring_config table")
        
        # Insert default monitoring configuration
        default_configs = [
            ('system_thresholds', {
                'cpu_critical': 90,
                'cpu_warning': 80,
                'memory_critical': 95,
                'memory_warning': 85,
                'disk_critical': 95,
                'disk_warning': 85,
                'response_time_critical': 5000,
                'response_time_warning': 2000,
                'error_rate_critical': 0.1,
                'error_rate_warning': 0.05
            }, 'System performance thresholds'),
            ('monitoring_intervals', {
                'metrics_collection_interval': 30,
                'health_check_interval': 60,
                'alert_retention_hours': 24,
                'metrics_retention_hours': 168
            }, 'Monitoring collection intervals'),
            ('service_registry', {
                'sde_framework': {
                    'health_check_enabled': True,
                    'metrics_collection_enabled': True,
                    'alerting_enabled': True
                },
                'signal_generator': {
                    'health_check_enabled': True,
                    'metrics_collection_enabled': True,
                    'alerting_enabled': True
                },
                'database': {
                    'health_check_enabled': True,
                    'metrics_collection_enabled': True,
                    'alerting_enabled': True
                },
                'feature_store': {
                    'health_check_enabled': True,
                    'metrics_collection_enabled': True,
                    'alerting_enabled': True
                }
            }, 'Service registry configuration')
        ]
        
        for config_key, config_value, description in default_configs:
            await conn.execute("""
                INSERT INTO monitoring_config (config_key, config_value, description)
                VALUES ($1, $2, $3)
                ON CONFLICT (config_key) DO UPDATE SET
                config_value = EXCLUDED.config_value,
                description = EXCLUDED.description,
                updated_at = NOW()
            """, config_key, json.dumps(config_value), description)
        
        logger.info("✅ Inserted default monitoring configuration")
        
        # Insert sample performance analytics
        sample_analytics = [
            ('cpu_usage', 45.2, 'percent', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'system'),
            ('memory_usage', 67.8, 'percent', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'system'),
            ('disk_usage', 23.4, 'percent', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'system'),
            ('response_time', 125.5, 'milliseconds', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'sde_framework'),
            ('error_rate', 0.02, 'ratio', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'signal_generator'),
            ('signals_generated', 45, 'count', datetime.now() - timedelta(hours=1), datetime.now(), 'sum', 'signal_generator'),
            ('database_connections', 12, 'count', datetime.now() - timedelta(hours=1), datetime.now(), 'avg', 'database')
        ]
        
        for metric_name, metric_value, metric_unit, period_start, period_end, aggregation_type, service_name in sample_analytics:
            await conn.execute("""
                INSERT INTO performance_analytics 
                (metric_name, metric_value, metric_unit, period_start, period_end, aggregation_type, service_name)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
            """, metric_name, metric_value, metric_unit, period_start, period_end, aggregation_type, service_name)
        
        logger.info("✅ Inserted sample performance analytics")
        
        # Insert sample deployment status
        sample_deployments = [
            ('deploy_001', 'v1.0.0', 'active', datetime.now() - timedelta(hours=2), datetime.now() - timedelta(hours=1, minutes=55), 10, 0, ['sde_framework', 'signal_generator', 'database']),
            ('deploy_002', 'v1.1.0', 'deploying', datetime.now() - timedelta(minutes=30), None, 5, 1, ['sde_framework', 'signal_generator']),
            ('deploy_003', 'v0.9.5', 'failed', datetime.now() - timedelta(days=1), datetime.now() - timedelta(days=1, minutes=5), 2, 8, ['sde_framework'])
        ]
        
        for deployment_id, version, status, start_time, end_time, health_checks_passed, health_checks_failed, services in sample_deployments:
            await conn.execute("""
                INSERT INTO deployment_status 
                (deployment_id, version, status, start_time, end_time, health_checks_passed, health_checks_failed, services)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (deployment_id) DO NOTHING
            """, deployment_id, version, status, start_time, end_time, health_checks_passed, health_checks_failed, services)
        
        logger.info("✅ Inserted sample deployment status")
        
        # Create indexes for performance
        indexes = [
            ("idx_production_metrics_timestamp", "production_metrics", "(timestamp DESC)"),
            ("idx_service_health_service_time", "service_health", "(service_name, last_check DESC)"),
            ("idx_service_health_status", "service_health", "(status, last_check DESC)"),
            ("idx_system_alerts_timestamp", "system_alerts", "(timestamp DESC)"),
            ("idx_system_alerts_severity", "system_alerts", "(severity, timestamp DESC)"),
            ("idx_system_alerts_service", "system_alerts", "(service, timestamp DESC)"),
            ("idx_deployment_status_time", "deployment_status", "(start_time DESC)"),
            ("idx_deployment_status_status", "deployment_status", "(status, start_time DESC)"),
            ("idx_performance_analytics_metric", "performance_analytics", "(metric_name, period_start DESC)"),
            ("idx_performance_analytics_service", "performance_analytics", "(service_name, period_start DESC)"),
            ("idx_monitoring_config_key", "monitoring_config", "(config_key)")
        ]
        
        for index_name, table_name, columns in indexes:
            try:
                await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} {columns}")
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.warning(f"⚠️ Index {index_name} already exists or failed: {e}")
        
        # Create TimescaleDB hypertables for time-series data
        try:
            # Drop existing tables if they exist to recreate with proper structure
            await conn.execute("DROP TABLE IF EXISTS production_metrics CASCADE")
            await conn.execute("DROP TABLE IF EXISTS service_health CASCADE")
            await conn.execute("DROP TABLE IF EXISTS system_alerts CASCADE")
            await conn.execute("DROP TABLE IF EXISTS performance_analytics CASCADE")
            
            # Recreate tables with TimescaleDB-compatible structure
            # 1. Production Metrics Table (timestamp as primary key component)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS production_metrics (
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    cpu_percent DECIMAL(5,2) NOT NULL,
                    memory_percent DECIMAL(5,2) NOT NULL,
                    disk_percent DECIMAL(5,2) NOT NULL,
                    network_io JSONB NOT NULL,
                    active_connections INTEGER NOT NULL,
                    process_count INTEGER NOT NULL,
                    uptime_seconds DECIMAL(10,2) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (timestamp)
                )
            """)
            
            # 2. Service Health Table (last_check as primary key component)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    last_check TIMESTAMP WITH TIME ZONE NOT NULL,
                    service_name VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    response_time_ms DECIMAL(8,2) NOT NULL,
                    error_rate DECIMAL(5,4) NOT NULL,
                    error_message TEXT,
                    metrics JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (last_check, service_name)
                )
            """)
            
            # 3. System Alerts Table (timestamp as primary key component)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    alert_id VARCHAR(100) UNIQUE NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
                    resolved BOOLEAN NOT NULL DEFAULT FALSE,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (timestamp, alert_id)
                )
            """)
            
            # 4. Performance Analytics Table (period_start as primary key component)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(10,4) NOT NULL,
                    metric_unit VARCHAR(20),
                    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    aggregation_type VARCHAR(20) NOT NULL,
                    service_name VARCHAR(50),
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (period_start, metric_name, service_name)
                )
            """)
            
            # Convert to hypertables
            await conn.execute("SELECT create_hypertable('production_metrics', 'timestamp', if_not_exists => TRUE)")
            logger.info("✅ Created TimescaleDB hypertable for production_metrics")
            
            await conn.execute("SELECT create_hypertable('service_health', 'last_check', if_not_exists => TRUE)")
            logger.info("✅ Created TimescaleDB hypertable for service_health")
            
            await conn.execute("SELECT create_hypertable('system_alerts', 'timestamp', if_not_exists => TRUE)")
            logger.info("✅ Created TimescaleDB hypertable for system_alerts")
            
            await conn.execute("SELECT create_hypertable('performance_analytics', 'period_start', if_not_exists => TRUE)")
            logger.info("✅ Created TimescaleDB hypertable for performance_analytics")
            
        except Exception as e:
            logger.warning(f"⚠️ TimescaleDB hypertable creation failed (may not be TimescaleDB): {e}")
            # Fallback: recreate tables with original structure
            logger.info("🔄 Falling back to standard PostgreSQL tables")
            
            # Recreate tables with original structure
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS production_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    cpu_percent DECIMAL(5,2) NOT NULL,
                    memory_percent DECIMAL(5,2) NOT NULL,
                    disk_percent DECIMAL(5,2) NOT NULL,
                    network_io JSONB NOT NULL,
                    active_connections INTEGER NOT NULL,
                    process_count INTEGER NOT NULL,
                    uptime_seconds DECIMAL(10,2) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    id SERIAL PRIMARY KEY,
                    service_name VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    response_time_ms DECIMAL(8,2) NOT NULL,
                    error_rate DECIMAL(5,4) NOT NULL,
                    last_check TIMESTAMP WITH TIME ZONE NOT NULL,
                    error_message TEXT,
                    metrics JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id SERIAL PRIMARY KEY,
                    alert_id VARCHAR(100) UNIQUE NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
                    resolved BOOLEAN NOT NULL DEFAULT FALSE,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(10,4) NOT NULL,
                    metric_unit VARCHAR(20),
                    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    aggregation_type VARCHAR(20) NOT NULL,
                    service_name VARCHAR(50),
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
        
        # Verify table creation
        tables = [
            'production_metrics',
            'service_health',
            'system_alerts',
            'deployment_status',
            'performance_analytics',
            'monitoring_config'
        ]
        
        for table in tables:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"✅ Table {table}: {result} rows")
        
        await conn.close()
        logger.info("✅ Phase 10 migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
