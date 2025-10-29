#!/usr/bin/env python3
"""
Phase 4.1: Performance Optimization Database Migrations
Adds performance tracking columns and indexes for ultra-low latency signal generation
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase4_1PerformanceMigrations:
    """Phase 4.1 Performance Optimization Database Migrations"""
    
    def __init__(self):
        self.conn = None
        self.migration_version = "4.1.0"
        self.migration_name = "performance_optimization"
        
    async def connect(self):
        """Connect to TimescaleDB"""
        try:
            self.conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711'
            )
            logger.info("✅ Connected to TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            await self.conn.close()
            logger.info("✅ Disconnected from database")
    
    async def check_migration_status(self) -> bool:
        """Check if Phase 4.1 migration has been applied"""
        try:
            result = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM migration_history 
                    WHERE version = $1 AND name = $2
                )
            """, self.migration_version, self.migration_name)
            
            if result:
                logger.info(f"✅ Phase 4.1 migration already applied (version {self.migration_version})")
                return True
            else:
                logger.info(f"📋 Phase 4.1 migration not yet applied (version {self.migration_version})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking migration status: {e}")
            return False
    
    async def apply_migrations(self) -> bool:
        """Apply Phase 4.1 performance optimization migrations"""
        try:
            logger.info("🚀 Starting Phase 4.1: Performance Optimization Migrations")
            
            # Check if migration already applied
            if await self.check_migration_status():
                return True
            
            # 1. Add performance tracking columns to enhanced_signals table
            await self.add_performance_columns()
            
            # 2. Create performance indexes
            await self.create_performance_indexes()
            
            # 3. Create performance views
            await self.create_performance_views()
            
            # 4. Create performance functions
            await self.create_performance_functions()
            
            # 5. Create performance triggers
            await self.create_performance_triggers()
            
            # 6. Record migration
            await self.record_migration()
            
            logger.info("✅ Phase 4.1: Performance Optimization Migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 4.1 migration failed: {e}")
            return False
    
    async def add_performance_columns(self):
        """Add performance tracking columns to enhanced_signals table"""
        try:
            logger.info("📊 Adding performance tracking columns...")
            
            await self.conn.execute("""
                ALTER TABLE enhanced_signals
                ADD COLUMN IF NOT EXISTS processing_time_ms FLOAT,
                ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS cache_key VARCHAR(255),
                ADD COLUMN IF NOT EXISTS performance_score FLOAT,
                ADD COLUMN IF NOT EXISTS memory_usage_mb FLOAT,
                ADD COLUMN IF NOT EXISTS cpu_usage_percent FLOAT,
                ADD COLUMN IF NOT EXISTS queue_size INTEGER,
                ADD COLUMN IF NOT EXISTS async_processing BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS parallel_analysis BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS optimization_level VARCHAR(50) DEFAULT 'ultra_low_latency',
                ADD COLUMN IF NOT EXISTS target_latency_ms FLOAT DEFAULT 100.0,
                ADD COLUMN IF NOT EXISTS actual_latency_ms FLOAT,
                ADD COLUMN IF NOT EXISTS latency_percentile_50 FLOAT,
                ADD COLUMN IF NOT EXISTS latency_percentile_95 FLOAT,
                ADD COLUMN IF NOT EXISTS latency_percentile_99 FLOAT,
                ADD COLUMN IF NOT EXISTS throughput_per_second FLOAT,
                ADD COLUMN IF NOT EXISTS phase_4_1_features BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS performance_metadata JSONB
            """)
            
            logger.info("✅ Performance tracking columns added successfully")
            
        except Exception as e:
            logger.error(f"❌ Error adding performance columns: {e}")
            raise
    
    async def create_performance_indexes(self):
        """Create performance optimization indexes"""
        try:
            logger.info("📊 Creating performance indexes...")
            
            # Index for processing time queries
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_processing_time 
                ON enhanced_signals (processing_time_ms)
            """)
            
            # Index for cache performance
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_cache_hit 
                ON enhanced_signals (cache_hit, timestamp)
            """)
            
            # Index for performance score
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_performance_score 
                ON enhanced_signals (performance_score DESC)
            """)
            
            # Index for latency analysis
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_latency 
                ON enhanced_signals (actual_latency_ms, target_latency_ms)
            """)
            
            # Index for optimization level
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_optimization 
                ON enhanced_signals (optimization_level, timestamp)
            """)
            
            # Composite index for performance analysis
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_performance_composite 
                ON enhanced_signals (symbol, timestamp, processing_time_ms, performance_score)
            """)
            
            # Index for memory and CPU usage
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_resource_usage 
                ON enhanced_signals (memory_usage_mb, cpu_usage_percent, timestamp)
            """)
            
            # Index for async processing
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_async_processing 
                ON enhanced_signals (async_processing, parallel_analysis, timestamp)
            """)
            
            # Index for phase 4.1 features
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_phase_4_1 
                ON enhanced_signals (phase_4_1_features, timestamp)
            """)
            
            logger.info("✅ Performance indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating performance indexes: {e}")
            raise
    
    async def create_performance_views(self):
        """Create performance monitoring views"""
        try:
            logger.info("📊 Creating performance monitoring views...")
            
            # View for performance metrics summary
            await self.conn.execute("""
                CREATE OR REPLACE VIEW performance_metrics_summary AS
                SELECT 
                    symbol,
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total_signals,
                    AVG(processing_time_ms) as avg_processing_time,
                    MAX(processing_time_ms) as max_processing_time,
                    MIN(processing_time_ms) as min_processing_time,
                    AVG(performance_score) as avg_performance_score,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    AVG(cpu_usage_percent) as avg_cpu_usage,
                    COUNT(CASE WHEN cache_hit THEN 1 END) as cache_hits,
                    COUNT(CASE WHEN NOT cache_hit THEN 1 END) as cache_misses,
                    COUNT(CASE WHEN actual_latency_ms <= target_latency_ms THEN 1 END) as target_met_count,
                    COUNT(CASE WHEN actual_latency_ms > target_latency_ms THEN 1 END) as target_exceeded_count
                FROM enhanced_signals 
                WHERE phase_4_1_features = TRUE
                GROUP BY symbol, DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC, symbol
            """)
            
            # View for latency analysis
            await self.conn.execute("""
                CREATE OR REPLACE VIEW latency_analysis AS
                SELECT 
                    symbol,
                    DATE_TRUNC('minute', timestamp) as minute,
                    AVG(actual_latency_ms) as avg_latency,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_latency_ms) as p50_latency,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY actual_latency_ms) as p95_latency,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY actual_latency_ms) as p99_latency,
                    AVG(target_latency_ms) as avg_target_latency,
                    COUNT(CASE WHEN actual_latency_ms <= target_latency_ms THEN 1 END) as target_met,
                    COUNT(*) as total_requests
                FROM enhanced_signals 
                WHERE phase_4_1_features = TRUE AND actual_latency_ms IS NOT NULL
                GROUP BY symbol, DATE_TRUNC('minute', timestamp)
                ORDER BY minute DESC, symbol
            """)
            
            # View for cache performance
            await self.conn.execute("""
                CREATE OR REPLACE VIEW cache_performance AS
                SELECT 
                    symbol,
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(CASE WHEN cache_hit THEN 1 END) as cache_hits,
                    COUNT(CASE WHEN NOT cache_hit THEN 1 END) as cache_misses,
                    COUNT(*) as total_requests,
                    ROUND(
                        COUNT(CASE WHEN cache_hit THEN 1 END) * 100.0 / COUNT(*), 2
                    ) as cache_hit_rate,
                    AVG(CASE WHEN cache_hit THEN processing_time_ms END) as avg_cache_hit_time,
                    AVG(CASE WHEN NOT cache_hit THEN processing_time_ms END) as avg_cache_miss_time
                FROM enhanced_signals 
                WHERE phase_4_1_features = TRUE
                GROUP BY symbol, DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC, symbol
            """)
            
            # View for resource usage
            await self.conn.execute("""
                CREATE OR REPLACE VIEW resource_usage AS
                SELECT 
                    symbol,
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    MAX(memory_usage_mb) as max_memory_usage,
                    AVG(cpu_usage_percent) as avg_cpu_usage,
                    MAX(cpu_usage_percent) as max_cpu_usage,
                    AVG(queue_size) as avg_queue_size,
                    MAX(queue_size) as max_queue_size,
                    COUNT(*) as total_requests
                FROM enhanced_signals 
                WHERE phase_4_1_features = TRUE
                GROUP BY symbol, DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC, symbol
            """)
            
            logger.info("✅ Performance monitoring views created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating performance views: {e}")
            raise
    
    async def create_performance_functions(self):
        """Create performance monitoring functions"""
        try:
            logger.info("📊 Creating performance monitoring functions...")
            
            # Function to calculate performance statistics
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_performance_stats(
                    p_symbol VARCHAR DEFAULT NULL,
                    p_hours INTEGER DEFAULT 24
                ) RETURNS TABLE (
                    symbol VARCHAR,
                    total_signals BIGINT,
                    avg_processing_time FLOAT,
                    max_processing_time FLOAT,
                    min_processing_time FLOAT,
                    avg_performance_score FLOAT,
                    cache_hit_rate FLOAT,
                    target_met_rate FLOAT,
                    avg_memory_usage FLOAT,
                    avg_cpu_usage FLOAT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        COALESCE(p_symbol, es.symbol) as symbol,
                        COUNT(*) as total_signals,
                        AVG(es.processing_time_ms) as avg_processing_time,
                        MAX(es.processing_time_ms) as max_processing_time,
                        MIN(es.processing_time_ms) as min_processing_time,
                        AVG(es.performance_score) as avg_performance_score,
                        ROUND(
                            COUNT(CASE WHEN es.cache_hit THEN 1 END) * 100.0 / COUNT(*), 2
                        ) as cache_hit_rate,
                        ROUND(
                            COUNT(CASE WHEN es.actual_latency_ms <= es.target_latency_ms THEN 1 END) * 100.0 / COUNT(*), 2
                        ) as target_met_rate,
                        AVG(es.memory_usage_mb) as avg_memory_usage,
                        AVG(es.cpu_usage_percent) as avg_cpu_usage
                    FROM enhanced_signals es
                    WHERE es.phase_4_1_features = TRUE
                        AND es.timestamp >= NOW() - INTERVAL '1 hour' * p_hours
                        AND (p_symbol IS NULL OR es.symbol = p_symbol)
                    GROUP BY COALESCE(p_symbol, es.symbol)
                    ORDER BY avg_processing_time ASC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to get performance alerts
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION get_performance_alerts(
                    p_threshold_ms FLOAT DEFAULT 100.0
                ) RETURNS TABLE (
                    symbol VARCHAR,
                    timestamp TIMESTAMPTZ,
                    processing_time_ms FLOAT,
                    target_latency_ms FLOAT,
                    performance_score FLOAT,
                    alert_type VARCHAR
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        es.symbol,
                        es.timestamp,
                        es.processing_time_ms,
                        es.target_latency_ms,
                        es.performance_score,
                        CASE 
                            WHEN es.processing_time_ms > p_threshold_ms THEN 'HIGH_LATENCY'
                            WHEN es.performance_score < 50.0 THEN 'LOW_PERFORMANCE'
                            WHEN es.memory_usage_mb > 500.0 THEN 'HIGH_MEMORY'
                            WHEN es.cpu_usage_percent > 80.0 THEN 'HIGH_CPU'
                            ELSE 'NORMAL'
                        END as alert_type
                    FROM enhanced_signals es
                    WHERE es.phase_4_1_features = TRUE
                        AND es.timestamp >= NOW() - INTERVAL '1 hour'
                        AND (
                            es.processing_time_ms > p_threshold_ms OR
                            es.performance_score < 50.0 OR
                            es.memory_usage_mb > 500.0 OR
                            es.cpu_usage_percent > 80.0
                        )
                    ORDER BY es.timestamp DESC;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to update performance metadata
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION update_performance_metadata(
                    p_signal_id VARCHAR,
                    p_metadata JSONB
                ) RETURNS BOOLEAN AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET performance_metadata = p_metadata
                    WHERE id = p_signal_id AND phase_4_1_features = TRUE;
                    
                    RETURN FOUND;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Performance monitoring functions created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating performance functions: {e}")
            raise
    
    async def create_performance_triggers(self):
        """Create performance monitoring triggers"""
        try:
            logger.info("📊 Creating performance monitoring triggers...")
            
            # Trigger function to log performance metrics
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION log_performance_metrics()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Calculate performance score
                    IF NEW.actual_latency_ms IS NOT NULL AND NEW.target_latency_ms IS NOT NULL THEN
                        IF NEW.actual_latency_ms <= NEW.target_latency_ms THEN
                            NEW.performance_score = 100.0;
                        ELSE
                            NEW.performance_score = GREATEST(0, 100 - (NEW.actual_latency_ms - NEW.target_latency_ms));
                        END IF;
                    END IF;
                    
                    -- Set default values for phase 4.1 features
                    NEW.phase_4_1_features = TRUE;
                    NEW.optimization_level = 'ultra_low_latency';
                    
                    -- Log performance alert if needed
                    IF NEW.processing_time_ms > NEW.target_latency_ms THEN
                        RAISE LOG 'Performance Alert: Signal % for % took %.2fms (target: %.2fms)', 
                            NEW.id, NEW.symbol, NEW.processing_time_ms, NEW.target_latency_ms;
                    END IF;
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Trigger to automatically calculate performance metrics
            await self.conn.execute("""
                DROP TRIGGER IF EXISTS trigger_performance_metrics ON enhanced_signals;
                CREATE TRIGGER trigger_performance_metrics
                    BEFORE INSERT OR UPDATE ON enhanced_signals
                    FOR EACH ROW
                    EXECUTE FUNCTION log_performance_metrics();
            """)
            
            logger.info("✅ Performance monitoring triggers created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating performance triggers: {e}")
            raise
    
    async def record_migration(self):
        """Record the migration in migration_history"""
        try:
            await self.conn.execute("""
                INSERT INTO migration_history (version, name, applied_at, description)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (version, name) DO NOTHING
            """, self.migration_version, self.migration_name, datetime.now(), 
                 "Phase 4.1: Performance Optimization - Ultra-low latency signal generation")
            
            logger.info(f"✅ Migration recorded: {self.migration_version} - {self.migration_name}")
            
        except Exception as e:
            logger.error(f"❌ Error recording migration: {e}")
            raise
    
    async def verify_migration(self) -> bool:
        """Verify that all Phase 4.1 migrations were applied correctly"""
        try:
            logger.info("🔍 Verifying Phase 4.1 migrations...")
            
            # Check if performance columns exist
            columns = await self.conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name IN (
                    'processing_time_ms', 'cache_hit', 'performance_score', 
                    'phase_4_1_features', 'optimization_level', 'target_latency_ms'
                )
            """)
            
            if len(columns) < 6:
                logger.error(f"❌ Missing performance columns. Found: {len(columns)}")
                return False
            
            # Check if performance indexes exist
            indexes = await self.conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'enhanced_signals' 
                AND indexname LIKE '%performance%'
            """)
            
            if len(indexes) < 5:
                logger.error(f"❌ Missing performance indexes. Found: {len(indexes)}")
                return False
            
            # Check if performance views exist
            views = await self.conn.fetch("""
                SELECT viewname 
                FROM pg_views 
                WHERE viewname LIKE '%performance%'
            """)
            
            if len(views) < 4:
                logger.error(f"❌ Missing performance views. Found: {len(views)}")
                return False
            
            # Check if performance functions exist
            functions = await self.conn.fetch("""
                SELECT proname 
                FROM pg_proc 
                WHERE proname LIKE '%performance%'
            """)
            
            if len(functions) < 3:
                logger.error(f"❌ Missing performance functions. Found: {len(functions)}")
                return False
            
            logger.info("✅ Phase 4.1 migration verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False

async def main():
    """Main migration function"""
    migrator = Phase4_1PerformanceMigrations()
    
    try:
        # Connect to database
        if not await migrator.connect():
            return 1
        
        # Apply migrations
        if not await migrator.apply_migrations():
            return 1
        
        # Verify migrations
        if not await migrator.verify_migration():
            return 1
        
        logger.info("🎉 Phase 4.1: Performance Optimization Migrations completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        return 1
        
    finally:
        await migrator.disconnect()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
