#!/usr/bin/env python3
"""
Phase 4.2: Memory & CPU Optimization Database Migrations
Adds performance tracking and resource monitoring capabilities to the enhanced_signals table
"""
import asyncio
import logging
from app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_phase4_2_migrations():
    """Run Phase 4.2 Memory & CPU Optimization migrations"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        logger.info("🔧 Applying Phase 4.2: Memory & CPU Optimization migrations...")
        
        async with db_manager.get_connection() as conn:
            # Add Phase 4.2 Memory & CPU Optimization columns
            logger.info("🧠 Adding Phase 4.2: Memory & CPU Optimization columns...")
            
            await conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS memory_usage_mb FLOAT,
                ADD COLUMN IF NOT EXISTS cpu_usage_percent FLOAT,
                ADD COLUMN IF NOT EXISTS cache_size INTEGER,
                ADD COLUMN IF NOT EXISTS gc_collections INTEGER,
                ADD COLUMN IF NOT EXISTS gc_objects_freed INTEGER,
                ADD COLUMN IF NOT EXISTS memory_cleanups INTEGER,
                ADD COLUMN IF NOT EXISTS cache_hit_rate FLOAT,
                ADD COLUMN IF NOT EXISTS processing_time_ms FLOAT,
                ADD COLUMN IF NOT EXISTS throughput_per_second FLOAT,
                ADD COLUMN IF NOT EXISTS latency_percentile_95 FLOAT,
                ADD COLUMN IF NOT EXISTS resource_alerts JSONB,
                ADD COLUMN IF NOT EXISTS optimization_enabled JSONB,
                ADD COLUMN IF NOT EXISTS memory_pressure_level FLOAT,
                ADD COLUMN IF NOT EXISTS cpu_throttling_applied BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS cache_compression_applied BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS gc_optimization_applied BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS phase_4_2_features BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS memory_optimization_metadata JSONB
            """)
            
            logger.info("✅ Added Phase 4.2 memory and CPU optimization columns")
            
            # Create Phase 4.2 performance indexes
            logger.info("📊 Creating Phase 4.2 performance indexes...")
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_memory_usage 
                ON enhanced_signals (memory_usage_mb)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_cpu_usage 
                ON enhanced_signals (cpu_usage_percent)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_cache_performance 
                ON enhanced_signals (cache_size, cache_hit_rate)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_processing_performance 
                ON enhanced_signals (processing_time_ms, throughput_per_second)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_gc_performance 
                ON enhanced_signals (gc_collections, gc_objects_freed)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_memory_pressure 
                ON enhanced_signals (memory_pressure_level)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_phase4_2_composite 
                ON enhanced_signals (memory_usage_mb, cpu_usage_percent, processing_time_ms, phase_4_2_features)
            """)
            
            logger.info("✅ Created Phase 4.2 performance indexes")
            
            # Create Phase 4.2 performance monitoring views
            logger.info("👁️ Creating Phase 4.2 performance monitoring views...")
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_2_performance_summary AS
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total_signals,
                    AVG(memory_usage_mb) as avg_memory_usage_mb,
                    AVG(cpu_usage_percent) as avg_cpu_usage_percent,
                    AVG(processing_time_ms) as avg_processing_time_ms,
                    AVG(cache_hit_rate) as avg_cache_hit_rate,
                    AVG(throughput_per_second) as avg_throughput_per_second,
                    SUM(gc_collections) as total_gc_collections,
                    SUM(gc_objects_freed) as total_objects_freed,
                    SUM(memory_cleanups) as total_memory_cleanups,
                    AVG(memory_pressure_level) as avg_memory_pressure
                FROM enhanced_signals 
                WHERE phase_4_2_features = TRUE
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_2_resource_alerts AS
                SELECT 
                    timestamp,
                    symbol,
                    memory_usage_mb,
                    cpu_usage_percent,
                    processing_time_ms,
                    resource_alerts,
                    memory_pressure_level,
                    cpu_throttling_applied,
                    cache_compression_applied,
                    gc_optimization_applied
                FROM enhanced_signals 
                WHERE phase_4_2_features = TRUE 
                AND (memory_usage_mb > 1000 OR cpu_usage_percent > 80 OR processing_time_ms > 200)
                ORDER BY timestamp DESC
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_2_optimization_stats AS
                SELECT 
                    DATE_TRUNC('day', timestamp) as day,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN cpu_throttling_applied THEN 1 END) as throttling_applied_count,
                    COUNT(CASE WHEN cache_compression_applied THEN 1 END) as compression_applied_count,
                    COUNT(CASE WHEN gc_optimization_applied THEN 1 END) as gc_optimization_count,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    AVG(cpu_usage_percent) as avg_cpu_usage,
                    AVG(processing_time_ms) as avg_processing_time
                FROM enhanced_signals 
                WHERE phase_4_2_features = TRUE
                GROUP BY DATE_TRUNC('day', timestamp)
                ORDER BY day DESC
            """)
            
            logger.info("✅ Created Phase 4.2 performance monitoring views")
            
            # Create Phase 4.2 performance analysis functions
            logger.info("⚙️ Creating Phase 4.2 performance analysis functions...")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION get_phase4_2_performance_stats(
                    p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
                    p_end_time TIMESTAMPTZ DEFAULT NOW()
                ) RETURNS JSONB AS $$
                DECLARE
                    result JSONB;
                BEGIN
                    SELECT jsonb_build_object(
                        'total_signals', COUNT(*),
                        'avg_memory_usage_mb', AVG(memory_usage_mb),
                        'avg_cpu_usage_percent', AVG(cpu_usage_percent),
                        'avg_processing_time_ms', AVG(processing_time_ms),
                        'avg_cache_hit_rate', AVG(cache_hit_rate),
                        'avg_throughput_per_second', AVG(throughput_per_second),
                        'total_gc_collections', SUM(gc_collections),
                        'total_objects_freed', SUM(gc_objects_freed),
                        'total_memory_cleanups', SUM(memory_cleanups),
                        'avg_memory_pressure', AVG(memory_pressure_level),
                        'throttling_applied_count', COUNT(CASE WHEN cpu_throttling_applied THEN 1 END),
                        'compression_applied_count', COUNT(CASE WHEN cache_compression_applied THEN 1 END),
                        'gc_optimization_count', COUNT(CASE WHEN gc_optimization_applied THEN 1 END)
                    ) INTO result
                    FROM enhanced_signals 
                    WHERE phase_4_2_features = TRUE 
                    AND timestamp BETWEEN p_start_time AND p_end_time;
                    
                    RETURN result;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_phase4_2_optimization_metadata(
                    p_signal_id VARCHAR(50),
                    p_memory_usage_mb FLOAT,
                    p_cpu_usage_percent FLOAT,
                    p_cache_size INTEGER,
                    p_gc_collections INTEGER,
                    p_gc_objects_freed INTEGER,
                    p_memory_cleanups INTEGER,
                    p_cache_hit_rate FLOAT,
                    p_processing_time_ms FLOAT,
                    p_throughput_per_second FLOAT,
                    p_latency_percentile_95 FLOAT,
                    p_resource_alerts JSONB,
                    p_optimization_enabled JSONB,
                    p_memory_pressure_level FLOAT,
                    p_cpu_throttling_applied BOOLEAN,
                    p_cache_compression_applied BOOLEAN,
                    p_gc_optimization_applied BOOLEAN
                ) RETURNS VOID AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET 
                        memory_usage_mb = p_memory_usage_mb,
                        cpu_usage_percent = p_cpu_usage_percent,
                        cache_size = p_cache_size,
                        gc_collections = p_gc_collections,
                        gc_objects_freed = p_gc_objects_freed,
                        memory_cleanups = p_memory_cleanups,
                        cache_hit_rate = p_cache_hit_rate,
                        processing_time_ms = p_processing_time_ms,
                        throughput_per_second = p_throughput_per_second,
                        latency_percentile_95 = p_latency_percentile_95,
                        resource_alerts = p_resource_alerts,
                        optimization_enabled = p_optimization_enabled,
                        memory_pressure_level = p_memory_pressure_level,
                        cpu_throttling_applied = p_cpu_throttling_applied,
                        cache_compression_applied = p_cache_compression_applied,
                        gc_optimization_applied = p_gc_optimization_applied,
                        phase_4_2_features = TRUE,
                        memory_optimization_metadata = jsonb_build_object(
                            'updated_at', NOW(),
                            'optimization_version', '4.2',
                            'memory_optimization_enabled', TRUE,
                            'cpu_optimization_enabled', TRUE
                        )
                    WHERE id = p_signal_id;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Created Phase 4.2 performance analysis functions")
            
            # Create Phase 4.2 performance monitoring trigger
            logger.info("🔔 Creating Phase 4.2 performance monitoring trigger...")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION log_phase4_2_performance_metrics() RETURNS TRIGGER AS $$
                BEGIN
                    -- Log performance metrics when memory or CPU usage is high
                    IF NEW.memory_usage_mb > 1000 OR NEW.cpu_usage_percent > 80 THEN
                        INSERT INTO performance_logs (
                            timestamp,
                            signal_id,
                            log_type,
                            message,
                            metadata
                        ) VALUES (
                            NOW(),
                            NEW.id,
                            'phase4_2_alert',
                            CASE 
                                WHEN NEW.memory_usage_mb > 1000 THEN 'High memory usage detected'
                                WHEN NEW.cpu_usage_percent > 80 THEN 'High CPU usage detected'
                                ELSE 'Performance alert'
                            END,
                            jsonb_build_object(
                                'memory_usage_mb', NEW.memory_usage_mb,
                                'cpu_usage_percent', NEW.cpu_usage_percent,
                                'processing_time_ms', NEW.processing_time_ms,
                                'memory_pressure_level', NEW.memory_pressure_level
                            )
                        );
                    END IF;
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            await conn.execute("""
                DROP TRIGGER IF EXISTS trigger_phase4_2_performance ON enhanced_signals;
                CREATE TRIGGER trigger_phase4_2_performance
                AFTER INSERT OR UPDATE ON enhanced_signals
                FOR EACH ROW
                WHEN (NEW.phase_4_2_features = TRUE)
                EXECUTE FUNCTION log_phase4_2_performance_metrics()
            """)
            
            logger.info("✅ Created Phase 4.2 performance monitoring trigger")
            
            # Verify migrations
            logger.info("🔍 Verifying Phase 4.2 migrations...")
            
            # Check Phase 4.2 columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name IN (
                    'memory_usage_mb', 'cpu_usage_percent', 'cache_size', 'gc_collections',
                    'gc_objects_freed', 'memory_cleanups', 'cache_hit_rate', 'processing_time_ms',
                    'throughput_per_second', 'latency_percentile_95', 'resource_alerts',
                    'optimization_enabled', 'memory_pressure_level', 'cpu_throttling_applied',
                    'cache_compression_applied', 'gc_optimization_applied', 'phase_4_2_features',
                    'memory_optimization_metadata'
                )
                ORDER BY column_name
            """)
            
            phase4_2_columns = [row['column_name'] for row in result]
            expected_columns = [
                'memory_usage_mb', 'cpu_usage_percent', 'cache_size', 'gc_collections',
                'gc_objects_freed', 'memory_cleanups', 'cache_hit_rate', 'processing_time_ms',
                'throughput_per_second', 'latency_percentile_95', 'resource_alerts',
                'optimization_enabled', 'memory_pressure_level', 'cpu_throttling_applied',
                'cache_compression_applied', 'gc_optimization_applied', 'phase_4_2_features',
                'memory_optimization_metadata'
            ]
            
            missing_columns = set(expected_columns) - set(phase4_2_columns)
            if missing_columns:
                logger.error(f"❌ Missing Phase 4.2 columns: {missing_columns}")
                return False
            
            logger.info(f"✅ All {len(phase4_2_columns)} Phase 4.2 columns verified")
            
            # Check views
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('phase4_2_performance_summary', 'phase4_2_resource_alerts', 'phase4_2_optimization_stats')
                ORDER BY viewname
            """)
            
            views = [row['viewname'] for row in result]
            if len(views) == 3:
                logger.info("✅ All Phase 4.2 views verified")
            else:
                logger.error(f"❌ Missing views. Found: {views}")
                return False
            
            # Check functions
            result = await conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('get_phase4_2_performance_stats', 'update_phase4_2_optimization_metadata', 'log_phase4_2_performance_metrics')
                ORDER BY proname
            """)
            
            functions = [row['proname'] for row in result]
            if len(functions) == 3:
                logger.info("✅ All Phase 4.2 functions verified")
            else:
                logger.error(f"❌ Missing functions. Found: {functions}")
                return False
        
        logger.info("🎉 Phase 4.2 Memory & CPU Optimization migrations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.2 migration failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase4_2_migrations())
