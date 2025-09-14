#!/usr/bin/env python3
"""
Phase 4.3: Database Integration Optimization Migrations
Adds TimescaleDB optimizations, performance indexes, and database enhancements
"""
import asyncio
import logging
from app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_phase4_3_migrations():
    """Run Phase 4.3 Database Integration Optimization migrations"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        logger.info("🔧 Applying Phase 4.3: Database Integration Optimization migrations...")
        
        async with db_manager.get_connection() as conn:
            # Phase 4.3: Add database optimization columns
            logger.info("🗄️ Adding Phase 4.3 Database Optimization columns...")
            
            await conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS phase_4_3_features BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS batch_processed BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS database_optimization_metadata JSONB,
                ADD COLUMN IF NOT EXISTS query_optimization_level VARCHAR(20) DEFAULT 'standard',
                ADD COLUMN IF NOT EXISTS compression_status VARCHAR(20) DEFAULT 'pending',
                ADD COLUMN IF NOT EXISTS index_usage_stats JSONB,
                ADD COLUMN IF NOT EXISTS hypertable_chunk_count INTEGER,
                ADD COLUMN IF NOT EXISTS batch_insert_timestamp TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS database_performance_score FLOAT,
                ADD COLUMN IF NOT EXISTS query_execution_time_ms FLOAT,
                ADD COLUMN IF NOT EXISTS index_hit_ratio FLOAT,
                ADD COLUMN IF NOT EXISTS compression_ratio FLOAT,
                ADD COLUMN IF NOT EXISTS chunk_compression_status JSONB,
                ADD COLUMN IF NOT EXISTS retention_policy_status VARCHAR(20) DEFAULT 'active'
            """)
            
            logger.info("✅ Added Phase 4.3 database optimization columns")
            
            # Phase 4.3: Create TimescaleDB hypertable optimizations
            logger.info("📊 Creating TimescaleDB hypertable optimizations...")
            
            # Drop existing primary key if it doesn't include timestamp
            await conn.execute("""
                ALTER TABLE enhanced_signals DROP CONSTRAINT IF EXISTS enhanced_signals_pkey;
            """)
            
            # Create new composite primary key including timestamp
            await conn.execute("""
                ALTER TABLE enhanced_signals ADD CONSTRAINT enhanced_signals_pkey 
                PRIMARY KEY (timestamp, id);
            """)
            
            # Create hypertable if it doesn't exist (with data migration)
            await conn.execute("""
                SELECT create_hypertable('enhanced_signals', 'timestamp', 
                    if_not_exists => TRUE, 
                    chunk_time_interval => INTERVAL '1 day',
                    migrate_data => TRUE
                );
            """)
            
            # Enable compression
            await conn.execute("""
                ALTER TABLE enhanced_signals SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)
            
            # Add compression policy
            await conn.execute("""
                SELECT add_compression_policy('enhanced_signals', INTERVAL '7 days');
            """)
            
            # Add retention policy
            await conn.execute("""
                SELECT add_retention_policy('enhanced_signals', INTERVAL '90 days');
            """)
            
            logger.info("✅ TimescaleDB hypertable optimizations applied")
            
            # Phase 4.3: Create performance indexes
            logger.info("📈 Creating Phase 4.3 performance indexes...")
            
            performance_indexes = [
                # Time-based query optimization
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_time_symbol_confidence ON enhanced_signals (timestamp DESC, symbol, confidence DESC)",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_symbol_time_side ON enhanced_signals (symbol, timestamp DESC, side)",
                
                # Batch processing optimization
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_batch_processing ON enhanced_signals (batch_processed, timestamp DESC) WHERE batch_processed = FALSE",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_recent_signals ON enhanced_signals (timestamp DESC, confidence DESC) WHERE timestamp > NOW() - INTERVAL '24 hours'",
                
                # Performance monitoring indexes
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_performance_score ON enhanced_signals (database_performance_score DESC, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_query_optimization ON enhanced_signals (query_optimization_level, query_execution_time_ms)",
                
                # Compression and retention indexes
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_compression_status ON enhanced_signals (compression_status, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_retention_status ON enhanced_signals (retention_policy_status, timestamp DESC)",
                
                # JSONB optimization indexes
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_database_metadata_gin ON enhanced_signals USING GIN (database_optimization_metadata)",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_index_usage_stats_gin ON enhanced_signals USING GIN (index_usage_stats)",
                "CREATE INDEX IF NOT EXISTS idx_phase4_3_chunk_compression_gin ON enhanced_signals USING GIN (chunk_compression_status)"
            ]
            
            for index_sql in performance_indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            logger.info("✅ Phase 4.3 performance indexes created")
            
            # Phase 4.3: Create database optimization views
            logger.info("👁️ Creating Phase 4.3 database optimization views...")
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_3_performance_summary AS
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as total_signals,
                    AVG(database_performance_score) as avg_performance_score,
                    AVG(query_execution_time_ms) as avg_query_time_ms,
                    AVG(index_hit_ratio) as avg_index_hit_ratio,
                    AVG(compression_ratio) as avg_compression_ratio,
                    COUNT(CASE WHEN batch_processed THEN 1 END) as batch_processed_count,
                    COUNT(CASE WHEN compression_status = 'compressed' THEN 1 END) as compressed_count,
                    COUNT(CASE WHEN query_optimization_level = 'optimized' THEN 1 END) as optimized_query_count
                FROM enhanced_signals 
                WHERE phase_4_3_features = TRUE
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour DESC
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_3_batch_processing_status AS
                SELECT 
                    DATE_TRUNC('day', batch_insert_timestamp) as day,
                    COUNT(*) as total_batch_inserts,
                    AVG(EXTRACT(EPOCH FROM (batch_insert_timestamp - timestamp))) as avg_batch_delay_seconds,
                    COUNT(CASE WHEN batch_processed THEN 1 END) as processed_count,
                    COUNT(CASE WHEN NOT batch_processed THEN 1 END) as pending_count
                FROM enhanced_signals 
                WHERE phase_4_3_features = TRUE AND batch_insert_timestamp IS NOT NULL
                GROUP BY DATE_TRUNC('day', batch_insert_timestamp)
                ORDER BY day DESC
            """)
            
            await conn.execute("""
                CREATE OR REPLACE VIEW phase4_3_compression_analysis AS
                SELECT 
                    symbol,
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN compression_status = 'compressed' THEN 1 END) as compressed_signals,
                    COUNT(CASE WHEN compression_status = 'pending' THEN 1 END) as pending_signals,
                    AVG(compression_ratio) as avg_compression_ratio,
                    AVG(hypertable_chunk_count) as avg_chunk_count
                FROM enhanced_signals 
                WHERE phase_4_3_features = TRUE
                GROUP BY symbol
                ORDER BY total_signals DESC
            """)
            
            logger.info("✅ Phase 4.3 database optimization views created")
            
            # Phase 4.3: Create database optimization functions
            logger.info("⚙️ Creating Phase 4.3 database optimization functions...")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION get_phase4_3_performance_stats(
                    p_start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
                    p_end_time TIMESTAMPTZ DEFAULT NOW()
                ) RETURNS JSONB AS $$
                DECLARE
                    result JSONB;
                BEGIN
                    SELECT jsonb_build_object(
                        'total_signals', COUNT(*),
                        'avg_performance_score', AVG(database_performance_score),
                        'avg_query_time_ms', AVG(query_execution_time_ms),
                        'avg_index_hit_ratio', AVG(index_hit_ratio),
                        'avg_compression_ratio', AVG(compression_ratio),
                        'batch_processed_count', COUNT(CASE WHEN batch_processed THEN 1 END),
                        'compressed_count', COUNT(CASE WHEN compression_status = 'compressed' THEN 1 END),
                        'optimized_query_count', COUNT(CASE WHEN query_optimization_level = 'optimized' THEN 1 END),
                        'total_chunks', SUM(hypertable_chunk_count),
                        'avg_chunk_count', AVG(hypertable_chunk_count)
                    ) INTO result
                    FROM enhanced_signals 
                    WHERE phase_4_3_features = TRUE 
                    AND timestamp BETWEEN p_start_time AND p_end_time;
                    
                    RETURN result;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_phase4_3_optimization_metadata(
                    p_signal_id VARCHAR(50),
                    p_batch_processed BOOLEAN,
                    p_database_performance_score FLOAT,
                    p_query_execution_time_ms FLOAT,
                    p_index_hit_ratio FLOAT,
                    p_compression_ratio FLOAT,
                    p_hypertable_chunk_count INTEGER,
                    p_database_optimization_metadata JSONB
                ) RETURNS VOID AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET 
                        batch_processed = p_batch_processed,
                        database_performance_score = p_database_performance_score,
                        query_execution_time_ms = p_query_execution_time_ms,
                        index_hit_ratio = p_index_hit_ratio,
                        compression_ratio = p_compression_ratio,
                        hypertable_chunk_count = p_hypertable_chunk_count,
                        database_optimization_metadata = p_database_optimization_metadata,
                        phase_4_3_features = TRUE
                    WHERE id = p_signal_id;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION optimize_database_queries(
                    p_symbol VARCHAR(20) DEFAULT NULL,
                    p_time_range INTERVAL DEFAULT INTERVAL '24 hours'
                ) RETURNS JSONB AS $$
                DECLARE
                    result JSONB;
                    query_start TIMESTAMPTZ;
                    query_end TIMESTAMPTZ;
                BEGIN
                    query_start = NOW();
                    
                    IF p_symbol IS NOT NULL THEN
                        SELECT jsonb_build_object(
                            'symbol', p_symbol,
                            'signals_count', COUNT(*),
                            'avg_confidence', AVG(confidence),
                            'performance_score', AVG(database_performance_score),
                            'query_time_ms', EXTRACT(EPOCH FROM (NOW() - query_start)) * 1000
                        ) INTO result
                        FROM enhanced_signals 
                        WHERE phase_4_3_features = TRUE 
                        AND symbol = p_symbol 
                        AND timestamp > NOW() - p_time_range;
                    ELSE
                        SELECT jsonb_build_object(
                            'total_signals', COUNT(*),
                            'avg_confidence', AVG(confidence),
                            'performance_score', AVG(database_performance_score),
                            'query_time_ms', EXTRACT(EPOCH FROM (NOW() - query_start)) * 1000
                        ) INTO result
                        FROM enhanced_signals 
                        WHERE phase_4_3_features = TRUE 
                        AND timestamp > NOW() - p_time_range;
                    END IF;
                    
                    RETURN result;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Phase 4.3 database optimization functions created")
            
            # Phase 4.3: Create database monitoring triggers
            logger.info("🔔 Creating Phase 4.3 database monitoring triggers...")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION log_phase4_3_database_metrics() RETURNS TRIGGER AS $$
                BEGIN
                    -- Log database performance metrics
                    IF NEW.phase_4_3_features = TRUE THEN
                        INSERT INTO performance_logs (
                            timestamp,
                            signal_id,
                            log_type,
                            message,
                            metadata
                        ) VALUES (
                            NOW(),
                            NEW.id,
                            'phase4_3_database_optimization',
                            CASE 
                                WHEN NEW.database_performance_score > 0.8 THEN 'High database performance'
                                WHEN NEW.query_execution_time_ms < 10 THEN 'Fast query execution'
                                WHEN NEW.compression_ratio > 0.5 THEN 'Good compression ratio'
                                ELSE 'Database optimization applied'
                            END,
                            jsonb_build_object(
                                'performance_score', NEW.database_performance_score,
                                'query_time_ms', NEW.query_execution_time_ms,
                                'index_hit_ratio', NEW.index_hit_ratio,
                                'compression_ratio', NEW.compression_ratio,
                                'batch_processed', NEW.batch_processed
                            )
                        );
                    END IF;
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            await conn.execute("""
                DROP TRIGGER IF EXISTS trigger_phase4_3_database_metrics ON enhanced_signals;
                CREATE TRIGGER trigger_phase4_3_database_metrics
                AFTER INSERT OR UPDATE ON enhanced_signals
                FOR EACH ROW
                WHEN (NEW.phase_4_3_features = TRUE)
                EXECUTE FUNCTION log_phase4_3_database_metrics()
            """)
            
            logger.info("✅ Phase 4.3 database monitoring triggers created")
            
            # Verify migrations
            logger.info("🔍 Verifying Phase 4.3 migrations...")
            
            # Check Phase 4.3 columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name IN (
                    'phase_4_3_features', 'batch_processed', 'database_optimization_metadata',
                    'query_optimization_level', 'compression_status', 'index_usage_stats',
                    'hypertable_chunk_count', 'batch_insert_timestamp', 'database_performance_score',
                    'query_execution_time_ms', 'index_hit_ratio', 'compression_ratio',
                    'chunk_compression_status', 'retention_policy_status'
                )
                ORDER BY column_name
            """)
            
            phase4_3_columns = [row['column_name'] for row in result]
            expected_columns = [
                'phase_4_3_features', 'batch_processed', 'database_optimization_metadata',
                'query_optimization_level', 'compression_status', 'index_usage_stats',
                'hypertable_chunk_count', 'batch_insert_timestamp', 'database_performance_score',
                'query_execution_time_ms', 'index_hit_ratio', 'compression_ratio',
                'chunk_compression_status', 'retention_policy_status'
            ]
            
            missing_columns = set(expected_columns) - set(phase4_3_columns)
            if missing_columns:
                logger.error(f"❌ Missing Phase 4.3 columns: {missing_columns}")
                return False
            
            logger.info(f"✅ All {len(phase4_3_columns)} Phase 4.3 columns verified")
            
            # Check views
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname IN ('phase4_3_performance_summary', 'phase4_3_batch_processing_status', 'phase4_3_compression_analysis')
                ORDER BY viewname
            """)
            
            views = [row['viewname'] for row in result]
            if len(views) == 3:
                logger.info("✅ All Phase 4.3 views verified")
            else:
                logger.error(f"❌ Missing views. Found: {views}")
                return False
            
            # Check functions
            result = await conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('get_phase4_3_performance_stats', 'update_phase4_3_optimization_metadata', 'optimize_database_queries', 'log_phase4_3_database_metrics')
                ORDER BY proname
            """)
            
            functions = [row['proname'] for row in result]
            if len(functions) == 4:
                logger.info("✅ All Phase 4.3 functions verified")
            else:
                logger.error(f"❌ Missing functions. Found: {functions}")
                return False
        
        logger.info("🎉 Phase 4.3 Database Integration Optimization migrations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4.3 migration failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase4_3_migrations())
