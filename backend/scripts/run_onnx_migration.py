#!/usr/bin/env python3
"""
Script to run the ONNX optimization tables migration for Priority 1
"""

import asyncio
import logging
from sqlalchemy import text
from ..database.connection import TimescaleDBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_onnx_optimization_tables():
    """Create the ONNX optimization tracking tables"""
    
    logger.info("🚀 Creating ONNX optimization tables for Priority 1...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            
            # Create onnx_optimization_metrics table
            create_optimization_metrics_sql = """
            CREATE TABLE IF NOT EXISTS onnx_optimization_metrics (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                optimization_type VARCHAR(50) NOT NULL,
                original_model_size_mb FLOAT,
                optimized_model_size_mb FLOAT,
                size_reduction_percent FLOAT,
                original_inference_time_ms FLOAT,
                optimized_inference_time_ms FLOAT,
                speedup_factor FLOAT,
                memory_usage_reduction_percent FLOAT,
                optimization_level VARCHAR(20) NOT NULL DEFAULT 'balanced',
                optimization_time_seconds FLOAT,
                optimization_success BOOLEAN NOT NULL DEFAULT TRUE,
                fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                execution_provider VARCHAR(50),
                hardware_capabilities JSONB,
                optimization_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                onnx_version VARCHAR(20),
                runtime_version VARCHAR(20),
                notes TEXT
            );
            """
            
            await session.execute(text(create_optimization_metrics_sql))
            logger.info("✅ Created onnx_optimization_metrics table")
            
            # Create onnx_model_registry table
            create_model_registry_sql = """
            CREATE TABLE IF NOT EXISTS onnx_model_registry (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL UNIQUE,
                model_type VARCHAR(50) NOT NULL,
                original_model_path VARCHAR(500),
                onnx_model_path VARCHAR(500),
                quantized_model_path VARCHAR(500),
                input_shape JSONB,
                output_shape JSONB,
                model_parameters JSONB,
                feature_names JSONB,
                is_optimized BOOLEAN NOT NULL DEFAULT FALSE,
                optimization_date TIMESTAMPTZ,
                last_used TIMESTAMPTZ,
                usage_count INTEGER NOT NULL DEFAULT 0,
                avg_inference_time_ms FLOAT,
                total_inferences INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
            
            await session.execute(text(create_model_registry_sql))
            logger.info("✅ Created onnx_model_registry table")
            
            # Create onnx_performance_logs table
            create_performance_logs_sql = """
            CREATE TABLE IF NOT EXISTS onnx_performance_logs (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                optimization_type VARCHAR(50),
                batch_size INTEGER,
                input_size INTEGER,
                preprocessing_time_ms FLOAT,
                inference_time_ms FLOAT NOT NULL,
                postprocessing_time_ms FLOAT,
                total_time_ms FLOAT NOT NULL,
                memory_usage_mb FLOAT,
                cpu_usage_percent FLOAT,
                gpu_usage_percent FLOAT,
                execution_provider VARCHAR(50),
                session_options JSONB,
                error_message TEXT,
                success BOOLEAN NOT NULL DEFAULT TRUE,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                request_id VARCHAR(100),
                user_id VARCHAR(100)
            );
            """
            
            await session.execute(text(create_performance_logs_sql))
            logger.info("✅ Created onnx_performance_logs table")
            
            # Create indexes for onnx_optimization_metrics
            optimization_metrics_indexes = [
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_model_name ON onnx_optimization_metrics(model_name);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_optimization_type ON onnx_optimization_metrics(optimization_type);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_optimization_date ON onnx_optimization_metrics(optimization_date);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_speedup_factor ON onnx_optimization_metrics(speedup_factor);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_model_type ON onnx_optimization_metrics(model_name, optimization_type);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_optimization_metrics_date_type ON onnx_optimization_metrics(optimization_date, optimization_type);"
            ]
            
            for index_sql in optimization_metrics_indexes:
                await session.execute(text(index_sql))
            logger.info("✅ Created indexes for onnx_optimization_metrics")
            
            # Create indexes for onnx_model_registry
            model_registry_indexes = [
                "CREATE INDEX IF NOT EXISTS ix_onnx_model_registry_model_name ON onnx_model_registry(model_name);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_model_registry_model_type ON onnx_model_registry(model_type);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_model_registry_is_optimized ON onnx_model_registry(is_optimized);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_model_registry_last_used ON onnx_model_registry(last_used);"
            ]
            
            for index_sql in model_registry_indexes:
                await session.execute(text(index_sql))
            logger.info("✅ Created indexes for onnx_model_registry")
            
            # Create indexes for onnx_performance_logs
            performance_logs_indexes = [
                "CREATE INDEX IF NOT EXISTS ix_onnx_performance_logs_model_name ON onnx_performance_logs(model_name);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_performance_logs_timestamp ON onnx_performance_logs(timestamp);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_performance_logs_inference_time ON onnx_performance_logs(inference_time_ms);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_performance_logs_success ON onnx_performance_logs(success);",
                "CREATE INDEX IF NOT EXISTS ix_onnx_performance_logs_model_timestamp ON onnx_performance_logs(model_name, timestamp);"
            ]
            
            for index_sql in performance_logs_indexes:
                await session.execute(text(index_sql))
            logger.info("✅ Created indexes for onnx_performance_logs")
            
            # Convert to TimescaleDB hypertables
            try:
                await session.execute(text("SELECT create_hypertable('onnx_optimization_metrics', 'optimization_date', if_not_exists => TRUE);"))
                await session.execute(text("SELECT create_hypertable('onnx_performance_logs', 'timestamp', if_not_exists => TRUE);"))
                logger.info("✅ Converted to TimescaleDB hypertables")
            except Exception as e:
                logger.warning(f"Could not convert to hypertables (TimescaleDB might not be available): {e}")
            
            # Set compression policies
            try:
                await session.execute(text("""
                    ALTER TABLE onnx_optimization_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'model_name,optimization_type',
                        timescaledb.compress_orderby = 'optimization_date DESC'
                    );
                """))
                
                await session.execute(text("""
                    ALTER TABLE onnx_performance_logs SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'model_name,optimization_type',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                logger.info("✅ Set compression policies")
            except Exception as e:
                logger.warning(f"Could not set compression policies: {e}")
            
            # Add compression and retention policies
            try:
                await session.execute(text("SELECT add_compression_policy('onnx_optimization_metrics', INTERVAL '7 days');"))
                await session.execute(text("SELECT add_compression_policy('onnx_performance_logs', INTERVAL '7 days');"))
                await session.execute(text("SELECT add_retention_policy('onnx_optimization_metrics', INTERVAL '1 year');"))
                await session.execute(text("SELECT add_retention_policy('onnx_performance_logs', INTERVAL '1 year');"))
                logger.info("✅ Added compression and retention policies")
            except Exception as e:
                logger.warning(f"Could not add compression/retention policies: {e}")
            
            await session.commit()
            logger.info("🎉 ONNX optimization tables created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating ONNX optimization tables: {e}")
        raise

async def main():
    """Main function"""
    await create_onnx_optimization_tables()

if __name__ == "__main__":
    asyncio.run(main())
