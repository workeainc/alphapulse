#!/usr/bin/env python3
"""
Simple script to run the latency metrics table migration
"""

import asyncio
import logging
from sqlalchemy import text
from ..database.connection import TimescaleDBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_latency_metrics_table():
    """Create the latency_metrics table"""
    
    logger.info("🚀 Creating latency_metrics table...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            # Create the table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS latency_metrics (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR(100),
                operation_type VARCHAR(50) NOT NULL,
                fetch_time_ms FLOAT,
                preprocess_time_ms FLOAT,
                inference_time_ms FLOAT,
                postprocess_time_ms FLOAT,
                total_latency_ms FLOAT NOT NULL,
                symbol VARCHAR(20),
                strategy_name VARCHAR(100),
                success BOOLEAN NOT NULL DEFAULT TRUE,
                error_message TEXT,
                metadata_json JSONB,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
            
            await session.execute(text(create_table_sql))
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_model_id ON latency_metrics(model_id);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_operation_type ON latency_metrics(operation_type);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_symbol ON latency_metrics(symbol);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_strategy_name ON latency_metrics(strategy_name);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_timestamp ON latency_metrics(timestamp);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_symbol_timestamp ON latency_metrics(symbol, timestamp);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_strategy_timestamp ON latency_metrics(strategy_name, timestamp);",
                "CREATE INDEX IF NOT EXISTS ix_latency_metrics_model_timestamp ON latency_metrics(model_id, timestamp);"
            ]
            
            for index_sql in indexes:
                await session.execute(text(index_sql))
            
            # Convert to TimescaleDB hypertable
            try:
                await session.execute(text("SELECT create_hypertable('latency_metrics', 'timestamp', if_not_exists => TRUE);"))
                logger.info("✅ Converted to TimescaleDB hypertable")
            except Exception as e:
                logger.warning(f"Could not convert to hypertable (TimescaleDB might not be available): {e}")
            
            # Set compression policy
            try:
                await session.execute(text("""
                    ALTER TABLE latency_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'operation_type,model_id',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                logger.info("✅ Set compression policy")
            except Exception as e:
                logger.warning(f"Could not set compression policy: {e}")
            
            await session.commit()
            logger.info("✅ latency_metrics table created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating latency_metrics table: {e}")
        raise

async def main():
    """Main function"""
    await create_latency_metrics_table()

if __name__ == "__main__":
    asyncio.run(main())
