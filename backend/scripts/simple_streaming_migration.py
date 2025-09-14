#!/usr/bin/env python3
"""
Simple Streaming Infrastructure Migration
Creates core streaming tables without complex SQL functions
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import TimescaleDBConnection
from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_simple_migration():
    """Run simplified streaming infrastructure migration"""
    logger.info("🚀 Starting Simple Streaming Infrastructure Migration...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection({
            'host': settings.TIMESCALEDB_HOST,
            'port': settings.TIMESCALEDB_PORT,
            'database': settings.TIMESCALEDB_DATABASE,
            'username': settings.TIMESCALEDB_USERNAME,
            'password': settings.TIMESCALEDB_PASSWORD,
            'pool_size': 5,
            'max_overflow': 10
        })
        
        await db_connection.initialize()
        logger.info("✅ Database connection established")
        
        # Core streaming tables SQL
        core_tables_sql = """
        -- Enable TimescaleDB extension
        CREATE EXTENSION IF NOT EXISTS timescaledb;
        
        -- Stream messages table
        CREATE TABLE IF NOT EXISTS stream_messages (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) UNIQUE NOT NULL,
            stream_key VARCHAR(200) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            source VARCHAR(100) NOT NULL,
            partition INTEGER DEFAULT 0,
            priority INTEGER DEFAULT 0,
            data JSONB NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Normalized data table
        CREATE TABLE IF NOT EXISTS normalized_data (
            id SERIAL PRIMARY KEY,
            original_message_id VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            validation_status VARCHAR(20) NOT NULL,
            confidence_score DECIMAL(5,4) NOT NULL,
            processing_time_ms DECIMAL(10,2) NOT NULL,
            normalized_data JSONB NOT NULL,
            validation_errors JSONB,
            metadata JSONB,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Real-time candles table
        CREATE TABLE IF NOT EXISTS realtime_candles (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            open_time TIMESTAMPTZ NOT NULL,
            close_time TIMESTAMPTZ NOT NULL,
            open_price DECIMAL(20,8) NOT NULL,
            high_price DECIMAL(20,8) NOT NULL,
            low_price DECIMAL(20,8) NOT NULL,
            close_price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8) NOT NULL,
            trade_count INTEGER DEFAULT 0,
            vwap DECIMAL(20,8) DEFAULT 0,
            metadata JSONB,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, timeframe, open_time)
        );
        
        -- Technical indicators table
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            indicator_name VARCHAR(50) NOT NULL,
            value DECIMAL(20,8) NOT NULL,
            parameters JSONB NOT NULL,
            metadata JSONB,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- System metrics table
        CREATE TABLE IF NOT EXISTS system_metrics (
            id SERIAL PRIMARY KEY,
            cpu_percent DECIMAL(5,2) NOT NULL,
            memory_percent DECIMAL(5,2) NOT NULL,
            memory_used_mb DECIMAL(10,2) NOT NULL,
            memory_available_mb DECIMAL(10,2) NOT NULL,
            disk_usage_percent DECIMAL(5,2) NOT NULL,
            network_bytes_sent BIGINT NOT NULL,
            network_bytes_recv BIGINT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Processing results table
        CREATE TABLE IF NOT EXISTS processing_results (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) NOT NULL,
            success BOOLEAN NOT NULL,
            processing_time_ms DECIMAL(10,2) NOT NULL,
            components_processed JSONB NOT NULL,
            errors JSONB,
            metadata JSONB,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Execute migration
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            
            # Split SQL into statements
            statements = [stmt.strip() for stmt in core_tables_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements):
                try:
                    await session.execute(text(statement))
                    logger.info(f"✅ Executed statement {i+1}/{len(statements)}")
                except Exception as e:
                    logger.warning(f"⚠️ Statement {i+1} failed (may already exist): {e}")
            
            await session.commit()
        
        logger.info("✅ Core tables migration completed")
        
        # Convert to TimescaleDB hypertables
        hypertables_sql = """
        -- Convert to TimescaleDB hypertables
        SELECT create_hypertable('stream_messages', 'timestamp', if_not_exists => TRUE);
        SELECT create_hypertable('normalized_data', 'timestamp', if_not_exists => TRUE);
        SELECT create_hypertable('realtime_candles', 'open_time', if_not_exists => TRUE);
        SELECT create_hypertable('technical_indicators', 'timestamp', if_not_exists => TRUE);
        SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);
        SELECT create_hypertable('processing_results', 'timestamp', if_not_exists => TRUE);
        
        -- Set compression policies
        SELECT add_compression_policy('stream_messages', INTERVAL '1 day');
        SELECT add_compression_policy('normalized_data', INTERVAL '1 day');
        SELECT add_compression_policy('realtime_candles', INTERVAL '1 day');
        SELECT add_compression_policy('technical_indicators', INTERVAL '1 day');
        SELECT add_compression_policy('system_metrics', INTERVAL '1 hour');
        SELECT add_compression_policy('processing_results', INTERVAL '1 day');
        """
        
        async with db_connection.async_session() as session:
            statements = [stmt.strip() for stmt in hypertables_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements):
                try:
                    await session.execute(text(statement))
                    logger.info(f"✅ TimescaleDB statement {i+1}/{len(statements)}")
                except Exception as e:
                    logger.warning(f"⚠️ TimescaleDB statement {i+1} failed: {e}")
            
            await session.commit()
        
        logger.info("✅ TimescaleDB configuration completed")
        
        # Validate migration
        await validate_simple_migration(db_connection)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False
    finally:
        if 'db_connection' in locals():
            await db_connection.close()

async def validate_simple_migration(db_connection):
    """Validate the simple migration"""
    logger.info("🔍 Validating migration...")
    
    try:
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            
            # Check if tables exist
            tables_to_check = [
                'stream_messages',
                'normalized_data', 
                'realtime_candles',
                'technical_indicators',
                'system_metrics',
                'processing_results'
            ]
            
            for table in tables_to_check:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    );
                """))
                exists = result.scalar()
                if exists:
                    logger.info(f"✅ Table {table} exists")
                else:
                    logger.error(f"❌ Table {table} not found")
            
            # Test TimescaleDB extension
            result = await session.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb';"))
            if result.fetchone():
                logger.info("✅ TimescaleDB extension is enabled")
            else:
                logger.warning("⚠️ TimescaleDB extension not found")
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")

async def test_streaming_components():
    """Test streaming components initialization"""
    logger.info("🧪 Testing streaming components...")
    
    try:
        # Add streaming directory to path
        import sys
        streaming_path = backend_path / "streaming"
        if streaming_path.exists():
            sys.path.insert(0, str(streaming_path))
        
        from streaming.stream_processor import StreamProcessor
        from streaming.stream_metrics import StreamMetrics
        from core.config import STREAMING_CONFIG
        
        # Test stream processor
        stream_processor = StreamProcessor(STREAMING_CONFIG)
        await stream_processor.initialize()
        logger.info("✅ Stream processor initialized")
        
        # Test stream metrics
        stream_metrics = StreamMetrics(STREAMING_CONFIG)
        await stream_metrics.initialize()
        logger.info("✅ Stream metrics initialized")
        
        # Get metrics
        metrics = stream_metrics.get_current_metrics()
        logger.info(f"📊 Current metrics: {len(metrics)} components")
        
        # Cleanup
        await stream_metrics.shutdown()
        await stream_processor.shutdown()
        
        logger.info("✅ Streaming components test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming components test failed: {e}")
        return False

async def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("SIMPLE STREAMING INFRASTRUCTURE MIGRATION")
    logger.info("=" * 60)
    
    # Run migration
    migration_success = await run_simple_migration()
    
    if migration_success:
        # Test streaming components
        components_success = await test_streaming_components()
        
        if components_success:
            logger.info("🎉 Simple Streaming Infrastructure Migration Completed Successfully!")
            logger.info("📋 Next steps:")
            logger.info("   1. Start Redis server (if not already running)")
            logger.info("   2. Run streaming tests: python tests/test_streaming_infrastructure.py")
            logger.info("   3. Integrate with existing market data services")
            logger.info("   4. Configure monitoring and alerting")
        else:
            logger.error("❌ Streaming components test failed")
            sys.exit(1)
    else:
        logger.error("❌ Migration failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
