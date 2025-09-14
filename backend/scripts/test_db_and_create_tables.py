#!/usr/bin/env python3
"""
Test database connection and create ML pipeline tables
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_and_create_tables():
    """Test database connection and create tables"""
    
    try:
        # Import the database connection
        from database.connection import TimescaleDBConnection
        
        logger.info("🔌 Testing database connection...")
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        await db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            # Test connection
            result = await session.execute("SELECT 1 as test")
            test_value = result.scalar()
            logger.info(f"✅ Database connection successful: {test_value}")
            
            # Create ml_predictions table
            logger.info("📊 Creating ml_predictions table...")
            await session.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    prediction DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION,
                    features_used JSONB,
                    inference_latency_ms DOUBLE PRECISION,
                    cache_hit BOOLEAN,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            logger.info("✅ ml_predictions table created")
            
            # Create ml_signals table
            logger.info("📊 Creating ml_signals table...")
            await session.execute("""
                CREATE TABLE IF NOT EXISTS ml_signals (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    signal_strength DOUBLE PRECISION NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    prediction_id BIGINT REFERENCES ml_predictions(id) ON DELETE SET NULL,
                    confidence DOUBLE PRECISION,
                    risk_score DOUBLE PRECISION,
                    market_conditions JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            logger.info("✅ ml_signals table created")
            
            # Create ml_model_performance table
            logger.info("📊 Creating ml_model_performance table...")
            await session.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(50) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    sample_count INTEGER,
                    time_window_minutes INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            logger.info("✅ ml_model_performance table created")
            
            # Convert to TimescaleDB hypertables
            logger.info("🔄 Converting to TimescaleDB hypertables...")
            
            try:
                await session.execute("SELECT create_hypertable('ml_predictions', 'timestamp', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);")
                logger.info("✅ ml_predictions converted to hypertable")
            except Exception as e:
                logger.warning(f"⚠️  Could not convert ml_predictions to hypertable: {e}")
            
            try:
                await session.execute("SELECT create_hypertable('ml_signals', 'timestamp', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);")
                logger.info("✅ ml_signals converted to hypertable")
            except Exception as e:
                logger.warning(f"⚠️  Could not convert ml_signals to hypertable: {e}")
            
            try:
                await session.execute("SELECT create_hypertable('ml_model_performance', 'timestamp', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);")
                logger.info("✅ ml_model_performance converted to hypertable")
            except Exception as e:
                logger.warning(f"⚠️  Could not convert ml_model_performance to hypertable: {e}")
            
            # Create indexes
            logger.info("📈 Creating indexes...")
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions (timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions (symbol);",
                "CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions (model_name);",
                "CREATE INDEX IF NOT EXISTS idx_ml_signals_timestamp ON ml_signals (timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_ml_signals_symbol ON ml_signals (symbol);",
                "CREATE INDEX IF NOT EXISTS idx_ml_signals_type ON ml_signals (signal_type);",
                "CREATE INDEX IF NOT EXISTS idx_ml_model_performance_timestamp ON ml_model_performance (timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_ml_model_performance_model ON ml_model_performance (model_name);",
                "CREATE INDEX IF NOT EXISTS idx_ml_model_performance_metric ON ml_model_performance (metric_name);"
            ]
            
            for index_sql in indexes:
                try:
                    await session.execute(index_sql)
                except Exception as e:
                    logger.warning(f"⚠️  Could not create index: {e}")
            
            logger.info("✅ Indexes created")
            
            await session.commit()
            
            # Verify tables exist
            logger.info("🔍 Verifying tables...")
            
            tables = ['ml_predictions', 'ml_signals', 'ml_model_performance']
            
            for table in tables:
                result = await session.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}');")
                exists = result.scalar()
                if exists:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.error(f"❌ Table '{table}' does not exist")
                    return False
            
            logger.info("🎉 All ML Pipeline tables created successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False

async def main():
    """Main function"""
    
    logger.info("=" * 60)
    logger.info("DATABASE TEST AND ML PIPELINE TABLES CREATION")
    logger.info("=" * 60)
    
    success = await test_and_create_tables()
    
    if success:
        logger.info("✅ ML Pipeline tables are ready for use!")
    else:
        logger.error("❌ Failed to create ML Pipeline tables!")
    
    logger.info("=" * 60)
    logger.info("PROCESS COMPLETE")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
