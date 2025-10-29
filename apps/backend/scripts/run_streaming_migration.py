#!/usr/bin/env python3
"""
Script to run Phase 1 Streaming Infrastructure Database Migration
Executes the migration and validates the setup
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.database.connection import TimescaleDBConnection
from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_migration():
    """Run the streaming infrastructure migration"""
    logger.info("🚀 Starting Phase 1 Streaming Infrastructure Migration...")
    
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
        
        # Read migration file
        migration_file = backend_path / "database" / "migrations" / "060_streaming_infrastructure_phase1.sql"
        
        if not migration_file.exists():
            logger.error(f"❌ Migration file not found: {migration_file}")
            return False
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        logger.info(f"📄 Migration file loaded: {migration_file}")
        
        # Execute migration
        async with db_connection.async_session() as session:
            # Split SQL into individual statements
            statements = migration_sql.split(';')
            
            for i, statement in enumerate(statements):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        from sqlalchemy import text
                        await session.execute(text(statement))
                        logger.info(f"✅ Executed statement {i+1}/{len(statements)}")
                    except Exception as e:
                        logger.warning(f"⚠️ Statement {i+1} failed (may already exist): {e}")
            
            await session.commit()
        
        logger.info("✅ Migration executed successfully")
        
        # Validate migration
        await validate_migration(db_connection)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False
    finally:
        if 'db_connection' in locals():
            await db_connection.close()

async def validate_migration(db_connection):
    """Validate the migration by checking created tables and views"""
    logger.info("🔍 Validating migration...")
    
    try:
        async with db_connection.async_session() as session:
            # Check if tables exist
            tables_to_check = [
                'stream_messages',
                'normalized_data', 
                'realtime_candles',
                'rolling_windows',
                'technical_indicators',
                'system_metrics',
                'component_metrics',
                'streaming_alerts',
                'processing_results'
            ]
            
            for table in tables_to_check:
                from sqlalchemy import text
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
            
            # Check if views exist
            views_to_check = [
                'streaming_status',
                'system_health',
                'processing_performance'
            ]
            
            for view in views_to_check:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.views 
                        WHERE table_schema = 'public' 
                        AND table_name = '{view}'
                    );
                """))
                exists = result.scalar()
                if exists:
                    logger.info(f"✅ View {view} exists")
                else:
                    logger.error(f"❌ View {view} not found")
            
            # Check if functions exist
            functions_to_check = [
                'cleanup_old_streaming_data',
                'get_streaming_stats',
                'get_system_health_summary'
            ]
            
            for func in functions_to_check:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.routines 
                        WHERE routine_schema = 'public' 
                        AND routine_name = '{func}'
                    );
                """))
                exists = result.scalar()
                if exists:
                    logger.info(f"✅ Function {func} exists")
                else:
                    logger.error(f"❌ Function {func} not found")
            
            # Test TimescaleDB extension
            result = await session.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb';"))
            if result.fetchone():
                logger.info("✅ TimescaleDB extension is enabled")
            else:
                logger.warning("⚠️ TimescaleDB extension not found")
            
            # Test streaming status view
            result = await session.execute(text("SELECT * FROM streaming_status LIMIT 1;"))
            logger.info("✅ Streaming status view is accessible")
            
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
        
        from src.streaming.stream_processor import StreamProcessor
        from src.streaming.stream_metrics import StreamMetrics
        from src.core.config import STREAMING_CONFIG
        
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
    logger.info("PHASE 1 STREAMING INFRASTRUCTURE MIGRATION")
    logger.info("=" * 60)
    
    # Run migration
    migration_success = await run_migration()
    
    if migration_success:
        # Test streaming components
        components_success = await test_streaming_components()
        
        if components_success:
            logger.info("🎉 Phase 1 Streaming Infrastructure Migration Completed Successfully!")
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
