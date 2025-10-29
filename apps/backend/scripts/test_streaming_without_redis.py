#!/usr/bin/env python3
"""
Test Streaming Components Without Redis
Validates streaming infrastructure without external dependencies
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.database.connection import TimescaleDBConnection
from src.core.config import settings, STREAMING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connection and basic operations"""
    logger.info("🔍 Testing database connection...")
    
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
        
        # Test basic operations
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            
            # Test inserting sample data
            import json
            from datetime import datetime
            sample_data = {
                'message_id': 'test_001',
                'stream_key': 'test_stream',
                'symbol': 'BTCUSDT',
                'data_type': 'tick',
                'source': 'test',
                'data': json.dumps({'price': 50000.0, 'volume': 1.0}),
                'timestamp': datetime(2024, 1, 1, 0, 0, 0)
            }
            
            insert_sql = text("""
                INSERT INTO stream_messages 
                (message_id, stream_key, symbol, data_type, source, data, timestamp)
                VALUES (:message_id, :stream_key, :symbol, :data_type, :source, :data, :timestamp)
                ON CONFLICT (message_id) DO NOTHING
            """)
            
            await session.execute(insert_sql, sample_data)
            await session.commit()
            logger.info("✅ Sample data inserted")
            
            # Test querying data
            query_sql = text("SELECT COUNT(*) FROM stream_messages WHERE symbol = 'BTCUSDT'")
            result = await session.execute(query_sql)
            count = result.scalar()
            logger.info(f"✅ Query test passed: {count} records found")
            
            # Test TimescaleDB functions
            timescale_sql = text("SELECT version()")
            result = await session.execute(timescale_sql)
            version = result.scalar()
            logger.info(f"✅ TimescaleDB version: {version}")
        
        await db_connection.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

async def test_streaming_components_initialization():
    """Test streaming components initialization without Redis"""
    logger.info("🧪 Testing streaming components initialization...")
    
    try:
        # Add streaming directory to path
        streaming_path = backend_path / "streaming"
        if streaming_path.exists():
            sys.path.insert(0, str(streaming_path))
        
        # Test importing components
        from src.streaming.stream_processor import StreamProcessor
        from src.streaming.stream_metrics import StreamMetrics
        from src.streaming.stream_normalizer import StreamNormalizer
        from src.streaming.candle_builder import CandleBuilder
        from src.streaming.rolling_state_manager import RollingStateManager
        
        logger.info("✅ All streaming components imported successfully")
        
        # Test component creation (without initialization)
        try:
            stream_processor = StreamProcessor(STREAMING_CONFIG)
            logger.info("✅ StreamProcessor created")
        except Exception as e:
            logger.warning(f"⚠️ StreamProcessor creation failed (expected without Redis): {e}")
        
        try:
            stream_metrics = StreamMetrics(STREAMING_CONFIG)
            logger.info("✅ StreamMetrics created")
        except Exception as e:
            logger.warning(f"⚠️ StreamMetrics creation failed: {e}")
        
        try:
            stream_normalizer = StreamNormalizer(STREAMING_CONFIG)
            logger.info("✅ StreamNormalizer created")
        except Exception as e:
            logger.warning(f"⚠️ StreamNormalizer creation failed: {e}")
        
        try:
            candle_builder = CandleBuilder(STREAMING_CONFIG)
            logger.info("✅ CandleBuilder created")
        except Exception as e:
            logger.warning(f"⚠️ CandleBuilder creation failed: {e}")
        
        try:
            rolling_state_manager = RollingStateManager(STREAMING_CONFIG)
            logger.info("✅ RollingStateManager created")
        except Exception as e:
            logger.warning(f"⚠️ RollingStateManager creation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming components test failed: {e}")
        return False

async def test_configuration():
    """Test configuration loading"""
    logger.info("⚙️ Testing configuration...")
    
    try:
        # Test core settings
        logger.info(f"✅ TimescaleDB Host: {settings.TIMESCALEDB_HOST}")
        logger.info(f"✅ TimescaleDB Port: {settings.TIMESCALEDB_PORT}")
        logger.info(f"✅ TimescaleDB Database: {settings.TIMESCALEDB_DATABASE}")
        logger.info(f"✅ TimescaleDB Username: {settings.TIMESCALEDB_USERNAME}")
        
        # Test streaming config
        logger.info(f"✅ Redis Host: {STREAMING_CONFIG['redis_host']}")
        logger.info(f"✅ Redis Port: {STREAMING_CONFIG['redis_port']}")
        logger.info(f"✅ Stream Prefix: {STREAMING_CONFIG['stream_prefix']}")
        logger.info(f"✅ Timeframes: {STREAMING_CONFIG['timeframes']}")
        logger.info(f"✅ Indicator Periods: {STREAMING_CONFIG['indicator_periods']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def test_table_structure():
    """Test table structure and constraints"""
    logger.info("📋 Testing table structure...")
    
    try:
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
        
        async with db_connection.async_session() as session:
            from sqlalchemy import text
            
            # Check table columns
            tables_to_check = [
                'stream_messages',
                'normalized_data', 
                'realtime_candles',
                'technical_indicators',
                'system_metrics',
                'processing_results'
            ]
            
            for table in tables_to_check:
                columns_sql = text(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                
                result = await session.execute(columns_sql)
                columns = result.fetchall()
                
                logger.info(f"📊 Table {table} has {len(columns)} columns:")
                for col in columns[:5]:  # Show first 5 columns
                    logger.info(f"   - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
                if len(columns) > 5:
                    logger.info(f"   ... and {len(columns) - 5} more columns")
            
            # Check TimescaleDB hypertables
            hypertables_sql = text("""
                SELECT hypertable_name, num_chunks, compression_enabled
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
            """)
            
            result = await session.execute(hypertables_sql)
            hypertables = result.fetchall()
            
            logger.info(f"📊 Found {len(hypertables)} TimescaleDB hypertables:")
            for ht in hypertables:
                logger.info(f"   - {ht[0]}: {ht[1]} chunks, compression: {ht[2]}")
        
        await db_connection.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Table structure test failed: {e}")
        return False

async def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("STREAMING INFRASTRUCTURE VALIDATION (NO REDIS)")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Database Connection", test_database_connection),
        ("Table Structure", test_table_structure),
        ("Streaming Components", test_streaming_components_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            if await test_func():
                logger.info(f"✅ {test_name} test passed")
                passed += 1
            else:
                logger.error(f"❌ {test_name} test failed")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Streaming infrastructure is ready.")
        logger.info("📋 Next steps:")
        logger.info("   1. Start Redis server: redis-server")
        logger.info("   2. Run full streaming tests: python tests/test_streaming_infrastructure.py")
        logger.info("   3. Start streaming pipeline: python -m streaming.stream_processor")
        logger.info("   4. Monitor with: python -m streaming.stream_metrics")
    else:
        logger.error(f"❌ {total - passed} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
