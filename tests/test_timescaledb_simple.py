#!/usr/bin/env python3
"""
Simplified TimescaleDB Test for Hard Example Buffer
Phase 5C: Misclassification Capture Implementation

Tests basic TimescaleDB connectivity and operations
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_simple_connection():
    """Test simple TimescaleDB connection"""
    try:
        logger.info("🧪 Testing simple TimescaleDB connection...")
        
        from ..database.connection_simple import get_simple_connection
        
        # Get connection
        db_connection = get_simple_connection()
        logger.info("✅ Simple connection created")
        
        # Test health check
        health = await db_connection.health_check()
        logger.info(f"📊 Database health: {health}")
        
        if health['healthy']:
            logger.info("✅ TimescaleDB connection test passed")
            return True
        else:
            logger.error(f"❌ Database unhealthy: {health}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple connection test failed: {e}")
        return False

async def test_basic_timescaledb_operations():
    """Test basic TimescaleDB operations"""
    try:
        logger.info("🧪 Testing basic TimescaleDB operations...")
        
        from ..database.connection_simple import get_simple_connection
        from sqlalchemy import text
        
        db_connection = get_simple_connection()
        
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Test 1: Basic query
            logger.info("📊 Test 1: Basic SELECT query...")
            result = await session.execute(text("SELECT version()"))
            version = result.fetchone()
            logger.info(f"✅ PostgreSQL version: {version[0] if version else 'Unknown'}")
            
            # Test 2: Check if TimescaleDB extension is available
            logger.info("📊 Test 2: Checking TimescaleDB extension...")
            result = await session.execute(text("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'timescaledb'
            """))
            timescaledb_info = result.fetchone()
            
            if timescaledb_info:
                logger.info(f"✅ TimescaleDB extension found: version {timescaledb_info[1]}")
            else:
                logger.warning("⚠️ TimescaleDB extension not found")
            
            # Test 3: Check if our tables exist
            logger.info("📊 Test 3: Checking table existence...")
            result = await session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('signals', 'candles', 'retrain_queue')
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            existing_tables = [row[0] for row in tables]
            logger.info(f"✅ Found tables: {existing_tables}")
            
            # Test 4: Check table structure
            if 'signals' in existing_tables:
                logger.info("📊 Test 4: Checking signals table structure...")
                result = await session.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'signals'
                    ORDER BY ordinal_position
                """))
                
                columns = result.fetchall()
                logger.info(f"✅ Signals table columns: {len(columns)}")
                for col in columns[:5]:  # Show first 5 columns
                    logger.info(f"   - {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Basic TimescaleDB operations test failed: {e}")
        return False

async def test_timescaledb_hypertable_features():
    """Test TimescaleDB hypertable features"""
    try:
        logger.info("🧪 Testing TimescaleDB hypertable features...")
        
        from ..database.connection_simple import get_simple_connection
        from sqlalchemy import text
        
        db_connection = get_simple_connection()
        
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Test 1: Check if signals table is a hypertable
            logger.info("📊 Test 1: Checking hypertable status...")
            result = await session.execute(text("""
                SELECT 
                    hypertable_name,
                    num_chunks,
                    compression_enabled
                FROM timescaledb_information.hypertables 
                WHERE hypertable_name = 'signals'
            """))
            
            hypertable_info = result.fetchone()
            if hypertable_info:
                logger.info(f"✅ Signals is a hypertable:")
                logger.info(f"   - Chunks: {hypertable_info[1]}")
                logger.info(f"   - Compression: {'Enabled' if hypertable_info[2] else 'Disabled'}")
            else:
                logger.warning("⚠️ Signals table is not a hypertable")
            
            # Test 2: Check chunk information
            if hypertable_info:
                logger.info("📊 Test 2: Checking chunk information...")
                try:
                    result = await session.execute(text("""
                        SELECT 
                            chunk_name,
                            range_start,
                            range_end,
                            is_compressed
                        FROM timescaledb_information.chunks 
                        WHERE hypertable_name = 'signals'
                        ORDER BY range_start DESC
                        LIMIT 3
                    """))
                    
                    chunks = result.fetchall()
                    logger.info(f"✅ Found {len(chunks)} chunks")
                    for chunk in chunks:
                        logger.info(f"   - {chunk[0]}: {chunk[1]} to {chunk[2]} (compressed: {chunk[3]})")
                except Exception as e:
                    logger.warning(f"⚠️ Chunk query failed: {e}")
            else:
                logger.info("📊 Test 2: Skipping chunk check (not a hypertable)")
            
            # Test 3: Test time bucket function
            logger.info("📊 Test 3: Testing time_bucket function...")
            try:
                result = await session.execute(text("""
                    SELECT 
                        time_bucket('1 hour', NOW()) as current_hour
                """))
                
                time_bucket = result.fetchone()
                if time_bucket:
                    logger.info(f"✅ Time bucket function works: {time_bucket[0]}")
                else:
                    logger.warning("⚠️ Time bucket function returned no results")
                    
            except Exception as e:
                logger.warning(f"⚠️ Time bucket function not available: {e}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ TimescaleDB hypertable features test failed: {e}")
        return False

async def test_data_insertion_and_retrieval():
    """Test data insertion and retrieval"""
    try:
        logger.info("🧪 Testing data insertion and retrieval...")
        
        from ..database.connection_simple import get_simple_connection
        from sqlalchemy import text
        
        db_connection = get_simple_connection()
        
        session_factory = await db_connection.get_async_session()
        async with session_factory as session:
            # Test 1: Insert test signal
            logger.info("📊 Test 1: Inserting test signal...")
            test_signal = {
                'label': 'buy',
                'pred': 'sell',
                'proba': 0.45,
                'ts': datetime.now(),
                'symbol': 'TEST_BTC',
                'tf': '1h',
                'features': json.dumps({'rsi': 30, 'macd': -0.5}),
                'model_id': 'test_model_v1',
                'outcome': None,
                'realized_rr': None,
                'latency_ms': 150
            }
            
            insert_query = text("""
                INSERT INTO signals (
                    label, pred, proba, ts, symbol, tf, features, 
                    model_id, outcome, realized_rr, latency_ms
                ) VALUES (
                    :label, :pred, :proba, :ts, :symbol, :tf, :features,
                    :model_id, :outcome, :realized_rr, :latency_ms
                ) RETURNING id, created_at
            """)
            
            result = await session.execute(insert_query, test_signal)
            inserted_row = result.fetchone()
            signal_id = inserted_row[0]
            
            await session.commit()
            logger.info(f"✅ Inserted test signal with ID: {signal_id}")
            
            # Test 2: Retrieve the inserted signal
            logger.info("📊 Test 2: Retrieving inserted signal...")
            select_query = text("""
                SELECT id, label, pred, proba, symbol, tf, created_at
                FROM signals 
                WHERE id = :signal_id
            """)
            
            result = await session.execute(select_query, {'signal_id': signal_id})
            retrieved_signal = result.fetchone()
            
            if retrieved_signal:
                logger.info(f"✅ Retrieved signal: ID={retrieved_signal[0]}, Label={retrieved_signal[1]}, Pred={retrieved_signal[2]}")
            else:
                logger.error("❌ Failed to retrieve inserted signal")
                return False
            
            # Test 3: Test time-based query
            logger.info("📊 Test 3: Testing time-based query...")
            time_query = text("""
                SELECT COUNT(*) as signal_count
                FROM signals 
                WHERE ts >= :start_time
                AND symbol = :symbol
            """)
            
            start_time = datetime.now() - timedelta(hours=1)
            result = await session.execute(time_query, {
                'start_time': start_time,
                'symbol': 'TEST_BTC'
            })
            
            count_row = result.fetchone()
            signal_count = count_row[0] if count_row else 0
            logger.info(f"✅ Time-based query: {signal_count} signals in last hour")
            
            # Clean up test data
            logger.info("🧹 Cleaning up test data...")
            delete_query = text("DELETE FROM signals WHERE id = :signal_id")
            await session.execute(delete_query, {'signal_id': signal_id})
            await session.commit()
            logger.info(f"✅ Cleaned up test signal {signal_id}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Data insertion and retrieval test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Simplified TimescaleDB Tests")
    
    test_results = []
    
    # Test 1: Simple Connection
    test_results.append(("Simple Connection", await test_simple_connection()))
    
    # Test 2: Basic TimescaleDB Operations
    test_results.append(("Basic Operations", await test_basic_timescaledb_operations()))
    
    # Test 3: Hypertable Features
    test_results.append(("Hypertable Features", await test_timescaledb_hypertable_features()))
    
    # Test 4: Data Operations
    test_results.append(("Data Operations", await test_data_insertion_and_retrieval()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All TimescaleDB tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
