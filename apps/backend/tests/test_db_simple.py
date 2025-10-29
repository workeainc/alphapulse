#!/usr/bin/env python3
"""
Simple database connection test for data versioning tables
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def test_database_connection():
    """Test basic database connection"""
    try:
        logger.info("🧪 Testing database connection...")
        
        engine = create_async_engine(DATABASE_URL)
        
        async with engine.begin() as conn:
            # Test basic connection
            result = await conn.execute(text("SELECT version()"))
            version = result.fetchone()
            logger.info(f"✅ Connected to database: {version[0][:50]}...")
            
            # Test if our tables exist
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name IN ('signals', 'candles', 'retrain_queue')
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            logger.info(f"✅ Found {len(tables)} tables:")
            for table in tables:
                logger.info(f"   - {table[0]}")
            
            # Test if hypertables exist
            result = await conn.execute(text("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_name IN ('signals', 'candles')
                ORDER BY hypertable_name
            """))
            
            hypertables = result.fetchall()
            logger.info(f"✅ Found {len(hypertables)} hypertables:")
            for ht in hypertables:
                logger.info(f"   - {ht[0]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False
    finally:
        await engine.dispose()

async def test_basic_operations():
    """Test basic CRUD operations"""
    try:
        logger.info("🧪 Testing basic operations...")
        
        engine = create_async_engine(DATABASE_URL)
        
        async with engine.begin() as conn:
            # Test inserting a signal
            test_signal = {
                'label': 'BUY',
                'pred': 'BUY',
                'proba': 0.85,
                'ts': datetime.now(),
                'symbol': 'BTCUSDT',
                'tf': '1h',
                'features': '{"rsi": 25.5, "macd": 0.0023}',
                'model_id': 'test_model_v1',
                'outcome': None,
                'realized_rr': None,
                'latency_ms': 45
            }
            
            insert_query = text("""
                INSERT INTO signals (
                    label, pred, proba, ts, symbol, tf, features, 
                    model_id, outcome, realized_rr, latency_ms
                ) VALUES (
                    :label, :pred, :proba, :ts, :symbol, :tf, :features,
                    :model_id, :outcome, :realized_rr, :latency_ms
                ) RETURNING id
            """)
            
            result = await conn.execute(insert_query, test_signal)
            signal_id = result.fetchone()[0]
            logger.info(f"✅ Inserted signal with ID: {signal_id}")
            
            # Test retrieving the signal
            select_query = text("""
                SELECT id, label, symbol, tf, model_id
                FROM signals
                WHERE id = :signal_id
            """)
            
            result = await conn.execute(select_query, {'signal_id': signal_id})
            signal = result.fetchone()
            logger.info(f"✅ Retrieved signal: {signal}")
            
            # Test updating the signal
            update_query = text("""
                UPDATE signals 
                SET outcome = 'win', realized_rr = 2.5
                WHERE id = :signal_id
            """)
            
            await conn.execute(update_query, {'signal_id': signal_id})
            logger.info(f"✅ Updated signal {signal_id}")
            
            # Test inserting a candle
            test_candle = {
                'symbol': 'BTCUSDT',
                'tf': '1h',
                'ts': datetime.now(),
                'o': 45000.0,
                'h': 45100.0,
                'l': 44900.0,
                'c': 45050.0,
                'v': 1250.5,
                'vwap': 45025.0,
                'taker_buy_vol': 800.3,
                'features': '{"ema_20": 44900.0, "rsi": 55.2}'
            }
            
            candle_query = text("""
                INSERT INTO candles (
                    symbol, tf, ts, o, h, l, c, v, vwap, taker_buy_vol, features
                ) VALUES (
                    :symbol, :tf, :ts, :o, :h, :l, :c, :v, :vwap, :taker_buy_vol, :features
                ) RETURNING id
            """)
            
            result = await conn.execute(candle_query, test_candle)
            candle_id = result.fetchone()[0]
            logger.info(f"✅ Inserted candle with ID: {candle_id}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Basic operations test failed: {e}")
        return False
    finally:
        await engine.dispose()

async def main():
    """Main test function"""
    logger.info("🚀 Starting simple database tests")
    logger.info("=" * 50)
    
    # Test 1: Database connection
    connection_ok = await test_database_connection()
    
    # Test 2: Basic operations
    operations_ok = await test_basic_operations()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Database Connection: {'✅ PASSED' if connection_ok else '❌ FAILED'}")
    logger.info(f"Basic Operations:   {'✅ PASSED' if operations_ok else '❌ FAILED'}")
    
    if connection_ok and operations_ok:
        logger.info("🎉 ALL TESTS PASSED! Database is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Please review the implementation.")
    
    return connection_ok and operations_ok

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        exit(1)
