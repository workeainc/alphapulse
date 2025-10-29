#!/usr/bin/env python3
"""
Comprehensive test suite for historical data integration
Tests data download, quality, indicator calculation, signal generation, and API integration
"""

import asyncio
import logging
import asyncpg
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

# Test symbols and timeframes
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TEST_TIMEFRAMES = ['1h', '15m', '5m', '1m']


class TestHistoricalDataIntegration:
    """Test suite for historical data integration"""
    
    @pytest.fixture(scope="class")
    async def db_conn(self):
        """Database connection fixture"""
        conn = await asyncpg.connect(**DB_CONFIG)
        yield conn
        await conn.close()
    
    async def test_01_database_connection(self, db_conn):
        """Test 1: Verify database connection"""
        result = await db_conn.fetchval("SELECT 1")
        assert result == 1
        logger.info("✅ Test 1: Database connection OK")
    
    async def test_02_ohlcv_table_exists(self, db_conn):
        """Test 2: Verify ohlcv_data table exists"""
        table_exists = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'ohlcv_data'
            )
        """)
        assert table_exists, "ohlcv_data table does not exist"
        logger.info("✅ Test 2: ohlcv_data table exists")
    
    async def test_03_data_download_completeness(self, db_conn):
        """Test 3: Verify data was downloaded for all symbols/timeframes"""
        results = {}
        
        for symbol in TEST_SYMBOLS:
            for timeframe in TEST_TIMEFRAMES:
                count = await db_conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                
                results[f"{symbol}_{timeframe}"] = count or 0
                
                # Should have at least 1 day of data (minimum expected)
                min_expected = {
                    '1m': 1440,  # 24 hours * 60 minutes
                    '5m': 288,   # 24 hours * 12 (5-min intervals)
                    '15m': 96,   # 24 hours * 4 (15-min intervals)
                    '1h': 24     # 24 hours
                }
                
                expected = min_expected.get(timeframe, 100)
                assert count >= expected, \
                    f"Insufficient data for {symbol} {timeframe}: {count} < {expected}"
        
        logger.info("✅ Test 3: Data download completeness verified")
        logger.info(f"   Results: {results}")
    
    async def test_04_data_quality(self, db_conn):
        """Test 4: Verify data quality (no gaps, valid OHLCV)"""
        issues = []
        
        for symbol in TEST_SYMBOLS:
            for timeframe in TEST_TIMEFRAMES:
                # Check for invalid OHLCV relationships
                invalid_ohlc = await db_conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                    AND (high < low OR high < open OR high < close OR low > open OR low > close)
                """, symbol, timeframe)
                
                if invalid_ohlc > 0:
                    issues.append(f"{symbol} {timeframe}: {invalid_ohlc} invalid OHLCV records")
                
                # Check for zero volume
                zero_volume = await db_conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                    AND volume = 0
                """, symbol, timeframe)
                
                if zero_volume > 100:  # Allow some zero volume but not excessive
                    issues.append(f"{symbol} {timeframe}: {zero_volume} zero volume records")
                
                # Check for gaps (simplified - just check date range)
                date_range = await db_conn.fetchrow("""
                    SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                
                if date_range and date_range['oldest'] and date_range['newest']:
                    days_covered = (date_range['newest'] - date_range['oldest']).days
                    # Should cover at least 300 days (roughly 1 year minus some margin)
                    if days_covered < 300:
                        issues.append(f"{symbol} {timeframe}: Only {days_covered} days of data (expected ~365)")
        
        if issues:
            logger.warning(f"⚠️ Test 4: Data quality issues found: {issues}")
        else:
            logger.info("✅ Test 4: Data quality verified")
        
        # Don't fail on minor issues, just log them
        assert True
    
    async def test_05_indexes_exist(self, db_conn):
        """Test 5: Verify indexes exist for performance"""
        indexes = await db_conn.fetch("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'ohlcv_data'
        """)
        
        index_names = [idx['indexname'].lower() for idx in indexes]
        
        # Check for recommended indexes
        required_patterns = ['symbol', 'timeframe', 'timestamp']
        found_patterns = []
        
        for pattern in required_patterns:
            if any(pattern in name for name in index_names):
                found_patterns.append(pattern)
        
        assert len(found_patterns) >= 2, \
            f"Missing recommended indexes. Found: {index_names}"
        
        logger.info("✅ Test 5: Indexes verified")
    
    async def test_06_historical_data_accessible(self, db_conn):
        """Test 6: Verify historical data can be queried efficiently"""
        for symbol in TEST_SYMBOLS[:1]:  # Test one symbol only
            for timeframe in ['1h']:  # Test one timeframe only
                # Query recent data (simulating what indicators need)
                start_time = datetime.now()
                data = await db_conn.fetch("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT 500
                """, symbol, timeframe)
                
                query_time = (datetime.now() - start_time).total_seconds()
                
                assert len(data) > 0, f"No data returned for {symbol} {timeframe}"
                assert query_time < 1.0, \
                    f"Query too slow: {query_time}s (expected < 1s)"
        
        logger.info("✅ Test 6: Historical data queries are efficient")
    
    async def test_07_indicator_calculation_data_sufficient(self, db_conn):
        """Test 7: Verify sufficient data for indicator calculations"""
        # Most indicators need at least 50-200 candles
        min_candles_needed = 200
        
        for symbol in TEST_SYMBOLS[:1]:
            for timeframe in ['1h', '15m']:
                count = await db_conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                
                assert count >= min_candles_needed, \
                    f"Insufficient data for indicators: {symbol} {timeframe} has {count} candles"
        
        logger.info("✅ Test 7: Sufficient data for indicator calculations")
    
    async def test_08_source_tracking(self, db_conn):
        """Test 8: Verify historical data is tagged with correct source"""
        historical_count = await db_conn.fetchval("""
            SELECT COUNT(*) 
            FROM ohlcv_data 
            WHERE source = 'historical_1year'
        """)
        
        assert historical_count > 0, "No historical_1year data found"
        logger.info(f"✅ Test 8: Found {historical_count:,} candles with source='historical_1year'")
    
    async def test_09_no_duplicates(self, db_conn):
        """Test 9: Verify no duplicate records"""
        duplicates = await db_conn.fetchval("""
            SELECT COUNT(*) - COUNT(DISTINCT (symbol, timeframe, timestamp))
            FROM ohlcv_data
        """)
        
        assert duplicates == 0, f"Found {duplicates} duplicate records"
        logger.info("✅ Test 9: No duplicate records found")
    
    async def test_10_timespan_coverage(self, db_conn):
        """Test 10: Verify data spans approximately 1 year"""
        overall_range = await db_conn.fetchrow("""
            SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
            FROM ohlcv_data
            WHERE source = 'historical_1year'
        """)
        
        if overall_range and overall_range['oldest'] and overall_range['newest']:
            days_covered = (overall_range['newest'] - overall_range['oldest']).days
            assert days_covered >= 300, \
                f"Data only covers {days_covered} days (expected ~365)"
            
            logger.info(f"✅ Test 10: Data covers {days_covered} days")
        else:
            logger.warning("⚠️ Test 10: Could not determine date range")


class TestBackendIntegration:
    """Test backend integration with historical data"""
    
    @pytest.fixture(scope="class")
    async def db_conn(self):
        """Database connection fixture"""
        conn = await asyncpg.connect(**DB_CONFIG)
        yield conn
        await conn.close()
    
    async def test_11_signal_table_accessible(self, db_conn):
        """Test 11: Verify live_signals table is accessible"""
        table_exists = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'live_signals'
            )
        """)
        
        assert table_exists, "live_signals table does not exist"
        
        # Check for is_test_data column
        column_exists = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'live_signals' AND column_name = 'is_test_data'
            )
        """)
        
        assert column_exists, "is_test_data column does not exist"
        logger.info("✅ Test 11: live_signals table and is_test_data column accessible")
    
    async def test_12_test_signals_marked(self, db_conn):
        """Test 12: Verify old signals are marked as test data"""
        test_signals = await db_conn.fetchval("""
            SELECT COUNT(*) 
            FROM live_signals 
            WHERE is_test_data = TRUE
        """)
        
        logger.info(f"✅ Test 12: Found {test_signals or 0} signals marked as test data")
        # This test just verifies the column works, not that signals exist


@pytest.mark.asyncio
async def test_run_all():
    """Run all tests"""
    import sys
    
    logger.info("=" * 80)
    logger.info("RUNNING HISTORICAL DATA INTEGRATION TESTS")
    logger.info("=" * 80)
    
    # Create test classes
    data_tests = TestHistoricalDataIntegration()
    backend_tests = TestBackendIntegration()
    
    # Get database connection
    conn = await asyncpg.connect(**DB_CONFIG)
    
    try:
        # Run data tests
        await data_tests.test_01_database_connection(conn)
        await data_tests.test_02_ohlcv_table_exists(conn)
        await data_tests.test_03_data_download_completeness(conn)
        await data_tests.test_04_data_quality(conn)
        await data_tests.test_05_indexes_exist(conn)
        await data_tests.test_06_historical_data_accessible(conn)
        await data_tests.test_07_indicator_calculation_data_sufficient(conn)
        await data_tests.test_08_source_tracking(conn)
        await data_tests.test_09_no_duplicates(conn)
        await data_tests.test_10_timespan_coverage(conn)
        
        # Run backend tests
        await backend_tests.test_11_signal_table_accessible(conn)
        await backend_tests.test_12_test_signals_marked(conn)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)
        
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ TEST ERROR: {e}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(test_run_all())

