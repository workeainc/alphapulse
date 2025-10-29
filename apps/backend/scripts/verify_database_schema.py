#!/usr/bin/env python3
"""
Verify database schema and connection
Ensures ohlcv_data table exists with correct structure and indexes
"""

import asyncio
import logging
import asyncpg
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration (matching main.py)
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def verify_connection() -> bool:
    """Verify database connection works"""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        await conn.execute("SELECT 1")
        await conn.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def verify_table_exists(conn, table_name: str) -> bool:
    """Check if table exists"""
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = $1
        )
    """
    exists = await conn.fetchval(query, table_name)
    return exists

async def get_table_columns(conn, table_name: str) -> List[Dict[str, Any]]:
    """Get all columns for a table"""
    query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
    """
    rows = await conn.fetch(query, table_name)
    return [dict(row) for row in rows]

async def verify_ohlcv_table_structure(conn) -> Dict[str, Any]:
    """Verify ohlcv_data table has correct structure"""
    results = {
        'table_exists': False,
        'has_required_columns': False,
        'columns': [],
        'missing_columns': [],
        'is_hypertable': False,
        'indexes': []
    }
    
    # Check if table exists
    results['table_exists'] = await verify_table_exists(conn, 'ohlcv_data')
    
    if not results['table_exists']:
        logger.error("❌ Table 'ohlcv_data' does not exist!")
        return results
    
    logger.info("✅ Table 'ohlcv_data' exists")
    
    # Get columns
    results['columns'] = await get_table_columns(conn, 'ohlcv_data')
    column_names = [col['column_name'] for col in results['columns']]
    
    # Required columns
    required_columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source']
    results['missing_columns'] = [col for col in required_columns if col not in column_names]
    
    if results['missing_columns']:
        logger.warning(f"⚠️ Missing columns: {results['missing_columns']}")
    else:
        results['has_required_columns'] = True
        logger.info("✅ All required columns present")
    
    # Check if it's a TimescaleDB hypertable
    try:
        hypertable_check = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'ohlcv_data'
            )
        """)
        results['is_hypertable'] = hypertable_check
        if hypertable_check:
            logger.info("✅ Table is a TimescaleDB hypertable")
        else:
            logger.warning("⚠️ Table is NOT a TimescaleDB hypertable (performance may be suboptimal)")
    except Exception as e:
        logger.warning(f"⚠️ Could not check hypertable status (TimescaleDB may not be installed): {e}")
    
    # Check indexes
    try:
        indexes = await conn.fetch("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'ohlcv_data'
            ORDER BY indexname
        """)
        results['indexes'] = [{'name': idx['indexname'], 'def': idx['indexdef']} for idx in indexes]
        
        # Check for recommended indexes
        index_names = [idx['name'].lower() for idx in results['indexes']]
        recommended_indexes = [
            ('idx_ohlcv_symbol_timeframe', 'symbol, timeframe'),
            ('idx_ohlcv_timestamp', 'timestamp'),
            ('idx_ohlcv_symbol_timestamp', 'symbol, timestamp')
        ]
        
        for idx_name, idx_cols in recommended_indexes:
            if any(idx_name.lower() in name for name in index_names):
                logger.info(f"✅ Index found: {idx_name}")
            else:
                logger.warning(f"⚠️ Recommended index missing: {idx_name} on ({idx_cols})")
        
        logger.info(f"✅ Found {len(results['indexes'])} indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not check indexes: {e}")
    
    return results

async def check_unique_constraint(conn) -> bool:
    """Check if unique constraint exists on (symbol, timeframe, timestamp)"""
    try:
        constraints = await conn.fetch("""
            SELECT conname, contype
            FROM pg_constraint
            WHERE conrelid = 'ohlcv_data'::regclass
            AND contype = 'u'
        """)
        
        if constraints:
            logger.info(f"✅ Unique constraint(s) found: {[c['conname'] for c in constraints]}")
            return True
        else:
            logger.warning("⚠️ No unique constraint on (symbol, timeframe, timestamp) - duplicates may occur")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Could not check constraints: {e}")
        return False

async def get_data_counts(conn) -> Dict[str, int]:
    """Get counts of existing data"""
    counts = {}
    
    try:
        # Total candles
        total = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_data")
        counts['total_candles'] = total or 0
        
        # By symbol
        symbol_counts = await conn.fetch("""
            SELECT symbol, COUNT(*) as count
            FROM ohlcv_data
            GROUP BY symbol
            ORDER BY count DESC
        """)
        counts['by_symbol'] = {row['symbol']: row['count'] for row in symbol_counts}
        
        # By timeframe
        tf_counts = await conn.fetch("""
            SELECT timeframe, COUNT(*) as count
            FROM ohlcv_data
            GROUP BY timeframe
            ORDER BY count DESC
        """)
        counts['by_timeframe'] = {row['timeframe']: row['count'] for row in tf_counts}
        
        # By source
        source_counts = await conn.fetch("""
            SELECT source, COUNT(*) as count
            FROM ohlcv_data
            GROUP BY source
            ORDER BY count DESC
        """)
        counts['by_source'] = {row['source']: row['count'] for row in source_counts}
        
        # Latest timestamp
        latest = await conn.fetchval("""
            SELECT MAX(timestamp) FROM ohlcv_data
        """)
        counts['latest_timestamp'] = latest
        
        # Oldest timestamp
        oldest = await conn.fetchval("""
            SELECT MIN(timestamp) FROM ohlcv_data
        """)
        counts['oldest_timestamp'] = oldest
        
    except Exception as e:
        logger.warning(f"⚠️ Could not get data counts: {e}")
    
    return counts

async def verify_schema() -> Dict[str, Any]:
    """Main verification function"""
    results = {
        'connection_ok': False,
        'ohlcv_table': {},
        'unique_constraint': False,
        'data_counts': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Test connection
        if not await verify_connection():
            results['errors'].append("Database connection failed")
            return results
        
        results['connection_ok'] = True
        
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        
        try:
            # Verify ohlcv_data table
            results['ohlcv_table'] = await verify_ohlcv_table_structure(conn)
            
            if not results['ohlcv_table']['table_exists']:
                results['errors'].append("ohlcv_data table does not exist")
            elif not results['ohlcv_table']['has_required_columns']:
                results['errors'].append(f"Missing columns: {results['ohlcv_table']['missing_columns']}")
            
            # Check unique constraint
            results['unique_constraint'] = await check_unique_constraint(conn)
            
            # Get data counts
            results['data_counts'] = await get_data_counts(conn)
            
        finally:
            await conn.close()
            
    except Exception as e:
        logger.error(f"❌ Schema verification failed: {e}")
        results['errors'].append(str(e))
    
    return results

async def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("DATABASE SCHEMA VERIFICATION")
    logger.info("=" * 80)
    logger.info("Verifying database connection and ohlcv_data table structure")
    logger.info("=" * 80)
    
    results = await verify_schema()
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 80)
    
    # Connection
    if results['connection_ok']:
        logger.info("✅ Database connection: OK")
    else:
        logger.error("❌ Database connection: FAILED")
    
    # Table structure
    ohlcv = results['ohlcv_table']
    if ohlcv.get('table_exists'):
        logger.info("✅ ohlcv_data table: EXISTS")
        
        if ohlcv.get('has_required_columns'):
            logger.info("✅ Required columns: PRESENT")
        else:
            logger.error(f"❌ Missing columns: {ohlcv.get('missing_columns', [])}")
        
        if ohlcv.get('is_hypertable'):
            logger.info("✅ TimescaleDB hypertable: YES")
        else:
            logger.warning("⚠️ TimescaleDB hypertable: NO")
        
        logger.info(f"✅ Indexes found: {len(ohlcv.get('indexes', []))}")
    
    # Data counts
    counts = results['data_counts']
    if counts:
        logger.info(f"\n📊 Existing Data:")
        logger.info(f"   Total candles: {counts.get('total_candles', 0):,}")
        
        if counts.get('by_symbol'):
            logger.info(f"   By symbol:")
            for symbol, count in counts['by_symbol'].items():
                logger.info(f"     {symbol}: {count:,}")
        
        if counts.get('by_timeframe'):
            logger.info(f"   By timeframe:")
            for tf, count in counts['by_timeframe'].items():
                logger.info(f"     {tf}: {count:,}")
        
        if counts.get('oldest_timestamp') and counts.get('latest_timestamp'):
            logger.info(f"   Date range: {counts['oldest_timestamp']} to {counts['latest_timestamp']}")
    
    # Errors
    if results['errors']:
        logger.error(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            logger.error(f"   - {error}")
    
    logger.info("=" * 80)
    
    # Return success status
    if results['errors']:
        logger.error("\n❌ Schema verification FAILED")
        return False
    else:
        logger.info("\n✅ Schema verification PASSED")
        return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

