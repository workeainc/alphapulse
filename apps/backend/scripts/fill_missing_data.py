#!/usr/bin/env python3
"""Fill missing historical data gaps"""
import asyncio
import asyncpg
import logging
from datetime import datetime, timezone, timedelta
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

BINANCE_BASE_URL = 'https://api.binance.com/api/v3/klines'

TIMEFRAME_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}

async def find_gaps(conn, symbol, timeframe):
    """Find missing date ranges"""
    rows = await conn.fetch("""
        WITH date_series AS (
            SELECT generate_series(
                (SELECT MIN(timestamp) FROM ohlcv_data WHERE symbol=$1 AND timeframe=$2),
                (SELECT MAX(timestamp) FROM ohlcv_data WHERE symbol=$1 AND timeframe=$2),
                (%s || ' minutes')::interval
            ) as expected_time
        )
        SELECT expected_time
        FROM date_series
        WHERE NOT EXISTS (
            SELECT 1 FROM ohlcv_data 
            WHERE symbol=$1 AND timeframe=$2 
            AND timestamp = expected_time
        )
        ORDER BY expected_time
    """, symbol, timeframe, TIMEFRAME_MINUTES[timeframe])
    
    return [r['expected_time'] for r in rows]

async def fetch_gap_data(session, symbol, timeframe, start_date, end_date):
    """Fetch data for a specific gap"""
    interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'}
    url = f"{BINANCE_BASE_URL}?symbol={symbol}&interval={interval_map[timeframe]}&startTime={int(start_date.timestamp() * 1000)}&endTime={int(end_date.timestamp() * 1000)}&limit=1000"
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                logger.warning(f"API error {response.status} for {symbol} {timeframe}")
                return []
    except Exception as e:
        logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
        return []

async def fill_missing(symbol, timeframe, expected_count):
    """Fill missing data for one symbol/timeframe"""
    conn = await asyncpg.connect(**DB_CONFIG)
    session = aiohttp.ClientSession()
    
    try:
        # Get current count
        initial_count = await conn.fetchval("""
            SELECT COUNT(*) FROM ohlcv_data 
            WHERE symbol=$1 AND timeframe=$2 AND source='historical_1year'
        """, symbol, timeframe) or 0
        
        missing = expected_count - initial_count
        if missing <= 0:
            logger.info(f"[OK] {symbol} {timeframe}: Complete ({initial_count:,} candles)")
            return
        
        logger.info(f"[FILL] {symbol} {timeframe}: Missing {missing:,} candles ({initial_count:,}/{expected_count:,})")
        
        # Get date range
        result = await conn.fetchrow("""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM ohlcv_data 
            WHERE symbol=$1 AND timeframe=$2
        """, symbol, timeframe)
        
        if not result or not result['min_ts']:
            logger.warning(f"[SKIP] {symbol} {timeframe}: No existing data to determine gaps")
            return
        
        min_ts = result['min_ts']
        max_ts = result['max_ts']
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)
        
        # Download missing period
        if max_ts < end_date - timedelta(hours=1):
            # Missing recent data
            gap_start = max_ts + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
            logger.info(f"[FILL] {symbol} {timeframe}: Filling gap from {gap_start} to {end_date}")
            
            current_date = gap_start
            fetched_total = 0
            consecutive_errors = 0
            
            while current_date < end_date and consecutive_errors < 10:
                next_end = min(current_date + timedelta(days=30), end_date)
                klines = await fetch_gap_data(session, symbol, timeframe, current_date, next_end)
                
                if klines and len(klines) > 0:
                    values = []
                    for k in klines:
                        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                        values.append((
                            symbol, timeframe, ts,
                            float(k[1]), float(k[2]), float(k[3]),
                            float(k[4]), float(k[5]), 'historical_1year'
                        ))
                    
                    if values:
                        try:
                            await conn.executemany("""
                                INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                            """, values)
                            fetched_total += len(values)
                            consecutive_errors = 0  # Reset on success
                            if fetched_total % 100 == 0 or len(values) > 50:
                                logger.info(f"  [PROGRESS] Fetched {fetched_total:,} candles...")
                        except Exception as e:
                            logger.error(f"  [ERROR] Storage failed: {e}")
                            consecutive_errors += 1
                    
                    if klines:
                        last_ts = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc)
                        current_date = last_ts + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
                    else:
                        break
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.warning(f"  [WARN] Too many consecutive API failures, skipping gap")
                        break
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                
                await asyncio.sleep(0.5)
            
            # Verify
            final_count = await conn.fetchval("""
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol=$1 AND timeframe=$2 AND source='historical_1year'
            """, symbol, timeframe) or 0
            
            added = final_count - initial_count
            logger.info(f"[DONE] {symbol} {timeframe}: Now has {final_count:,} candles (added {added:,}, still missing {expected_count - final_count:,})")
        else:
            logger.info(f"[OK] {symbol} {timeframe}: Data is up to date (max timestamp: {max_ts})")
        
    finally:
        await conn.close()
        await session.close()

async def main():
    # Define what needs to be filled
    tasks = [
        ('SOLUSDT', '1h', 8760),
        ('ADAUSDT', '1m', 525600),
        ('BNBUSDT', '1m', 525600),
        ('ETHUSDT', '1m', 525600),
        ('ETHUSDT', '15m', 35040),
    ]
    
    for symbol, timeframe, expected in tasks:
        await fill_missing(symbol, timeframe, expected)
        await asyncio.sleep(2)

asyncio.run(main())

