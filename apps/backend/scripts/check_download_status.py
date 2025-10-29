#!/usr/bin/env python3
"""Quick script to check download status"""
import asyncio
import asyncpg

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def check():
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Check BTCUSDT 1m
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM ohlcv_data WHERE symbol='BTCUSDT' AND timeframe='1m'"
        )
        print(f"BTCUSDT 1m: {count:,} candles")
        
        # Check all symbols/timeframes
        print("\nAll data:")
        rows = await conn.fetch("""
            SELECT symbol, timeframe, COUNT(*) as count
            FROM ohlcv_data
            WHERE source = 'historical_1year'
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        for row in rows:
            print(f"  {row['symbol']} {row['timeframe']}: {row['count']:,}")
    finally:
        await conn.close()

asyncio.run(check())

