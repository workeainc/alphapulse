#!/usr/bin/env python3
"""
Check 4H and 1D data status
"""

import asyncio
import asyncpg

async def check_data():
    db_url = 'postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse'
    conn = await asyncpg.connect(db_url)
    
    print("🔍 Checking 4H and 1D data status...")
    
    # Check 4H data
    for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
        count_4h = await conn.fetchval(f"SELECT COUNT(*) FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '4h'")
        count_1d = await conn.fetchval(f"SELECT COUNT(*) FROM ohlcv_data WHERE symbol = '{symbol}' AND timeframe = '1d'")
        print(f"✅ {symbol}: 4H={count_4h}, 1D={count_1d}")
    
    # Check if ohlcv_4h and ohlcv_1d tables exist
    try:
        count_4h_table = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_4h")
        count_1d_table = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_1d")
        print(f"📊 ohlcv_4h table: {count_4h_table} rows")
        print(f"📊 ohlcv_1d table: {count_1d_table} rows")
    except Exception as e:
        print(f"⚠️ Separate tables don't exist: {e}")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_data())
