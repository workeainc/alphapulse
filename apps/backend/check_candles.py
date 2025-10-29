import asyncio
import asyncpg
from datetime import datetime, timedelta

async def check():
    pool = await asyncpg.create_pool(
        host='localhost',
        port=55433,
        database='alphapulse',
        user='alpha_emon',
        password='Emon_@17711'
    )
    
    async with pool.acquire() as conn:
        # Total websocket candles
        try:
            total_ws = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_data WHERE source='websocket'")
            print(f"\n💾 Total websocket candles stored: {total_ws}")
        except Exception as e:
            print(f"\n❌ Error counting websocket candles: {e}")
            total_ws = 0
        
        # If we have websocket candles, show recent ones
        if total_ws > 0:
            try:
                recent = await conn.fetch("""
                    SELECT symbol, timeframe, timestamp 
                    FROM ohlcv_data 
                    WHERE source='websocket' 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                print(f"\n📊 Recent WebSocket candles:")
                for r in recent:
                    print(f"  {r['symbol']} {r['timeframe']} @ {r['timestamp']}")
            except Exception as e:
                print(f"❌ Error fetching recent candles: {e}")
        else:
            print("\n⚠️ NO WEBSOCKET CANDLES FOUND IN DATABASE")
            print("   This means callbacks are NOT being triggered")
        
        # Check historical data
        try:
            total_hist = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_data WHERE source='historical_1year_download'")
            print(f"\n📦 Historical candles: {total_hist:,}")
        except:
            print("\n📦 Historical candles: (error counting)")
    
    await pool.close()

asyncio.run(check())

