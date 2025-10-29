#!/usr/bin/env python3
"""Kill stuck INSERT queries on ohlcv_data"""
import asyncio
import asyncpg

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def kill_stuck():
    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        # Find stuck INSERT queries
        stuck = await conn.fetch("""
            SELECT pid, now() - query_start as duration, query
            FROM pg_stat_activity
            WHERE query LIKE '%INSERT INTO ohlcv_data%'
            AND state = 'active'
            AND query_start < now() - interval '1 minute'
            ORDER BY query_start
        """)
        
        if not stuck:
            print("No stuck queries found")
            return
        
        print(f"Found {len(stuck)} stuck INSERT queries:")
        for q in stuck:
            print(f"  PID {q['pid']}: Running for {q['duration']}")
        
        # Kill them
        for q in stuck:
            try:
                await conn.execute(f"SELECT pg_terminate_backend({q['pid']})")
                print(f"  [OK] Killed PID {q['pid']}")
            except Exception as e:
                print(f"  [ERROR] Failed to kill PID {q['pid']}: {e}")
        
        print("\nStuck queries killed. Database should be responsive now.")
        
    finally:
        await conn.close()

asyncio.run(kill_stuck())

