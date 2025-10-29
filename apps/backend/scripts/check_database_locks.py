#!/usr/bin/env python3
"""Check for database locks that might be blocking inserts"""
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
        # Check for locks
        locks = await conn.fetch("""
            SELECT 
                l.locktype,
                l.relation::regclass as table_name,
                l.mode,
                l.granted,
                a.query,
                a.state,
                a.wait_event_type,
                a.wait_event
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation = 'ohlcv_data'::regclass::oid
               OR a.query LIKE '%ohlcv_data%'
            ORDER BY l.granted, a.query_start
        """)
        
        if locks:
            print(f"Found {len(locks)} locks:")
            for lock in locks:
                print(f"  - {lock['table_name']}: {lock['mode']} ({'granted' if lock['granted'] else 'WAITING'})")
                if lock['query']:
                    print(f"    Query: {lock['query'][:100]}...")
        else:
            print("No locks found on ohlcv_data table")
        
        # Check connection count
        conns = await conn.fetchval("SELECT COUNT(*) FROM pg_stat_activity")
        print(f"\nActive connections: {conns}")
        
        # Check for long-running queries
        long_queries = await conn.fetch("""
            SELECT 
                pid,
                now() - query_start as duration,
                state,
                query
            FROM pg_stat_activity
            WHERE state = 'active'
            AND query_start < now() - interval '5 seconds'
            ORDER BY query_start
        """)
        
        if long_queries:
            print(f"\nLong-running queries ({len(long_queries)}):")
            for q in long_queries:
                print(f"  - Duration: {q['duration']}, Query: {q['query'][:80]}...")
        
    finally:
        await conn.close()

asyncio.run(check())

