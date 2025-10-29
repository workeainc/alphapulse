"""
Quick script to check database contents
"""
import asyncpg
import asyncio
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def check_database():
    """Check what's in the database"""
    
    conn = await asyncpg.connect(**DB_CONFIG)
    
    try:
        # Get count and date range
        row = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_signals,
                MIN(created_at) as oldest_signal,
                MAX(created_at) as newest_signal
            FROM live_signals
        """)
        
        print("=" * 80)
        print("DATABASE CONTENTS SUMMARY")
        print("=" * 80)
        print(f"Total Signals: {row['total_signals']}")
        print(f"Oldest Signal: {row['oldest_signal']}")
        print(f"Newest Signal: {row['newest_signal']}")
        
        if row['total_signals'] > 0:
            # Check if signals are recent (within last 24 hours) or old
            newest = row['newest_signal']
            age_hours = (datetime.now(newest.tzinfo) - newest).total_seconds() / 3600
            print(f"Newest signal age: {age_hours:.1f} hours ago")
        
        print("\n" + "=" * 80)
        print("SAMPLE SIGNALS (First 3)")
        print("=" * 80)
        
        # Get sample signals
        samples = await conn.fetch("""
            SELECT 
                signal_id,
                symbol,
                direction,
                confidence,
                pattern_type,
                created_at,
                sde_consensus::text as sde_data_preview
            FROM live_signals
            ORDER BY created_at DESC
            LIMIT 3
        """)
        
        for i, sig in enumerate(samples, 1):
            print(f"\nSignal {i}:")
            print(f"  ID: {sig['signal_id']}")
            print(f"  Symbol: {sig['symbol']}")
            print(f"  Direction: {sig['direction']}")
            print(f"  Confidence: {sig['confidence']:.2%}")
            print(f"  Pattern: {sig['pattern_type']}")
            print(f"  Created: {sig['created_at']}")
            
            # Check SDE data structure
            sde_preview = sig['sde_data_preview'][:200] if sig['sde_data_preview'] else 'None'
            print(f"  SDE Data Preview: {sde_preview}...")
        
        print("\n" + "=" * 80)
        print("CHECKING FOR MOCK DATA INDICATORS")
        print("=" * 80)
        
        # Check for indicators that suggest mock data
        mock_indicators = await conn.fetch("""
            SELECT 
                COUNT(*) FILTER (WHERE pattern_type LIKE '%mock%') as mock_pattern_count,
                COUNT(*) FILTER (WHERE pattern_type LIKE '%test%') as test_pattern_count,
                COUNT(*) FILTER (WHERE signal_id LIKE 'MOCK%') as mock_id_count,
                COUNT(*) FILTER (WHERE signal_id LIKE 'TEST%') as test_id_count,
                COUNT(*) as total
            FROM live_signals
        """)
        
        mock = mock_indicators[0]
        print(f"Signals with 'mock' in pattern_type: {mock['mock_pattern_count']}")
        print(f"Signals with 'test' in pattern_type: {mock['test_pattern_count']}")
        print(f"Signals with 'MOCK' in signal_id: {mock['mock_id_count']}")
        print(f"Signals with 'TEST' in signal_id: {mock['test_id_count']}")
        
        is_mock = (mock['mock_pattern_count'] > 0 or 
                   mock['test_pattern_count'] > 0 or 
                   mock['mock_id_count'] > 0 or 
                   mock['test_id_count'] > 0)
        
        print("\n" + "=" * 80)
        if is_mock:
            print("✅ VERDICT: Database contains MOCK/TEST data")
            print("   → Safe to clear and regenerate with real data")
        else:
            print("⚠️  VERDICT: Database does NOT show obvious mock indicators")
            print("   → Need to verify if this is historical Binance data")
            print("   → Check the pattern_type and signal creation logic")
        print("=" * 80)
        
        # Check symbols
        print("\n" + "=" * 80)
        print("SYMBOLS IN DATABASE")
        print("=" * 80)
        symbol_counts = await conn.fetch("""
            SELECT symbol, COUNT(*) as count
            FROM live_signals
            GROUP BY symbol
            ORDER BY count DESC
        """)
        
        for sym in symbol_counts:
            print(f"  {sym['symbol']}: {sym['count']} signals")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_database())

