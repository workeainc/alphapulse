#!/usr/bin/env python3
"""
Manual script to check for data gaps without filling them
Useful for diagnostics and monitoring
"""

import asyncio
import asyncpg
from datetime import datetime, timezone
import sys

async def check_gaps():
    """Check for gaps in data"""
    
    try:
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=55433,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711'
        )
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
        
        print("\n" + "="*80)
        print("📊 DATA GAP ANALYSIS")
        print("="*80 + "\n")
        
        total_gaps = 0
        
        for symbol in symbols:
            # Get last candle info
            query = """
                SELECT 
                    timestamp,
                    close,
                    volume
                FROM ohlcv_data
                WHERE symbol = $1 AND timeframe = '1m'
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            result = await db_pool.fetchrow(query, symbol)
            
            if result:
                last_time = result['timestamp']
                last_close = result['close']
                current_time = datetime.now(timezone.utc)
                gap_minutes = int((current_time - last_time).total_seconds() / 60)
                gap_hours = gap_minutes / 60
                gap_days = gap_hours / 24
                
                # Count total candles
                count_query = "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = $1 AND timeframe = '1m'"
                total_candles = await db_pool.fetchval(count_query, symbol)
                
                # Get first candle time
                first_query = """
                    SELECT timestamp 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m'
                    ORDER BY timestamp ASC
                    LIMIT 1
                """
                first_time = await db_pool.fetchval(first_query, symbol)
                
                # Calculate data coverage
                if first_time:
                    total_duration = (current_time - first_time).total_seconds() / 60
                    coverage_percent = (total_candles / total_duration * 100) if total_duration > 0 else 0
                else:
                    coverage_percent = 0
                
                # Determine status
                if gap_minutes < 5:
                    status = "✅ CURRENT"
                elif gap_minutes < 60:
                    status = "⚠️ MINOR GAP"
                    total_gaps += 1
                elif gap_minutes < 1440:
                    status = "⚠️ MODERATE GAP"
                    total_gaps += 1
                else:
                    status = "❌ MAJOR GAP"
                    total_gaps += 1
                
                print(f"{status} - {symbol}")
                print(f"   Last candle: {last_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   Last price: ${last_close:,.2f}")
                print(f"   Gap: {gap_minutes} minutes ({gap_hours:.1f}h / {gap_days:.2f}d)")
                print(f"   Total candles: {total_candles:,}")
                print(f"   First candle: {first_time.strftime('%Y-%m-%d %H:%M:%S UTC') if first_time else 'N/A'}")
                print(f"   Data coverage: {coverage_percent:.1f}%")
                print()
            else:
                print(f"❌ NO DATA - {symbol}")
                print(f"   No data in database - needs initial load")
                print()
                total_gaps += 1
        
        print("="*80)
        print(f"Summary: {total_gaps} symbols need backfill")
        print("="*80 + "\n")
        
        await db_pool.close()
        
        return total_gaps
        
    except Exception as e:
        print(f"\n❌ Error checking gaps: {e}\n")
        import traceback
        traceback.print_exc()
        return -1

if __name__ == "__main__":
    result = asyncio.run(check_gaps())
    sys.exit(0 if result >= 0 else 1)

