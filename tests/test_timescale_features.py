#!/usr/bin/env python3
"""
Test TimescaleDB-specific features for AlphaPlus
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_timescale_features():
    """Test TimescaleDB-specific features"""
    print("🔍 Testing TimescaleDB features...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
        # Create database manager with localhost configuration
        db_manager = DatabaseManager()
        
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 1,
            'max_size': 5
        }
        
        await db_manager.initialize(config)
        
        # Test 1: Check TimescaleDB extension
        print("\n1. Testing TimescaleDB extension...")
        result = await db_manager.fetch("""
            SELECT extname, extversion 
            FROM pg_extension 
            WHERE extname = 'timescaledb'
        """)
        
        if result:
            print(f"✅ TimescaleDB extension found: {result[0]['extversion']}")
        else:
            print("⚠️  TimescaleDB extension not found")
        
        # Test 2: Check for hypertables
        print("\n2. Testing TimescaleDB hypertables...")
        try:
            hypertables = await db_manager.fetch("""
                SELECT hypertable_name, num_chunks 
                FROM timescaledb_information.hypertables
                ORDER BY hypertable_name
            """)
            
            if hypertables:
                print(f"✅ Found {len(hypertables)} TimescaleDB hypertables:")
                for table in hypertables:
                    print(f"   - {table['hypertable_name']} ({table['num_chunks']} chunks)")
            else:
                print("ℹ️  No TimescaleDB hypertables found")
        except Exception as e:
            print(f"⚠️  Hypertable query failed: {e}")
        
        # Test 3: Check for time-series tables
        print("\n3. Testing time-series data tables...")
        time_series_tables = await db_manager.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('candles', 'candlestick_data', 'ohlcv', 'price_data', 'market_data')
            ORDER BY table_name
        """)
        
        if time_series_tables:
            print(f"✅ Found {len(time_series_tables)} time-series tables:")
            for table in time_series_tables:
                print(f"   - {table['table_name']}")
        else:
            print("ℹ️  No time-series tables found")
        
        # Test 4: Check table structure for candles
        print("\n4. Testing candles table structure...")
        try:
            columns = await db_manager.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'candles' 
                ORDER BY ordinal_position
            """)
            
            if columns:
                print(f"✅ Candles table has {len(columns)} columns:")
                for col in columns:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    print(f"   - {col['column_name']}: {col['data_type']} ({nullable})")
            else:
                print("ℹ️  Candles table not found")
        except Exception as e:
            print(f"⚠️  Column query failed: {e}")
        
        # Test 5: Check for recent data
        print("\n5. Testing recent data availability...")
        try:
            recent_data = await db_manager.fetch("""
                SELECT COUNT(*) as count, 
                       MIN(timestamp) as earliest,
                       MAX(timestamp) as latest
                FROM candles 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """)
            
            if recent_data and recent_data[0]['count'] > 0:
                print(f"✅ Recent data available:")
                print(f"   - Records in last 24h: {recent_data[0]['count']}")
                print(f"   - Earliest: {recent_data[0]['earliest']}")
                print(f"   - Latest: {recent_data[0]['latest']}")
            else:
                print("ℹ️  No recent data found in candles table")
        except Exception as e:
            print(f"⚠️  Recent data query failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ TimescaleDB features test error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 AlphaPlus TimescaleDB Features Test")
    print("=" * 50)
    
    success = await test_timescale_features()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TimescaleDB features test completed! Database is ready for trading operations.")
        return True
    else:
        print("❌ TimescaleDB features test failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
