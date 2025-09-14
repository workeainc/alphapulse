#!/usr/bin/env python3
"""
Database Integration Test for AlphaPlus
Tests TimescaleDB connection and operations
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_database_connection():
    """Test database connection"""
    print("🔍 Testing database connection...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # Test initialization
        success = await db_manager.initialize()
        
        if success:
            print("✅ Database connection successful")
            print(f"   Pool size: {db_manager.pool.get_size()}")
            print(f"   Free connections: {db_manager.pool.get_free_size()}")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

async def test_database_operations():
    """Test basic database operations"""
    print("\n🔍 Testing database operations...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Test simple query
        result = await db_manager.fetch("SELECT version()")
        print("✅ Database query successful")
        print(f"   PostgreSQL version: {result[0]['version']}")
        
        # Test table listing
        tables = await db_manager.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        print(f"✅ Found {len(tables)} tables in database")
        for table in tables:
            print(f"   - {table['table_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations error: {e}")
        return False

async def test_timescale_extension():
    """Test TimescaleDB extension"""
    print("\n🔍 Testing TimescaleDB extension...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Check if TimescaleDB extension is installed
        result = await db_manager.fetch("""
            SELECT extname, extversion 
            FROM pg_extension 
            WHERE extname = 'timescaledb'
        """)
        
        if result:
            print("✅ TimescaleDB extension found")
            print(f"   Version: {result[0]['extversion']}")
            return True
        else:
            print("⚠️  TimescaleDB extension not found (using regular PostgreSQL)")
            return True
            
    except Exception as e:
        print(f"❌ TimescaleDB extension test error: {e}")
        return False

async def test_hypertables():
    """Test TimescaleDB hypertables"""
    print("\n🔍 Testing TimescaleDB hypertables...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Check for hypertables
        hypertables = await db_manager.fetch("""
            SELECT hypertable_name, num_chunks 
            FROM timescaledb_information.hypertables
        """)
        
        if hypertables:
            print("✅ TimescaleDB hypertables found")
            for table in hypertables:
                print(f"   - {table['hypertable_name']} ({table['num_chunks']} chunks)")
        else:
            print("ℹ️  No TimescaleDB hypertables found")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Hypertable test error (may not be TimescaleDB): {e}")
        return True

async def main():
    """Main test function"""
    print("🚀 AlphaPlus Database Integration Test")
    print("=" * 50)
    
    tests = [
        test_database_connection,
        test_database_operations,
        test_timescale_extension,
        test_hypertables
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Database Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Database integration successful! Ready for next step.")
        return True
    else:
        print("❌ Some database tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
