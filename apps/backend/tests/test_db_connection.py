#!/usr/bin/env python3
"""
Test Database Connection
"""
import asyncio
import asyncpg

async def test_db_connection():
    """Test database connection"""
    try:
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            database='alphapulse',
            user='alpha_emon',
            password='Emon_@17711'
        )
        print("✅ Database connection successful!")
        await conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_db_connection())
