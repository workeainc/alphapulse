#!/usr/bin/env python3
"""
Simple database connection test
"""

import psycopg2
import asyncio
import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

# Test different connection strings
DB_URLS = [
    "postgresql://alpha_emon:Emon%4017711@localhost:5432/alphapulse",
    "postgresql://postgres:Emon%4017711@localhost:5432/alphapulse",
    "postgresql://alpha_emon:Emon%4017711@127.0.0.1:5432/alphapulse",
    "postgresql://postgres:Emon%4017711@127.0.0.1:5432/alphapulse"
]

def test_psycopg2_connection(url):
    """Test connection using psycopg2"""
    try:
        print(f"Testing psycopg2: {url}")
        conn = psycopg2.connect(url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ psycopg2 connection successful: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ psycopg2 connection failed: {e}")
        return False

def test_sqlalchemy_connection(url):
    """Test connection using SQLAlchemy"""
    try:
        print(f"Testing SQLAlchemy: {url}")
        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute("SELECT version();")
            version = result.fetchone()
            print(f"✅ SQLAlchemy connection successful: {version[0]}")
        return True
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {e}")
        return False

async def test_asyncpg_connection(url):
    """Test connection using asyncpg"""
    try:
        print(f"Testing asyncpg: {url}")
        # Convert postgresql:// to postgresql+asyncpg://
        async_url = url.replace('postgresql://', 'postgresql+asyncpg://')
        conn = await asyncpg.connect(async_url)
        version = await conn.fetchval("SELECT version();")
        print(f"✅ asyncpg connection successful: {version}")
        await conn.close()
        return True
    except Exception as e:
        print(f"❌ asyncpg connection failed: {e}")
        return False

async def test_sqlalchemy_async_connection(url):
    """Test connection using SQLAlchemy async"""
    try:
        print(f"Testing SQLAlchemy async: {url}")
        # Convert postgresql:// to postgresql+asyncpg://
        async_url = url.replace('postgresql://', 'postgresql+asyncpg://')
        engine = create_async_engine(async_url)
        async with engine.connect() as conn:
            result = await conn.execute("SELECT version();")
            version = result.fetchone()
            print(f"✅ SQLAlchemy async connection successful: {version[0]}")
        await engine.dispose()
        return True
    except Exception as e:
        print(f"❌ SQLAlchemy async connection failed: {e}")
        return False

async def main():
    """Run all connection tests"""
    print("🔍 Testing Database Connections")
    print("=" * 50)
    
    for url in DB_URLS:
        print(f"\n🧪 Testing URL: {url}")
        print("-" * 30)
        
        # Test psycopg2
        test_psycopg2_connection(url)
        
        # Test SQLAlchemy sync
        test_sqlalchemy_connection(url)
        
        # Test asyncpg
        await test_asyncpg_connection(url)
        
        # Test SQLAlchemy async
        await test_sqlalchemy_async_connection(url)
        
        print()

if __name__ == "__main__":
    asyncio.run(main())
