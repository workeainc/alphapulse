#!/usr/bin/env python3
"""
Quick test for enhanced sentiment configuration
"""

import os
import sys
sys.path.append('.')

def test_config():
    """Test the enhanced sentiment configuration"""
    print("🧪 Testing Enhanced Sentiment Configuration...")
    
    try:
        from config.enhanced_sentiment_config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Test API configurations
        print(f"\n🔑 API Configurations:")
        print(f"   News API: {'✅ Configured' if config.is_news_configured() else '❌ Not configured'}")
        print(f"   Twitter API: {'✅ Configured' if config.is_twitter_configured() else '❌ Not configured'}")
        print(f"   Reddit API: {'✅ Configured' if config.is_reddit_configured() else '❌ Not configured'}")
        
        # Test Redis configuration
        print(f"\n📊 Redis Configuration:")
        print(f"   URL: {config.get_redis_url()}")
        
        # Test database configuration
        print(f"\n🗄️ Database Configuration:")
        print(f"   Host: {config.database.host}")
        print(f"   Database: {config.database.database}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_redis():
    """Test Redis connection"""
    print("\n🔍 Testing Redis Connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis connection successful")
        
        # Test basic operations
        r.set('test_sentiment', 'test_value')
        value = r.get('test_sentiment')
        r.delete('test_sentiment')
        
        if value == b'test_value':
            print("✅ Redis read/write operations successful")
            return True
        else:
            print("❌ Redis read/write operations failed")
            return False
            
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_database():
    """Test database connection"""
    print("\n🔍 Testing Database Connection...")
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711'
            )
            
            result = await conn.fetchval('SELECT 1')
            await conn.close()
            
            if result == 1:
                print("✅ Database connection successful")
                return True
            else:
                print("❌ Database query failed")
                return False
        
        return asyncio.run(test_connection())
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Quick Enhanced Sentiment Configuration Test")
    print("=" * 50)
    
    config_ok = test_config()
    redis_ok = test_redis()
    db_ok = test_database()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Redis Connection: {'✅ PASS' if redis_ok else '❌ FAIL'}")
    print(f"Database Connection: {'✅ PASS' if db_ok else '❌ FAIL'}")
    
    if config_ok and redis_ok and db_ok:
        print("\n🎉 All tests passed! Environment is ready for enhanced sentiment analysis.")
        return True
    else:
        print("\n⚠️ Some tests failed.")
        return False

if __name__ == "__main__":
    main()
