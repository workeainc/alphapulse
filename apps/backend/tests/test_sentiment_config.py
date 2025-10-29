#!/usr/bin/env python3
"""
Test script for Enhanced Sentiment Analysis Configuration
"""

import sys
import os
sys.path.append('.')

def test_configuration():
    """Test the enhanced sentiment configuration"""
    print("🧪 Testing Enhanced Sentiment Analysis Configuration...")
    print("=" * 60)
    
    try:
        from config.enhanced_sentiment_config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Test Redis configuration
        print(f"\n📊 Redis Configuration:")
        print(f"   Host: {config.redis.host}")
        print(f"   Port: {config.redis.port}")
        print(f"   Database: {config.redis.db}")
        print(f"   URL: {config.get_redis_url()}")
        
        # Test Database configuration
        print(f"\n🗄️ Database Configuration:")
        print(f"   Host: {config.database.host}")
        print(f"   Port: {config.database.port}")
        print(f"   Database: {config.database.database}")
        print(f"   Username: {config.database.username}")
        
        # Test API configurations
        print(f"\n🔑 API Configurations:")
        print(f"   Twitter: {'✅ Configured' if config.is_twitter_configured() else '❌ Not configured'}")
        print(f"   Reddit: {'✅ Configured' if config.is_reddit_configured() else '❌ Not configured'}")
        print(f"   News: {'✅ Configured' if config.is_news_configured() else '❌ Not configured'}")
        print(f"   Telegram: {'✅ Configured' if config.is_telegram_configured() else '❌ Not configured'}")
        print(f"   Discord: {'✅ Configured' if config.is_discord_configured() else '❌ Not configured'}")
        
        # Test model configuration
        print(f"\n🤖 Model Configuration:")
        print(f"   Default Model: {config.model.default_model}")
        print(f"   Device: {config.model.device}")
        print(f"   Max Length: {config.model.max_text_length}")
        
        # Test processing configuration
        print(f"\n⚙️ Processing Configuration:")
        print(f"   Collection Interval: {config.processing.collection_interval_seconds}s")
        print(f"   Aggregation Interval: {config.processing.aggregation_interval_seconds}s")
        print(f"   Cache Timeout: {config.processing.cache_timeout_seconds}s")
        print(f"   Window Sizes: {config.processing.window_sizes}")
        
        # Test alert configuration
        print(f"\n🚨 Alert Configuration:")
        print(f"   Spike Threshold: {config.alerts.sentiment_spike_threshold}")
        print(f"   Webhook URL: {'✅ Configured' if config.alerts.webhook_url else '❌ Not configured'}")
        
        print(f"\n✅ Configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print("\n🔍 Testing Redis Connection...")
    try:
        import redis
        from config.enhanced_sentiment_config import get_config
        
        config = get_config()
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            socket_timeout=config.redis.socket_timeout,
            socket_connect_timeout=config.redis.socket_connect_timeout
        )
        
        # Test connection
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
        print("\n📋 Redis Setup Instructions:")
        print("1. Install Redis Server:")
        print("   - Windows: Download from https://redis.io/download")
        print("   - Or use Docker: docker run -d -p 6379:6379 redis:alpine")
        print("2. Start Redis Server:")
        print("   - Windows: redis-server")
        print("   - Or ensure Docker container is running")
        print("3. Test connection: redis-cli ping")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing Database Connection...")
    try:
        import asyncpg
        import asyncio
        from config.enhanced_sentiment_config import get_config
        
        config = get_config()
        
        async def test_connection():
            conn = await asyncpg.connect(
                host=config.database.host,
                port=config.database.port,
                database=config.database.database,
                user=config.database.username,
                password=config.database.password
            )
            
            # Test basic query
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
    print("🚀 Enhanced Sentiment Analysis Configuration Test")
    print("=" * 60)
    
    # Test configuration loading
    config_ok = test_configuration()
    
    # Test Redis connection
    redis_ok = test_redis_connection()
    
    # Test database connection
    db_ok = test_database_connection()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Redis Connection: {'✅ PASS' if redis_ok else '❌ FAIL'}")
    print(f"Database Connection: {'✅ PASS' if db_ok else '❌ FAIL'}")
    
    if config_ok and redis_ok and db_ok:
        print("\n🎉 All tests passed! Environment is ready for enhanced sentiment analysis.")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration and setup.")
        print("\n📋 Next Steps:")
        print("1. Install and start Redis Server")
        print("2. Ensure PostgreSQL/TimescaleDB is running")
        print("3. Set up API keys in environment variables (optional)")
        print("4. Run this test again")

if __name__ == "__main__":
    main()
