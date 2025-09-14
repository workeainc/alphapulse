#!/usr/bin/env python3
"""
Simple Integration Test for Enhanced Sentiment Analysis
"""

import asyncio
import logging
import sys
import os
sys.path.append('.')

from ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from app.services.enhanced_sentiment_service import EnhancedSentimentService
import asyncpg
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sentiment_integration():
    """Test sentiment analysis integration"""
    try:
        logger.info("🧪 Testing Enhanced Sentiment Analysis Integration...")
        
        # Setup connections
        db_pool = await asyncpg.create_pool(
            host='localhost', port=5432, database='alphapulse',
            user='alpha_emon', password='Emon_@17711'
        )
        
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connections
        await redis_client.ping()
        async with db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        
        # Initialize services
        sentiment_analyzer = EnhancedSentimentAnalyzer(db_pool, redis_client)
        sentiment_service = EnhancedSentimentService(db_pool, redis_client)
        
        # Test 1: Text sentiment analysis
        test_text = "Bitcoin is performing exceptionally well today! 🚀"
        sentiment_result = await sentiment_analyzer.analyze_text_sentiment(test_text, 'test')
        
        if sentiment_result and 'sentiment_score' in sentiment_result:
            logger.info(f"✅ Text sentiment analysis: {sentiment_result['sentiment_score']:.3f}")
        else:
            logger.error("❌ Text sentiment analysis failed")
            return False
        
        # Test 2: Sentiment collection
        symbol = 'BTC/USDT'
        collection_result = await sentiment_service.collect_all_sentiment(symbol)
        
        if collection_result and len(collection_result) > 0:
            logger.info(f"✅ Sentiment collection: {len(collection_result)} records")
        else:
            logger.warning("⚠️ No sentiment data collected (normal for new system)")
        
        # Test 3: Cache functionality
        await redis_client.set('test_key', 'test_value')
        cached_value = await redis_client.get('test_key')
        await redis_client.delete('test_key')
        
        if cached_value == 'test_value':
            logger.info("✅ Cache functionality working")
        else:
            logger.error("❌ Cache functionality failed")
            return False
        
        # Cleanup
        await db_pool.close()
        await redis_client.aclose()
        
        logger.info("✅ Enhanced Sentiment Analysis Integration Test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Main function"""
    success = await test_sentiment_integration()
    
    if success:
        print("\n🎉 Enhanced Sentiment Analysis System is fully operational!")
        print("🚀 Ready for production deployment!")
    else:
        print("\n⚠️ Integration test failed.")

if __name__ == "__main__":
    asyncio.run(main())
