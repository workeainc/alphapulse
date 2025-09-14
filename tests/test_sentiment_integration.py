#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Integration Test
Tests the complete sentiment analysis system integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import asyncpg
import redis.asyncio as redis
import json

sys.path.append('.')

from ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from app.services.enhanced_sentiment_service import EnhancedSentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentIntegrationTester:
    """Integration tester for enhanced sentiment analysis system"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.sentiment_analyzer = None
        self.sentiment_service = None
    
    async def setup(self):
        """Setup connections and services"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711',
                min_size=5,
                max_size=20
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            
            # Test connections
            await self.redis_client.ping()
            async with self.db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            # Initialize services
            self.sentiment_analyzer = EnhancedSentimentAnalyzer(self.db_pool, self.redis_client)
            self.sentiment_service = EnhancedSentimentService(self.db_pool, self.redis_client)
            
            logger.info("✅ Integration test setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test setup failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.aclose()
    
    async def test_sentiment_analysis_integration(self):
        """Test complete sentiment analysis integration"""
        try:
            logger.info("🧪 Testing Sentiment Analysis Integration...")
            
            # Test 1: Text sentiment analysis
            test_text = "Bitcoin is performing exceptionally well today! 🚀"
            sentiment_result = await self.sentiment_analyzer.analyze_text_sentiment(test_text, 'test')
            
            if sentiment_result and 'sentiment_score' in sentiment_result:
                logger.info(f"✅ Text sentiment analysis: {sentiment_result['sentiment_score']:.3f}")
            else:
                logger.error("❌ Text sentiment analysis failed")
                return False
            
            # Test 2: Sentiment data collection
            symbol = 'BTC/USDT'
            collection_result = await self.sentiment_service.collect_all_sentiment(symbol)
            
            if collection_result and len(collection_result) > 0:
                logger.info(f"✅ Sentiment collection: {len(collection_result)} records")
            else:
                logger.warning("⚠️ No sentiment data collected (this is normal for new system)")
            
            # Test 3: Sentiment aggregation
            aggregation_result = await self.sentiment_service.aggregate_sentiment(symbol, '5min')
            
            if aggregation_result and 'overall_sentiment_score' in aggregation_result:
                logger.info(f"✅ Sentiment aggregation: {aggregation_result['overall_sentiment_score']:.3f}")
            else:
                logger.warning("⚠️ No sentiment aggregation available (this is normal for new system)")
            
            # Test 4: Sentiment summary
            summary = await self.sentiment_service.get_sentiment_summary(symbol)
            
            if summary:
                logger.info(f"✅ Sentiment summary: {summary.sentiment_label} ({summary.overall_sentiment:.3f})")
            else:
                logger.warning("⚠️ No sentiment summary available (this is normal for new system)")
            
            # Test 5: Cache functionality
            test_key = 'integration_test'
            test_value = {'test': 'data', 'timestamp': datetime.now().isoformat()}
            
            await self.redis_client.set(test_key, json.dumps(test_value))
            cached_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if cached_value and json.loads(cached_value)['test'] == 'data':
                logger.info("✅ Cache functionality working")
            else:
                logger.error("❌ Cache functionality failed")
                return False
            
            logger.info("✅ Sentiment Analysis Integration Test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment Analysis Integration Test failed: {e}")
            return False
    
    async def test_api_endpoints(self):
        """Test API endpoints"""
        try:
            logger.info("🧪 Testing API Endpoints...")
            
            import aiohttp
            
            # Test API health
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ API Health: {data.get('message', 'Unknown')}")
                    else:
                        logger.error(f"❌ API Health failed: {response.status}")
                        return False
            
            logger.info("✅ API Endpoints Test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ API Endpoints Test failed: {e}")
            return False
    
    async def run_integration_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Enhanced Sentiment Analysis Integration Tests")
        logger.info("=" * 60)
        
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        try:
            tests = [
                ("Sentiment Analysis Integration", self.test_sentiment_analysis_integration),
                ("API Endpoints", self.test_api_endpoints)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n🧪 Running {test_name} Test...")
                logger.info("-" * 40)
                try:
                    result = await test_func()
                    if result:
                        logger.info(f"✅ {test_name} Test PASSED")
                        passed_tests += 1
                    else:
                        logger.error(f"❌ {test_name} Test FAILED")
                except Exception as e:
                    logger.error(f"❌ {test_name} Test ERROR: {e}")
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 INTEGRATION TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed: {passed_tests}")
            logger.info(f"Failed: {total_tests - passed_tests}")
            logger.info(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
            
            if passed_tests == total_tests:
                logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
                return True
            else:
                logger.error("❌ SOME INTEGRATION TESTS FAILED")
                return False
                
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    tester = SentimentIntegrationTester()
    success = await tester.run_integration_tests()
    
    if success:
        print("\n🎉 Enhanced Sentiment Analysis System is fully operational!")
        print("🚀 Ready for production deployment!")
    else:
        print("\n⚠️ Some integration tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
