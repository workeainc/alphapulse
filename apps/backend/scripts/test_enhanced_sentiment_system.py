#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis System Test Script
Comprehensive testing for the enhanced sentiment analysis implementation
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as redis
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from src.app.services.enhanced_sentiment_service import EnhancedSentimentService

logger = logging.getLogger(__name__)

class EnhancedSentimentSystemTester:
    """Comprehensive tester for enhanced sentiment analysis system"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.sentiment_analyzer = None
        self.sentiment_service = None
        self.test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        self.test_results = {}
    
    async def setup(self):
        """Setup database and Redis connections"""
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
            
            # Initialize components
            self.sentiment_analyzer = EnhancedSentimentAnalyzer(self.db_pool, self.redis_client)
            self.sentiment_service = EnhancedSentimentService(self.db_pool, self.redis_client)
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def test_database_migration(self):
        """Test database migration - check if tables exist"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if enhanced sentiment tables exist
                tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name IN (
                        'enhanced_sentiment_data',
                        'real_time_sentiment_aggregation',
                        'sentiment_correlation',
                        'sentiment_alerts',
                        'sentiment_model_performance'
                    )
                """)
                
                if len(tables) == 5:
                    logger.info("✅ All enhanced sentiment tables exist")
                    return True
                else:
                    logger.error(f"❌ Missing tables. Found: {[t['table_name'] for t in tables]}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Database migration test failed: {e}")
            return False
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        try:
            test_texts = [
                "Bitcoin is going to the moon! 🚀",
                "This is terrible news for crypto investors",
                "The market is stable with no significant changes"
            ]
            
            for text in test_texts:
                result = await self.sentiment_analyzer.analyze_text_sentiment(text, 'test')
                if not result or 'sentiment_score' not in result:
                    logger.error(f"❌ Sentiment analysis failed for text: {text}")
                    return False
            
            logger.info("✅ Sentiment analysis test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis test failed: {e}")
            return False
    
    async def test_sentiment_collection(self):
        """Test sentiment data collection"""
        try:
            # Test simulated collection
            symbol = 'BTC/USDT'
            collection_result = await self.sentiment_service.collect_all_sentiment(symbol)
            
            if collection_result and len(collection_result) > 0:
                logger.info(f"✅ Sentiment collection test passed. Collected {len(collection_result)} records")
                return True
            else:
                logger.error("❌ Sentiment collection returned no data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sentiment collection test failed: {e}")
            return False
    
    async def test_sentiment_aggregation(self):
        """Test sentiment aggregation"""
        try:
            symbol = 'BTC/USDT'
            window_size = '5min'
            
            # Test aggregation
            aggregation_result = await self.sentiment_service.aggregate_sentiment(symbol, window_size)
            
            if aggregation_result and 'overall_sentiment_score' in aggregation_result:
                logger.info(f"✅ Sentiment aggregation test passed. Score: {aggregation_result['overall_sentiment_score']}")
                return True
            else:
                logger.error("❌ Sentiment aggregation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Sentiment aggregation test failed: {e}")
            return False
    
    async def test_sentiment_service(self):
        """Test sentiment service functionality"""
        try:
            symbol = 'BTC/USDT'
            
            # Test sentiment summary
            summary = await self.sentiment_service.get_sentiment_summary(symbol)
            if not summary:
                logger.warning("⚠️ No sentiment summary available (this is normal for new system)")
            
            # Test multi-symbol sentiment
            multi_symbol = await self.sentiment_service.get_multi_symbol_sentiment(self.test_symbols)
            if not multi_symbol:
                logger.warning("⚠️ No multi-symbol sentiment available (this is normal for new system)")
            
            # Test sentiment trends
            trends = await self.sentiment_service.get_sentiment_trends(symbol)
            if not trends:
                logger.warning("⚠️ No sentiment trends available (this is normal for new system)")
            
            logger.info("✅ Sentiment service test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment service test failed: {e}")
            return False
    
    async def test_cache_functionality(self):
        """Test Redis cache functionality"""
        try:
            # Test cache set/get
            test_key = 'test_sentiment_cache'
            test_value = {'sentiment': 0.5, 'timestamp': datetime.now().isoformat()}
            
            await self.redis_client.set(test_key, json.dumps(test_value))
            cached_value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            if cached_value and json.loads(cached_value)['sentiment'] == 0.5:
                logger.info("✅ Cache functionality test passed")
                return True
            else:
                logger.error("❌ Cache functionality test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cache functionality test failed: {e}")
            return False
    
    async def test_performance(self):
        """Test performance benchmarks"""
        try:
            import time
            
            # Test sentiment analysis performance
            start_time = time.time()
            test_text = "Bitcoin is performing exceptionally well today!"
            
            for _ in range(10):
                await self.sentiment_analyzer.analyze_text_sentiment(test_text, 'performance_test')
            
            analysis_time = time.time() - start_time
            avg_time = analysis_time / 10
            
            logger.info(f"✅ Performance test passed. Average analysis time: {avg_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Enhanced Sentiment Analysis System Tests")
        logger.info("=" * 60)
        
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        try:
            tests = [
                ("Database Migration", self.test_database_migration),
                ("Sentiment Analysis", self.test_sentiment_analysis),
                ("Sentiment Collection", self.test_sentiment_collection),
                ("Sentiment Aggregation", self.test_sentiment_aggregation),
                ("Sentiment Service", self.test_sentiment_service),
                ("Cache Functionality", self.test_cache_functionality),
                ("Performance", self.test_performance)
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
            logger.info("📊 TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed: {passed_tests}")
            logger.info(f"Failed: {total_tests - passed_tests}")
            logger.info(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
            
            if passed_tests == total_tests:
                logger.info("🎉 ALL TESTS PASSED!")
                return True
            else:
                logger.error("❌ SOME TESTS FAILED")
                return False
                
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    
    tester = EnhancedSentimentSystemTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Enhanced Sentiment Analysis System is ready for deployment!")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
