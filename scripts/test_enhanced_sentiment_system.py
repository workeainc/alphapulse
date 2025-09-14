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

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from app.services.enhanced_sentiment_service import EnhancedSentimentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSentimentSystemTester:
    """Comprehensive tester for enhanced sentiment analysis system"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.sentiment_analyzer = None
        self.sentiment_service = None
        
        # Test configuration
        self.test_symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        self.test_results = {}
        
    async def setup(self):
        """Setup database and Redis connections"""
        try:
            logger.info("🔧 Setting up test environment...")
            
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
            )
            logger.info("✅ Database connection established")
            
            # Redis connection
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = EnhancedSentimentAnalyzer(self.db_pool, self.redis_client)
            logger.info("✅ Enhanced sentiment analyzer initialized")
            
            # Initialize sentiment service
            self.sentiment_service = EnhancedSentimentService(self.db_pool, self.redis_client)
            logger.info("✅ Enhanced sentiment service initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup connections"""
        try:
            if self.sentiment_service:
                await self.sentiment_service.stop_service()
            
            if self.sentiment_analyzer:
                await self.sentiment_analyzer.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    async def test_database_migration(self):
        """Test database migration and table creation"""
        try:
            logger.info("🗄️ Testing database migration...")
            
            # Check if tables exist
            async with self.db_pool.acquire() as conn:
                # Check enhanced_sentiment_data table
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'enhanced_sentiment_data'
                    )
                """)
                
                if not result:
                    logger.error("❌ enhanced_sentiment_data table not found")
                    return False
                
                # Check real_time_sentiment_aggregation table
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'real_time_sentiment_aggregation'
                    )
                """)
                
                if not result:
                    logger.error("❌ real_time_sentiment_aggregation table not found")
                    return False
                
                # Check sentiment_correlation table
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sentiment_correlation'
                    )
                """)
                
                if not result:
                    logger.error("❌ sentiment_correlation table not found")
                    return False
                
                # Check sentiment_alerts table
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'sentiment_alerts'
                    )
                """)
                
                if not result:
                    logger.error("❌ sentiment_alerts table not found")
                    return False
                
                logger.info("✅ All sentiment tables exist")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database migration test failed: {e}")
            return False
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        try:
            logger.info("🧠 Testing sentiment analysis...")
            
            test_texts = [
                "Bitcoin is going to the moon! 🚀",
                "Market is looking bearish today",
                "Great analysis on crypto fundamentals",
                "This is terrible news for investors",
                "Neutral market conditions continue"
            ]
            
            results = []
            
            for text in test_texts:
                result = await self.sentiment_analyzer.analyze_text_sentiment(text, 'twitter')
                results.append({
                    'text': text,
                    'sentiment_score': result['sentiment_score'],
                    'sentiment_label': result['sentiment_label'],
                    'confidence': result['confidence'],
                    'sarcasm_detected': result['sarcasm_detected'],
                    'topic_classification': result['topic_classification']
                })
                
                logger.info(f"Text: '{text[:50]}...' -> {result['sentiment_label']} ({result['sentiment_score']:.3f})")
            
            # Check if results are reasonable
            valid_labels = ['positive', 'negative', 'neutral']
            valid_topics = ['price_moving', 'news', 'opinion', 'noise']
            
            for result in results:
                if result['sentiment_label'] not in valid_labels:
                    logger.error(f"❌ Invalid sentiment label: {result['sentiment_label']}")
                    return False
                
                if result['topic_classification'] not in valid_topics:
                    logger.error(f"❌ Invalid topic classification: {result['topic_classification']}")
                    return False
                
                if not (0 <= result['confidence'] <= 1):
                    logger.error(f"❌ Invalid confidence score: {result['confidence']}")
                    return False
            
            logger.info("✅ Sentiment analysis test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis test failed: {e}")
            return False
    
    async def test_sentiment_collection(self):
        """Test sentiment data collection"""
        try:
            logger.info("📊 Testing sentiment data collection...")
            
            for symbol in self.test_symbols:
                logger.info(f"Collecting sentiment for {symbol}...")
                
                # Collect sentiment data
                sentiment_data = await self.sentiment_analyzer.collect_all_sentiment(symbol)
                
                if not sentiment_data:
                    logger.warning(f"⚠️ No sentiment data collected for {symbol}")
                    continue
                
                logger.info(f"✅ Collected {len(sentiment_data)} sentiment records for {symbol}")
                
                # Verify data structure
                for data in sentiment_data:
                    required_fields = [
                        'symbol', 'timestamp', 'source', 'sentiment_score',
                        'sentiment_label', 'confidence', 'volume', 'keywords',
                        'raw_text', 'processed_text', 'topic_classification',
                        'sarcasm_detected', 'context_score'
                    ]
                    
                    for field in required_fields:
                        if not hasattr(data, field):
                            logger.error(f"❌ Missing field {field} in sentiment data")
                            return False
            
            logger.info("✅ Sentiment collection test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment collection test failed: {e}")
            return False
    
    async def test_sentiment_aggregation(self):
        """Test sentiment aggregation"""
        try:
            logger.info("📈 Testing sentiment aggregation...")
            
            for symbol in self.test_symbols:
                for window_size in ['1min', '5min', '15min', '1hour']:
                    logger.info(f"Aggregating {window_size} sentiment for {symbol}...")
                    
                    aggregation = await self.sentiment_analyzer.aggregate_sentiment(symbol, window_size)
                    
                    if not aggregation:
                        logger.warning(f"⚠️ No aggregation data for {symbol} ({window_size})")
                        continue
                    
                    # Verify aggregation structure
                    required_fields = [
                        'symbol', 'timestamp', 'window_size', 'overall_sentiment_score',
                        'positive_sentiment_score', 'negative_sentiment_score',
                        'neutral_sentiment_score', 'source_breakdown', 'volume_metrics',
                        'confidence_weighted_score', 'sentiment_trend', 'trend_strength'
                    ]
                    
                    for field in required_fields:
                        if not hasattr(aggregation, field):
                            logger.error(f"❌ Missing field {field} in aggregation data")
                            return False
                    
                    logger.info(f"✅ Aggregated {window_size} sentiment for {symbol}")
            
            logger.info("✅ Sentiment aggregation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment aggregation test failed: {e}")
            return False
    
    async def test_sentiment_service(self):
        """Test sentiment service functionality"""
        try:
            logger.info("🔧 Testing sentiment service...")
            
            # Start service
            await self.sentiment_service.start_service()
            logger.info("✅ Sentiment service started")
            
            # Wait for some data collection
            await asyncio.sleep(5)
            
            # Test sentiment summary
            for symbol in self.test_symbols:
                logger.info(f"Getting sentiment summary for {symbol}...")
                
                summary = await self.sentiment_service.get_sentiment_summary(symbol)
                
                if summary:
                    logger.info(f"✅ Sentiment summary for {symbol}: {summary.sentiment_label} ({summary.overall_sentiment:.3f})")
                else:
                    logger.warning(f"⚠️ No sentiment summary for {symbol}")
            
            # Test multi-symbol sentiment
            logger.info("Testing multi-symbol sentiment...")
            multi_results = await self.sentiment_service.get_multi_symbol_sentiment(self.test_symbols)
            logger.info(f"✅ Multi-symbol results: {len(multi_results)} symbols")
            
            # Test sentiment trends
            for symbol in self.test_symbols:
                logger.info(f"Getting sentiment trends for {symbol}...")
                trends = await self.sentiment_service.get_sentiment_trends(symbol, hours=1)
                
                if trends:
                    logger.info(f"✅ Sentiment trends for {symbol}: {trends.get('data_points', 0)} data points")
                else:
                    logger.warning(f"⚠️ No sentiment trends for {symbol}")
            
            # Test sentiment alerts
            for symbol in self.test_symbols:
                logger.info(f"Getting sentiment alerts for {symbol}...")
                alerts = await self.sentiment_service.get_sentiment_alerts(symbol, threshold=0.1)
                logger.info(f"✅ Sentiment alerts for {symbol}: {len(alerts)} alerts")
            
            # Stop service
            await self.sentiment_service.stop_service()
            logger.info("✅ Sentiment service stopped")
            
            logger.info("✅ Sentiment service test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment service test failed: {e}")
            return False
    
    async def test_cache_functionality(self):
        """Test Redis cache functionality"""
        try:
            logger.info("💾 Testing cache functionality...")
            
            # Test cache set/get
            test_key = "test_sentiment_cache"
            test_data = {
                "symbol": "BTC/USDT",
                "sentiment": 0.75,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Set cache
            await self.redis_client.setex(test_key, 60, json.dumps(test_data))
            logger.info("✅ Cache set successful")
            
            # Get cache
            cached_data = await self.redis_client.get(test_key)
            if cached_data:
                parsed_data = json.loads(cached_data)
                if parsed_data['symbol'] == test_data['symbol']:
                    logger.info("✅ Cache get successful")
                else:
                    logger.error("❌ Cache data mismatch")
                    return False
            else:
                logger.error("❌ Cache get failed")
                return False
            
            # Clean up
            await self.redis_client.delete(test_key)
            logger.info("✅ Cache cleanup successful")
            
            logger.info("✅ Cache functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache functionality test failed: {e}")
            return False
    
    async def test_performance(self):
        """Test system performance"""
        try:
            logger.info("⚡ Testing system performance...")
            
            import time
            
            # Test sentiment analysis performance
            start_time = time.time()
            test_text = "Bitcoin is showing strong bullish momentum with increasing volume"
            
            for _ in range(10):
                await self.sentiment_analyzer.analyze_text_sentiment(test_text, 'twitter')
            
            analysis_time = time.time() - start_time
            avg_analysis_time = analysis_time / 10
            
            logger.info(f"✅ Average sentiment analysis time: {avg_analysis_time:.3f}s")
            
            if avg_analysis_time > 1.0:  # Should be under 1 second
                logger.warning(f"⚠️ Sentiment analysis is slow: {avg_analysis_time:.3f}s")
            
            # Test aggregation performance
            start_time = time.time()
            
            for symbol in self.test_symbols:
                await self.sentiment_analyzer.aggregate_sentiment(symbol, '5min')
            
            aggregation_time = time.time() - start_time
            avg_aggregation_time = aggregation_time / len(self.test_symbols)
            
            logger.info(f"✅ Average aggregation time: {avg_aggregation_time:.3f}s")
            
            if avg_aggregation_time > 0.5:  # Should be under 0.5 seconds
                logger.warning(f"⚠️ Sentiment aggregation is slow: {avg_aggregation_time:.3f}s")
            
            logger.info("✅ Performance test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Enhanced Sentiment Analysis System Tests")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        try:
            # Run tests
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
            
            # Summary
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
    """Main test function"""
    tester = EnhancedSentimentSystemTester()
    
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        await tester.cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        await tester.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
