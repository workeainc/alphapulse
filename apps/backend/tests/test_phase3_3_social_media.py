#!/usr/bin/env python3
"""
Phase 3.3: Social Media Integration - Comprehensive Test Script
Tests implementation, integration, database, end-to-end functionality, and performance
"""

import asyncio
import asyncpg
import logging
import time
import sys
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase33SocialMediaTester:
    """Phase 3.3 Social Media Integration Tester"""
    
    def __init__(self):
        self.conn = None
        self.test_results = {}
        
    async def connect_database(self) -> bool:
        """Connect to database"""
        try:
            self.conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711'
            )
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def close_database(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
    
    async def test_implementation(self) -> bool:
        """Test Phase 3.3 implementation"""
        logger.info("🔍 Testing Phase 3.3 Implementation...")
        
        try:
            # Test sentiment service social media methods
            from src.app.services.sentiment_service import SentimentService
            
            sentiment_service = SentimentService()
            
            # Test social media sentiment analysis
            social_sentiment = await sentiment_service.get_social_media_sentiment("BTCUSDT")
            
            required_keys = [
                'twitter_sentiment', 'reddit_sentiment', 'social_sentiment',
                'social_trends', 'social_momentum', 'social_impact_score',
                'social_volume', 'social_engagement', 'social_volatility',
                'social_correlation', 'phase_3_3_features'
            ]
            
            for key in required_keys:
                if key not in social_sentiment:
                    logger.error(f"❌ Missing key in social sentiment: {key}")
                    return False
            
            # Test enhanced sentiment with social media
            enhanced_sentiment = await sentiment_service.get_enhanced_sentiment_with_social_media("BTCUSDT")
            
            social_keys = [
                'social_media', 'social_enhanced_confidence', 'social_filtered_sentiment',
                'phase_3_3_features'
            ]
            
            for key in social_keys:
                if key not in enhanced_sentiment:
                    logger.error(f"❌ Missing key in enhanced sentiment: {key}")
                    return False
            
            logger.info("✅ Phase 3.3 Implementation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Implementation test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test Phase 3.3 integration"""
        logger.info("🔍 Testing Phase 3.3 Integration...")
        
        try:
            # Test real-time signal generator integration
            from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            signal_generator = RealTimeSignalGenerator()
            await signal_generator.initialize_sentiment_service()
            
            # Test social media predictions
            social_predictions = await signal_generator.analyze_social_media_predictions("BTCUSDT")
            
            required_keys = [
                'social_impact_score', 'social_sentiment_score', 'social_confidence',
                'social_trends', 'social_momentum', 'social_volume', 'social_engagement',
                'social_aware_signal', 'social_enhanced_confidence', 'phase_3_3_features'
            ]
            
            for key in required_keys:
                if key not in social_predictions:
                    logger.error(f"❌ Missing key in social predictions: {key}")
                    return False
            
            # Test enhanced sentiment analysis integration
            enhanced_analysis = await signal_generator.analyze_enhanced_sentiment_predictions("BTCUSDT")
            
            if 'social_enhanced_confidence' not in enhanced_analysis:
                logger.error("❌ Missing social_enhanced_confidence in enhanced analysis")
                return False
            
            logger.info("✅ Phase 3.3 Integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    async def test_database(self) -> bool:
        """Test Phase 3.3 database functionality"""
        logger.info("🔍 Testing Phase 3.3 Database...")
        
        if not self.conn:
            logger.error("❌ No database connection")
            return False
        
        try:
            # Check for social media columns
            required_columns = [
                'social_media_sentiment', 'social_impact_score', 'social_sentiment_score',
                'social_confidence', 'social_trends', 'social_momentum', 'social_volume',
                'social_engagement', 'social_volatility', 'social_correlation',
                'social_aware_signal', 'social_enhanced_confidence', 'social_filtered_sentiment',
                'twitter_sentiment_data', 'reddit_sentiment_data', 'social_sentiment_history',
                'social_media_last_updated'
            ]
            
            columns_result = await self.conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND table_schema = 'public'
                AND column_name IN (
                    'social_media_sentiment', 'social_impact_score', 'social_sentiment_score',
                    'social_confidence', 'social_trends', 'social_momentum', 'social_volume',
                    'social_engagement', 'social_volatility', 'social_correlation',
                    'social_aware_signal', 'social_enhanced_confidence', 'social_filtered_sentiment',
                    'twitter_sentiment_data', 'reddit_sentiment_data', 'social_sentiment_history',
                    'social_media_last_updated'
                )
            """)
            
            actual_columns = [row['column_name'] for row in columns_result]
            missing_columns = [col for col in required_columns if col not in actual_columns]
            
            if missing_columns:
                logger.error(f"❌ Missing social media columns: {missing_columns}")
                return False
            
            # Check for social media indexes
            indexes_result = await self.conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'enhanced_signals' 
                AND indexname LIKE '%social_%'
            """)
            
            if len(indexes_result) < 10:  # Should have multiple social media indexes
                logger.error(f"❌ Insufficient social media indexes: {len(indexes_result)}")
                return False
            
            # Check for social media view
            view_exists = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.views 
                    WHERE table_name = 'social_media_enhanced_signals'
                )
            """)
            
            if not view_exists:
                logger.error("❌ social_media_enhanced_signals view not found")
                return False
            
            # Check for social media functions
            functions_result = await self.conn.fetch("""
                SELECT proname 
                FROM pg_proc 
                WHERE proname LIKE '%social_%'
            """)
            
            if len(functions_result) < 3:  # Should have multiple social media functions
                logger.error(f"❌ Insufficient social media functions: {len(functions_result)}")
                return False
            
            # Test social media function
            quality_score = await self.conn.fetchval("""
                SELECT calculate_social_media_sentiment_quality(0.8, 0.7, '{"volume_score": 0.6}'::jsonb, '{"engagement_score": 0.5}'::jsonb)
            """)
            
            if quality_score is None or quality_score <= 0:
                logger.error("❌ Social media quality function test failed")
                return False
            
            logger.info("✅ Phase 3.3 Database test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database test failed: {e}")
            return False
    
    async def test_end_to_end(self) -> bool:
        """Test Phase 3.3 end-to-end functionality"""
        logger.info("🔍 Testing Phase 3.3 End-to-End Functionality...")
        
        try:
            # Test complete social media workflow
            from src.app.services.sentiment_service import SentimentService
            from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            sentiment_service = SentimentService()
            signal_generator = RealTimeSignalGenerator()
            await signal_generator.initialize_sentiment_service()
            
            # Test social media sentiment analysis
            social_sentiment = await sentiment_service.get_social_media_sentiment("BTCUSDT")
            
            # Test enhanced sentiment with social media
            enhanced_sentiment = await sentiment_service.get_enhanced_sentiment_with_social_media("BTCUSDT")
            
            # Test signal generator integration
            social_predictions = await signal_generator.analyze_social_media_predictions("BTCUSDT")
            
            # Verify data flow
            if not all([
                social_sentiment.get('social_impact_score', 0) >= 0,
                enhanced_sentiment.get('social_enhanced_confidence', 0) >= 0,
                social_predictions.get('social_aware_signal') is not None
            ]):
                logger.error("❌ End-to-end data flow test failed")
                return False
            
            logger.info("✅ Phase 3.3 End-to-End test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            return False
    
    async def test_performance(self) -> bool:
        """Test Phase 3.3 performance"""
        logger.info("🔍 Testing Phase 3.3 Performance...")
        
        try:
            from src.app.services.sentiment_service import SentimentService
            
            sentiment_service = SentimentService()
            
            # Performance test
            start_time = time.time()
            
            # Run multiple social media analyses
            for i in range(5):
                await sentiment_service.get_social_media_sentiment("BTCUSDT")
                await sentiment_service.get_enhanced_sentiment_with_social_media("BTCUSDT")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time (5 seconds for 10 operations)
            if execution_time > 5.0:
                logger.error(f"❌ Performance test failed: {execution_time:.2f}s")
                return False
            
            logger.info(f"✅ Phase 3.3 Performance test passed: {execution_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive Phase 3.3 test"""
        logger.info("🚀 Starting Phase 3.3: Social Media Integration Comprehensive Test...")
        
        # Connect to database
        db_connected = await self.connect_database()
        
        # Run all tests
        tests = {
            "Implementation": await self.test_implementation(),
            "Integration": await self.test_integration(),
            "Database": await self.test_database() if db_connected else False,
            "End-to-End": await self.test_end_to_end(),
            "Performance": await self.test_performance()
        }
        
        # Close database connection
        await self.close_database()
        
        # Calculate overall success
        successful_tests = sum(tests.values())
        total_tests = len(tests)
        
        # Generate report
        logger.info("\n" + "="*60)
        logger.info("📊 PHASE 3.3 TEST REPORT")
        logger.info("="*60)
        
        for test_name, success in tests.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name:<20} {status}")
        
        logger.info("-"*60)
        overall_status = "✅ SUCCESS" if successful_tests == total_tests else "❌ FAILED"
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        if successful_tests == total_tests:
            logger.info("🎉 Phase 3.3: Social Media Integration test completed successfully!")
        else:
            logger.error("💥 Phase 3.3 test has issues that need to be addressed")
        
        return {
            "overall_success": successful_tests == total_tests,
            "success_rate": successful_tests / total_tests,
            "tests": tests,
            "database_connected": db_connected
        }

async def main():
    """Main test function"""
    tester = Phase33SocialMediaTester()
    result = await tester.run_comprehensive_test()
    
    if result["overall_success"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
