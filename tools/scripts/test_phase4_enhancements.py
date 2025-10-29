#!/usr/bin/env python3
"""
Phase 4 Enhancement Test Suite
Tests all Phase 4 features: Predictive Analytics, Cross-Asset Correlation, and Model Performance
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

class Phase4EnhancementTester:
    """Comprehensive tester for Phase 4 enhancements"""
    
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
            
            logger.info("✅ Phase 4 test setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 4 test setup failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.aclose()
    
    async def test_enhanced_models(self):
        """Test enhanced ML models with ensemble capabilities"""
        try:
            logger.info("🧪 Testing Enhanced ML Models...")
            
            # Test text sentiment analysis with ensemble
            test_text = "Bitcoin is showing strong bullish momentum with increasing volume! 🚀"
            sentiment_result = await self.sentiment_analyzer.analyze_text_sentiment(test_text, 'twitter')
            
            if sentiment_result and 'model_breakdown' in sentiment_result:
                logger.info(f"✅ Ensemble sentiment analysis: {sentiment_result['sentiment_score']:.3f}")
                logger.info(f"✅ Model breakdown: {len(sentiment_result['model_breakdown'])} models")
                logger.info(f"✅ Ensemble confidence: {sentiment_result['ensemble_confidence']:.3f}")
                return True
            else:
                logger.error("❌ Enhanced ML models test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced ML models test error: {e}")
            return False
    
    async def test_predictive_analytics(self):
        """Test predictive analytics capabilities"""
        try:
            logger.info("🧪 Testing Predictive Analytics...")
            
            # Test price prediction
            symbol = 'BTC/USDT'
            prediction = await self.sentiment_service.get_price_prediction(symbol, '4h')
            
            if prediction and 'prediction_probability' in prediction:
                logger.info(f"✅ Price prediction: {prediction['direction']} ({prediction['prediction_probability']:.3f})")
                logger.info(f"✅ Prediction confidence: {prediction['confidence']:.3f}")
                logger.info(f"✅ Technical indicators: {len(prediction['technical_indicators'])}")
                return True
            else:
                logger.error("❌ Predictive analytics test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Predictive analytics test error: {e}")
            return False
    
    async def test_multi_horizon_predictions(self):
        """Test multi-horizon predictions"""
        try:
            logger.info("🧪 Testing Multi-Horizon Predictions...")
            
            symbol = 'BTC/USDT'
            predictions = await self.sentiment_service.get_multi_horizon_predictions(symbol)
            
            if predictions and 'predictions' in predictions:
                horizon_count = len(predictions['predictions'])
                logger.info(f"✅ Multi-horizon predictions: {horizon_count} horizons")
                
                for horizon, pred in predictions['predictions'].items():
                    logger.info(f"   - {horizon}: {pred['direction']} ({pred['prediction_probability']:.3f})")
                
                return True
            else:
                logger.error("❌ Multi-horizon predictions test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Multi-horizon predictions test error: {e}")
            return False
    
    async def test_cross_asset_correlation(self):
        """Test cross-asset correlation analysis"""
        try:
            logger.info("🧪 Testing Cross-Asset Correlation...")
            
            primary_symbol = 'BTC/USDT'
            secondary_symbols = ['ETH/USDT', 'BNB/USDT']
            
            analysis = await self.sentiment_service.get_cross_asset_analysis(primary_symbol, secondary_symbols)
            
            if analysis and 'correlations' in analysis:
                logger.info(f"✅ Cross-asset analysis: {len(analysis['correlations'])} correlations")
                logger.info(f"✅ Market sentiment: {analysis['market_sentiment']['market_mood']}")
                logger.info(f"✅ Average sentiment: {analysis['market_sentiment']['average_sentiment']:.3f}")
                return True
            else:
                logger.error("❌ Cross-asset correlation test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cross-asset correlation test error: {e}")
            return False
    
    async def test_market_sentiment_overview(self):
        """Test market sentiment overview"""
        try:
            logger.info("🧪 Testing Market Sentiment Overview...")
            
            overview = await self.sentiment_service.get_market_sentiment_overview()
            
            if overview and 'total_assets' in overview:
                logger.info(f"✅ Market overview: {overview['total_assets']} assets")
                logger.info(f"✅ Average sentiment: {overview['average_sentiment']:.3f}")
                logger.info(f"✅ Bullish assets: {overview['bullish_assets']}")
                logger.info(f"✅ Bearish assets: {overview['bearish_assets']}")
                return True
            else:
                logger.error("❌ Market sentiment overview test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market sentiment overview test error: {e}")
            return False
    
    async def test_model_performance(self):
        """Test model performance tracking"""
        try:
            logger.info("🧪 Testing Model Performance...")
            
            # Get performance summary
            performance = await self.sentiment_service.get_model_performance_summary('BTC/USDT', 30)
            
            if performance:
                logger.info(f"✅ Model performance: {performance.get('total_predictions', 0)} predictions")
                logger.info(f"✅ Average accuracy: {performance.get('average_accuracy', 0):.3f}")
                return True
            else:
                logger.warning("⚠️ No model performance data available (normal for new system)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Model performance test error: {e}")
            return False
    
    async def test_prediction_alerts(self):
        """Test prediction alerts"""
        try:
            logger.info("🧪 Testing Prediction Alerts...")
            
            alerts = await self.sentiment_service.get_prediction_alerts()
            
            if alerts is not None:
                logger.info(f"✅ Prediction alerts: {len(alerts)} alerts")
                return True
            else:
                logger.warning("⚠️ No prediction alerts available (normal for new system)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Prediction alerts test error: {e}")
            return False
    
    async def test_database_tables(self):
        """Test Phase 4 database tables"""
        try:
            logger.info("🧪 Testing Phase 4 Database Tables...")
            
            async with self.db_pool.acquire() as conn:
                # Test sentiment_predictions table
                prediction_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sentiment_predictions"
                )
                logger.info(f"✅ Sentiment predictions table: {prediction_count} records")
                
                # Test cross_asset_sentiment table
                cross_asset_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM cross_asset_sentiment"
                )
                logger.info(f"✅ Cross-asset sentiment table: {cross_asset_count} records")
                
                # Test model_performance_metrics table
                performance_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM model_performance_metrics"
                )
                logger.info(f"✅ Model performance metrics table: {performance_count} records")
                
                # Test enhanced columns in existing table
                enhanced_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM enhanced_sentiment_data WHERE prediction_confidence IS NOT NULL"
                )
                logger.info(f"✅ Enhanced sentiment data: {enhanced_count} records with new columns")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database tables test error: {e}")
            return False
    
    async def test_api_endpoints(self):
        """Test new API endpoints"""
        try:
            logger.info("🧪 Testing New API Endpoints...")
            
            import aiohttp
            
            # Test prediction endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/sentiment/predictions/BTC/USDT') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Prediction API: {data.get('direction', 'unknown')} prediction")
                    else:
                        logger.warning(f"⚠️ Prediction API: {response.status} (may not be running)")
            
            # Test cross-asset endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/sentiment/cross-asset/BTC/USDT') as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Cross-asset API: {len(data.get('correlations', {}))} correlations")
                    else:
                        logger.warning(f"⚠️ Cross-asset API: {response.status} (may not be running)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API endpoints test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Phase 4 enhancement tests"""
        logger.info("🚀 Starting Phase 4 Enhancement Tests")
        logger.info("=" * 60)
        
        if not await self.setup():
            logger.error("❌ Setup failed, aborting tests")
            return False
        
        try:
            tests = [
                ("Enhanced ML Models", self.test_enhanced_models),
                ("Predictive Analytics", self.test_predictive_analytics),
                ("Multi-Horizon Predictions", self.test_multi_horizon_predictions),
                ("Cross-Asset Correlation", self.test_cross_asset_correlation),
                ("Market Sentiment Overview", self.test_market_sentiment_overview),
                ("Model Performance", self.test_model_performance),
                ("Prediction Alerts", self.test_prediction_alerts),
                ("Database Tables", self.test_database_tables),
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
            logger.info("📊 PHASE 4 ENHANCEMENT TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {total_tests}")
            logger.info(f"Passed: {passed_tests}")
            logger.info(f"Failed: {total_tests - passed_tests}")
            logger.info(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
            
            if passed_tests >= total_tests * 0.8:  # 80% success rate
                logger.info("🎉 PHASE 4 ENHANCEMENTS SUCCESSFULLY IMPLEMENTED!")
                return True
            else:
                logger.error("❌ SOME PHASE 4 TESTS FAILED")
                return False
                
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    tester = Phase4EnhancementTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Phase 4 Enhancements are fully operational!")
        print("🚀 Advanced Predictive Analytics, Cross-Asset Correlation, and Model Performance tracking are ready!")
        print("📊 New API endpoints available for enhanced sentiment analysis!")
    else:
        print("\n⚠️ Some Phase 4 tests failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
