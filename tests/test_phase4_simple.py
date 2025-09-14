#!/usr/bin/env python3
"""
Simple Phase 4 Enhancement Test
Tests core Phase 4 features from backend directory
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import asyncpg
import redis.asyncio as redis
import json

# Add current directory to path
sys.path.append('.')

from ai.enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
from app.services.enhanced_sentiment_service import EnhancedSentimentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phase4_enhancements():
    """Test Phase 4 enhancements"""
    try:
        logger.info("🚀 Testing Phase 4 Enhancements...")
        
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
        
        logger.info("✅ Connections established")
        
        # Test 1: Enhanced ML Models
        logger.info("🧪 Testing Enhanced ML Models...")
        test_text = "Bitcoin is showing strong bullish momentum! 🚀"
        sentiment_result = await sentiment_analyzer.analyze_text_sentiment(test_text, 'twitter')
        
        if sentiment_result and 'model_breakdown' in sentiment_result:
            logger.info(f"✅ Ensemble sentiment analysis: {sentiment_result['sentiment_score']:.3f}")
            logger.info(f"✅ Model breakdown: {len(sentiment_result['model_breakdown'])} models")
            logger.info(f"✅ Ensemble confidence: {sentiment_result['ensemble_confidence']:.3f}")
        else:
            logger.error("❌ Enhanced ML models test failed")
            return False
        
        # Test 2: Predictive Analytics
        logger.info("🧪 Testing Predictive Analytics...")
        prediction = await sentiment_service.get_price_prediction('BTC/USDT', '4h')
        
        if prediction and 'prediction_probability' in prediction:
            logger.info(f"✅ Price prediction: {prediction['direction']} ({prediction['prediction_probability']:.3f})")
            logger.info(f"✅ Prediction confidence: {prediction['confidence']:.3f}")
        else:
            logger.error("❌ Predictive analytics test failed")
            return False
        
        # Test 3: Cross-Asset Analysis
        logger.info("🧪 Testing Cross-Asset Analysis...")
        analysis = await sentiment_service.get_cross_asset_analysis('BTC/USDT', ['ETH/USDT', 'BNB/USDT'])
        
        if analysis and 'correlations' in analysis:
            logger.info(f"✅ Cross-asset analysis: {len(analysis['correlations'])} correlations")
            logger.info(f"✅ Market sentiment: {analysis['market_sentiment']['market_mood']}")
        else:
            logger.error("❌ Cross-asset analysis test failed")
            return False
        
        # Test 4: Database Tables
        logger.info("🧪 Testing Database Tables...")
        async with db_pool.acquire() as conn:
            # Test sentiment_predictions table
            prediction_count = await conn.fetchval("SELECT COUNT(*) FROM sentiment_predictions")
            logger.info(f"✅ Sentiment predictions table: {prediction_count} records")
            
            # Test cross_asset_sentiment table
            cross_asset_count = await conn.fetchval("SELECT COUNT(*) FROM cross_asset_sentiment")
            logger.info(f"✅ Cross-asset sentiment table: {cross_asset_count} records")
            
            # Test model_performance_metrics table
            performance_count = await conn.fetchval("SELECT COUNT(*) FROM model_performance_metrics")
            logger.info(f"✅ Model performance metrics table: {performance_count} records")
        
        # Cleanup
        await db_pool.close()
        await redis_client.aclose()
        
        logger.info("🎉 Phase 4 Enhancements Test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4 test failed: {e}")
        return False

async def main():
    """Main function"""
    success = await test_phase4_enhancements()
    
    if success:
        print("\n🎉 Phase 4 Enhancements are fully operational!")
        print("🚀 Advanced Predictive Analytics, Cross-Asset Correlation, and Model Performance tracking are ready!")
    else:
        print("\n⚠️ Phase 4 test failed. Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
