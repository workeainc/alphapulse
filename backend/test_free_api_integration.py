#!/usr/bin/env python3
"""
Free API Integration Test Script
Tests all free API integrations for AlphaPlus
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.free_api_manager import FreeAPIManager
from services.free_api_integration_service import FreeAPIIntegrationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_free_api_manager():
    """Test FreeAPIManager directly"""
    print("🔍 Testing FreeAPIManager...")
    print("=" * 50)
    
    api_manager = FreeAPIManager()
    
    # Test news sentiment
    print("📰 Testing news sentiment...")
    try:
        news_sentiment = await api_manager.get_news_sentiment('BTC')
        print(f"✅ News Sentiment: {news_sentiment.get('sentiment', 'unknown')}")
        print(f"   Source: {news_sentiment.get('source', 'unknown')}")
        print(f"   Articles: {len(news_sentiment.get('articles', []))}")
    except Exception as e:
        print(f"❌ News sentiment failed: {e}")
    
    # Test social sentiment
    print("\n📱 Testing social sentiment...")
    try:
        social_sentiment = await api_manager.get_social_sentiment('BTC')
        print(f"✅ Social Sentiment: {social_sentiment.get('reddit', {}).get('sentiment', 'unknown')}")
        print(f"   Posts: {social_sentiment.get('reddit', {}).get('posts', 0)}")
    except Exception as e:
        print(f"❌ Social sentiment failed: {e}")
    
    # Test market data
    print("\n📊 Testing market data...")
    try:
        market_data = await api_manager.get_market_data('BTC')
        print(f"✅ Market Data Source: {market_data.get('source', 'unknown')}")
        price = market_data.get('data', {}).get('price', 0)
        print(f"   Price: ${price:,.2f}")
    except Exception as e:
        print(f"❌ Market data failed: {e}")
    
    # Test liquidation data
    print("\n💧 Testing liquidation data...")
    try:
        liquidation_data = await api_manager.get_liquidation_data('BTC')
        print(f"✅ Liquidation Data:")
        print(f"   Long Liquidations: {liquidation_data.get('long_liquidations', 0)}")
        print(f"   Short Liquidations: {liquidation_data.get('short_liquidations', 0)}")
    except Exception as e:
        print(f"❌ Liquidation data failed: {e}")

async def test_free_api_integration_service():
    """Test FreeAPIIntegrationService"""
    print("\n🔗 Testing FreeAPIIntegrationService...")
    print("=" * 50)
    
    integration_service = FreeAPIIntegrationService()
    
    # Test enhanced sentiment
    print("🎯 Testing enhanced sentiment...")
    try:
        enhanced_sentiment = await integration_service.get_enhanced_sentiment('BTC')
        print(f"✅ Enhanced Sentiment: {enhanced_sentiment.get('overall_sentiment', 'unknown')}")
        print(f"   Score: {enhanced_sentiment.get('overall_sentiment_score', 0):.3f}")
        print(f"   Sources: {list(enhanced_sentiment.get('sentiment_scores', {}).keys())}")
    except Exception as e:
        print(f"❌ Enhanced sentiment failed: {e}")
    
    # Test enhanced market data
    print("\n📈 Testing enhanced market data...")
    try:
        enhanced_market_data = await integration_service.get_enhanced_market_data('BTC')
        print(f"✅ Enhanced Market Data:")
        print(f"   Price: ${enhanced_market_data.get('price', 0):,.2f}")
        print(f"   Volume 24h: ${enhanced_market_data.get('volume_24h', 0):,.0f}")
        print(f"   Fear & Greed Index: {enhanced_market_data.get('fear_greed_index', 50)}")
    except Exception as e:
        print(f"❌ Enhanced market data failed: {e}")
    
    # Test comprehensive signal data
    print("\n🚀 Testing comprehensive signal data...")
    try:
        comprehensive_data = await integration_service.get_comprehensive_signal_data('BTC')
        print(f"✅ Comprehensive Signal Data:")
        print(f"   Status: {comprehensive_data.get('status', 'unknown')}")
        print(f"   Sentiment: {comprehensive_data.get('sentiment', {}).get('overall_sentiment', 'unknown')}")
        print(f"   Market Data Available: {'price' in comprehensive_data.get('market_data', {})}")
    except Exception as e:
        print(f"❌ Comprehensive signal data failed: {e}")
    
    # Test API status
    print("\n📊 Testing API status...")
    try:
        api_status = await integration_service.get_api_status()
        print(f"✅ API Status: {api_status.get('overall_status', 'unknown')}")
        
        api_status_details = api_status.get('api_status', {})
        for api_name, status in api_status_details.items():
            print(f"   {api_name}: {status.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ API status failed: {e}")

async def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⏱️ Testing rate limiting...")
    print("=" * 50)
    
    api_manager = FreeAPIManager()
    
    # Test multiple requests to check rate limiting
    print("Testing multiple requests to check rate limiting...")
    
    for i in range(5):
        try:
            news_data = await api_manager.get_news_sentiment('BTC')
            print(f"Request {i+1}: {len(news_data.get('articles', []))} articles")
        except Exception as e:
            print(f"Request {i+1}: Failed - {e}")
        
        # Small delay between requests
        await asyncio.sleep(1)

async def main():
    """Main test function"""
    print("🚀 Starting Free API Integration Tests")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Test FreeAPIManager
        await test_free_api_manager()
        
        # Test FreeAPIIntegrationService
        await test_free_api_integration_service()
        
        # Test rate limiting
        await test_rate_limiting()
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.error(f"Test suite error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
