#!/usr/bin/env python3
"""
Free API Integration Service for AlphaPlus
Integrates free APIs into the existing sentiment and market data services
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.free_api_manager import FreeAPIManager
from src.app.services.sentiment_service import SentimentService
from src.app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

class FreeAPIIntegrationService:
    """Integrates free APIs into existing AlphaPlus services"""
    
    def __init__(self):
        self.free_api_manager = FreeAPIManager()
        self.sentiment_service = SentimentService()
        self.market_data_service = MarketDataService()
        
        logger.info("Free API Integration Service initialized")
    
    async def get_enhanced_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced sentiment using free APIs"""
        try:
            # Get news sentiment from free sources
            news_sentiment = await self.free_api_manager.get_news_sentiment(symbol)
            
            # Get social sentiment from free sources
            social_sentiment = await self.free_api_manager.get_social_sentiment(symbol)
            
            # Combine with existing sentiment service
            existing_sentiment = await self.sentiment_service.get_sentiment(symbol)
            
            # Aggregate sentiment scores
            sentiment_scores = {
                'news': self._convert_sentiment_to_score(news_sentiment.get('sentiment', 'neutral')),
                'social': self._convert_sentiment_to_score(social_sentiment.get('reddit', {}).get('sentiment', 'neutral')),
                'existing': existing_sentiment.get('overall_sentiment_score', 0)
            }
            
            # Calculate weighted average
            weights = {'news': 0.4, 'social': 0.3, 'existing': 0.3}
            overall_score = sum(sentiment_scores[key] * weights[key] for key in weights)
            
            # Determine overall sentiment
            if overall_score > 0.1:
                overall_sentiment = 'bullish'
            elif overall_score < -0.1:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            return {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment,
                'overall_sentiment_score': overall_score,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'existing_sentiment': existing_sentiment,
                'sentiment_scores': sentiment_scores,
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration'
            }
            
        except Exception as e:
            logger.error(f"Enhanced sentiment error: {e}")
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'overall_sentiment_score': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration'
            }
    
    async def get_enhanced_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced market data using free APIs"""
        try:
            # Get market data from free sources
            free_market_data = await self.free_api_manager.get_market_data(symbol)
            
            # Get liquidation data
            liquidation_data = await self.free_api_manager.get_liquidation_data(symbol)
            
            # Combine with existing market data service
            existing_market_data = await self.market_data_service.get_market_data(symbol)
            
            # Merge data sources
            enhanced_data = {
                'symbol': symbol,
                'price': free_market_data.get('data', {}).get('price', existing_market_data.get('price', 0)),
                'volume_24h': free_market_data.get('data', {}).get('volume_24h', existing_market_data.get('volume_24h', 0)),
                'price_change_24h': free_market_data.get('data', {}).get('price_change_24h', existing_market_data.get('price_change_24h', 0)),
                'market_cap': free_market_data.get('data', {}).get('market_cap', existing_market_data.get('market_cap', 0)),
                'fear_greed_index': free_market_data.get('data', {}).get('fear_greed_index', 50),
                'liquidation_data': liquidation_data,
                'free_api_data': free_market_data,
                'existing_data': existing_market_data,
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration'
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Enhanced market data error: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration'
            }
    
    async def get_comprehensive_signal_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive signal data using free APIs"""
        try:
            # Get all data in parallel
            sentiment_task = self.get_enhanced_sentiment(symbol)
            market_data_task = self.get_enhanced_market_data(symbol)
            
            sentiment_data, market_data = await asyncio.gather(
                sentiment_task,
                market_data_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(sentiment_data, Exception):
                logger.error(f"Sentiment data error: {sentiment_data}")
                sentiment_data = {'error': str(sentiment_data)}
            
            if isinstance(market_data, Exception):
                logger.error(f"Market data error: {market_data}")
                market_data = {'error': str(market_data)}
            
            # Combine all data
            comprehensive_data = {
                'symbol': symbol,
                'sentiment': sentiment_data,
                'market_data': market_data,
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration',
                'status': 'success'
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Comprehensive signal data error: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'free_api_integration',
                'status': 'error'
            }
    
    def _convert_sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment string to numerical score"""
        sentiment_map = {
            'bullish': 0.5,
            'bearish': -0.5,
            'neutral': 0.0,
            'positive': 0.3,
            'negative': -0.3
        }
        return sentiment_map.get(sentiment.lower(), 0.0)
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get status of all free APIs"""
        try:
            # Test each API
            api_status = {}
            
            # Test NewsAPI
            try:
                news_test = await self.free_api_manager.get_news_sentiment('BTC')
                api_status['newsapi'] = {
                    'status': 'working',
                    'last_test': datetime.now().isoformat(),
                    'articles_count': len(news_test.get('articles', []))
                }
            except Exception as e:
                api_status['newsapi'] = {
                    'status': 'error',
                    'error': str(e),
                    'last_test': datetime.now().isoformat()
                }
            
            # Test Reddit API
            try:
                reddit_test = await self.free_api_manager.get_social_sentiment('BTC')
                api_status['reddit'] = {
                    'status': 'working',
                    'last_test': datetime.now().isoformat(),
                    'posts_count': reddit_test.get('reddit', {}).get('posts', 0)
                }
            except Exception as e:
                api_status['reddit'] = {
                    'status': 'error',
                    'error': str(e),
                    'last_test': datetime.now().isoformat()
                }
            
            # Test CoinGecko API
            try:
                coingecko_test = await self.free_api_manager.get_market_data('BTC')
                api_status['coingecko'] = {
                    'status': 'working',
                    'last_test': datetime.now().isoformat(),
                    'has_price': 'price' in coingecko_test.get('data', {})
                }
            except Exception as e:
                api_status['coingecko'] = {
                    'status': 'error',
                    'error': str(e),
                    'last_test': datetime.now().isoformat()
                }
            
            # Test Binance API
            try:
                binance_test = await self.free_api_manager.get_liquidation_data('BTC')
                api_status['binance'] = {
                    'status': 'working',
                    'last_test': datetime.now().isoformat(),
                    'liquidations_count': binance_test.get('recent_liquidations', 0)
                }
            except Exception as e:
                api_status['binance'] = {
                    'status': 'error',
                    'error': str(e),
                    'last_test': datetime.now().isoformat()
                }
            
            return {
                'api_status': api_status,
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'working' if all(api.get('status') == 'working' for api in api_status.values()) else 'partial'
            }
            
        except Exception as e:
            logger.error(f"API status check error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error'
            }

# Example usage
async def main():
    """Example usage of FreeAPIIntegrationService"""
    integration_service = FreeAPIIntegrationService()
    
    # Test comprehensive signal data
    print("Testing comprehensive signal data...")
    signal_data = await integration_service.get_comprehensive_signal_data('BTC')
    print(f"Signal Data: {signal_data}")
    
    # Test API status
    print("\nTesting API status...")
    api_status = await integration_service.get_api_status()
    print(f"API Status: {api_status}")

if __name__ == "__main__":
    asyncio.run(main())
