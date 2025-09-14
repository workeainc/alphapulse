#!/usr/bin/env python3
"""
Comprehensive External API Testing Script for AlphaPlus
Tests all external APIs including WebSocket, CoinMarketCap, CoinGecko, News API, Twitter, etc.
"""

import asyncio
import aiohttp
import websockets
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExternalAPITester:
    """Comprehensive external API testing class"""
    
    def __init__(self):
        self.results = {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_news_api(self) -> Dict[str, Any]:
        """Test News API"""
        logger.info("🔍 Testing News API...")
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'cryptocurrency Bitcoin',
                'apiKey': settings.NEWS_API_KEY,
                'sortBy': 'publishedAt',
                'pageSize': 5
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    logger.info(f"✅ News API: Retrieved {len(articles)} articles")
                    return {
                        'status': 'success',
                        'articles_count': len(articles),
                        'sample_title': articles[0].get('title', '') if articles else '',
                        'response_time': response.headers.get('X-RateLimit-Remaining', 'N/A')
                    }
                else:
                    logger.error(f"❌ News API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ News API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_coinmarketcap_api(self) -> Dict[str, Any]:
        """Test CoinMarketCap API"""
        logger.info("🔍 Testing CoinMarketCap API...")
        try:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            headers = {
                'X-CMC_PRO_API_KEY': settings.COINMARKETCAP_API_KEY,
                'Accept': 'application/json'
            }
            params = {
                'start': '1',
                'limit': '5',
                'convert': 'USD'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    cryptocurrencies = data.get('data', [])
                    logger.info(f"✅ CoinMarketCap API: Retrieved {len(cryptocurrencies)} cryptocurrencies")
                    return {
                        'status': 'success',
                        'crypto_count': len(cryptocurrencies),
                        'sample_crypto': cryptocurrencies[0].get('name', '') if cryptocurrencies else '',
                        'response_time': response.headers.get('X-RateLimit-Remaining', 'N/A')
                    }
                else:
                    logger.error(f"❌ CoinMarketCap API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ CoinMarketCap API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_coingecko_api(self) -> Dict[str, Any]:
        """Test CoinGecko API"""
        logger.info("🔍 Testing CoinGecko API...")
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 5,
                'page': 1,
                'sparkline': 'false'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ CoinGecko API: Retrieved {len(data)} cryptocurrencies")
                    return {
                        'status': 'success',
                        'crypto_count': len(data),
                        'sample_crypto': data[0].get('name', '') if data else '',
                        'response_time': 'N/A'
                    }
                else:
                    logger.error(f"❌ CoinGecko API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ CoinGecko API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_polygon_api(self) -> Dict[str, Any]:
        """Test Polygon API"""
        logger.info("🔍 Testing Polygon API...")
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
            params = {
                'adjusted': 'true',
                'apikey': settings.POLYGON_API_KEY
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Polygon API: Retrieved market data")
                    return {
                        'status': 'success',
                        'ticker': data.get('ticker', ''),
                        'response_time': 'N/A'
                    }
                else:
                    logger.error(f"❌ Polygon API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ Polygon API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_coinglass_api(self) -> Dict[str, Any]:
        """Test CoinGlass API"""
        logger.info("🔍 Testing CoinGlass API...")
        try:
            url = "https://open-api.coinglass.com/public/v2/futures/longShort_chart"
            params = {
                'symbol': 'BTC',
                'time_type': 'h1',
                'api_key': settings.COINGLASS_API_KEY
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ CoinGlass API: Retrieved futures data")
                    return {
                        'status': 'success',
                        'data_type': 'futures_long_short',
                        'response_time': 'N/A'
                    }
                else:
                    logger.error(f"❌ CoinGlass API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ CoinGlass API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_binance_websocket(self) -> Dict[str, Any]:
        """Test Binance WebSocket connection"""
        logger.info("🔍 Testing Binance WebSocket...")
        try:
            uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
            
            async with websockets.connect(uri) as websocket:
                # Wait for a message
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(message)
                
                logger.info("✅ Binance WebSocket: Received ticker data")
                return {
                    'status': 'success',
                    'symbol': data.get('s', ''),
                    'price': data.get('c', ''),
                    'connection_time': 'N/A'
                }
                
        except asyncio.TimeoutError:
            logger.error("❌ Binance WebSocket: Connection timeout")
            return {'status': 'error', 'error': 'Connection timeout'}
        except Exception as e:
            logger.error(f"❌ Binance WebSocket Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_huggingface_api(self) -> Dict[str, Any]:
        """Test Hugging Face API"""
        logger.info("🔍 Testing Hugging Face API...")
        try:
            url = "https://api-inference.huggingface.co/models/sentiment-analysis"
            headers = {
                'Authorization': f'Bearer {settings.HUGGINGFACE_API_KEY}',
                'Content-Type': 'application/json'
            }
            data = {
                'inputs': 'Bitcoin is going to the moon!'
            }
            
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("✅ Hugging Face API: Sentiment analysis successful")
                    return {
                        'status': 'success',
                        'sentiment': result[0][0].get('label', '') if result else '',
                        'confidence': result[0][0].get('score', 0) if result else 0
                    }
                else:
                    logger.error(f"❌ Hugging Face API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ Hugging Face API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_twitter_api(self) -> Dict[str, Any]:
        """Test Twitter API (Basic connectivity)"""
        logger.info("🔍 Testing Twitter API...")
        try:
            # Note: Twitter API v2 requires OAuth 2.0 Bearer Token
            # This is a basic connectivity test
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {settings.TWITTER_API_KEY}',
                'Content-Type': 'application/json'
            }
            params = {
                'query': 'cryptocurrency',
                'max_results': 10
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tweets = data.get('data', [])
                    logger.info(f"✅ Twitter API: Retrieved {len(tweets)} tweets")
                    return {
                        'status': 'success',
                        'tweets_count': len(tweets),
                        'response_time': 'N/A'
                    }
                else:
                    logger.error(f"❌ Twitter API: HTTP {response.status}")
                    return {'status': 'error', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"❌ Twitter API Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def test_internal_apis(self) -> Dict[str, Any]:
        """Test internal AlphaPlus APIs"""
        logger.info("🔍 Testing Internal AlphaPlus APIs...")
        results = {}
        
        # Test health endpoint
        try:
            async with self.session.get("http://localhost:8000/api/v1/production/health") as response:
                if response.status == 200:
                    data = await response.json()
                    results['health'] = {'status': 'success', 'components': data.get('components', {})}
                else:
                    results['health'] = {'status': 'error', 'error': f'HTTP {response.status}'}
        except Exception as e:
            results['health'] = {'status': 'error', 'error': str(e)}
        
        # Test status endpoint
        try:
            async with self.session.get("http://localhost:8000/api/v1/production/status") as response:
                if response.status == 200:
                    data = await response.json()
                    results['status'] = {'status': 'success', 'phases': data.get('phases_completed', [])}
                else:
                    results['status'] = {'status': 'error', 'error': f'HTTP {response.status}'}
        except Exception as e:
            results['status'] = {'status': 'error', 'error': str(e)}
        
        # Test metrics endpoint
        try:
            async with self.session.get("http://localhost:8000/api/v1/production/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    results['metrics'] = {'status': 'success', 'metrics_count': len(data)}
                else:
                    results['metrics'] = {'status': 'error', 'error': f'HTTP {response.status}'}
        except Exception as e:
            results['metrics'] = {'status': 'error', 'error': str(e)}
        
        logger.info(f"✅ Internal APIs: Tested {len(results)} endpoints")
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests"""
        logger.info("🚀 Starting comprehensive external API testing...")
        start_time = time.time()
        
        # Run all tests concurrently
        tasks = [
            self.test_news_api(),
            self.test_coinmarketcap_api(),
            self.test_coingecko_api(),
            self.test_polygon_api(),
            self.test_coinglass_api(),
            self.test_binance_websocket(),
            self.test_huggingface_api(),
            self.test_twitter_api(),
            self.test_internal_apis()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        api_names = [
            'news_api',
            'coinmarketcap_api',
            'coingecko_api',
            'polygon_api',
            'coinglass_api',
            'binance_websocket',
            'huggingface_api',
            'twitter_api',
            'internal_apis'
        ]
        
        self.results = dict(zip(api_names, results))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        successful_apis = sum(1 for result in results if isinstance(result, dict) and result.get('status') == 'success')
        total_apis = len(results)
        
        logger.info(f"✅ API Testing Complete: {successful_apis}/{total_apis} APIs working")
        logger.info(f"⏱️ Total testing time: {total_time:.2f} seconds")
        
        return {
            'summary': {
                'total_apis': total_apis,
                'successful_apis': successful_apis,
                'failed_apis': total_apis - successful_apis,
                'success_rate': f"{(successful_apis/total_apis)*100:.1f}%",
                'total_time': f"{total_time:.2f}s"
            },
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Main testing function"""
    print("🔍 AlphaPlus External API Testing Suite")
    print("=" * 50)
    
    async with ExternalAPITester() as tester:
        results = await tester.run_all_tests()
        
        # Print detailed results
        print("\n📊 Detailed Results:")
        print("-" * 30)
        
        for api_name, result in results['results'].items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                status_icon = "✅" if status == 'success' else "❌"
                print(f"{status_icon} {api_name.replace('_', ' ').title()}: {status}")
                
                if status == 'error':
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                elif api_name == 'internal_apis':
                    # Show internal API details
                    for internal_api, internal_result in result.items():
                        internal_status = internal_result.get('status', 'unknown')
                        internal_icon = "✅" if internal_status == 'success' else "❌"
                        print(f"   {internal_icon} {internal_api}: {internal_status}")
            else:
                print(f"❌ {api_name.replace('_', ' ').title()}: Exception occurred")
        
        # Print summary
        print(f"\n📈 Summary:")
        print(f"   Total APIs: {results['summary']['total_apis']}")
        print(f"   Successful: {results['summary']['successful_apis']}")
        print(f"   Failed: {results['summary']['failed_apis']}")
        print(f"   Success Rate: {results['summary']['success_rate']}")
        print(f"   Total Time: {results['summary']['total_time']}")
        
        # Save results to file
        with open('api_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: api_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())
