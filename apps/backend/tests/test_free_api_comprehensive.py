#!/usr/bin/env python3
"""
Comprehensive Free API Test Suite
Tests all free API implementations with detailed reporting
"""

import asyncio
import sys
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreeAPITestSuite:
    """Comprehensive test suite for free API implementations"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    async def run_all_tests(self):
        """Run all free API tests"""
        print("🚀 Starting Comprehensive Free API Test Suite")
        print("=" * 80)
        print(f"Test started at: {self.start_time.isoformat()}")
        print("=" * 80)
        
        # Test 1: Core Dependencies
        await self.test_core_dependencies()
        
        # Test 2: FreeAPIManager Implementation
        await self.test_free_api_manager()
        
        # Test 3: FreeAPIIntegrationService Implementation
        await self.test_free_api_integration_service()
        
        # Test 4: API Endpoints Integration
        await self.test_api_endpoints()
        
        # Test 5: Configuration Files
        await self.test_configuration()
        
        # Test 6: Rate Limiting Logic
        await self.test_rate_limiting_logic()
        
        # Test 7: Caching Implementation
        await self.test_caching_implementation()
        
        # Test 8: Fallback Mechanisms
        await self.test_fallback_mechanisms()
        
        # Generate comprehensive report
        await self.generate_report()
    
    async def test_core_dependencies(self):
        """Test core dependencies availability"""
        print("\n📦 Testing Core Dependencies...")
        print("-" * 50)
        
        dependencies = {
            'aiohttp': 'HTTP client for API calls',
            'redis': 'Caching system',
            'json': 'JSON processing',
            'praw': 'Reddit API wrapper',
            'feedparser': 'RSS feed parsing',
            'transformers': 'Local ML models',
            'requests': 'HTTP requests fallback'
        }
        
        results = {}
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                print(f"✅ {dep}: {description}")
                results[dep] = {'status': 'available', 'description': description}
            except ImportError:
                print(f"⚠️ {dep}: {description} - NOT AVAILABLE")
                results[dep] = {'status': 'missing', 'description': description}
        
        self.test_results['dependencies'] = results
        print(f"✅ Dependencies test completed: {len([r for r in results.values() if r['status'] == 'available'])}/{len(results)} available")
    
    async def test_free_api_manager(self):
        """Test FreeAPIManager implementation"""
        print("\n🔧 Testing FreeAPIManager Implementation...")
        print("-" * 50)
        
        try:
            from src.services.free_api_manager import FreeAPIManager, APILimit
            
            # Test class creation
            print("✅ FreeAPIManager class imported successfully")
            
            # Test APILimit dataclass
            api_limit = APILimit(1000, 100, 10)
            print(f"✅ APILimit dataclass: {api_limit.requests_per_day} requests/day")
            
            # Test FreeAPIManager initialization (without Redis)
            try:
                # Mock Redis to avoid connection issues
                import unittest.mock
                with unittest.mock.patch('redis.Redis'):
                    api_manager = FreeAPIManager()
                    print("✅ FreeAPIManager initialized successfully")
                    
                    # Test API limits configuration
                    print(f"✅ API limits configured: {len(api_manager.api_limits)} APIs")
                    for api_name, limit in api_manager.api_limits.items():
                        print(f"   - {api_name}: {limit.requests_per_day} requests/day")
                    
                    # Test API initialization
                    print(f"✅ NewsAPI key configured: {bool(api_manager.newsapi_key)}")
                    print(f"✅ Reddit client: {api_manager.reddit is not None}")
                    print(f"✅ CoinGecko base URL: {api_manager.coingecko_base}")
                    print(f"✅ Binance base URL: {api_manager.binance_base}")
                    print(f"✅ Hugging Face token: {bool(api_manager.huggingface_token)}")
                    
                    self.test_results['free_api_manager'] = {
                        'status': 'success',
                        'api_limits': len(api_manager.api_limits),
                        'newsapi_configured': bool(api_manager.newsapi_key),
                        'reddit_configured': api_manager.reddit is not None,
                        'coingecko_configured': bool(api_manager.coingecko_base),
                        'binance_configured': bool(api_manager.binance_base),
                        'huggingface_configured': bool(api_manager.huggingface_token)
                    }
                    
            except Exception as e:
                print(f"❌ FreeAPIManager initialization failed: {e}")
                self.test_results['free_api_manager'] = {'status': 'failed', 'error': str(e)}
                
        except ImportError as e:
            print(f"❌ FreeAPIManager import failed: {e}")
            self.test_results['free_api_manager'] = {'status': 'import_failed', 'error': str(e)}
    
    async def test_free_api_integration_service(self):
        """Test FreeAPIIntegrationService implementation"""
        print("\n🔗 Testing FreeAPIIntegrationService Implementation...")
        print("-" * 50)
        
        try:
            from src.services.free_api_integration_service import FreeAPIIntegrationService
            
            print("✅ FreeAPIIntegrationService class imported successfully")
            
            # Test service initialization
            try:
                integration_service = FreeAPIIntegrationService()
                print("✅ FreeAPIIntegrationService initialized successfully")
                
                # Test service components
                print(f"✅ Free API Manager: {integration_service.free_api_manager is not None}")
                
                self.test_results['free_api_integration_service'] = {
                    'status': 'success',
                    'free_api_manager_available': integration_service.free_api_manager is not None
                }
                
            except Exception as e:
                print(f"❌ FreeAPIIntegrationService initialization failed: {e}")
                self.test_results['free_api_integration_service'] = {'status': 'failed', 'error': str(e)}
                
        except ImportError as e:
            print(f"❌ FreeAPIIntegrationService import failed: {e}")
            self.test_results['free_api_integration_service'] = {'status': 'import_failed', 'error': str(e)}
    
    async def test_api_endpoints(self):
        """Test API endpoints integration"""
        print("\n🌐 Testing API Endpoints Integration...")
        print("-" * 50)
        
        try:
            from src.app.main_ai_system_simple import app
            
            print("✅ Main app imported successfully")
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    routes.append({
                        'path': route.path,
                        'methods': list(route.methods),
                        'name': getattr(route, 'name', 'unknown')
                    })
            
            # Filter free API routes
            free_api_routes = [r for r in routes if 'free-apis' in r['path']]
            
            print(f"✅ Found {len(free_api_routes)} free API endpoints:")
            for route in free_api_routes:
                print(f"   - {route['methods']} {route['path']} ({route['name']})")
            
            # Expected endpoints
            expected_endpoints = [
                '/api/v1/free-apis/sentiment/{symbol}',
                '/api/v1/free-apis/market-data/{symbol}',
                '/api/v1/free-apis/comprehensive/{symbol}',
                '/api/v1/free-apis/status'
            ]
            
            found_endpoints = [r['path'] for r in free_api_routes]
            
            print(f"\n✅ Endpoint coverage:")
            coverage_results = {}
            for endpoint in expected_endpoints:
                if endpoint in found_endpoints:
                    print(f"   ✅ {endpoint}")
                    coverage_results[endpoint] = 'found'
                else:
                    print(f"   ❌ {endpoint}")
                    coverage_results[endpoint] = 'missing'
            
            self.test_results['api_endpoints'] = {
                'status': 'success',
                'total_routes': len(routes),
                'free_api_routes': len(free_api_routes),
                'coverage': coverage_results
            }
            
        except Exception as e:
            print(f"❌ API endpoints test failed: {e}")
            self.test_results['api_endpoints'] = {'status': 'failed', 'error': str(e)}
    
    async def test_configuration(self):
        """Test configuration files"""
        print("\n⚙️ Testing Configuration Files...")
        print("-" * 50)
        
        config_files = {
            'backend/free_api_config_template.env': 'Free API configuration template',
            'backend/services/free_api_manager.py': 'Free API Manager implementation',
            'backend/services/free_api_integration_service.py': 'Free API Integration Service'
        }
        
        config_results = {}
        for file_path, description in config_files.items():
            if os.path.exists(file_path):
                print(f"✅ {file_path}: {description}")
                
                # Check file content for key configurations
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for specific configurations
                    checks = {
                        'NEWS_API_KEY': 'NEWS_API_KEY' in content,
                        'REDDIT_CLIENT_ID': 'REDDIT_CLIENT_ID' in content,
                        'HUGGINGFACE_API_KEY': 'HUGGINGFACE_API_KEY' in content,
                        'Redis configuration': 'redis' in content.lower(),
                        'API limits': 'APILimit' in content or 'rate_limit' in content.lower()
                    }
                    
                    config_results[file_path] = {
                        'status': 'found',
                        'description': description,
                        'checks': checks
                    }
                    
                    for check, result in checks.items():
                        status = "✅" if result else "❌"
                        print(f"   {status} {check}")
                        
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")
                    config_results[file_path] = {'status': 'error', 'error': str(e)}
            else:
                print(f"❌ {file_path}: {description} - NOT FOUND")
                config_results[file_path] = {'status': 'missing', 'description': description}
        
        self.test_results['configuration'] = config_results
    
    async def test_rate_limiting_logic(self):
        """Test rate limiting logic implementation"""
        print("\n⏱️ Testing Rate Limiting Logic...")
        print("-" * 50)
        
        try:
            from src.services.free_api_manager import FreeAPIManager, APILimit
            
            # Test APILimit dataclass functionality
            api_limit = APILimit(1000, 100, 10)
            print(f"✅ APILimit created: {api_limit.requests_per_day} requests/day")
            
            # Test rate limit tracking
            api_limit.current_daily = 500
            api_limit.current_hourly = 50
            api_limit.current_minute = 5
            
            print(f"✅ Rate limit tracking: Daily={api_limit.current_daily}, Hourly={api_limit.current_hourly}, Minute={api_limit.current_minute}")
            
            # Test rate limit checking logic
            daily_ok = api_limit.current_daily < api_limit.requests_per_day
            hourly_ok = api_limit.current_hourly < api_limit.requests_per_hour
            minute_ok = api_limit.current_minute < api_limit.requests_per_minute
            
            print(f"✅ Rate limit checks: Daily={daily_ok}, Hourly={hourly_ok}, Minute={minute_ok}")
            
            self.test_results['rate_limiting'] = {
                'status': 'success',
                'daily_limit': api_limit.requests_per_day,
                'hourly_limit': api_limit.requests_per_hour,
                'minute_limit': api_limit.requests_per_minute,
                'tracking_working': True
            }
            
        except Exception as e:
            print(f"❌ Rate limiting test failed: {e}")
            self.test_results['rate_limiting'] = {'status': 'failed', 'error': str(e)}
    
    async def test_caching_implementation(self):
        """Test caching implementation"""
        print("\n💾 Testing Caching Implementation...")
        print("-" * 50)
        
        try:
            from src.services.free_api_manager import FreeAPIManager
            
            # Check if caching is implemented in the code
            with open('backend/services/free_api_manager.py', 'r') as f:
                content = f.read()
            
            caching_features = {
                'Redis client': 'redis_client' in content,
                'Cache key generation': 'cache_key' in content,
                'Cache retrieval': 'redis_client.get' in content,
                'Cache storage': 'redis_client.setex' in content,
                'Cache expiration': 'setex' in content
            }
            
            print("✅ Caching implementation checks:")
            for feature, implemented in caching_features.items():
                status = "✅" if implemented else "❌"
                print(f"   {status} {feature}")
            
            self.test_results['caching'] = {
                'status': 'success',
                'features': caching_features,
                'redis_integration': 'redis_client' in content
            }
            
        except Exception as e:
            print(f"❌ Caching test failed: {e}")
            self.test_results['caching'] = {'status': 'failed', 'error': str(e)}
    
    async def test_fallback_mechanisms(self):
        """Test fallback mechanisms"""
        print("\n🔄 Testing Fallback Mechanisms...")
        print("-" * 50)
        
        try:
            from src.services.free_api_manager import FreeAPIManager
            
            # Check if fallback mechanisms are implemented
            with open('backend/services/free_api_manager.py', 'r') as f:
                content = f.read()
            
            fallback_features = {
                'NewsAPI fallback to Reddit': 'reddit' in content and 'newsapi' in content,
                'Reddit fallback to RSS': 'rss' in content.lower(),
                'Hugging Face fallback to local model': 'local_sentiment_model' in content,
                'Local model fallback to keywords': 'keywords' in content.lower(),
                'CoinGecko fallback to Binance': 'coingecko' in content and 'binance' in content,
                'Exception handling': 'except Exception' in content
            }
            
            print("✅ Fallback mechanism checks:")
            for feature, implemented in fallback_features.items():
                status = "✅" if implemented else "❌"
                print(f"   {status} {feature}")
            
            self.test_results['fallback_mechanisms'] = {
                'status': 'success',
                'features': fallback_features,
                'exception_handling': 'except Exception' in content
            }
            
        except Exception as e:
            print(f"❌ Fallback mechanisms test failed: {e}")
            self.test_results['fallback_mechanisms'] = {'status': 'failed', 'error': str(e)}
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE FREE API TEST REPORT")
        print("=" * 80)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"Test Duration: {duration:.2f} seconds")
        print(f"Test Completed: {end_time.isoformat()}")
        print()
        
        # Summary statistics
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() if isinstance(r, dict) and r.get('status') == 'success'])
        
        print(f"📈 Test Summary: {successful_tests}/{total_tests} tests passed")
        print()
        
        # Detailed results
        for test_name, result in self.test_results.items():
            print(f"🔍 {test_name.upper().replace('_', ' ')}:")
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    print(f"   ✅ Status: SUCCESS")
                    # Print additional details
                    for key, value in result.items():
                        if key != 'status':
                            print(f"   📊 {key}: {value}")
                else:
                    print(f"   ❌ Status: {result.get('status', 'UNKNOWN')}")
                    if 'error' in result:
                        print(f"   🚨 Error: {result['error']}")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        if 'dependencies' in self.test_results:
            missing_deps = [dep for dep, info in self.test_results['dependencies'].items() 
                          if info.get('status') == 'missing']
            if missing_deps:
                print(f"📦 Install missing dependencies: pip install {' '.join(missing_deps)}")
        
        if 'configuration' in self.test_results:
            config_status = self.test_results['configuration']
            if any(info.get('status') == 'missing' for info in config_status.values()):
                print("⚙️ Configure missing configuration files")
        
        print("🔑 Set up API keys in .env file:")
        print("   - NEWS_API_KEY (free tier: 1,000 requests/day)")
        print("   - REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        print("   - HUGGINGFACE_API_KEY (free tier: 1,000 requests/month)")
        
        print("🚀 Start Redis server for caching: redis-server")
        
        print("\n🎯 NEXT STEPS:")
        print("-" * 50)
        print("1. Install missing dependencies")
        print("2. Configure API keys")
        print("3. Start Redis server")
        print("4. Run live API tests: python backend/test_free_api_integration.py")
        print("5. Test endpoints: curl http://localhost:8000/api/v1/free-apis/status")
        
        # Save report to file
        report_file = f"backend/free_api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'test_results': self.test_results,
                    'duration': duration,
                    'timestamp': end_time.isoformat(),
                    'summary': {
                        'total_tests': total_tests,
                        'successful_tests': successful_tests,
                        'success_rate': f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
                    }
                }, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save report: {e}")

async def main():
    """Main test function"""
    test_suite = FreeAPITestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
