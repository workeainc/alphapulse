#!/usr/bin/env python3
"""
Simple Free API Test - No External Dependencies
Tests core functionality without requiring external API keys
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_core_imports():
    """Test core imports without external dependencies"""
    print("🔍 Testing Core Imports...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("Testing basic imports...")
        import aiohttp
        print("✅ aiohttp imported")
        
        import json
        print("✅ json imported")
        
        import redis
        print("✅ redis imported")
        
        # Test if praw is available
        try:
            import praw
            print("✅ praw imported")
        except ImportError:
            print("⚠️ praw not available - will use fallback")
        
        # Test if feedparser is available
        try:
            import feedparser
            print("✅ feedparser imported")
        except ImportError:
            print("⚠️ feedparser not available - will use fallback")
        
        # Test if transformers is available
        try:
            import transformers
            print("✅ transformers imported")
        except ImportError:
            print("⚠️ transformers not available - will use fallback")
        
        print("\n✅ Core imports test completed")
        return True
        
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

async def test_free_api_manager_creation():
    """Test FreeAPIManager creation"""
    print("\n🔧 Testing FreeAPIManager Creation...")
    print("=" * 50)
    
    try:
        # Import with fallback handling
        try:
            from src.services.free_api_manager import FreeAPIManager
            print("✅ FreeAPIManager imported successfully")
            
            # Create instance
            api_manager = FreeAPIManager()
            print("✅ FreeAPIManager instance created")
            
            # Test basic attributes
            print(f"✅ API limits configured: {len(api_manager.api_limits)} APIs")
            print(f"✅ Redis client initialized: {api_manager.redis_client is not None}")
            
            return True
            
        except ImportError as e:
            print(f"❌ FreeAPIManager import failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ FreeAPIManager creation failed: {e}")
        return False

async def test_free_api_integration_service():
    """Test FreeAPIIntegrationService creation"""
    print("\n🔗 Testing FreeAPIIntegrationService Creation...")
    print("=" * 50)
    
    try:
        # Import with fallback handling
        try:
            from src.services.free_api_integration_service import FreeAPIIntegrationService
            print("✅ FreeAPIIntegrationService imported successfully")
            
            # Create instance
            integration_service = FreeAPIIntegrationService()
            print("✅ FreeAPIIntegrationService instance created")
            
            # Test basic attributes
            print(f"✅ Free API Manager: {integration_service.free_api_manager is not None}")
            
            return True
            
        except ImportError as e:
            print(f"❌ FreeAPIIntegrationService import failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ FreeAPIIntegrationService creation failed: {e}")
        return False

async def test_main_app_integration():
    """Test main app integration"""
    print("\n🚀 Testing Main App Integration...")
    print("=" * 50)
    
    try:
        # Test if main app can import the new services
        try:
            from src.app.main_ai_system_simple import app
            print("✅ Main app imported successfully")
            
            # Check if free API endpoints are available
            routes = [route.path for route in app.routes]
            free_api_routes = [route for route in routes if 'free-apis' in route]
            
            print(f"✅ Free API routes found: {len(free_api_routes)}")
            for route in free_api_routes:
                print(f"   - {route}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Main app import failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Main app integration failed: {e}")
        return False

async def test_api_endpoints():
    """Test API endpoints without external calls"""
    print("\n🌐 Testing API Endpoints...")
    print("=" * 50)
    
    try:
        # Test endpoint definitions
        from src.app.main_ai_system_simple import app
        
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
        for endpoint in expected_endpoints:
            if endpoint in found_endpoints:
                print(f"   ✅ {endpoint}")
            else:
                print(f"   ❌ {endpoint}")
        
        return len(free_api_routes) >= len(expected_endpoints)
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

async def test_configuration():
    """Test configuration files"""
    print("\n⚙️ Testing Configuration...")
    print("=" * 50)
    
    try:
        # Check if config template exists
        config_file = "backend/free_api_config_template.env"
        if os.path.exists(config_file):
            print(f"✅ Configuration template found: {config_file}")
            
            # Read and check content
            with open(config_file, 'r') as f:
                content = f.read()
                
            # Check for key configurations
            required_configs = [
                'NEWS_API_KEY',
                'REDDIT_CLIENT_ID',
                'HUGGINGFACE_API_KEY',
                'REDIS_HOST',
                'DB_HOST'
            ]
            
            print("✅ Configuration template contains:")
            for config in required_configs:
                if config in content:
                    print(f"   ✅ {config}")
                else:
                    print(f"   ❌ {config}")
            
            return True
        else:
            print(f"❌ Configuration template not found: {config_file}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Simple Free API Integration Tests")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    test_results = []
    
    try:
        # Run all tests
        test_results.append(await test_core_imports())
        test_results.append(await test_free_api_manager_creation())
        test_results.append(await test_free_api_integration_service())
        test_results.append(await test_main_app_integration())
        test_results.append(await test_api_endpoints())
        test_results.append(await test_configuration())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        print("\n" + "=" * 60)
        print(f"🎉 Test Results: {passed_tests}/{total_tests} tests passed")
        print("=" * 60)
        
        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED! Free API integration is ready!")
        else:
            print(f"⚠️ {total_tests - passed_tests} tests failed. Check the output above.")
        
        print("\n📋 Next Steps:")
        print("1. Install missing dependencies: pip install praw feedparser")
        print("2. Configure API keys in .env file")
        print("3. Start Redis server for caching")
        print("4. Run full integration test with: python backend/test_free_api_integration.py")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logger.error(f"Test suite error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
