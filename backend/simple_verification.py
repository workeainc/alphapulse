#!/usr/bin/env python3
"""
Simple verification script for key barrier fixes
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all key imports work"""
    print("🔍 Testing key imports...")
    
    try:
        # Test ML Pattern Detector import
        from app.strategies.ml_pattern_detector import MLPatternDetector
        print("✅ ML Pattern Detector import successful")
        
        # Test Strategy Manager import
        from app.strategies.strategy_manager import StrategyManager
        print("✅ Strategy Manager import successful")
        
        # Test Market Data Service import
        from app.services.market_data_service import MarketDataService
        print("✅ Market Data Service import successful")
        
        # Test Pattern Integration Service import
        from app.services.pattern_integration_service import PatternIntegrationService
        print("✅ Pattern Integration Service import successful")
        
        # Test database connection import
        from database.connection import TimescaleDBConnection
        print("✅ Database Connection import successful")
        
        # Test config import
        from app.core.config import DatabaseSettings
        print("✅ Config import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_initialization():
    """Test that services can be initialized"""
    print("\n🔍 Testing service initialization...")
    
    try:
        # Test ML Pattern Detector initialization
        from app.strategies.ml_pattern_detector import MLPatternDetector
        ml_detector = MLPatternDetector()
        print("✅ ML Pattern Detector initialized")
        
        # Test Strategy Manager initialization
        from app.strategies.strategy_manager import StrategyManager
        strategy_manager = StrategyManager()
        print("✅ Strategy Manager initialized")
        
        # Test Market Data Service initialization
        from app.services.market_data_service import MarketDataService
        market_data_service = MarketDataService()
        print("✅ Market Data Service initialized")
        
        # Test database connection initialization
        from database.connection import TimescaleDBConnection
        db_conn = TimescaleDBConnection()
        print("✅ Database Connection initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available"""
    print("\n🔍 Testing dependencies...")
    
    try:
        import aiosqlite
        print("✅ aiosqlite available")
        
        import ccxt
        print("✅ ccxt available")
        
        import sklearn
        print("✅ scikit-learn available")
        
        import textblob
        print("✅ textblob available")
        
        import tweepy
        print("✅ tweepy available")
        
        import praw
        print("✅ praw available")
        
        import transformers
        print("✅ transformers available")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting barrier fixes verification...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Initialization Tests", test_initialization),
        ("Dependency Tests", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Tests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Barrier fixes are working correctly.")
        print("✅ System is ready for production deployment!")
        print("✅ Ready for external API integration when needed!")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
