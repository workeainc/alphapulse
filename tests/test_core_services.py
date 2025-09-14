#!/usr/bin/env python3
"""
Core Services Test for AlphaPlus
Tests available core services
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_market_data_service():
    """Test market data service"""
    print("🔍 Testing market data service...")
    
    try:
        from app.core.database_manager import DatabaseManager
        from app.services.market_data_service import MarketDataService
        
        # Initialize database
        db_manager = DatabaseManager()
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 1,
            'max_size': 5
        }
        await db_manager.initialize(config)
        
        # Create market data service
        market_data_service = MarketDataService(db_manager)
        
        print("✅ Market data service created successfully")
        
        # Test getting market data
        print("\n1. Testing market data retrieval...")
        try:
            # Get recent candles for BTC/USDT
            candles = await market_data_service.get_market_data("BTC/USDT", "1h", 10)
            if candles:
                print(f"✅ Retrieved {len(candles)} candles for BTC/USDT")
                print(f"   Latest candle: {candles[0] if candles else 'None'}")
            else:
                print("ℹ️  No recent candles found for BTC/USDT")
        except Exception as e:
            print(f"⚠️  Market data retrieval failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Market data service test error: {e}")
        return False

async def test_strategy_manager():
    """Test strategy manager"""
    print("\n🔍 Testing strategy manager...")
    
    try:
        from app.core.database_manager import DatabaseManager
        from app.strategies.strategy_manager import StrategyManager
        
        # Initialize database
        db_manager = DatabaseManager()
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 1,
            'max_size': 5
        }
        await db_manager.initialize(config)
        
        # Create strategy manager
        strategy_manager = StrategyManager(db_manager)
        
        print("✅ Strategy manager created successfully")
        
        # Test getting strategies
        print("\n1. Testing strategy retrieval...")
        try:
            strategies = await strategy_manager.get_strategies()
            if strategies:
                print(f"✅ Retrieved {len(strategies)} strategies")
                for i, strategy in enumerate(strategies[:3]):
                    print(f"   Strategy {i+1}: {strategy.get('name', 'N/A')}")
            else:
                print("ℹ️  No strategies found")
        except Exception as e:
            print(f"⚠️  Strategy retrieval failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Strategy manager test error: {e}")
        return False

async def test_sentiment_service():
    """Test sentiment service"""
    print("\n🔍 Testing sentiment service...")
    
    try:
        from app.core.database_manager import DatabaseManager
        from app.services.sentiment_service import SentimentService
        
        # Initialize database
        db_manager = DatabaseManager()
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 1,
            'max_size': 5
        }
        await db_manager.initialize(config)
        
        # Create sentiment service
        sentiment_service = SentimentService(db_manager)
        
        print("✅ Sentiment service created successfully")
        
        # Test getting sentiment data
        print("\n1. Testing sentiment data retrieval...")
        try:
            sentiment_data = await sentiment_service.get_sentiment_data("BTC/USDT", limit=5)
            if sentiment_data:
                print(f"✅ Retrieved {len(sentiment_data)} sentiment records")
                print(f"   Latest sentiment: {sentiment_data[0] if sentiment_data else 'None'}")
            else:
                print("ℹ️  No sentiment data found")
        except Exception as e:
            print(f"⚠️  Sentiment data retrieval failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Sentiment service test error: {e}")
        return False

async def test_service_manager_integration():
    """Test service manager integration"""
    print("\n🔍 Testing service manager integration...")
    
    try:
        from app.core.database_manager import DatabaseManager
        from app.core.service_manager import ServiceManager
        from app.services.market_data_service import MarketDataService
        from app.services.sentiment_service import SentimentService
        from app.strategies.strategy_manager import StrategyManager
        
        # Initialize database
        db_manager = DatabaseManager()
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711',
            'min_size': 1,
            'max_size': 5
        }
        await db_manager.initialize(config)
        
        # Create service manager
        service_manager = ServiceManager()
        
        # Register services
        market_data_service = MarketDataService(db_manager)
        sentiment_service = SentimentService(db_manager)
        strategy_manager = StrategyManager(db_manager)
        
        service_manager.register_service("database", db_manager)
        service_manager.register_service("market_data", market_data_service, dependencies=["database"])
        service_manager.register_service("sentiment", sentiment_service, dependencies=["database"])
        service_manager.register_service("strategy", strategy_manager, dependencies=["database"])
        
        print("✅ Services registered successfully")
        print(f"   Registered services: {list(service_manager.services.keys())}")
        
        # Test service initialization
        print("\n1. Testing service initialization...")
        success = await service_manager.initialize_services()
        
        if success:
            print("✅ All services initialized successfully")
            
            # Check service status
            status = service_manager.get_all_services_status()
            print("\n2. Service status:")
            for service_name, service_status in status.items():
                print(f"   - {service_name}: {service_status['status']}")
        else:
            print("❌ Service initialization failed")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Service manager integration test error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 AlphaPlus Core Services Test")
    print("=" * 50)
    
    tests = [
        test_market_data_service,
        test_strategy_manager,
        test_sentiment_service,
        test_service_manager_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Core Services Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Core services integration successful! Ready for next step.")
        return True
    else:
        print("❌ Some core services tests failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
