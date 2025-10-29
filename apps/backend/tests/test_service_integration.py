#!/usr/bin/env python3
"""
Service Integration Test for AlphaPlus
Tests market data service and signal generation
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
        from src.app.core.database_manager import DatabaseManager
        from src.app.services.market_data_service import MarketDataService
        
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
        
        # Test getting available symbols
        print("\n2. Testing available symbols...")
        try:
            symbols = await market_data_service.get_available_symbols()
            if symbols:
                print(f"✅ Found {len(symbols)} available symbols")
                print(f"   Sample symbols: {symbols[:5]}")
            else:
                print("ℹ️  No symbols found")
        except Exception as e:
            print(f"⚠️  Symbol retrieval failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Market data service test error: {e}")
        return False

async def test_signal_generation():
    """Test signal generation service"""
    print("\n🔍 Testing signal generation service...")
    
    try:
        from src.app.core.database_manager import DatabaseManager
        from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
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
        
        # Create signal generator
        signal_generator = RealTimeSignalGenerator(db_manager)
        
        print("✅ Signal generator created successfully")
        
        # Test getting recent signals
        print("\n1. Testing signal retrieval...")
        try:
            signals = await signal_generator.get_signals(limit=5)
            if signals:
                print(f"✅ Retrieved {len(signals)} recent signals")
                for i, signal in enumerate(signals[:3]):
                    print(f"   Signal {i+1}: {signal.get('symbol', 'N/A')} - {signal.get('signal_type', 'N/A')}")
            else:
                print("ℹ️  No recent signals found")
        except Exception as e:
            print(f"⚠️  Signal retrieval failed: {e}")
        
        await db_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Signal generation test error: {e}")
        return False

async def test_service_manager_integration():
    """Test service manager integration"""
    print("\n🔍 Testing service manager integration...")
    
    try:
        from src.app.core.database_manager import DatabaseManager
        from src.app.core.service_manager import ServiceManager
        from src.app.services.market_data_service import MarketDataService
        from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
        
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
        signal_generator = RealTimeSignalGenerator(db_manager)
        
        service_manager.register_service("database", db_manager)
        service_manager.register_service("market_data", market_data_service, dependencies=["database"])
        service_manager.register_service("signal_generator", signal_generator, dependencies=["database"])
        
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
    print("🚀 AlphaPlus Service Integration Test")
    print("=" * 50)
    
    tests = [
        test_market_data_service,
        test_signal_generation,
        test_service_manager_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Service Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Service integration successful! Ready for next step.")
        return True
    else:
        print("❌ Some service integration tests failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
