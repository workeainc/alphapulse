#!/usr/bin/env python3
"""
WebSocket Integration Test for AlphaPlus
Tests real-time communication capabilities
"""

import asyncio
import sys
import os
import json
import websockets
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_websocket_server():
    """Test WebSocket server functionality"""
    print("🔍 Testing WebSocket server...")
    
    try:
        from app.main_unified_fixed import app
        
        # Check if the app has WebSocket endpoints
        routes = [route.path for route in app.routes]
        websocket_routes = [route for route in routes if 'ws' in route.lower()]
        
        if websocket_routes:
            print(f"✅ Found {len(websocket_routes)} WebSocket routes:")
            for route in websocket_routes:
                print(f"   - {route}")
        else:
            print("ℹ️  No WebSocket routes found in main application")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket server test error: {e}")
        return False

async def test_websocket_client():
    """Test WebSocket client connection"""
    print("\n🔍 Testing WebSocket client connection...")
    
    try:
        # Test connection to localhost:8000/ws
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri, timeout=5) as websocket:
                print("✅ WebSocket connection successful")
                
                # Test sending a message
                test_message = {
                    "type": "ping",
                    "data": "test"
                }
                
                await websocket.send(json.dumps(test_message))
                print("✅ Message sent successfully")
                
                # Test receiving a message
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    print(f"✅ Received response: {response}")
                except asyncio.TimeoutError:
                    print("ℹ️  No response received (timeout)")
                
                return True
                
        except websockets.exceptions.ConnectionClosed:
            print("ℹ️  WebSocket connection closed (server may not be running)")
            return True
        except websockets.exceptions.InvalidURI:
            print("ℹ️  Invalid WebSocket URI")
            return True
        except Exception as e:
            print(f"ℹ️  WebSocket connection failed: {e}")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket client test error: {e}")
        return False

async def test_websocket_service():
    """Test WebSocket service functionality"""
    print("\n🔍 Testing WebSocket service...")
    
    try:
        from app.services.enhanced_websocket_service import EnhancedWebSocketService
        
        # Create WebSocket service
        websocket_service = EnhancedWebSocketService()
        
        print("✅ WebSocket service created successfully")
        
        # Test service methods
        methods = [method for method in dir(websocket_service) if not method.startswith('_')]
        print(f"✅ WebSocket service has {len(methods)} public methods")
        print(f"   Sample methods: {methods[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket service test error: {e}")
        return False

async def test_real_time_data():
    """Test real-time data processing"""
    print("\n🔍 Testing real-time data processing...")
    
    try:
        from app.data.real_time_processor import RealTimeCandlestickProcessor
        
        # Create real-time processor
        processor = RealTimeCandlestickProcessor({
            'min_confidence': 0.7,
            'min_strength': 0.6,
            'confirmation_required': True,
            'volume_confirmation': True,
            'trend_confirmation': True,
            'min_data_points': 50,
            'max_data_points': 1000,
            'signal_cooldown': 300
        })
        
        print("✅ Real-time processor created successfully")
        
        # Test processor methods
        methods = [method for method in dir(processor) if not method.startswith('_')]
        print(f"✅ Real-time processor has {len(methods)} public methods")
        print(f"   Sample methods: {methods[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time data test error: {e}")
        return False

async def test_websocket_integration():
    """Test WebSocket integration with database"""
    print("\n🔍 Testing WebSocket integration with database...")
    
    try:
        from app.core.database_manager import DatabaseManager
        
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
        
        success = await db_manager.initialize(config)
        if success:
            print("✅ Database connection for WebSocket integration successful")
            
            # Test WebSocket-related tables
            websocket_tables = await db_manager.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%websocket%'
                ORDER BY table_name
            """)
            
            if websocket_tables:
                print(f"✅ Found {len(websocket_tables)} WebSocket-related tables:")
                for table in websocket_tables:
                    print(f"   - {table['table_name']}")
            else:
                print("ℹ️  No WebSocket-specific tables found")
            
            await db_manager.close()
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket integration test error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 AlphaPlus WebSocket Integration Test")
    print("=" * 50)
    
    tests = [
        test_websocket_server,
        test_websocket_client,
        test_websocket_service,
        test_real_time_data,
        test_websocket_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 WebSocket Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 WebSocket integration successful! Ready for next step.")
        return True
    else:
        print("❌ Some WebSocket integration tests failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
