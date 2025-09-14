#!/usr/bin/env python3
"""
Test Unified WebSocket Implementation
Validates the consolidated WebSocket functionality and performance
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.unified_websocket_client import (
    UnifiedWebSocketClient, 
    UnifiedWebSocketManager, 
    WebSocketConfig, 
    PerformanceMode
)
from app.core.config import get_config

async def test_websocket_client_basic():
    """Test basic WebSocket client functionality"""
    print("🧪 Testing Basic WebSocket Client...")
    
    try:
        # Create basic configuration
        config = WebSocketConfig(
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.BASIC
        )
        
        # Create client
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        
        # Test callback functionality
        received_messages = []
        
        async def test_callback(data):
            received_messages.append(data)
            print(f"📨 Received message: {data.get('s', 'unknown')}")
        
        client.add_callback("test", test_callback)
        
        # Start client
        await client.start()
        
        # Wait for some messages
        await asyncio.sleep(10)
        
        # Check status
        status = client.get_status()
        print(f"✅ Basic client status: {status}")
        
        # Stop client
        await client.stop()
        
        print(f"✅ Basic WebSocket test passed. Received {len(received_messages)} messages")
        return True
        
    except Exception as e:
        print(f"❌ Basic WebSocket test failed: {e}")
        return False

async def test_websocket_client_enhanced():
    """Test enhanced WebSocket client functionality"""
    print("🧪 Testing Enhanced WebSocket Client...")
    
    try:
        # Create enhanced configuration
        config = WebSocketConfig(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["1m", "5m"],
            performance_mode=PerformanceMode.ENHANCED,
            batch_size=10,
            batch_timeout=0.5
        )
        
        # Create client
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        
        # Test callback functionality
        received_messages = []
        
        async def test_callback(data):
            received_messages.append(data)
            print(f"📨 Enhanced received: {data.get('s', 'unknown')}")
        
        client.add_callback("test", test_callback)
        
        # Start client
        await client.start()
        
        # Wait for some messages
        await asyncio.sleep(15)
        
        # Check metrics
        metrics = client.get_metrics()
        print(f"✅ Enhanced metrics: {metrics}")
        
        # Stop client
        await client.stop()
        
        print(f"✅ Enhanced WebSocket test passed. Received {len(received_messages)} messages")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced WebSocket test failed: {e}")
        return False

async def test_websocket_manager():
    """Test WebSocket manager functionality"""
    print("🧪 Testing WebSocket Manager...")
    
    try:
        # Create manager
        manager = UnifiedWebSocketManager(max_connections=2)
        await manager.start()
        
        # Create multiple clients
        config1 = WebSocketConfig(
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.BASIC
        )
        
        config2 = WebSocketConfig(
            symbols=["ETHUSDT"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.ENHANCED
        )
        
        # Create clients
        client1 = await manager.create_client("client1", config1)
        client2 = await manager.create_client("client2", config2)
        
        # Wait for connections
        await asyncio.sleep(10)
        
        # Check manager status
        status = manager.get_status()
        print(f"✅ Manager status: {status}")
        
        # Remove one client
        await manager.remove_client("client1")
        
        # Check status again
        status = manager.get_status()
        print(f"✅ Manager status after removal: {status}")
        
        # Stop manager
        await manager.stop()
        
        print("✅ WebSocket Manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket Manager test failed: {e}")
        return False

async def test_performance_modes():
    """Test different performance modes"""
    print("🧪 Testing Performance Modes...")
    
    results = {}
    
    # Test basic mode
    try:
        config = WebSocketConfig(
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.BASIC
        )
        
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        await client.start()
        
        await asyncio.sleep(5)
        metrics = client.get_metrics()
        results["basic"] = metrics
        
        await client.stop()
        print("✅ Basic mode test passed")
        
    except Exception as e:
        print(f"❌ Basic mode test failed: {e}")
        results["basic"] = None
    
    # Test enhanced mode
    try:
        config = WebSocketConfig(
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.ENHANCED
        )
        
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        await client.start()
        
        await asyncio.sleep(5)
        metrics = client.get_metrics()
        results["enhanced"] = metrics
        
        await client.stop()
        print("✅ Enhanced mode test passed")
        
    except Exception as e:
        print(f"❌ Enhanced mode test failed: {e}")
        results["enhanced"] = None
    
    # Compare performance
    print("\n📊 Performance Comparison:")
    for mode, metrics in results.items():
        if metrics:
            print(f"{mode.upper()}: {metrics.avg_latency_ms:.2f}ms avg latency, {metrics.messages_processed} messages")
    
    return len([r for r in results.values() if r is not None]) > 0

async def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing Configuration...")
    
    try:
        # Load configuration
        config = get_config()
        
        print(f"✅ Database config: {config.database.host}:{config.database.port}")
        print(f"✅ Redis config: {config.redis.enabled} - {config.redis.url}")
        print(f"✅ WebSocket config: {config.websocket.symbols} - {config.websocket.performance_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and reconnection"""
    print("🧪 Testing Error Handling...")
    
    try:
        # Create client with invalid configuration
        config = WebSocketConfig(
            symbols=["INVALID"],
            timeframes=["1m"],
            performance_mode=PerformanceMode.BASIC
        )
        
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        await client.start()
        
        # Wait for reconnection attempts
        await asyncio.sleep(10)
        
        # Check reconnection attempts
        status = client.get_status()
        print(f"✅ Reconnection attempts: {status['reconnect_attempts']}")
        
        await client.stop()
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Unified WebSocket Tests...")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Basic WebSocket Client", test_websocket_client_basic),
        ("Enhanced WebSocket Client", test_websocket_client_enhanced),
        ("WebSocket Manager", test_websocket_manager),
        ("Performance Modes", test_performance_modes),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified WebSocket implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
