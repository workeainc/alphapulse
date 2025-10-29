#!/usr/bin/env python3
"""
Test Notification System
Test the real-time notification system for AlphaPulse
"""

import asyncio
import json
import logging
import asyncpg
from datetime import datetime
import importlib.util

# Import production config
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_notification_system():
    """Test the notification system"""
    
    # Connect to database
    db_pool = await asyncpg.create_pool(
        host=production_config.DATABASE_CONFIG['host'],
        port=production_config.DATABASE_CONFIG['port'],
        database=production_config.DATABASE_CONFIG['database'],
        user=production_config.DATABASE_CONFIG['username'],
        password=production_config.DATABASE_CONFIG['password']
    )
    
    try:
        logger.info("Testing notification system...")
        
        # Import dashboard
        from src.monitoring.production_deployment_dashboard import ProductionDeploymentDashboard
        
        # Create dashboard instance
        dashboard = ProductionDeploymentDashboard(db_pool)
        
        # Test notification methods
        logger.info("Testing signal notification...")
        await dashboard.send_signal_notification("BTC/USDT", "Long", 91.5, 25200.0)
        
        logger.info("Testing TP notification...")
        await dashboard.send_tp_notification("BTC/USDT", 1, 25600.0)
        
        logger.info("Testing SL notification...")
        await dashboard.send_sl_notification("BTC/USDT", 24800.0)
        
        logger.info("Testing system alert...")
        await dashboard.send_system_alert("System performance is optimal", "low")
        
        logger.info("Testing market update...")
        await dashboard.send_market_update("Bullish", 0.15)
        
        # Check notification queue
        notifications = dashboard.get_notification_queue()
        logger.info(f"Notification queue has {len(notifications)} notifications")
        
        # Display last 3 notifications
        for i, notification in enumerate(notifications[-3:], 1):
            logger.info(f"Notification {i}: {notification['type']} - {notification['data'].get('message', 'No message')}")
        
        logger.info("✅ Notification system test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Notification system test failed: {e}")
        raise
    finally:
        await db_pool.close()

async def test_websocket_connection():
    """Test WebSocket connection"""
    import websockets
    
    try:
        logger.info("Testing WebSocket connection...")
        
        # Connect to WebSocket
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            logger.info("✅ WebSocket connected successfully!")
            
            # Send a test message
            await websocket.send(json.dumps({
                "type": "test",
                "data": {"message": "Test notification"},
                "timestamp": datetime.now().isoformat()
            }))
            
            # Wait for response
            response = await websocket.recv()
            logger.info(f"Received response: {response}")
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_notification_system())
    # asyncio.run(test_websocket_connection())  # Uncomment when dashboard is running
