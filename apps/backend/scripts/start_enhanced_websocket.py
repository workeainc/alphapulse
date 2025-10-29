#!/usr/bin/env python3
"""
Enhanced WebSocket Quick Start Script
AlphaPulse Trading System - Performance Optimized

This script provides a quick way to start the enhanced WebSocket system
with all necessary components and configurations.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.app.main_enhanced_websocket import app
from src.core.websocket_enhanced import EnhancedBinanceWebSocketClient, EnhancedWebSocketManager
from src.app.services.enhanced_websocket_service import EnhancedWebSocketService
from src.database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_websocket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedWebSocketStarter:
    """Quick start class for the enhanced WebSocket system"""
    
    def __init__(self):
        self.services = {}
        self.is_running = False
        
    async def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("🔍 Checking dependencies...")
        
        try:
            # Check Python version
            if sys.version_info < (3, 11):
                logger.error("❌ Python 3.11+ is required")
                return False
            
            # Check required packages
            required_packages = [
                'fastapi', 'uvicorn', 'websockets', 'orjson', 
                'aioredis', 'sqlalchemy', 'asyncpg'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
                logger.info("💡 Run: pip install -r requirements_enhanced_websocket.txt")
                return False
            
            logger.info("✅ All dependencies are available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking dependencies: {e}")
            return False
    
    async def check_services(self) -> bool:
        """Check if required services (Redis, TimescaleDB) are running"""
        logger.info("🔍 Checking external services...")
        
        try:
            # Check Redis
            import aioredis
            try:
                redis = await aioredis.from_url("redis://localhost:6379")
                await redis.ping()
                await redis.close()
                logger.info("✅ Redis is running")
            except Exception as e:
                logger.error(f"❌ Redis is not available: {e}")
                logger.info("💡 Start Redis: docker run -d --name redis -p 6379:6379 redis:7-alpine")
                return False
            
            # Check TimescaleDB
            try:
                db = TimescaleDBConnection()
                await db.initialize()
                await db.close()
                logger.info("✅ TimescaleDB is running")
            except Exception as e:
                logger.error(f"❌ TimescaleDB is not available: {e}")
                logger.info("💡 Start TimescaleDB: docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password -e POSTGRES_DB=alphapulse timescale/timescaledb:latest-pg14")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking services: {e}")
            return False
    
    async def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directories...")
        
        directories = [
            'logs',
            'data',
            'cache',
            'reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("✅ Directories created")
    
    async def setup_environment(self):
        """Setup environment variables"""
        logger.info("⚙️ Setting up environment...")
        
        # Default environment variables
        env_vars = {
            'DATABASE_URL': 'postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse',
            'REDIS_URL': 'redis://localhost:6379',
            'WEBSOCKET_UPDATE_INTERVAL': '3.0',
            'WEBSOCKET_MAX_CLIENTS': '1000',
            'BINANCE_MAX_CONNECTIONS': '5',
            'BATCH_SIZE': '50',
            'BATCH_TIMEOUT': '0.1',
            'MAX_QUEUE_SIZE': '10000'
        }
        
        # Set environment variables if not already set
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
        
        logger.info("✅ Environment configured")
    
    async def start_services(self):
        """Start all enhanced WebSocket services"""
        logger.info("🚀 Starting enhanced WebSocket services...")
        
        try:
            # Start the FastAPI application
            import uvicorn
            
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                reload=False
            )
            
            server = uvicorn.Server(config)
            
            # Start the server in the background
            self.services['fastapi'] = asyncio.create_task(server.serve())
            
            logger.info("✅ Enhanced WebSocket system started successfully")
            logger.info("🌐 Dashboard available at: http://localhost:8000")
            logger.info("📊 API documentation at: http://localhost:8000/docs")
            logger.info("🔌 WebSocket endpoint at: ws://localhost:8000/ws")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting services: {e}")
            return False
    
    async def run_health_check(self):
        """Run a health check on the system"""
        logger.info("🏥 Running health check...")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Check main endpoint
                response = await client.get("http://localhost:8000/api/status")
                if response.status_code == 200:
                    logger.info("✅ API is responding")
                else:
                    logger.error(f"❌ API health check failed: {response.status_code}")
                    return False
                
                # Check WebSocket endpoint
                try:
                    import websockets
                    async with websockets.connect("ws://localhost:8000/ws") as ws:
                        await ws.send('{"type": "ping"}')
                        response = await ws.recv()
                        logger.info("✅ WebSocket is responding")
                except Exception as e:
                    logger.error(f"❌ WebSocket health check failed: {e}")
                    return False
            
            logger.info("✅ All health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def show_status(self):
        """Show current system status"""
        logger.info("📊 System Status:")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/api/status")
                if response.status_code == 200:
                    status = response.json()
                    
                    print("\n" + "="*50)
                    print("🚀 ENHANCED WEBSOCKET SYSTEM STATUS")
                    print("="*50)
                    
                    # System info
                    if 'system' in status:
                        sys_info = status['system']
                        print(f"📋 Version: {sys_info.get('version', 'N/A')}")
                        print(f"⏱️  Uptime: {sys_info.get('uptime', 'N/A')} seconds")
                    
                    # WebSocket service info
                    if 'websocket_service' in status:
                        ws_info = status['websocket_service']
                        print(f"🔌 Active Clients: {ws_info.get('active_clients', 0)}")
                        print(f"📨 Messages Sent: {ws_info.get('messages_sent', 0)}")
                        print(f"⚡ Avg Latency: {ws_info.get('avg_latency_ms', 0):.2f}ms")
                    
                    # Binance WebSocket info
                    if 'binance_websocket' in status:
                        binance_info = status['binance_websocket']
                        print(f"📡 Active Connections: {binance_info.get('active_connections', 0)}")
                        print(f"📥 Messages Received: {binance_info.get('total_messages_received', 0)}")
                        print(f"❌ Total Errors: {binance_info.get('total_errors', 0)}")
                    
                    # Database info
                    if 'database' in status:
                        db_info = status['database']
                        print(f"🗄️  Database Status: {db_info.get('status', 'N/A')}")
                    
                    print("="*50)
                    
                else:
                    logger.error("❌ Could not retrieve system status")
                    
        except Exception as e:
            logger.error(f"❌ Error showing status: {e}")
    
    async def start(self):
        """Main start method"""
        logger.info("🚀 Starting Enhanced WebSocket System...")
        
        try:
            # Check dependencies
            if not await self.check_dependencies():
                return False
            
            # Check services
            if not await self.check_services():
                return False
            
            # Create directories
            await self.create_directories()
            
            # Setup environment
            await self.setup_environment()
            
            # Start services
            if not await self.start_services():
                return False
            
            # Wait a moment for services to start
            await asyncio.sleep(2)
            
            # Run health check
            if not await self.run_health_check():
                logger.warning("⚠️ Health check failed, but system may still be functional")
            
            # Show status
            await self.show_status()
            
            self.is_running = True
            
            # Keep the system running
            logger.info("🔄 System is running. Press Ctrl+C to stop.")
            
            try:
                while self.is_running:
                    await asyncio.sleep(30)
                    await self.show_status()
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down...")
                await self.stop()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            return False
    
    async def stop(self):
        """Stop all services"""
        logger.info("🛑 Stopping enhanced WebSocket system...")
        
        self.is_running = False
        
        # Cancel all running tasks
        for service_name, task in self.services.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("✅ Enhanced WebSocket system stopped")

async def main():
    """Main entry point"""
    starter = EnhancedWebSocketStarter()
    
    try:
        await starter.start()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Enhanced WebSocket Quick Start")
    print("=" * 40)
    print("This script will start the enhanced WebSocket system")
    print("with all necessary components and configurations.")
    print()
    
    # Run the main function
    asyncio.run(main())
