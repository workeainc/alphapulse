"""
WebSocket Client for AlphaPlus
Handles WebSocket connections for real-time data
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import websockets
import aiohttp

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """WebSocket client for Binance data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the WebSocket client"""
        self.config = config or {}
        self.is_running = False
        self.websocket = None
        self.callbacks = {}
        self.subscriptions = set()
        
        # Default configuration
        self.base_url = self.config.get('base_url', 'wss://stream.binance.com:9443/ws')
        self.reconnect_delay = self.config.get('reconnect_delay', 5)
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 10)
        
        logger.info("🚀 Binance WebSocket Client initialized")
    
    async def start(self):
        """Start the WebSocket client"""
        if self.is_running:
            logger.warning("WebSocket client is already running")
            return
            
        logger.info("🚀 Starting WebSocket client...")
        self.is_running = True
        
        # Start connection
        await self._connect()
        
        logger.info("✅ WebSocket client started successfully")
    
    async def stop(self):
        """Stop the WebSocket client"""
        if not self.is_running:
            logger.warning("WebSocket client is not running")
            return
            
        logger.info("🛑 Stopping WebSocket client...")
        self.is_running = False
        
        if self.websocket:
            await self.websocket.close()
        
        logger.info("✅ WebSocket client stopped successfully")
    
    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            # Create connection URL
            streams = list(self.subscriptions)
            if streams:
                url = f"{self.base_url}/{'@'.join(streams)}"
            else:
                url = f"{self.base_url}/btcusdt@kline_1m"  # Default stream
            
            logger.info(f"🔗 Connecting to {url}")
            
            self.websocket = await websockets.connect(url)
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            logger.info("✅ WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket: {e}")
            if self.is_running:
                await asyncio.sleep(self.reconnect_delay)
                await self._connect()
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed")
            if self.is_running:
                await asyncio.sleep(self.reconnect_delay)
                await self._connect()
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
            if self.is_running:
                await asyncio.sleep(self.reconnect_delay)
                await self._connect()
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming message"""
        try:
            # Extract stream name
            stream = data.get('stream', '')
            
            # Call appropriate callback
            if stream in self.callbacks:
                callback = self.callbacks[stream]
                await callback(data)
            else:
                logger.debug(f"📨 Received message for stream: {stream}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process message: {e}")
    
    async def subscribe(self, stream: str, callback: Optional[Callable] = None):
        """Subscribe to a data stream"""
        try:
            self.subscriptions.add(stream)
            
            if callback:
                self.callbacks[stream] = callback
            
            logger.info(f"✅ Subscribed to stream: {stream}")
            
            # Reconnect if already running to include new subscription
            if self.is_running and self.websocket:
                await self.websocket.close()
                await self._connect()
                
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {stream}: {e}")
    
    async def unsubscribe(self, stream: str):
        """Unsubscribe from a data stream"""
        try:
            self.subscriptions.discard(stream)
            self.callbacks.pop(stream, None)
            
            logger.info(f"✅ Unsubscribed from stream: {stream}")
            
        except Exception as e:
            logger.error(f"❌ Failed to unsubscribe from {stream}: {e}")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send a message through the WebSocket"""
        try:
            if self.websocket:
                await self.websocket.send(json.dumps(message))
                logger.debug(f"📤 Sent message: {message}")
            else:
                logger.warning("⚠️ WebSocket not connected")
                
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the WebSocket client"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'connected': self.websocket is not None,
            'subscriptions': list(self.subscriptions),
            'callbacks': list(self.callbacks.keys())
        }

class WebSocketManager:
    """Manager for multiple WebSocket connections"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the WebSocket manager"""
        self.config = config or {}
        self.clients = {}
        self.is_running = False
        
        logger.info("🚀 WebSocket Manager initialized")
    
    async def start(self):
        """Start the WebSocket manager"""
        if self.is_running:
            logger.warning("WebSocket manager is already running")
            return
            
        logger.info("🚀 Starting WebSocket manager...")
        self.is_running = True
        logger.info("✅ WebSocket manager started successfully")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        if not self.is_running:
            logger.warning("WebSocket manager is not running")
            return
            
        logger.info("🛑 Stopping WebSocket manager...")
        self.is_running = False
        
        # Stop all clients
        for client in self.clients.values():
            await client.stop()
        
        logger.info("✅ WebSocket manager stopped successfully")
    
    async def create_client(self, name: str, config: Optional[Dict[str, Any]] = None) -> BinanceWebSocketClient:
        """Create a new WebSocket client"""
        try:
            client = BinanceWebSocketClient(config)
            self.clients[name] = client
            
            if self.is_running:
                await client.start()
            
            logger.info(f"✅ Created WebSocket client: {name}")
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to create WebSocket client {name}: {e}")
            return None
    
    async def get_client(self, name: str) -> Optional[BinanceWebSocketClient]:
        """Get a WebSocket client by name"""
        return self.clients.get(name)
    
    async def remove_client(self, name: str):
        """Remove a WebSocket client"""
        try:
            if name in self.clients:
                client = self.clients[name]
                await client.stop()
                del self.clients[name]
                logger.info(f"✅ Removed WebSocket client: {name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to remove WebSocket client {name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of all WebSocket clients"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'clients': {
                name: client.get_status() 
                for name, client in self.clients.items()
            }
        }
