"""
Real-Time Data Manager for AlphaPulse
Handles real-time data feeds, websocket connections, and data streaming for production
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
import importlib.util
spec = importlib.util.spec_from_file_location('production', 'config/production.py')
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)
production_config = production_module.production_config

logger = logging.getLogger(__name__)

@dataclass
class DataStream:
    """Data stream configuration"""
    stream_id: str
    stream_type: str  # 'market_data', 'signals', 'alerts', 'performance'
    source: str
    enabled: bool = True
    last_update: Optional[datetime] = None
    subscribers: List[WebSocket] = field(default_factory=list)
    data_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    reconnect_attempts: int = 0

@dataclass
class ConnectionMetrics:
    """Connection metrics"""
    total_connections: int = 0
    active_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None

class RealTimeDataManager:
    """
    Real-time data manager for handling multiple data streams and websocket connections
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.is_running = False
        
        # Configuration
        self.config = production_config.REAL_TIME_CONFIG
        
        # Data streams
        self.data_streams: Dict[str, DataStream] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Metrics
        self.metrics = ConnectionMetrics()
        self.performance_metrics = defaultdict(int)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Initialize data streams
        self._initialize_data_streams()
        
        logger.info("Real-Time Data Manager initialized")
    
    def _initialize_data_streams(self):
        """Initialize data streams based on configuration"""
        streams_config = self.config["data_streams"]
        
        for stream_type, enabled in streams_config.items():
            stream_id = f"{stream_type}_stream"
            self.data_streams[stream_id] = DataStream(
                stream_id=stream_id,
                stream_type=stream_type,
                source=self._get_stream_source(stream_type),
                enabled=enabled
            )
        
        logger.info(f"Initialized {len(self.data_streams)} data streams")
    
    def _get_stream_source(self, stream_type: str) -> str:
        """Get stream source based on type"""
        sources = {
            "market_data": "market_data_collector",
            "signals": "sde_framework",
            "alerts": "monitoring_system",
            "performance": "performance_tracker"
        }
        return sources.get(stream_type, "unknown")
    
    async def start(self):
        """Start the real-time data manager"""
        if self.is_running:
            logger.warning("⚠️ Real-time data manager already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting real-time data manager...")
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._connection_cleanup_loop()),
            asyncio.create_task(self._data_stream_monitoring_loop())
        ]
        
        # Start data stream processors
        for stream in self.data_streams.values():
            if stream.enabled:
                task = asyncio.create_task(self._process_data_stream(stream))
                self.background_tasks.append(task)
        
        logger.info("✅ Real-time data manager started")
    
    async def stop(self):
        """Stop the real-time data manager"""
        if not self.is_running:
            logger.warning("⚠️ Real-time data manager not running")
            return
        
        self.is_running = False
        logger.info("🛑 Stopping real-time data manager...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close all websocket connections
        for websocket in self.websocket_connections.values():
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"❌ Error closing websocket: {e}")
        
        logger.info("✅ Real-time data manager stopped")
    
    async def handle_websocket_connection(self, websocket: WebSocket, client_id: str):
        """Handle new websocket connection"""
        try:
            await websocket.accept()
            self.websocket_connections[client_id] = websocket
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            self.metrics.last_activity = datetime.now()
            
            logger.info(f"✅ New websocket connection: {client_id}")
            
            # Send welcome message
            welcome_message = {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "available_streams": list(self.data_streams.keys())
            }
            await websocket.send_text(json.dumps(welcome_message))
            
            # Handle client messages
            async for message in websocket.iter_text():
                await self._handle_client_message(client_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"📤 Websocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"❌ Error handling websocket connection {client_id}: {e}")
            self.metrics.errors += 1
        finally:
            # Clean up connection
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]
            self.metrics.active_connections -= 1
    
    async def _handle_client_message(self, client_id: str, message: str):
        """Handle client message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                await self._handle_subscribe(client_id, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(client_id, data)
            elif message_type == "ping":
                await self._handle_ping(client_id)
            else:
                logger.warning(f"⚠️ Unknown message type: {message_type}")
            
            self.metrics.messages_received += 1
            
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON message from {client_id}")
        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")
    
    async def _handle_subscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        stream_id = data.get("stream_id")
        if stream_id in self.data_streams:
            stream = self.data_streams[stream_id]
            websocket = self.websocket_connections.get(client_id)
            
            if websocket and websocket not in stream.subscribers:
                stream.subscribers.append(websocket)
                logger.info(f"✅ Client {client_id} subscribed to {stream_id}")
                
                # Send subscription confirmation
                response = {
                    "type": "subscription_confirmed",
                    "stream_id": stream_id,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(response))
    
    async def _handle_unsubscribe(self, client_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        stream_id = data.get("stream_id")
        if stream_id in self.data_streams:
            stream = self.data_streams[stream_id]
            websocket = self.websocket_connections.get(client_id)
            
            if websocket in stream.subscribers:
                stream.subscribers.remove(websocket)
                logger.info(f"📤 Client {client_id} unsubscribed from {stream_id}")
    
    async def _handle_ping(self, client_id: str):
        """Handle ping message"""
        websocket = self.websocket_connections.get(client_id)
        if websocket:
            response = {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(response))
    
    async def broadcast_data(self, stream_id: str, data: Dict[str, Any]):
        """Broadcast data to all subscribers of a stream"""
        if stream_id not in self.data_streams:
            logger.warning(f"⚠️ Unknown stream: {stream_id}")
            return
        
        stream = self.data_streams[stream_id]
        if not stream.enabled:
            return
        
        # Add metadata to data
        message = {
            "stream_id": stream_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Store in buffer
        stream.data_buffer.append(message)
        stream.last_update = datetime.now()
        
        # Broadcast to subscribers
        disconnected_subscribers = []
        for websocket in stream.subscribers:
            try:
                await websocket.send_text(json.dumps(message))
                self.metrics.messages_sent += 1
            except Exception as e:
                logger.error(f"❌ Error sending to subscriber: {e}")
                disconnected_subscribers.append(websocket)
        
        # Remove disconnected subscribers
        for websocket in disconnected_subscribers:
            if websocket in stream.subscribers:
                stream.subscribers.remove(websocket)
        
        # Store in Redis for persistence
        await self._store_data_in_redis(stream_id, message)
    
    async def _store_data_in_redis(self, stream_id: str, data: Dict[str, Any]):
        """Store data in Redis for persistence"""
        try:
            key = f"realtime:{stream_id}:latest"
            await self.redis_client.setex(
                key,
                self.config["cache_ttl"],
                json.dumps(data)
            )
        except Exception as e:
            logger.error(f"❌ Error storing data in Redis: {e}")
    
    async def _process_data_stream(self, stream: DataStream):
        """Process data stream"""
        while self.is_running and stream.enabled:
            try:
                # Get data from source
                data = await self._get_stream_data(stream)
                if data:
                    await self.broadcast_data(stream.stream_id, data)
                
                # Wait before next update
                await asyncio.sleep(self._get_stream_interval(stream.stream_type))
                
            except Exception as e:
                logger.error(f"❌ Error processing stream {stream.stream_id}: {e}")
                stream.error_count += 1
                await asyncio.sleep(5)  # Wait before retry
    
    async def _get_stream_data(self, stream: DataStream) -> Optional[Dict[str, Any]]:
        """Get data from stream source"""
        try:
            if stream.stream_type == "market_data":
                return await self._get_market_data()
            elif stream.stream_type == "signals":
                return await self._get_signal_data()
            elif stream.stream_type == "alerts":
                return await self._get_alert_data()
            elif stream.stream_type == "performance":
                return await self._get_performance_data()
            else:
                return None
        except Exception as e:
            logger.error(f"❌ Error getting stream data: {e}")
            return None
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get market data"""
        # This would integrate with your existing market data collection
        return {
            "type": "market_data",
            "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_signal_data(self) -> Dict[str, Any]:
        """Get signal data"""
        # This would integrate with your SDE framework
        return {
            "type": "signals",
            "active_signals": 0,
            "signal_accuracy": 0.85,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_alert_data(self) -> Dict[str, Any]:
        """Get alert data"""
        # This would integrate with your monitoring system
        return {
            "type": "alerts",
            "active_alerts": 0,
            "critical_alerts": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        return {
            "type": "performance",
            "cpu_percent": 45.2,
            "memory_percent": 62.1,
            "active_connections": self.metrics.active_connections,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_stream_interval(self, stream_type: str) -> int:
        """Get update interval for stream type"""
        intervals = {
            "market_data": 1,      # 1 second
            "signals": 5,          # 5 seconds
            "alerts": 10,          # 10 seconds
            "performance": 30      # 30 seconds
        }
        return intervals.get(stream_type, 10)
    
    async def _heartbeat_loop(self):
        """Send heartbeat to all connections"""
        while self.is_running:
            try:
                heartbeat_message = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "active_connections": self.metrics.active_connections
                }
                
                for websocket in self.websocket_connections.values():
                    try:
                        await websocket.send_text(json.dumps(heartbeat_message))
                    except Exception:
                        pass  # Connection will be cleaned up
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"❌ Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self):
        """Collect and store metrics"""
        while self.is_running:
            try:
                metrics_data = {
                    "timestamp": datetime.now().isoformat(),
                    "active_connections": self.metrics.active_connections,
                    "total_connections": self.metrics.total_connections,
                    "messages_sent": self.metrics.messages_sent,
                    "messages_received": self.metrics.messages_received,
                    "errors": self.metrics.errors
                }
                
                # Store metrics in database
                await self._store_metrics(metrics_data)
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"❌ Error in metrics collection: {e}")
                await asyncio.sleep(60)
    
    async def _store_metrics(self, metrics_data: Dict[str, Any]):
        """Store metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO real_time_metrics (
                        timestamp, active_connections, total_connections,
                        messages_sent, messages_received, errors
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                datetime.fromisoformat(metrics_data["timestamp"]),
                metrics_data["active_connections"],
                metrics_data["total_connections"],
                metrics_data["messages_sent"],
                metrics_data["messages_received"],
                metrics_data["errors"]
                )
        except Exception as e:
            logger.error(f"❌ Error storing metrics: {e}")
    
    async def _connection_cleanup_loop(self):
        """Clean up inactive connections"""
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout = timedelta(seconds=self.config["connection_timeout"])
                
                disconnected_clients = []
                for client_id, websocket in self.websocket_connections.items():
                    if (self.metrics.last_activity and 
                        current_time - self.metrics.last_activity > timeout):
                        disconnected_clients.append(client_id)
                
                for client_id in disconnected_clients:
                    try:
                        await self.websocket_connections[client_id].close()
                        del self.websocket_connections[client_id]
                        self.metrics.active_connections -= 1
                        logger.info(f"📤 Cleaned up inactive connection: {client_id}")
                    except Exception as e:
                        logger.error(f"❌ Error cleaning up connection: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in connection cleanup: {e}")
                await asyncio.sleep(30)
    
    async def _data_stream_monitoring_loop(self):
        """Monitor data stream health"""
        while self.is_running:
            try:
                for stream in self.data_streams.values():
                    if stream.enabled:
                        # Check if stream is healthy
                        if stream.error_count > 10:
                            logger.warning(f"⚠️ Stream {stream.stream_id} has high error count: {stream.error_count}")
                        
                        # Reset error count periodically
                        if stream.error_count > 0:
                            stream.error_count = max(0, stream.error_count - 1)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in data stream monitoring: {e}")
                await asyncio.sleep(300)
    
    def get_status(self) -> Dict[str, Any]:
        """Get real-time data manager status"""
        return {
            "is_running": self.is_running,
            "active_connections": self.metrics.active_connections,
            "total_connections": self.metrics.total_connections,
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "errors": self.metrics.errors,
            "data_streams": {
                stream_id: {
                    "enabled": stream.enabled,
                    "subscribers": len(stream.subscribers),
                    "last_update": stream.last_update.isoformat() if stream.last_update else None,
                    "error_count": stream.error_count
                }
                for stream_id, stream in self.data_streams.items()
            }
        }
