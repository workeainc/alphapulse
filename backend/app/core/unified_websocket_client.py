#!/usr/bin/env python3
"""
Unified WebSocket Client for AlphaPlus
Consolidates all WebSocket functionality into a single, configurable client
Supports multiple performance modes: basic, enhanced, ultra-low-latency
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import websockets
import aiohttp
from enum import Enum

# Optional imports for enhanced features
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceMode(Enum):
    """Performance modes for the WebSocket client"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ULTRA_LOW_LATENCY = "ultra_low_latency"

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket client"""
    symbols: List[str] = None
    timeframes: List[str] = None
    base_url: str = "wss://stream.binance.com:9443/ws"
    performance_mode: PerformanceMode = PerformanceMode.ENHANCED
    max_queue_size: int = 10000
    batch_size: int = 50
    batch_timeout: float = 0.1
    reconnect_delay: float = 1.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30
    redis_url: Optional[str] = None
    enable_shared_memory: bool = False
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT"]
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h"]

@dataclass
class WebSocketMetrics:
    """Performance metrics for WebSocket client"""
    messages_received: int = 0
    messages_processed: int = 0
    processing_latency_ms: float = 0.0
    connection_uptime: float = 0.0
    last_message_time: Optional[datetime] = None
    reconnect_attempts: int = 0
    errors_count: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')

class UnifiedWebSocketClient:
    """
    Unified WebSocket client that consolidates all functionality
    Supports multiple performance modes and configurations
    """
    
    def __init__(self, config: WebSocketConfig):
        """Initialize the unified WebSocket client"""
        self.config = config
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_attempts = 0
        self.connection_start_time = None
        self.last_heartbeat = time.time()
        
        # Performance tracking
        self.metrics = WebSocketMetrics()
        self.processing_times = []
        
        # Message processing
        self.message_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.callbacks = {}
        self.subscriptions = set()
        
        # Enhanced features
        self.redis_client = None
        self.batch_processor = None
        
        # Stream management
        self.active_streams = {}
        self.stream_callbacks = {}
        
        logger.info(f"🚀 Unified WebSocket Client initialized in {config.performance_mode.value} mode")
    
    async def initialize(self):
        """Initialize the WebSocket client based on performance mode"""
        try:
            if self.config.performance_mode == PerformanceMode.ULTRA_LOW_LATENCY:
                await self._initialize_ultra_low_latency()
            elif self.config.performance_mode == PerformanceMode.ENHANCED:
                await self._initialize_enhanced()
            else:
                await self._initialize_basic()
            
            logger.info(f"✅ WebSocket client initialized in {self.config.performance_mode.value} mode")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket client: {e}")
            raise
    
    async def _initialize_basic(self):
        """Initialize basic mode - minimal features"""
        logger.info("Initializing basic WebSocket mode")
        # Basic mode requires no additional initialization
    
    async def _initialize_enhanced(self):
        """Initialize enhanced mode - with performance optimizations"""
        logger.info("Initializing enhanced WebSocket mode")
        
        # Initialize batch processor
        self.batch_processor = asyncio.create_task(self._batch_processor())
        
        # Initialize Redis if available and configured
        if self.config.redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = await aioredis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Redis connection established for enhanced mode")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available: {e}")
    
    async def _initialize_ultra_low_latency(self):
        """Initialize ultra-low latency mode - maximum performance"""
        logger.info("Initializing ultra-low latency WebSocket mode")
        
        # Initialize enhanced features
        await self._initialize_enhanced()
        
        # Initialize shared memory if enabled
        if self.config.enable_shared_memory and self.redis_client:
            await self._initialize_shared_memory()
        
        # Set up high-performance processing
        self.batch_size = 1  # Process messages immediately
        self.batch_timeout = 0.001  # 1ms timeout
    
    async def _initialize_shared_memory(self):
        """Initialize shared memory buffers for ultra-low latency"""
        try:
            buffer_configs = {
                'candlestick_data': {'maxlen': 1000, 'approximate': True},
                'pattern_detection': {'maxlen': 500, 'approximate': True},
                'signal_generation': {'maxlen': 200, 'approximate': True},
                'market_analysis': {'maxlen': 1000, 'approximate': True}
            }
            
            for buffer_name, config in buffer_configs.items():
                await self.redis_client.xgroup_create(
                    f"stream:{buffer_name}",
                    "alphapulse",
                    mkstream=True
                )
            
            logger.info("✅ Shared memory buffers initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared memory: {e}")
    
    async def start(self):
        """Start the WebSocket client"""
        if self.is_running:
            logger.warning("WebSocket client is already running")
            return
        
        logger.info("🚀 Starting unified WebSocket client...")
        self.is_running = True
        self.connection_start_time = time.time()
        
        # Start connection
        await self._connect()
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._metrics_collector())
        
        logger.info("✅ Unified WebSocket client started successfully")
    
    async def stop(self):
        """Stop the WebSocket client"""
        if not self.is_running:
            logger.warning("WebSocket client is not running")
            return
        
        logger.info("🛑 Stopping unified WebSocket client...")
        self.is_running = False
        
        # Stop background tasks
        if self.batch_processor:
            self.batch_processor.cancel()
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Unified WebSocket client stopped successfully")
    
    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            # Build stream names
            stream_names = []
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    stream_name = f"{symbol.lower()}@kline_{timeframe}"
                    stream_names.append(stream_name)
            
            # Create combined stream URL
            combined_streams = "/".join(stream_names)
            ws_url = f"{self.config.base_url}/{combined_streams}"
            
            logger.info(f"🔗 Connecting to {ws_url}")
            
            # Connect with appropriate timeout
            if self.config.performance_mode == PerformanceMode.ULTRA_LOW_LATENCY:
                # Use shorter timeout for ultra-low latency
                self.websocket = await asyncio.wait_for(
                    websockets.connect(ws_url),
                    timeout=5.0
                )
            else:
                self.websocket = await websockets.connect(ws_url)
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.active_streams = set(stream_names)
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            logger.info(f"✅ Connected to WebSocket with {len(stream_names)} streams")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket: {e}")
            await self._handle_connection_error()
    
    async def _handle_connection_error(self):
        """Handle connection errors with exponential backoff"""
        if not self.is_running:
            return
        
        self.is_connected = False
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts <= self.config.max_reconnect_attempts:
            delay = self.config.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            logger.info(f"🔄 Reconnecting in {delay}s (attempt {self.reconnect_attempts})")
            await asyncio.sleep(delay)
            await self._connect()
        else:
            logger.error("❌ Max reconnection attempts reached")
            self.is_running = False
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break
                
                start_time = time.time()
                
                try:
                    # Parse message based on performance mode
                    if ORJSON_AVAILABLE and self.config.performance_mode != PerformanceMode.BASIC:
                        data = orjson.loads(message)
                    else:
                        data = json.loads(message)
                    
                    # Update metrics
                    self.metrics.messages_received += 1
                    self.metrics.last_message_time = datetime.now(timezone.utc)
                    
                    # Process message based on mode
                    if self.config.performance_mode == PerformanceMode.ULTRA_LOW_LATENCY:
                        await self._process_message_ultra_low_latency(data)
                    elif self.config.performance_mode == PerformanceMode.ENHANCED:
                        await self.message_queue.put(data)
                    else:
                        await self._process_message_basic(data)
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    self._update_latency_metrics(latency_ms)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse message: {e}")
                    self.metrics.errors_count += 1
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    self.metrics.errors_count += 1
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed")
            if self.is_running:
                await self._handle_connection_error()
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
            if self.is_running:
                await self._handle_connection_error()
    
    async def _process_message_basic(self, data: Dict[str, Any]):
        """Process message in basic mode"""
        self.metrics.messages_processed += 1
        
        # Execute callbacks
        for callback in self.callbacks.values():
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
    
    async def _process_message_ultra_low_latency(self, data: Dict[str, Any]):
        """Process message in ultra-low latency mode"""
        self.metrics.messages_processed += 1
        
        # Store in shared memory if available
        if self.redis_client and self.config.enable_shared_memory:
            try:
                await self.redis_client.xadd(
                    "stream:candlestick_data",
                    {"data": json.dumps(data), "timestamp": str(time.time())}
                )
            except Exception as e:
                logger.error(f"❌ Shared memory error: {e}")
        
        # Execute callbacks immediately
        for callback in self.callbacks.values():
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
    
    async def _batch_processor(self):
        """Process messages in batches for enhanced mode"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Wait for message or timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=self.config.batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or 
                    (batch and current_time - last_batch_time >= self.config.batch_timeout)):
                    
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                        last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"❌ Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of messages"""
        self.metrics.messages_processed += len(batch)
        
        # Execute callbacks for each message
        for data in batch:
            for callback in self.callbacks.values():
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Callback error: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor connection health"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.is_connected:
                    current_time = time.time()
                    if current_time - self.last_heartbeat > self.config.heartbeat_interval * 2:
                        logger.warning("⚠️ No heartbeat detected, reconnecting...")
                        await self._handle_connection_error()
                    else:
                        self.last_heartbeat = current_time
                        
            except Exception as e:
                logger.error(f"❌ Heartbeat monitor error: {e}")
    
    async def _metrics_collector(self):
        """Collect and update performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                if self.connection_start_time:
                    self.metrics.connection_uptime = time.time() - self.connection_start_time
                
                # Log performance metrics
                if self.metrics.messages_received > 0:
                    logger.info(f"📊 WebSocket Metrics: "
                              f"Received={self.metrics.messages_received}, "
                              f"Processed={self.metrics.messages_processed}, "
                              f"Avg Latency={self.metrics.avg_latency_ms:.2f}ms, "
                              f"Errors={self.metrics.errors_count}")
                
            except Exception as e:
                logger.error(f"❌ Metrics collector error: {e}")
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update latency metrics"""
        try:
            # Ensure latency is a valid number
            if not (isinstance(latency_ms, (int, float)) and latency_ms >= 0):
                latency_ms = 0.0
            
            self.metrics.processing_latency_ms = latency_ms
            
            # Calculate average latency safely
            if self.metrics.messages_processed > 0:
                self.metrics.avg_latency_ms = (
                    (self.metrics.avg_latency_ms * (self.metrics.messages_processed - 1) + latency_ms) /
                    self.metrics.messages_processed
                )
                # Ensure the result is finite
                if not (isinstance(self.metrics.avg_latency_ms, (int, float)) and 
                       self.metrics.avg_latency_ms >= 0 and 
                       self.metrics.avg_latency_ms < float('inf')):
                    self.metrics.avg_latency_ms = 0.0
            
            # Update min/max safely
            if self.metrics.messages_processed == 1:
                self.metrics.min_latency_ms = latency_ms
                self.metrics.max_latency_ms = latency_ms
            else:
                self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, latency_ms)
                self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, latency_ms)
                
        except Exception as e:
            logger.error(f"❌ Error updating latency metrics: {e}")
            # Reset to safe values
            self.metrics.avg_latency_ms = 0.0
            self.metrics.processing_latency_ms = 0.0
    
    def add_callback(self, name: str, callback: Callable):
        """Add a callback for message processing"""
        self.callbacks[name] = callback
        logger.info(f"✅ Added callback: {name}")
    
    def remove_callback(self, name: str):
        """Remove a callback"""
        if name in self.callbacks:
            del self.callbacks[name]
            logger.info(f"✅ Removed callback: {name}")
    
    def subscribe(self, stream: str):
        """Subscribe to a stream"""
        self.subscriptions.add(stream)
        logger.info(f"✅ Subscribed to: {stream}")
    
    def unsubscribe(self, stream: str):
        """Unsubscribe from a stream"""
        self.subscriptions.discard(stream)
        logger.info(f"✅ Unsubscribed from: {stream}")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send a message through the WebSocket"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(message))
                logger.debug(f"📤 Sent message: {message}")
            except Exception as e:
                logger.error(f"❌ Failed to send message: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the WebSocket client"""
        try:
            # Convert metrics to dict safely
            metrics_dict = asdict(self.metrics)
            
            # Ensure all float values are JSON serializable
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    if not (value >= 0 and value < float('inf')):
                        metrics_dict[key] = 0.0
            
            return {
                "is_running": self.is_running,
                "is_connected": self.is_connected,
                "performance_mode": self.config.performance_mode.value,
                "active_streams": len(self.active_streams),
                "callbacks": len(self.callbacks),
                "reconnect_attempts": self.reconnect_attempts,
                "metrics": metrics_dict
            }
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {
                "is_running": self.is_running,
                "is_connected": self.is_connected,
                "performance_mode": "basic",
                "active_streams": 0,
                "callbacks": 0,
                "reconnect_attempts": 0,
                "metrics": {
                    "messages_received": 0,
                    "messages_processed": 0,
                    "errors_count": 0,
                    "avg_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "processing_latency_ms": 0.0
                }
            }
    
    def get_metrics(self) -> WebSocketMetrics:
        """Get performance metrics"""
        return self.metrics

class UnifiedWebSocketManager:
    """Manager for multiple WebSocket clients"""
    
    def __init__(self, max_connections: int = 3):
        self.max_connections = max_connections
        self.clients = {}
        self.is_running = False
        
        logger.info(f"🚀 Unified WebSocket Manager initialized (max connections: {max_connections})")
    
    async def start(self):
        """Start the WebSocket manager"""
        if self.is_running:
            logger.warning("WebSocket manager is already running")
            return
        
        logger.info("🚀 Starting unified WebSocket manager...")
        self.is_running = True
        logger.info("✅ Unified WebSocket manager started")
    
    async def stop(self):
        """Stop the WebSocket manager"""
        if not self.is_running:
            logger.warning("WebSocket manager is not running")
            return
        
        logger.info("🛑 Stopping unified WebSocket manager...")
        self.is_running = False
        
        # Stop all clients
        for client in self.clients.values():
            await client.stop()
        
        self.clients.clear()
        logger.info("✅ Unified WebSocket manager stopped")
    
    async def create_client(self, name: str, config: WebSocketConfig) -> UnifiedWebSocketClient:
        """Create a new WebSocket client"""
        if len(self.clients) >= self.max_connections:
            raise ValueError(f"Maximum connections ({self.max_connections}) reached")
        
        if name in self.clients:
            raise ValueError(f"Client '{name}' already exists")
        
        client = UnifiedWebSocketClient(config)
        await client.initialize()
        
        if self.is_running:
            await client.start()
        
        self.clients[name] = client
        logger.info(f"✅ Created WebSocket client: {name}")
        
        return client
    
    def get_client(self, name: str) -> Optional[UnifiedWebSocketClient]:
        """Get a WebSocket client by name"""
        return self.clients.get(name)
    
    async def remove_client(self, name: str):
        """Remove a WebSocket client"""
        if name in self.clients:
            client = self.clients[name]
            await client.stop()
            del self.clients[name]
            logger.info(f"✅ Removed WebSocket client: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of all clients"""
        return {
            "is_running": self.is_running,
            "total_clients": len(self.clients),
            "max_connections": self.max_connections,
            "clients": {
                name: client.get_status() for name, client in self.clients.items()
            }
        }
