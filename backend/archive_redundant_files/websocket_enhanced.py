#!/usr/bin/env python3
"""
Enhanced WebSocket Implementation for AlphaPulse
Real-time market data streaming with TimescaleDB integration and performance optimizations
"""

import asyncio
import json
import logging
import time
import json
import websockets
import aiohttp
from typing import Dict, List, Optional, Callable, AsyncGenerator, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Local imports
from database.connection import TimescaleDBConnection
from database.models import Signal

logger = logging.getLogger(__name__)

@dataclass
class WebSocketMetrics:
    """WebSocket performance metrics"""
    messages_received: int = 0
    messages_processed: int = 0
    processing_latency_ms: float = 0.0
    connection_uptime: float = 0.0
    last_message_time: Optional[datetime] = None
    reconnect_attempts: int = 0
    errors_count: int = 0

@dataclass
class SignalData:
    """Standardized signal data structure"""
    symbol: str
    timeframe: str
    timestamp: datetime
    direction: str
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pattern_type: Optional[str] = None
    indicators: Optional[Dict] = None
    metadata: Optional[Dict] = None

class EnhancedBinanceWebSocketClient:
    """
    Enhanced Binance WebSocket client with performance optimizations
    - Zero-copy JSON parsing with orjson
    - Micro-batching for message processing
    - Backpressure handling with async queues
    - TimescaleDB integration for signal storage
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 timeframes: List[str] = None,
                 base_url: str = "wss://stream.binance.com:9443/ws",
                 max_queue_size: int = 10000,
                 batch_size: int = 50,
                 batch_timeout: float = 0.1):
        """
        Initialize enhanced WebSocket client
        
        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            base_url: Binance WebSocket base URL
            max_queue_size: Maximum queue size for backpressure handling
            batch_size: Number of messages to process in batch
            batch_timeout: Maximum time to wait for batch completion
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h"]
        self.base_url = base_url
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # WebSocket connection
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1
        self.connection_start_time = None
        
        # Performance tracking
        self.metrics = WebSocketMetrics()
        self.processing_times = []
        
        # Message processing
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_tasks = []
        
        # Callbacks and handlers
        self.signal_callbacks = []
        self.error_callbacks = []
        self.connection_callbacks = []
        
        # TimescaleDB integration
        self.db_connection = None
        self.signal_buffer = []
        self.buffer_size = 100
        self.flush_interval = 5.0  # seconds
        
        # Redis for broadcasting
        self.redis_client = None
        
        logger.info(f"Enhanced Binance WebSocket client initialized for {len(self.symbols)} symbols")
    
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize TimescaleDB connection (without table creation)
            self.db_connection = TimescaleDBConnection()
            await self.db_connection.initialize(create_tables=False)
            
            # Initialize Redis for broadcasting
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            
            logger.info("✅ Enhanced WebSocket client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced WebSocket client: {e}")
            raise
    
    async def connect(self) -> bool:
        """Establish WebSocket connection with enhanced error handling"""
        try:
            # Build stream names for all symbols and timeframes
            stream_names = []
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    stream_name = f"{symbol.lower()}@kline_{timeframe}"
                    stream_names.append(stream_name)
            
            # Create combined stream URL
            combined_streams = "/".join(stream_names)
            ws_url = f"{self.base_url}/{combined_streams}"
            
            logger.info(f"🔌 Connecting to enhanced Binance WebSocket: {ws_url}")
            
            # Connect with enhanced options
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,  # 1MB max message size
                compression=None  # Disable compression for performance
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.reconnect_delay = 1
            self.connection_start_time = time.time()
            self.metrics.connection_uptime = 0
            
            # Start background processing tasks
            await self._start_background_tasks()
            
            # Notify connection callbacks
            await self._notify_connection_callbacks("connected")
            
            logger.info(f"✅ Enhanced WebSocket connected with {len(stream_names)} streams")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to enhanced WebSocket: {e}")
            await self._handle_connection_error()
            return False
    
    async def disconnect(self):
        """Close WebSocket connection gracefully"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.websocket = None
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Flush remaining signals to database
            await self._flush_signal_buffer()
            
            logger.info("🔌 Enhanced WebSocket disconnected")
    
    async def listen(self) -> AsyncGenerator[Dict, None]:
        """Enhanced listening loop with micro-batching"""
        if not self.is_connected:
            await self.connect()
        
        try:
            async for message in self.websocket:
                if not self.is_connected:
                    break
                
                # Add message to processing queue
                try:
                    self.message_queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning("⚠️ Message queue full, dropping message")
                    self.metrics.errors_count += 1
                
                # Update metrics
                self.metrics.messages_received += 1
                self.metrics.last_message_time = datetime.now(timezone.utc)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ Enhanced WebSocket connection closed, attempting to reconnect...")
            await self._handle_connection_error()
        except Exception as e:
            logger.error(f"❌ Error in enhanced WebSocket listener: {e}")
            await self._handle_connection_error()
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        self.processing_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._signal_flusher()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._health_monitor())
        ]
        
        logger.info("✅ Background processing tasks started")
    
    async def _stop_background_tasks(self):
        """Stop background processing tasks"""
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        self.processing_tasks = []
        
        logger.info("✅ Background processing tasks stopped")
    
    async def _message_processor(self):
        """Process individual messages with zero-copy parsing"""
        while self.is_connected:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Parse with orjson for performance
                start_time = time.time()
                data = json.loads(message)
                parse_time = (time.time() - start_time) * 1000
                
                # Add to batch queue
                try:
                    self.batch_queue.put_nowait((data, parse_time))
                except asyncio.QueueFull:
                    logger.warning("⚠️ Batch queue full, dropping message")
                    self.metrics.errors_count += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                self.metrics.errors_count += 1
    
    async def _batch_processor(self):
        """Process messages in batches for efficiency"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_connected:
            try:
                # Collect messages for batch
                while len(batch) < self.batch_size:
                    try:
                        message_data, parse_time = await asyncio.wait_for(
                            self.batch_queue.get(), 
                            timeout=self.batch_timeout
                        )
                        batch.append((message_data, parse_time))
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Process batch
                    start_time = time.time()
                    await self._process_batch(batch)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Update metrics
                    self.metrics.messages_processed += len(batch)
                    self.metrics.processing_latency_ms = processing_time
                    self.processing_times.append(processing_time)
                    
                    # Keep only last 100 processing times
                    if len(self.processing_times) > 100:
                        self.processing_times = self.processing_times[-100:]
                    
                    batch = []
                    last_batch_time = time.time()
                
            except Exception as e:
                logger.error(f"❌ Error processing batch: {e}")
                self.metrics.errors_count += 1
                batch = []
    
    async def _process_batch(self, batch: List[tuple]):
        """Process a batch of messages"""
        for message_data, parse_time in batch:
            try:
                # Extract candlestick data
                if 'data' in message_data and 'k' in message_data['data']:
                    kline = message_data['data']['k']
                    
                    # Create signal data
                    signal_data = SignalData(
                        symbol=message_data['data']['s'],
                        timeframe=kline['i'],
                        timestamp=datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc),
                        direction='buy' if float(kline['c']) > float(kline['o']) else 'sell',
                        confidence=0.5,  # Placeholder - will be calculated by ML model
                        entry_price=float(kline['c']),
                        indicators={
                            'open': float(kline['o']),
                            'high': float(kline['h']),
                            'low': float(kline['l']),
                            'close': float(kline['c']),
                            'volume': float(kline['v']),
                            'is_complete': kline['x']
                        },
                        metadata={
                            'parse_time_ms': parse_time,
                            'source': 'binance_websocket',
                            'stream_type': 'kline'
                        }
                    )
                    
                    # Add to signal buffer
                    self.signal_buffer.append(signal_data)
                    
                    # Notify signal callbacks
                    await self._notify_signal_callbacks(signal_data)
                    
                    # Broadcast via Redis
                    await self._broadcast_signal(signal_data)
                    
            except Exception as e:
                logger.error(f"❌ Error processing message in batch: {e}")
                self.metrics.errors_count += 1
    
    async def _signal_flusher(self):
        """Periodically flush signals to TimescaleDB"""
        while self.is_connected:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_signal_buffer()
                
            except Exception as e:
                logger.error(f"❌ Error in signal flusher: {e}")
    
    async def _flush_signal_buffer(self):
        """Flush signal buffer to TimescaleDB"""
        if not self.signal_buffer:
            return
        
        try:
            # Convert signals to database format
            signals_to_insert = []
            for signal in self.signal_buffer:
                signals_to_insert.append({
                    'signal_id': f"ALPHA_{int(signal.timestamp.timestamp() * 1000000)}",
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'timeframe': signal.timeframe,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'tp1': signal.take_profit,
                    'pattern_type': signal.pattern_type,
                    'indicators': signal.indicators,
                    'metadata': signal.metadata,
                    'outcome': 'pending'
                })
            
            # Insert into TimescaleDB
            if self.db_connection:
                await self.db_connection.bulk_insert_signals(signals_to_insert)
                logger.info(f"✅ Flushed {len(signals_to_insert)} signals to TimescaleDB")
            
            # Clear buffer
            self.signal_buffer = []
            
        except Exception as e:
            logger.error(f"❌ Error flushing signals to database: {e}")
    
    async def _broadcast_signal(self, signal: SignalData):
        """Broadcast signal via Redis pub/sub"""
        try:
            if self.redis_client:
                signal_json = json.dumps(asdict(signal))
                await self.redis_client.publish('alphapulse_signals', signal_json)
                
        except Exception as e:
            logger.error(f"❌ Error broadcasting signal: {e}")
    
    async def _metrics_updater(self):
        """Update connection metrics"""
        while self.is_connected:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if self.connection_start_time:
                    self.metrics.connection_uptime = time.time() - self.connection_start_time
                
                # Log metrics
                logger.info(f"📊 WebSocket Metrics: "
                          f"Received={self.metrics.messages_received}, "
                          f"Processed={self.metrics.messages_processed}, "
                          f"Latency={self.metrics.processing_latency_ms:.2f}ms, "
                          f"Errors={self.metrics.errors_count}")
                
            except Exception as e:
                logger.error(f"❌ Error updating metrics: {e}")
    
    async def _health_monitor(self):
        """Monitor WebSocket health and trigger reconnection if needed"""
        while self.is_connected:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check if we're receiving messages
                if (self.metrics.last_message_time and 
                    (datetime.now(timezone.utc) - self.metrics.last_message_time).seconds > 120):
                    logger.warning("⚠️ No messages received for 2 minutes, reconnecting...")
                    await self._handle_connection_error()
                
            except Exception as e:
                logger.error(f"❌ Error in health monitor: {e}")
    
    async def _handle_connection_error(self):
        """Handle connection errors with exponential backoff"""
        self.is_connected = False
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)
            
            logger.info(f"🔄 Attempting to reconnect in {delay} seconds (attempt {self.reconnect_attempts})")
            
            await asyncio.sleep(delay)
            await self.connect()
        else:
            logger.error("❌ Max reconnection attempts reached")
            await self._notify_error_callbacks("Max reconnection attempts reached")
    
    # Callback management
    def add_signal_callback(self, callback: Callable):
        """Add signal callback"""
        self.signal_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add connection callback"""
        self.connection_callbacks.append(callback)
    
    async def _notify_signal_callbacks(self, signal: SignalData):
        """Notify signal callbacks"""
        for callback in self.signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"❌ Error in signal callback: {e}")
    
    async def _notify_error_callbacks(self, error: str):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"❌ Error in error callback: {e}")
    
    async def _notify_connection_callbacks(self, status: str):
        """Notify connection callbacks"""
        for callback in self.connection_callbacks:
            try:
                await callback(status)
            except Exception as e:
                logger.error(f"❌ Error in connection callback: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times) 
            if self.processing_times else 0
        )
        
        return {
            'messages_received': self.metrics.messages_received,
            'messages_processed': self.metrics.messages_processed,
            'processing_latency_ms': self.metrics.processing_latency_ms,
            'avg_processing_time_ms': avg_processing_time,
            'connection_uptime': self.metrics.connection_uptime,
            'reconnect_attempts': self.metrics.reconnect_attempts,
            'errors_count': self.metrics.errors_count,
            'queue_size': self.message_queue.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'signal_buffer_size': len(self.signal_buffer),
            'is_connected': self.is_connected
        }

class EnhancedWebSocketManager:
    """
    Manager for multiple enhanced WebSocket connections
    Handles connection pooling, load balancing, and failover
    """
    
    def __init__(self, max_connections: int = 5):
        """
        Initialize enhanced WebSocket manager
        
        Args:
            max_connections: Maximum number of concurrent connections
        """
        self.max_connections = max_connections
        self.connections = []
        self.connection_pool = asyncio.Queue(maxsize=max_connections)
        self.is_running = False
        self.health_check_interval = 30
        
        logger.info(f"Enhanced WebSocket Manager initialized with {max_connections} max connections")
    
    async def start(self):
        """Start the enhanced WebSocket manager"""
        self.is_running = True
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        logger.info("✅ Enhanced WebSocket Manager started")
    
    async def stop(self):
        """Stop the enhanced WebSocket manager"""
        self.is_running = False
        
        # Close all connections
        for connection in self.connections:
            await connection.disconnect()
        
        logger.info("✅ Enhanced WebSocket Manager stopped")
    
    async def get_connection(self) -> EnhancedBinanceWebSocketClient:
        """Get an available WebSocket connection"""
        try:
            # Try to get from pool
            connection = await asyncio.wait_for(self.connection_pool.get(), timeout=5.0)
            return connection
        except asyncio.TimeoutError:
            # Create new connection if pool is empty
            if len(self.connections) < self.max_connections:
                connection = EnhancedBinanceWebSocketClient()
                await connection.initialize()
                await connection.connect()
                self.connections.append(connection)
                return connection
            else:
                raise Exception("No available WebSocket connections")
    
    async def return_connection(self, connection: EnhancedBinanceWebSocketClient):
        """Return connection to pool"""
        try:
            self.connection_pool.put_nowait(connection)
        except asyncio.QueueFull:
            # Pool is full, close connection
            await connection.disconnect()
            self.connections.remove(connection)
    
    async def _health_monitor(self):
        """Monitor health of all connections"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for connection in self.connections:
                    metrics = connection.get_metrics()
                    
                    # Check for unhealthy connections
                    if (metrics['errors_count'] > 100 or 
                        metrics['processing_latency_ms'] > 1000):
                        logger.warning(f"⚠️ Unhealthy connection detected: {metrics}")
                        
                        # Replace connection
                        await connection.disconnect()
                        self.connections.remove(connection)
                        
                        new_connection = EnhancedBinanceWebSocketClient()
                        await new_connection.initialize()
                        await new_connection.connect()
                        self.connections.append(new_connection)
                
            except Exception as e:
                logger.error(f"❌ Error in health monitor: {e}")
    
    def get_manager_metrics(self) -> Dict[str, Any]:
        """Get manager metrics"""
        total_metrics = {
            'total_messages_received': 0,
            'total_messages_processed': 0,
            'total_errors': 0,
            'active_connections': len(self.connections),
            'pool_size': self.connection_pool.qsize()
        }
        
        for connection in self.connections:
            metrics = connection.get_metrics()
            total_metrics['total_messages_received'] += metrics['messages_received']
            total_metrics['total_messages_processed'] += metrics['messages_processed']
            total_metrics['total_errors'] += metrics['errors_count']
        
        return total_metrics
