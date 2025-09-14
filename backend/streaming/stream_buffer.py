"""
Redis Streams Buffer for AlphaPulse
Real-time data ingestion with TimescaleDB integration and enterprise features
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

# Import existing database connection
try:
    from ..database.connection import TimescaleDBConnection
    from ..core.config import settings
except ImportError:
    # Fallback for standalone testing
    from database.connection import TimescaleDBConnection
    from core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class StreamMessage:
    """Stream message structure"""
    id: str
    timestamp: datetime
    symbol: str
    data_type: str  # 'tick', 'candle', 'signal', 'outcome'
    data: Dict[str, Any]
    source: str
    partition: int = 0
    priority: int = 0  # Higher number = higher priority

@dataclass
class StreamMetrics:
    """Stream performance metrics"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_mps: float = 0.0
    buffer_size: int = 0
    last_message_time: Optional[datetime] = None
    error_count: int = 0
    reconnection_count: int = 0

class StreamBuffer:
    """
    Redis Streams buffer for real-time data ingestion
    
    Features:
    - Redis Streams for high-throughput data ingestion
    - Automatic partitioning by symbol
    - Connection pooling and failover
    - Backpressure handling
    - Real-time metrics tracking
    - TimescaleDB integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Redis configuration
        self.redis_host = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_db = self.config.get('redis_db', 0)
        self.redis_password = self.config.get('redis_password', None)
        self.redis_ssl = self.config.get('redis_ssl', False)
        
        # Stream configuration
        self.stream_prefix = self.config.get('stream_prefix', 'alphapulse')
        self.max_stream_length = self.config.get('max_stream_length', 10000)
        self.batch_size = self.config.get('batch_size', 100)
        self.flush_interval = self.config.get('flush_interval', 1.0)  # seconds
        
        # Performance settings
        self.connection_pool_size = self.config.get('connection_pool_size', 10)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        
        # State management
        self.redis_client = None
        self.is_connected = False
        self.is_running = False
        self.metrics = StreamMetrics()
        self.message_buffer = deque(maxlen=10000)
        self.processing_callbacks = {}
        self.health_check_task = None
        
        # TimescaleDB integration
        self.timescaledb = None
        self.db_batch_size = self.config.get('db_batch_size', 1000)
        self.db_flush_interval = self.config.get('db_flush_interval', 5.0)
        self.db_batch = []
        self.db_flush_task = None
        
        logger.info("StreamBuffer initialized")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current stream metrics"""
        return {
            'messages_received': self.metrics.messages_received,
            'messages_processed': self.metrics.messages_processed,
            'messages_failed': self.metrics.messages_failed,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_mps': self.metrics.throughput_mps,
            'buffer_size': self.metrics.buffer_size,
            'last_message_time': self.metrics.last_message_time.isoformat() if self.metrics.last_message_time else None,
            'error_count': self.metrics.error_count,
            'reconnection_count': self.metrics.reconnection_count,
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'message_buffer_size': len(self.message_buffer)
        }
    
    async def initialize(self):
        """Initialize Redis connection and TimescaleDB integration"""
        try:
            # Initialize Redis connection
            await self._connect_redis()
            
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("✅ StreamBuffer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StreamBuffer: {e}")
            raise
    
    async def _connect_redis(self):
        """Establish Redis connection with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    ssl=self.redis_ssl,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=self.health_check_interval
                )
                
                # Test connection
                await self.redis_client.ping()
                self.is_connected = True
                logger.info(f"✅ Redis connected to {self.redis_host}:{self.redis_port}")
                return
                
            except (RedisError, ConnectionError) as e:
                self.metrics.reconnection_count += 1
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.warning(f"⚠️ Failed to connect to Redis after {self.retry_attempts} attempts")
                    logger.info("🔄 Continuing without Redis connection...")
                    self.redis_client = None
                    self.redis_connected = False
                    return
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        try:
            self.timescaledb = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 10,
                'max_overflow': 20
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            # Don't fail completely - can work without DB
            self.timescaledb = None
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start health check
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start DB flush task
        if self.timescaledb:
            self.db_flush_task = asyncio.create_task(self._db_flush_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while self.is_running:
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await self._reconnect_redis()
    
    async def _reconnect_redis(self):
        """Reconnect to Redis"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            await self._connect_redis()
            logger.info("✅ Redis reconnected")
        except Exception as e:
            logger.error(f"❌ Redis reconnection failed: {e}")
    
    async def _db_flush_loop(self):
        """Periodic database flush loop"""
        while self.is_running:
            try:
                if self.db_batch:
                    await self._flush_db_batch()
                await asyncio.sleep(self.db_flush_interval)
            except Exception as e:
                logger.error(f"DB flush error: {e}")
    
    async def _flush_db_batch(self):
        """Flush buffered data to TimescaleDB"""
        if not self.db_batch or not self.timescaledb:
            return
        
        try:
            batch = self.db_batch.copy()
            self.db_batch.clear()
            
            # Insert batch into TimescaleDB
            async with self.timescaledb.async_session() as session:
                for message in batch:
                    # Convert to appropriate table format
                    if message.data_type == 'tick':
                        await self._insert_tick_data(session, message)
                    elif message.data_type == 'candle':
                        await self._insert_candle_data(session, message)
                    elif message.data_type == 'signal':
                        await self._insert_signal_data(session, message)
                
                await session.commit()
            
            logger.debug(f"Flushed {len(batch)} messages to TimescaleDB")
            
        except Exception as e:
            logger.error(f"DB flush failed: {e}")
            # Re-add messages to batch for retry
            self.db_batch.extend(batch)
    
    async def _insert_tick_data(self, session, message: StreamMessage):
        """Insert tick data into TimescaleDB"""
        # Implementation depends on your tick table schema
        pass
    
    async def _insert_candle_data(self, session, message: StreamMessage):
        """Insert candle data into TimescaleDB"""
        # Implementation depends on your candle table schema
        pass
    
    async def _insert_signal_data(self, session, message: StreamMessage):
        """Insert signal data into TimescaleDB"""
        # Implementation depends on your signal table schema
        pass
    
    async def publish_message(self, message: StreamMessage) -> str:
        """
        Publish message to Redis Streams
        
        Args:
            message: StreamMessage to publish
            
        Returns:
            Message ID
        """
        if not self.is_connected:
            raise Exception("StreamBuffer not connected")
        
        try:
            # Generate stream key
            stream_key = f"{self.stream_prefix}:{message.symbol}:{message.data_type}"
            
            # Prepare message data
            message_data = {
                'id': message.id,
                'timestamp': message.timestamp.isoformat(),
                'symbol': message.symbol,
                'data_type': message.data_type,
                'data': json.dumps(message.data),
                'source': message.source,
                'partition': str(message.partition),
                'priority': str(message.priority)
            }
            
            # Add to Redis Stream
            message_id = await self.redis_client.xadd(
                stream_key,
                message_data,
                maxlen=self.max_stream_length,
                approximate=True
            )
            
            # Update metrics
            self.metrics.messages_received += 1
            self.metrics.last_message_time = datetime.now(timezone.utc)
            self.metrics.buffer_size = len(self.message_buffer)
            
            # Add to DB batch if TimescaleDB is available
            if self.timescaledb:
                self.db_batch.append(message)
                if len(self.db_batch) >= self.db_batch_size:
                    await self._flush_db_batch()
            
            # Trigger processing callbacks
            await self._trigger_callbacks(message)
            
            return message_id
            
        except Exception as e:
            self.metrics.messages_failed += 1
            self.metrics.error_count += 1
            logger.error(f"Failed to publish message: {e}")
            raise
    
    async def _trigger_callbacks(self, message: StreamMessage):
        """Trigger registered processing callbacks"""
        callbacks = self.processing_callbacks.get(message.data_type, [])
        for callback in callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Callback error for {message.data_type}: {e}")
    
    async def subscribe_to_stream(self, symbol: str, data_type: str, callback: Callable):
        """
        Subscribe to a specific stream
        
        Args:
            symbol: Symbol to subscribe to
            data_type: Type of data to subscribe to
            callback: Async callback function to process messages
        """
        stream_key = f"{self.stream_prefix}:{symbol}:{data_type}"
        
        if data_type not in self.processing_callbacks:
            self.processing_callbacks[data_type] = []
        
        self.processing_callbacks[data_type].append(callback)
        logger.info(f"Subscribed to {stream_key}")
    
    async def get_stream_info(self, symbol: str, data_type: str) -> Dict[str, Any]:
        """Get stream information"""
        stream_key = f"{self.stream_prefix}:{symbol}:{data_type}"
        
        try:
            info = await self.redis_client.xinfo_stream(stream_key)
            return {
                'stream_key': stream_key,
                'length': info.get('length', 0),
                'radix_tree_keys': info.get('radix-tree-keys', 0),
                'radix_tree_nodes': info.get('radix-tree-nodes', 0),
                'groups': info.get('groups', 0),
                'last_generated_id': info.get('last-generated-id', '0-0'),
                'first_entry': info.get('first-entry', []),
                'last_entry': info.get('last-entry', [])
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'is_connected': self.is_connected,
            'is_running': self.is_running,
            'messages_received': self.metrics.messages_received,
            'messages_processed': self.metrics.messages_processed,
            'messages_failed': self.metrics.messages_failed,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_mps': self.metrics.throughput_mps,
            'buffer_size': self.metrics.buffer_size,
            'last_message_time': self.metrics.last_message_time.isoformat() if self.metrics.last_message_time else None,
            'error_count': self.metrics.error_count,
            'reconnection_count': self.metrics.reconnection_count,
            'db_batch_size': len(self.db_batch) if self.db_batch else 0
        }
    
    async def get_messages(self, symbol: str = None, data_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages from the buffer"""
        try:
            if not self.is_connected or not self.redis_client:
                return []
            
            messages = []
            
            # Get from Redis streams
            if symbol and data_type:
                stream_name = f"{self.stream_prefix}:{symbol}:{data_type}"
                try:
                    # Read recent messages from stream
                    stream_messages = await self.redis_client.xrevrange(
                        stream_name, count=limit, max='+', min='-'
                    )
                    
                    for msg_id, fields in stream_messages:
                        message_data = {
                            'id': msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                            'symbol': symbol,
                            'data_type': data_type,
                            'data': {k.decode() if isinstance(k, bytes) else k: 
                                    v.decode() if isinstance(v, bytes) else v 
                                    for k, v in fields.items()},
                            'timestamp': datetime.now(timezone.utc)
                        }
                        messages.append(message_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to read from stream {stream_name}: {e}")
            
            # Also check local buffer
            if len(messages) < limit:
                buffer_messages = list(self.message_buffer)[-limit:]
                for msg in buffer_messages:
                    if isinstance(msg, dict):
                        messages.append(msg)
            
            return messages[:limit]
            
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the stream buffer"""
        self.is_running = False
        
        # Cancel background tasks
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.db_flush_task:
            self.db_flush_task.cancel()
        
        # Flush remaining data
        if self.db_batch and self.timescaledb:
            await self._flush_db_batch()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 StreamBuffer shutdown complete")

# Global instance
stream_buffer = StreamBuffer()
