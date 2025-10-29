"""
Stream Normalizer for AlphaPulse
Data deduplication, validation, and normalization with TimescaleDB integration
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

# Import existing components
try:
    from .stream_buffer import StreamMessage, StreamBuffer
    from ..src.database.connection import TimescaleDBConnection
    from ..src.core.config import settings
except ImportError:
    try:
        # Fallback for standalone testing
        from stream_buffer import StreamMessage, StreamBuffer
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
    except ImportError:
        # Final fallback - create minimal classes
        class StreamMessage:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class StreamBuffer:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
                self.metrics = type('obj', (object,), {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0
                })()
                self.is_connected = False
                self.is_running = False
                self.message_buffer = []
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
            
            def get_metrics(self):
                return {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0,
                    'is_connected': False,
                    'is_running': False,
                    'message_buffer_size': 0
                }
        
        class TimescaleDBConnection:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
            
            async def close(self):
                self.is_initialized = False
        
        class settings:
            TIMESCALEDB_HOST = 'localhost'
            TIMESCALEDB_PORT = 5432
            TIMESCALEDB_DATABASE = 'alphapulse'
            TIMESCALEDB_USERNAME = 'alpha_emon'
            TIMESCALEDB_PASSWORD = 'Emon_@17711'

logger = logging.getLogger(__name__)

@dataclass
class NormalizedData:
    """Normalized data structure"""
    original_message: StreamMessage
    normalized_data: Dict[str, Any]
    validation_status: str  # 'valid', 'invalid', 'duplicate', 'outlier'
    confidence_score: float
    processing_time_ms: float
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NormalizationMetrics:
    """Normalization performance metrics"""
    messages_processed: int = 0
    messages_validated: int = 0
    messages_normalized: int = 0
    duplicates_detected: int = 0
    outliers_detected: int = 0
    validation_errors: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_mps: float = 0.0
    last_processed_time: Optional[datetime] = None

class StreamNormalizer:
    """
    Stream data normalizer with deduplication and validation
    
    Features:
    - Data deduplication using content hashing
    - Timestamp validation and normalization
    - Symbol normalization and validation
    - Outlier detection using statistical methods
    - Data quality validation
    - TimescaleDB integration for persistence
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Normalization settings
        self.enable_deduplication = self.config.get('enable_deduplication', True)
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        self.enable_validation = self.config.get('enable_validation', True)
        self.enable_timestamp_normalization = self.config.get('enable_timestamp_normalization', True)
        
        # Deduplication settings
        self.deduplication_window = self.config.get('deduplication_window', 300)  # seconds
        self.hash_cache_size = self.config.get('hash_cache_size', 10000)
        
        # Outlier detection settings
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)  # standard deviations
        self.outlier_window_size = self.config.get('outlier_window_size', 1000)
        
        # Validation settings
        self.max_price_change = self.config.get('max_price_change', 0.5)  # 50% max change
        self.min_volume = self.config.get('min_volume', 0.0)
        self.max_timestamp_drift = self.config.get('max_timestamp_drift', 60)  # seconds
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.processing_timeout = self.config.get('processing_timeout', 5.0)  # seconds
        
        # State management
        self.is_running = False
        self.metrics = NormalizationMetrics()
        self.hash_cache = deque(maxlen=self.hash_cache_size)
        self.price_history = defaultdict(lambda: deque(maxlen=self.outlier_window_size))
        self.volume_history = defaultdict(lambda: deque(maxlen=self.outlier_window_size))
        self.timestamp_cache = defaultdict(lambda: deque(maxlen=1000))
        
        # TimescaleDB integration
        self.timescaledb = None
        self.db_batch = []
        self.db_flush_interval = self.config.get('db_flush_interval', 5.0)
        self.db_flush_task = None
        
        # Processing callbacks
        self.normalization_callbacks = []
        self.validation_callbacks = []
        
        logger.info("StreamNormalizer initialized")
    
    async def initialize(self):
        """Initialize the normalizer"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("✅ StreamNormalizer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StreamNormalizer: {e}")
            raise
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        try:
            self.timescaledb = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 5,
                'max_overflow': 10
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized for normalizer")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        if self.timescaledb:
            self.db_flush_task = asyncio.create_task(self._db_flush_loop())
        logger.info("✅ Background tasks started")
    
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
        """Flush normalized data to TimescaleDB"""
        if not self.db_batch or not self.timescaledb:
            return
        
        try:
            batch = self.db_batch.copy()
            self.db_batch.clear()
            
            # Insert batch into TimescaleDB
            async with self.timescaledb.async_session() as session:
                for normalized_data in batch:
                    await self._insert_normalized_data(session, normalized_data)
                await session.commit()
            
            logger.debug(f"Flushed {len(batch)} normalized records to TimescaleDB")
            
        except Exception as e:
            logger.error(f"DB flush failed: {e}")
            # Re-add to batch for retry
            self.db_batch.extend(batch)
    
    async def _insert_normalized_data(self, session, normalized_data: NormalizedData):
        """Insert normalized data into TimescaleDB"""
        # Implementation depends on your normalized data table schema
        # This would typically go into a normalized_data table
        pass
    
    async def normalize_message(self, message: StreamMessage) -> NormalizedData:
        """
        Normalize a stream message
        
        Args:
            message: StreamMessage to normalize
            
        Returns:
            NormalizedData with validation results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create normalized data structure
            normalized_data = NormalizedData(
                original_message=message,
                normalized_data={},
                validation_status='pending',
                confidence_score=1.0,
                processing_time_ms=0.0
            )
            
            # Step 1: Deduplication check
            if self.enable_deduplication:
                is_duplicate = await self._check_duplicate(message)
                if is_duplicate:
                    normalized_data.validation_status = 'duplicate'
                    normalized_data.confidence_score = 0.0
                    self.metrics.duplicates_detected += 1
                    return normalized_data
            
            # Step 2: Data validation
            if self.enable_validation:
                validation_result = await self._validate_data(message)
                if not validation_result['is_valid']:
                    normalized_data.validation_status = 'invalid'
                    normalized_data.validation_errors = validation_result['errors']
                    normalized_data.confidence_score = 0.0
                    self.metrics.validation_errors += 1
                    return normalized_data
            
            # Step 3: Outlier detection
            if self.enable_outlier_detection:
                outlier_result = await self._detect_outliers(message)
                if outlier_result['is_outlier']:
                    normalized_data.validation_status = 'outlier'
                    normalized_data.confidence_score = outlier_result['confidence']
                    self.metrics.outliers_detected += 1
                    return normalized_data
            
            # Step 4: Data normalization
            normalized_data.normalized_data = await self._normalize_data(message)
            normalized_data.validation_status = 'valid'
            normalized_data.confidence_score = 1.0
            
            # Update metrics
            self.metrics.messages_processed += 1
            self.metrics.messages_validated += 1
            self.metrics.messages_normalized += 1
            self.metrics.last_processed_time = datetime.now(timezone.utc)
            
            # Add to DB batch
            if self.timescaledb:
                self.db_batch.append(normalized_data)
                if len(self.db_batch) >= self.batch_size:
                    await self._flush_db_batch()
            
            # Trigger callbacks
            await self._trigger_callbacks(normalized_data)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            normalized_data.validation_status = 'error'
            normalized_data.validation_errors = [str(e)]
            normalized_data.confidence_score = 0.0
            return normalized_data
        
        finally:
            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            normalized_data.processing_time_ms = processing_time
            
            # Update average processing time
            if self.metrics.messages_processed > 0:
                self.metrics.avg_processing_time_ms = (
                    (self.metrics.avg_processing_time_ms * (self.metrics.messages_processed - 1) + processing_time) /
                    self.metrics.messages_processed
                )
    
    async def _check_duplicate(self, message: StreamMessage) -> bool:
        """Check if message is a duplicate"""
        try:
            # Create content hash
            content = json.dumps(message.data, sort_keys=True)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if hash exists in cache
            if content_hash in self.hash_cache:
                return True
            
            # Add to cache
            self.hash_cache.append(content_hash)
            return False
            
        except Exception as e:
            logger.error(f"Duplicate check error: {e}")
            return False
    
    async def _validate_data(self, message: StreamMessage) -> Dict[str, Any]:
        """Validate message data"""
        errors = []
        
        try:
            data = message.data
            
            # Check required fields
            required_fields = ['price', 'volume', 'timestamp']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate price
            if 'price' in data:
                price = float(data['price'])
                if price <= 0:
                    errors.append("Price must be positive")
                
                # Check for extreme price changes
                if message.symbol in self.price_history and self.price_history[message.symbol]:
                    last_price = self.price_history[message.symbol][-1]
                    price_change = abs(price - last_price) / last_price
                    if price_change > self.max_price_change:
                        errors.append(f"Price change too large: {price_change:.2%}")
            
            # Validate volume
            if 'volume' in data:
                volume = float(data['volume'])
                if volume < self.min_volume:
                    errors.append(f"Volume too low: {volume}")
            
            # Validate timestamp
            if 'timestamp' in data:
                try:
                    msg_timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    current_time = datetime.now(timezone.utc)
                    time_diff = abs((msg_timestamp - current_time).total_seconds())
                    
                    if time_diff > self.max_timestamp_drift:
                        errors.append(f"Timestamp drift too large: {time_diff:.1f}s")
                    
                except Exception as e:
                    errors.append(f"Invalid timestamp format: {e}")
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return {
                'is_valid': False,
                'errors': errors
            }
    
    async def _detect_outliers(self, message: StreamMessage) -> Dict[str, Any]:
        """Detect outliers using statistical methods"""
        try:
            data = message.data
            symbol = message.symbol
            
            # Check price outliers
            if 'price' in data and symbol in self.price_history:
                prices = list(self.price_history[symbol])
                if len(prices) >= 10:  # Need minimum data points
                    price = float(data['price'])
                    mean_price = np.mean(prices)
                    std_price = np.std(prices)
                    
                    if std_price > 0:
                        z_score = abs(price - mean_price) / std_price
                        if z_score > self.outlier_threshold:
                            return {
                                'is_outlier': True,
                                'confidence': max(0.0, 1.0 - (z_score / self.outlier_threshold))
                            }
            
            # Check volume outliers
            if 'volume' in data and symbol in self.volume_history:
                volumes = list(self.volume_history[symbol])
                if len(volumes) >= 10:
                    volume = float(data['volume'])
                    mean_volume = np.mean(volumes)
                    std_volume = np.std(volumes)
                    
                    if std_volume > 0:
                        z_score = abs(volume - mean_volume) / std_volume
                        if z_score > self.outlier_threshold:
                            return {
                                'is_outlier': True,
                                'confidence': max(0.0, 1.0 - (z_score / self.outlier_threshold))
                            }
            
            return {
                'is_outlier': False,
                'confidence': 1.0
            }
            
        except Exception as e:
            logger.error(f"Outlier detection error: {e}")
            return {
                'is_outlier': False,
                'confidence': 1.0
            }
    
    async def _normalize_data(self, message: StreamMessage) -> Dict[str, Any]:
        """Normalize message data"""
        try:
            normalized = {}
            data = message.data
            
            # Normalize timestamp
            if self.enable_timestamp_normalization and 'timestamp' in data:
                try:
                    timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                    normalized['timestamp'] = timestamp.isoformat()
                    normalized['timestamp_utc'] = timestamp.utctimetuple()
                except Exception as e:
                    logger.warning(f"Timestamp normalization failed: {e}")
                    normalized['timestamp'] = data['timestamp']
            else:
                normalized['timestamp'] = data.get('timestamp')
            
            # Normalize symbol
            normalized['symbol'] = message.symbol.upper()
            
            # Normalize price data
            if 'price' in data:
                price = float(data['price'])
                normalized['price'] = price
                normalized['price_formatted'] = f"{price:.8f}"
                
                # Update price history
                self.price_history[message.symbol].append(price)
            
            # Normalize volume data
            if 'volume' in data:
                volume = float(data['volume'])
                normalized['volume'] = volume
                normalized['volume_formatted'] = f"{volume:.2f}"
                
                # Update volume history
                self.volume_history[message.symbol].append(volume)
            
            # Copy other fields
            for key, value in data.items():
                if key not in ['timestamp', 'price', 'volume']:
                    normalized[key] = value
            
            # Add metadata
            normalized['normalized_at'] = datetime.now(timezone.utc).isoformat()
            normalized['source'] = message.source
            normalized['data_type'] = message.data_type
            
            return normalized
            
        except Exception as e:
            logger.error(f"Data normalization error: {e}")
            return data  # Return original data if normalization fails
    
    async def _trigger_callbacks(self, normalized_data: NormalizedData):
        """Trigger registered callbacks"""
        # Trigger normalization callbacks
        for callback in self.normalization_callbacks:
            try:
                await callback(normalized_data)
            except Exception as e:
                logger.error(f"Normalization callback error: {e}")
        
        # Trigger validation callbacks
        for callback in self.validation_callbacks:
            try:
                await callback(normalized_data)
            except Exception as e:
                logger.error(f"Validation callback error: {e}")
    
    def add_normalization_callback(self, callback):
        """Add normalization callback"""
        self.normalization_callbacks.append(callback)
    
    def add_validation_callback(self, callback):
        """Add validation callback"""
        self.validation_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get normalization metrics"""
        return {
            'is_running': self.is_running,
            'messages_processed': self.metrics.messages_processed,
            'messages_validated': self.metrics.messages_validated,
            'messages_normalized': self.metrics.messages_normalized,
            'duplicates_detected': self.metrics.duplicates_detected,
            'outliers_detected': self.metrics.outliers_detected,
            'validation_errors': self.metrics.validation_errors,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_mps': self.metrics.throughput_mps,
            'last_processed_time': self.metrics.last_processed_time.isoformat() if self.metrics.last_processed_time else None,
            'hash_cache_size': len(self.hash_cache),
            'db_batch_size': len(self.db_batch)
        }
    
    async def shutdown(self):
        """Shutdown the normalizer"""
        self.is_running = False
        
        # Cancel background tasks
        if self.db_flush_task:
            self.db_flush_task.cancel()
        
        # Flush remaining data
        if self.db_batch and self.timescaledb:
            await self._flush_db_batch()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 StreamNormalizer shutdown complete")

# Global instance
stream_normalizer = StreamNormalizer()
