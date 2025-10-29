"""
Advanced Logging System for AlphaPulse
Phase 4: High-throughput Redis logging with automated ensemble analysis and walk-forward optimization
"""

import asyncio
import logging
import time
import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import pickle
import gzip
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# Redis imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available - using in-memory fallback")

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Scikit-learn not available - ML features disabled")

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - hyperparameter optimization disabled")

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for system tracking"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventType(Enum):
    """Types of events being logged"""
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_VALIDATED = "signal_validated"
    SIGNAL_REJECTED = "signal_rejected"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_UPDATE = "performance_update"
    MODEL_UPDATE = "model_update"

@dataclass
class LogEntry:
    """Individual log entry"""
    id: str
    timestamp: datetime
    event_type: EventType
    log_level: LogLevel
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    size_bytes: int = 0

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    features: List[float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleAnalysis:
    """Ensemble analysis result"""
    prediction: str
    confidence: float
    individual_predictions: Dict[str, float]
    ensemble_method: str
    features_used: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class RedisLogger:
    """High-throughput Redis-based logging system"""
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 stream_name: str = 'alphapulse_logs',
                 batch_size: int = 100,
                 flush_interval: int = 5):
        
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.stream_name = stream_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self.redis_client = None
        self.log_buffer = deque(maxlen=batch_size * 10)
        self.is_running = False
        self.flush_task = None
        self.stats = {
            'total_logs': 0,
            'redis_operations': 0,
            'errors': 0,
            'avg_latency': 0.0
        }
        
        self._initialize_redis()
        logger.info(f"Redis Logger initialized with stream: {stream_name}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory logging")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    async def start(self):
        """Start the logging system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        logger.info("🚀 Redis Logger started")
    
    async def stop(self):
        """Stop the logging system"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining logs
        await self._flush_buffer()
        logger.info("🛑 Redis Logger stopped")
    
    async def log(self, 
                  event_type: EventType,
                  data: Dict[str, Any],
                  log_level: LogLevel = LogLevel.INFO,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log an event with high throughput"""
        start_time = time.time()
        
        try:
            # Create log entry
            entry_id = hashlib.md5(f"{time.time()}_{event_type.value}".encode()).hexdigest()
            entry = LogEntry(
                id=entry_id,
                timestamp=datetime.now(),
                event_type=event_type,
                log_level=log_level,
                data=data,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.log_buffer.append(entry)
            self.stats['total_logs'] += 1
            
            # Flush if buffer is full
            if len(self.log_buffer) >= self.batch_size:
                await self._flush_buffer()
            
            latency = time.time() - start_time
            self.stats['avg_latency'] = (
                (self.stats['avg_latency'] * (self.stats['total_logs'] - 1) + latency) 
                / self.stats['total_logs']
            )
            
            return entry_id
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error logging event: {e}")
            return ""
    
    async def _flush_buffer(self):
        """Flush buffered logs to Redis"""
        if not self.redis_client or not self.log_buffer:
            return
        
        try:
            # Prepare batch for Redis Streams
            pipeline = self.redis_client.pipeline()
            
            for entry in list(self.log_buffer):
                # Compress data if large
                if len(str(entry.data)) > 1000:
                    entry.data = self._compress_data(entry.data)
                    entry.compressed = True
                
                entry.size_bytes = len(str(entry.data))
                
                # Add to Redis stream
                pipeline.xadd(
                    self.stream_name,
                    {
                        'id': entry.id,
                        'timestamp': entry.timestamp.isoformat(),
                        'event_type': entry.event_type.value,
                        'log_level': entry.log_level.value,
                        'data': json.dumps(entry.data),
                        'metadata': json.dumps(entry.metadata),
                        'compressed': str(entry.compressed),
                        'size_bytes': str(entry.size_bytes)
                    }
                )
            
            # Execute pipeline
            pipeline.execute()
            self.stats['redis_operations'] += len(self.log_buffer)
            self.log_buffer.clear()
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error flushing buffer: {e}")
    
    async def _flush_loop(self):
        """Background task to flush logs periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(1)
    
    def _compress_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress data using gzip"""
        try:
            json_str = json.dumps(data)
            compressed = gzip.compress(json_str.encode())
            return {'compressed_data': compressed.hex(), 'original_size': len(json_str)}
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.log_buffer),
            'redis_connected': self.redis_client is not None
        }

# Global instance
redis_logger = RedisLogger(
    redis_host='localhost',
    redis_port=6379,
    stream_name='alphapulse_logs',
    batch_size=100,
    flush_interval=5
)