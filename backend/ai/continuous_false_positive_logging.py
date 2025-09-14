"""
Continuous False Positive Logging System for AlphaPulse
Phase 3: Advanced logging with in-memory databases, sampling, compression, and automated pattern dropping
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
try:
    import zstandard as zstd
    ZSTANDARD_AVAILABLE = True
except ImportError:
    ZSTANDARD_AVAILABLE = False
    zstd = None
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# In-memory database imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels for false positive tracking"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SignalType(Enum):
    """Types of signals being logged"""
    FALSE_POSITIVE = "false_positive"
    TRUE_POSITIVE = "true_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_NEGATIVE = "true_negative"
    UNCERTAIN = "uncertain"

@dataclass
class FalsePositiveEntry:
    """Individual false positive log entry"""
    id: str
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    threshold: float
    market_data: Dict[str, Any]
    filter_results: Dict[str, Any]
    pattern_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    log_level: LogLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    size_bytes: int = 0

@dataclass
class PatternAnalysis:
    """Pattern analysis result"""
    pattern_type: str
    frequency: int
    success_rate: float
    avg_confidence: float
    risk_score: float
    recommendation: str
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoggingMetrics:
    """Logging system performance metrics"""
    total_entries: int = 0
    compressed_entries: int = 0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    avg_processing_time: float = 0.0
    database_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_update: datetime = field(default_factory=datetime.now)

class InMemoryDatabase:
    """In-memory database for fast logging operations"""
    
    def __init__(self, 
                use_redis: bool = True,
                use_mongodb: bool = False,
                redis_host: str = 'localhost',
                redis_port: int = 6379,
                redis_db: int = 0,
                mongo_uri: str = 'mongodb://localhost:27017/'):
        
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.use_mongodb = use_mongodb and MONGODB_AVAILABLE
        
        # Redis connection
        self.redis_client = None
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.use_redis = False
        
        # MongoDB connection
        self.mongo_client = None
        self.mongo_db = None
        if self.use_mongodb:
            try:
                self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
                self.mongo_db = self.mongo_client['alphapulse_logs']
                # Test connection
                self.mongo_client.admin.command('ping')
                logger.info("MongoDB connection established")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
                self.use_mongodb = False
        
        # In-memory fallback
        self.memory_store = {}
        self.memory_index = defaultdict(list)
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        
        logger.info(f"InMemoryDatabase initialized (Redis: {self.use_redis}, MongoDB: {self.use_mongodb})")
    
    async def store_entry(self, entry: FalsePositiveEntry) -> bool:
        """Store log entry in database"""
        try:
            entry_dict = asdict(entry)
            entry_dict['timestamp'] = entry.timestamp.isoformat()
            
            # Generate unique key
            key = f"fp:{entry.symbol}:{entry.timestamp.timestamp()}:{entry.id}"
            
            # Try Redis first
            if self.use_redis:
                try:
                    # Set with TTL (24 hours)
                    self.redis_client.setex(key, 86400, json.dumps(entry_dict))
                    self.operation_count += 1
                    return True
                except Exception as e:
                    logger.warning(f"Redis store failed: {e}")
            
            # Try MongoDB
            if self.use_mongodb:
                try:
                    collection = self.mongo_db['false_positives']
                    collection.insert_one(entry_dict)
                    self.operation_count += 1
                    return True
                except Exception as e:
                    logger.warning(f"MongoDB store failed: {e}")
            
            # Fallback to memory
            self.memory_store[key] = entry_dict
            self.memory_index[entry.symbol].append(key)
            self.operation_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            self.error_count += 1
            return False
    
    async def retrieve_entries(self, symbol: str = None, limit: int = 100) -> List[FalsePositiveEntry]:
        """Retrieve log entries"""
        try:
            entries = []
            
            # Try Redis first
            if self.use_redis:
                try:
                    if symbol:
                        pattern = f"fp:{symbol}:*"
                    else:
                        pattern = "fp:*"
                    
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        # Get entries (limit to avoid memory issues)
                        keys = keys[:limit]
                        values = self.redis_client.mget(keys)
                        
                        for value in values:
                            if value:
                                entry_dict = json.loads(value)
                                entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                                entries.append(FalsePositiveEntry(**entry_dict))
                        
                        self.operation_count += 1
                        return entries
                except Exception as e:
                    logger.warning(f"Redis retrieve failed: {e}")
            
            # Try MongoDB
            if self.use_mongodb:
                try:
                    collection = self.mongo_db['false_positives']
                    query = {'symbol': symbol} if symbol else {}
                    cursor = collection.find(query).limit(limit)
                    
                    for doc in cursor:
                        doc['timestamp'] = datetime.fromisoformat(doc['timestamp'])
                        entries.append(FalsePositiveEntry(**doc))
                    
                    self.operation_count += 1
                    return entries
                except Exception as e:
                    logger.warning(f"MongoDB retrieve failed: {e}")
            
            # Fallback to memory
            if symbol:
                keys = self.memory_index.get(symbol, [])[:limit]
            else:
                keys = list(self.memory_store.keys())[:limit]
            
            for key in keys:
                if key in self.memory_store:
                    entry_dict = self.memory_store[key]
                    entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                    entries.append(FalsePositiveEntry(**entry_dict))
            
            self.operation_count += 1
            return entries
            
        except Exception as e:
            logger.error(f"Database retrieve error: {e}")
            self.error_count += 1
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'redis_available': self.use_redis,
            'mongodb_available': self.use_mongodb,
            'operation_count': self.operation_count,
            'error_count': self.error_count,
            'memory_store_size': len(self.memory_store),
            'memory_index_size': len(self.memory_index)
        }

class DataCompressor:
    """Data compression for log entries"""
    
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        if ZSTANDARD_AVAILABLE:
            self.compressor = zstd.ZstdCompressor(level=compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        else:
            self.compressor = None
            self.decompressor = None
        
        # Performance tracking
        self.compression_stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'avg_compression_ratio': 0.0
        }
        
        logger.info(f"DataCompressor initialized with level {compression_level}")
    
    def compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes with compression ratio"""
        try:
            original_size = len(data)
            
            if ZSTANDARD_AVAILABLE and self.compressor:
                compressed_data = self.compressor.compress(data)
                compressed_size = len(compressed_data)
            else:
                # Fallback to gzip compression
                compressed_data = gzip.compress(data)
                compressed_size = len(compressed_data)
            
            # Update statistics
            self.compression_stats['total_compressed'] += 1
            self.compression_stats['total_original_size'] += original_size
            self.compression_stats['total_compressed_size'] += compressed_size
            
            # Calculate compression ratio
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            self.compression_stats['avg_compression_ratio'] = (
                self.compression_stats['total_compressed_size'] / 
                self.compression_stats['total_original_size']
            )
            
            return compressed_data, compression_ratio
            
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return data, 1.0
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data"""
        try:
            if ZSTANDARD_AVAILABLE and self.decompressor:
                return self.decompressor.decompress(compressed_data)
            else:
                # Fallback to gzip decompression
                return gzip.decompress(compressed_data)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return compressed_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.compression_stats.copy()

class SamplingManager:
    """Intelligent sampling for log entries"""
    
    def __init__(self, 
                base_sampling_rate: float = 0.1,
                adaptive_sampling: bool = True,
                max_entries_per_minute: int = 1000):
        
        self.base_sampling_rate = base_sampling_rate
        self.adaptive_sampling = adaptive_sampling
        self.max_entries_per_minute = max_entries_per_minute
        
        # Sampling history
        self.sampling_history = deque(maxlen=1000)
        self.entry_counts = defaultdict(int)
        
        # Adaptive parameters
        self.current_sampling_rate = base_sampling_rate
        self.performance_threshold = 0.8
        
        logger.info(f"SamplingManager initialized (base_rate: {base_sampling_rate}, adaptive: {adaptive_sampling})")
    
    def should_sample(self, entry: FalsePositiveEntry) -> bool:
        """Determine if entry should be sampled"""
        try:
            # Check rate limiting
            current_minute = int(time.time() / 60)
            if self.entry_counts[current_minute] >= self.max_entries_per_minute:
                return False
            
            # Base sampling
            if np.random.random() > self.current_sampling_rate:
                return False
            
            # Adaptive sampling based on log level
            if self.adaptive_sampling:
                level_weights = {
                    LogLevel.LOW: 0.1,
                    LogLevel.MEDIUM: 0.3,
                    LogLevel.HIGH: 0.7,
                    LogLevel.CRITICAL: 1.0
                }
                
                weight = level_weights.get(entry.log_level, 0.5)
                if np.random.random() > weight:
                    return False
            
            # Update counters
            self.entry_counts[current_minute] += 1
            self.sampling_history.append({
                'timestamp': datetime.now(),
                'log_level': entry.log_level,
                'sampling_rate': self.current_sampling_rate
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Sampling decision error: {e}")
            return True  # Default to sampling on error
    
    def update_sampling_rate(self, performance_metrics: Dict[str, Any]):
        """Update sampling rate based on performance"""
        if not self.adaptive_sampling:
            return
        
        try:
            # Adjust based on system performance
            memory_usage = performance_metrics.get('memory_usage_percent', 50)
            cpu_usage = performance_metrics.get('cpu_usage_percent', 50)
            
            # Reduce sampling if system is under pressure
            if memory_usage > 80 or cpu_usage > 80:
                self.current_sampling_rate = max(0.01, self.current_sampling_rate * 0.8)
            elif memory_usage < 50 and cpu_usage < 50:
                self.current_sampling_rate = min(1.0, self.current_sampling_rate * 1.2)
            
            logger.debug(f"Sampling rate updated to {self.current_sampling_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Sampling rate update error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics"""
        return {
            'current_sampling_rate': self.current_sampling_rate,
            'base_sampling_rate': self.base_sampling_rate,
            'adaptive_sampling': self.adaptive_sampling,
            'total_sampled': len(self.sampling_history),
            'current_minute_entries': sum(self.entry_counts.values())
        }

class PatternDropper:
    """Automated pattern dropping using ensemble methods"""
    
    def __init__(self, 
                min_pattern_frequency: int = 10,
                min_success_rate: float = 0.3,
                max_risk_score: float = 0.7,
                enable_gpu_analysis: bool = True):
        
        self.min_pattern_frequency = min_pattern_frequency
        self.min_success_rate = min_success_rate
        self.max_risk_score = max_risk_score
        self.enable_gpu_analysis = enable_gpu_analysis
        
        # Pattern tracking
        self.pattern_history = defaultdict(list)
        self.dropped_patterns = set()
        self.pattern_analysis = {}
        
        # Ensemble models
        self.ensemble_models = {}
        self.feature_importance = {}
        
        logger.info("PatternDropper initialized")
    
    def add_pattern_data(self, pattern_type: str, entry: FalsePositiveEntry):
        """Add pattern data for analysis"""
        try:
            pattern_data = {
                'timestamp': entry.timestamp,
                'confidence': entry.confidence,
                'success': entry.signal_type in [SignalType.TRUE_POSITIVE, SignalType.TRUE_NEGATIVE],
                'risk_score': entry.performance_metrics.get('risk_score', 0.5),
                'market_conditions': entry.market_data.get('market_regime', 'normal'),
                'volatility': entry.market_data.get('volatility', 0.5)
            }
            
            self.pattern_history[pattern_type].append(pattern_data)
            
            # Keep only recent data (last 1000 entries per pattern)
            if len(self.pattern_history[pattern_type]) > 1000:
                self.pattern_history[pattern_type] = self.pattern_history[pattern_type][-1000:]
            
        except Exception as e:
            logger.error(f"Pattern data addition error: {e}")
    
    def analyze_patterns(self) -> Dict[str, PatternAnalysis]:
        """Analyze patterns and identify candidates for dropping"""
        try:
            analysis_results = {}
            
            for pattern_type, history in self.pattern_history.items():
                if len(history) < self.min_pattern_frequency:
                    continue
                
                # Calculate metrics
                frequency = len(history)
                success_count = sum(1 for entry in history if entry['success'])
                success_rate = success_count / frequency
                avg_confidence = np.mean([entry['confidence'] for entry in history])
                
                # Calculate risk score
                risk_scores = [entry['risk_score'] for entry in history]
                avg_risk = np.mean(risk_scores)
                risk_volatility = np.std(risk_scores)
                risk_score = avg_risk + risk_volatility
                
                # Determine recommendation
                if (success_rate < self.min_success_rate or 
                    risk_score > self.max_risk_score or
                    avg_confidence < 0.4):
                    recommendation = "drop"
                elif success_rate < 0.5 or risk_score > 0.6:
                    recommendation = "monitor"
                else:
                    recommendation = "keep"
                
                analysis = PatternAnalysis(
                    pattern_type=pattern_type,
                    frequency=frequency,
                    success_rate=success_rate,
                    avg_confidence=avg_confidence,
                    risk_score=risk_score,
                    recommendation=recommendation,
                    last_seen=max(entry['timestamp'] for entry in history),
                    metadata={
                        'avg_risk': avg_risk,
                        'risk_volatility': risk_volatility,
                        'market_conditions': self._get_market_distribution(history)
                    }
                )
                
                analysis_results[pattern_type] = analysis
                
                # Update dropped patterns
                if recommendation == "drop":
                    self.dropped_patterns.add(pattern_type)
                elif pattern_type in self.dropped_patterns:
                    self.dropped_patterns.remove(pattern_type)
            
            self.pattern_analysis = analysis_results
            return analysis_results
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return {}
    
    def _get_market_distribution(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get distribution of market conditions for pattern"""
        try:
            conditions = [entry['market_conditions'] for entry in history]
            total = len(conditions)
            
            distribution = defaultdict(int)
            for condition in conditions:
                distribution[condition] += 1
            
            return {k: v/total for k, v in distribution.items()}
            
        except Exception as e:
            logger.error(f"Market distribution calculation error: {e}")
            return {}
    
    def should_drop_pattern(self, pattern_type: str) -> bool:
        """Check if pattern should be dropped"""
        return pattern_type in self.dropped_patterns
    
    def get_dropped_patterns(self) -> List[str]:
        """Get list of dropped patterns"""
        return list(self.dropped_patterns)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pattern dropping statistics"""
        return {
            'total_patterns': len(self.pattern_history),
            'dropped_patterns': len(self.dropped_patterns),
            'analysis_count': len(self.pattern_analysis),
            'min_frequency': self.min_pattern_frequency,
            'min_success_rate': self.min_success_rate,
            'max_risk_score': self.max_risk_score
        }

class ContinuousFalsePositiveLogger:
    """Main continuous false positive logging system"""
    
    def __init__(self, 
                database_config: Dict[str, Any] = None,
                compression_level: int = 3,
                sampling_rate: float = 0.1,
                enable_pattern_dropping: bool = True,
                cleanup_interval: int = 3600):
        
        # Initialize components
        db_config = database_config or {}
        self.database = InMemoryDatabase(**db_config)
        self.compressor = DataCompressor(compression_level)
        self.sampling_manager = SamplingManager(sampling_rate)
        self.pattern_dropper = PatternDropper() if enable_pattern_dropping else None
        
        # Configuration
        self.cleanup_interval = cleanup_interval
        self.enable_compression = True
        self.enable_sampling = True
        
        # Performance tracking
        self.metrics = LoggingMetrics()
        self.processing_times = deque(maxlen=1000)
        
        # Background tasks
        self.cleanup_task = None
        self.analysis_task = None
        self.running = False
        
        logger.info("Continuous False Positive Logger initialized")
    
    async def start(self):
        """Start the logging system"""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self.pattern_dropper:
            self.analysis_task = asyncio.create_task(self._pattern_analysis_loop())
        
        logger.info("Continuous False Positive Logger started")
    
    async def stop(self):
        """Stop the logging system"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Continuous False Positive Logger stopped")
    
    async def log_false_positive(self, 
                               symbol: str,
                               signal_type: SignalType,
                               confidence: float,
                               threshold: float,
                               market_data: Dict[str, Any],
                               filter_results: Dict[str, Any],
                               pattern_data: Dict[str, Any],
                               performance_metrics: Dict[str, Any]) -> bool:
        """Log a false positive entry"""
        start_time = time.time()
        
        try:
            # Determine log level
            log_level = self._determine_log_level(confidence, threshold, performance_metrics)
            
            # Create entry
            entry = FalsePositiveEntry(
                id=hashlib.md5(f"{symbol}{time.time()}".encode()).hexdigest()[:8],
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                threshold=threshold,
                market_data=market_data,
                filter_results=filter_results,
                pattern_data=pattern_data,
                performance_metrics=performance_metrics,
                log_level=log_level
            )
            
            # Apply sampling
            if self.enable_sampling and not self.sampling_manager.should_sample(entry):
                return True  # Successfully skipped
            
            # Apply compression
            if self.enable_compression:
                entry_data = pickle.dumps(entry)
                compressed_data, compression_ratio = self.compressor.compress_data(entry_data)
                
                if compression_ratio < 0.9:  # Only use if compression is effective
                    entry.compressed = True
                    entry.size_bytes = len(compressed_data)
                    self.metrics.compressed_entries += 1
                    self.metrics.compressed_size_bytes += entry.size_bytes
                else:
                    entry.size_bytes = len(entry_data)
                
                self.metrics.total_size_bytes += entry.size_bytes
            
            # Store in database
            success = await self.database.store_entry(entry)
            
            # Update pattern data
            if self.pattern_dropper and pattern_data:
                pattern_type = pattern_data.get('pattern_type', 'unknown')
                self.pattern_dropper.add_pattern_data(pattern_type, entry)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.metrics.total_entries += 1
            self.metrics.avg_processing_time = np.mean(list(self.processing_times))
            
            return success
            
        except Exception as e:
            logger.error(f"False positive logging error: {e}")
            return False
    
    def _determine_log_level(self, confidence: float, threshold: float, performance_metrics: Dict[str, Any]) -> LogLevel:
        """Determine log level based on signal characteristics"""
        try:
            # Base level on confidence vs threshold
            confidence_ratio = confidence / threshold if threshold > 0 else 1.0
            
            # Consider performance metrics
            risk_score = performance_metrics.get('risk_score', 0.5)
            loss_amount = performance_metrics.get('loss_amount', 0.0)
            
            if confidence_ratio < 0.5 or risk_score > 0.8 or loss_amount > 1000:
                return LogLevel.CRITICAL
            elif confidence_ratio < 0.7 or risk_score > 0.6:
                return LogLevel.HIGH
            elif confidence_ratio < 0.9 or risk_score > 0.4:
                return LogLevel.MEDIUM
            else:
                return LogLevel.LOW
                
        except Exception as e:
            logger.error(f"Log level determination error: {e}")
            return LogLevel.MEDIUM
    
    async def get_recent_entries(self, symbol: str = None, limit: int = 100) -> List[FalsePositiveEntry]:
        """Get recent log entries"""
        return await self.database.retrieve_entries(symbol, limit)
    
    async def get_pattern_analysis(self) -> Dict[str, PatternAnalysis]:
        """Get current pattern analysis"""
        if self.pattern_dropper:
            return self.pattern_dropper.analyze_patterns()
        return {}
    
    def should_drop_pattern(self, pattern_type: str) -> bool:
        """Check if pattern should be dropped"""
        if self.pattern_dropper:
            return self.pattern_dropper.should_drop_pattern(pattern_type)
        return False
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while self.running:
            try:
                # Update sampling rate based on system performance
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                self.sampling_manager.update_sampling_rate({
                    'memory_usage_percent': memory_usage,
                    'cpu_usage_percent': cpu_usage
                })
                
                # Clear old processing times
                if len(self.processing_times) > 1000:
                    self.processing_times.clear()
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
    
    async def _pattern_analysis_loop(self):
        """Periodic pattern analysis loop"""
        while self.running:
            try:
                # Run pattern analysis
                analysis = await self.get_pattern_analysis()
                
                # Log analysis results
                if analysis:
                    logger.info(f"Pattern analysis completed: {len(analysis)} patterns analyzed")
                    
                    # Log dropped patterns
                    dropped = [p for p, a in analysis.items() if a.recommendation == "drop"]
                    if dropped:
                        logger.warning(f"Patterns recommended for dropping: {dropped}")
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Pattern analysis loop error: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive logging metrics"""
        return {
            'logging_metrics': asdict(self.metrics),
            'database_stats': self.database.get_stats(),
            'compression_stats': self.compressor.get_stats(),
            'sampling_stats': self.sampling_manager.get_stats(),
            'pattern_dropper_stats': self.pattern_dropper.get_stats() if self.pattern_dropper else {},
            'system_performance': {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        }

# Global continuous false positive logger instance
continuous_false_positive_logger = ContinuousFalsePositiveLogger(
    database_config={
        'use_redis': True,
        'use_mongodb': False,
        'redis_host': 'localhost',
        'redis_port': 6379
    },
    compression_level=3,
    sampling_rate=0.1,
    enable_pattern_dropping=True,
    cleanup_interval=3600
)
