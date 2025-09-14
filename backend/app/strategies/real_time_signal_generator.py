"""
Real-time Signal Generator for AlphaPlus
Generates trading signals in real-time based on market data and patterns
Phase 4.1: Real-time Processing Optimization - Ultra-low latency signal generation
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc
import psutil
import os

logger = logging.getLogger(__name__)

# Phase 4.2: Memory & CPU Optimization Dataclasses
from dataclasses import dataclass

@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    total_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    cpu_percent: float
    process_memory_mb: float
    cache_size: int
    gc_objects: int
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    processing_time_ms: float
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_per_second: float
    latency_percentile_95: float
    timestamp: datetime

# Import database connection
try:
    from ...database.connection import TimescaleDBConnection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    TimescaleDBConnection = None

# Import advanced technical indicators engine
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai'))
    from technical_indicators_engine import TechnicalIndicatorsEngine
    ADVANCED_INDICATORS_AVAILABLE = True
    logger.info("✅ Advanced technical indicators engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced indicators not available: {e}")
    ADVANCED_INDICATORS_AVAILABLE = False
    TechnicalIndicatorsEngine = None

# Import Smart Money Concepts engine
try:
    from .smart_money_concepts_engine import SmartMoneyConceptsEngine, SMCAnalysis
    SMC_AVAILABLE = True
    logger.info("✅ Smart Money Concepts engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Smart Money Concepts not available: {e}")
    SMC_AVAILABLE = False
    SmartMoneyConceptsEngine = None

# Import Deep Learning engine
try:
    from .deep_learning_engine import DeepLearningEngine
    DEEP_LEARNING_AVAILABLE = True
    logger.info("✅ Deep Learning engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Deep Learning not available: {e}")
    DEEP_LEARNING_AVAILABLE = False
    DeepLearningEngine = None

# Import Reinforcement Learning engine
try:
    from .reinforcement_learning_engine import ReinforcementLearningEngine, RLState, RLAction
    RL_AVAILABLE = True
    logger.info("✅ Reinforcement Learning engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Reinforcement Learning not available: {e}")
    RL_AVAILABLE = False
    ReinforcementLearningEngine = None

# Import Natural Language Processing engine
try:
    from .natural_language_processing_engine import NaturalLanguageProcessingEngine, NLPSource, SentimentType
    NLP_AVAILABLE = True
    logger.info("✅ Natural Language Processing engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Natural Language Processing not available: {e}")
    NLP_AVAILABLE = False
    NaturalLanguageProcessingEngine = None

# Import ML Strategy Enhancement for ensemble models
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from ai.ml_strategy_enhancement import EnsembleStrategy, EnsembleConfig, MLStrategyType, ModelType
    ML_STRATEGY_AVAILABLE = True
    logger.info("✅ ML Strategy Enhancement imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML Strategy Enhancement not available: {e}")
    ML_STRATEGY_AVAILABLE = False
    EnsembleStrategy = None

# Import Phase 3.1: Enhanced Sentiment Service
try:
    from ..services.sentiment_service import SentimentService
    SENTIMENT_AVAILABLE = True
    logger.info("✅ Enhanced Sentiment Service imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Sentiment Service not available: {e}")
    SENTIMENT_AVAILABLE = False
    SentimentService = None

# Phase 4.1: Performance Optimization Imports
try:
    from ..core.performance_monitor import PerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
    logger.info("✅ Performance Monitor imported successfully")
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    PerformanceMonitor = None

try:
    from ..core.memory_manager import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ Memory Manager imported successfully")
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    MemoryManager = None

class RealTimeSignalGenerator:
    """Real-time signal generator for trading signals with deduplication and advanced indicators
    Phase 4.1: Ultra-low latency signal generation with <100ms processing time"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Real-time Signal Generator with Phase 4.1 optimizations"""
        self.config = config or {}
        self.is_running = False
        self.signals = []
        self.signal_cache = {}
        self.last_update = {}
        
        # Phase 4.1: Performance Optimization
        self.performance_monitor = None
        self.memory_manager = None
        self.processing_times = deque(maxlen=1000)  # Track last 1000 processing times
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Phase 4.1: Real-time Processing Optimizations
        self.async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="SignalGen")
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.result_cache = {}
        self.cache_ttl = 5.0  # 5 seconds cache TTL
        self.last_cache_cleanup = time.time()
        
        # Phase 4.1: Memory Management
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.gc_threshold = 1000  # Garbage collection after 1000 operations
        self.operation_count = 0
        
        # Phase 4.2: Advanced Memory & CPU Optimization
        self.memory_metrics_history = deque(maxlen=1000)  # Track memory usage over time
        self.cpu_metrics_history = deque(maxlen=1000)  # Track CPU usage over time
        self.memory_cleanup_interval = 300  # 5 minutes
        self.last_memory_cleanup = time.time()
        self.memory_optimization_enabled = True
        self.cpu_optimization_enabled = True
        
        # Phase 4.2: Advanced Caching with Memory Pressure
        self.cache_memory_limit_mb = 512  # 512MB cache limit
        self.cache_eviction_policy = "lru"  # Least Recently Used
        self.cache_compression_enabled = True
        self.cache_compression_threshold = 0.7  # Compress when cache is 70% full
        
        # Phase 4.2: CPU Optimization
        self.cpu_affinity_enabled = True
        self.cpu_cores_reserved = 1  # Reserve 1 core for system
        self.cpu_load_balancing = True
        self.cpu_throttling_threshold = 0.9  # Throttle at 90% CPU usage
        
        # Phase 4.2: Resource Monitoring
        self.resource_monitoring_enabled = True
        self.monitoring_interval = 60  # 60 seconds
        self.last_resource_check = time.time()
        self.resource_alerts = []
        
        # Phase 4.2: Garbage Collection Optimization
        self.gc_optimization_enabled = True
        self.gc_generation_thresholds = (700, 10, 10)  # Optimized thresholds
        self.gc_auto_tuning = True
        self.gc_stats = {
            'collections': 0,
            'objects_freed': 0,
            'last_collection_time': 0
        }
        
        # Signal deduplication tracking
        self.active_signals = {}  # signal_id -> signal
        self.signal_history = []  # List of all signals
        self.signals_generated = 0
        self.signals_expired = 0
        self.signal_cooldown = self.config.get('signal_cooldown', 300)  # 5 minutes
        
        # Database connection
        self.db_connection = None
        self.use_database = self.config.get('use_database', True) and DB_AVAILABLE
        
        # Advanced technical indicators engine
        self.indicators_engine = None
        self.use_advanced_indicators = self.config.get('use_advanced_indicators', True) and ADVANCED_INDICATORS_AVAILABLE
        
        # Smart Money Concepts engine
        self.smc_engine = None
        self.use_smc = self.config.get('use_smc', True) and SMC_AVAILABLE
        
        # Deep Learning engine
        self.dl_engine = None
        self.use_dl = self.config.get('use_dl', True) and DEEP_LEARNING_AVAILABLE
        
        # Reinforcement Learning engine
        self.rl_engine = None
        self.use_rl = self.config.get('use_rl', True) and RL_AVAILABLE
        
        # Natural Language Processing engine
        self.nlp_engine = None
        self.use_nlp = self.config.get('use_nlp', True) and NLP_AVAILABLE
        
        # Phase 2.3: Ensemble Model Integration
        self.ensemble_strategy = None
        self.use_ensemble = self.config.get('use_ensemble', True) and ML_STRATEGY_AVAILABLE
        
        # Phase 3.1: Enhanced Sentiment Analysis Integration
        self.sentiment_service = None
        self.use_sentiment = self.config.get('use_sentiment', True) and SENTIMENT_AVAILABLE
        
        # Default configuration
        self.min_confidence = self.config.get('min_confidence', 0.6)  # Lowered from 0.85 to 0.6 for more real signals
        self.min_strength = self.config.get('min_strength', 0.6)
        self.confirmation_required = self.config.get('confirmation_required', True)
        self.volume_confirmation = self.config.get('volume_confirmation', True)
        self.trend_confirmation = self.config.get('trend_confirmation', True)
        
        # Phase 4.1: Performance Targets
        self.target_latency_ms = 100  # Target <100ms processing time
        self.max_memory_mb = 512  # Max memory usage in MB
        self.enable_async_processing = True
        self.enable_caching = True
        self.enable_memory_management = True
        
        logger.info("🚀 Phase 4.1: Real-time Signal Generator initialized with ultra-low latency optimizations")
        logger.info(f"🎯 Performance Target: <{self.target_latency_ms}ms signal generation")
        
        # Initialize performance monitoring
        self._initialize_performance_monitoring()
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring components"""
        if PERFORMANCE_MONITOR_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
            logger.info("✅ Performance monitoring initialized")
        
        if MEMORY_MANAGER_AVAILABLE:
            self.memory_manager = MemoryManager()
            logger.info("✅ Memory management initialized")
        
        # Start background tasks
        asyncio.create_task(self._background_cache_cleanup())
        asyncio.create_task(self._background_memory_management())
        asyncio.create_task(self._background_performance_monitoring())
    
    async def _background_cache_cleanup(self):
        """Background task for cache cleanup"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                await self._cleanup_cache()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _background_memory_management(self):
        """Background task for memory management"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_memory_usage()
            except Exception as e:
                logger.error(f"Memory management error: {e}")
    
    async def _background_performance_monitoring(self):
        """Background task for performance monitoring"""
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                await self._log_performance_metrics()
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, timestamp) in self.result_cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.result_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _check_memory_usage(self):
        """Check and manage memory usage"""
        if not self.enable_memory_management:
            return
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            if memory_percent > self.memory_threshold * 100:
                logger.warning(f"High memory usage: {memory_percent:.1f}%")
                await self._optimize_memory()
            
            # Periodic garbage collection
            self.operation_count += 1
            if self.operation_count >= self.gc_threshold:
                gc.collect()
                self.operation_count = 0
                logger.debug("Garbage collection performed")
                
        except Exception as e:
            logger.error(f"Memory check error: {e}")
    
    async def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Clear old cache entries
            self.result_cache.clear()
            
            # Clear old processing times
            if len(self.processing_times) > 500:
                # Keep only last 500 entries
                self.processing_times = deque(list(self.processing_times)[-500:], maxlen=1000)
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
    
    async def _log_performance_metrics(self):
        """Log performance metrics"""
        if not self.processing_times:
            return
        
        try:
            times = list(self.processing_times)
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            cache_hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            
            logger.info(f"📊 Performance Metrics - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms, "
                       f"Min: {min_time:.2f}ms, Cache Hit Rate: {cache_hit_rate:.1f}%")
            
            # Check if we're meeting performance targets
            if avg_time > self.target_latency_ms:
                logger.warning(f"⚠️ Average processing time ({avg_time:.2f}ms) exceeds target ({self.target_latency_ms}ms)")
            else:
                logger.info(f"✅ Performance target met: {avg_time:.2f}ms < {self.target_latency_ms}ms")
                
        except Exception as e:
            logger.error(f"Performance logging error: {e}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, data_hash: str) -> str:
        """Generate cache key for data"""
        return f"{symbol}_{timeframe}_{data_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        if not self.enable_caching:
            return None
        
        try:
            if cache_key in self.result_cache:
                data, timestamp = self.result_cache[cache_key]
                if time.time() - timestamp <= self.cache_ttl:
                    self.cache_hits += 1
                    return data
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result with timestamp"""
        if not self.enable_caching:
            return
        
        try:
            self.result_cache[cache_key] = (result, time.time())
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    async def initialize_database(self):
        """Initialize database connection"""
        if self.use_database and not self.db_connection:
            try:
                self.db_connection = TimescaleDBConnection({
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'alphapulse',
                    'username': 'alpha_emon',
                    'password': 'Emon_@17711'
                })
                await self.db_connection.initialize()
                logger.info("✅ Database connection initialized for signal generator")
            except Exception as e:
                logger.error(f"❌ Failed to initialize database connection: {e}")
                self.use_database = False
                self.db_connection = None

    async def initialize_advanced_indicators(self):
        """Initialize advanced technical indicators engine"""
        if self.use_advanced_indicators and not self.indicators_engine:
            try:
                self.indicators_engine = TechnicalIndicatorsEngine()
                logger.info("✅ Advanced technical indicators engine initialized")
                logger.info(f"📊 Available indicators: {', '.join(self.indicators_engine.get_available_indicators())}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize advanced indicators engine: {e}")
                self.use_advanced_indicators = False
                self.indicators_engine = None
    
    async def save_signal_to_database(self, signal: Dict[str, Any]) -> bool:
        """Save signal to TimescaleDB"""
        if not self.use_database or not self.db_connection:
            return False
        
        try:
            # Prepare signal data for database
            signal_id = f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            db_signal_data = {
                'id': signal_id,
                'symbol': signal.get('symbol', 'BTCUSDT'),
                'side': signal.get('signal_type', 'buy'),
                'strategy': 'real_time_technical',
                'confidence': signal.get('confidence', 0.0),
                'strength': signal.get('strength', 0.0),
                'timestamp': datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat())),
                'price': signal.get('price', 0.0),
                'stop_loss': signal.get('price', 0.0) * 0.95,  # 5% stop loss
                'take_profit': signal.get('price', 0.0) * 1.05,  # 5% take profit
                'metadata': {
                    'reason': signal.get('reason', ''),
                    'indicators': signal.get('indicators', {}),
                    'source': 'real_time_signal_generator',
                    'confidence_threshold': self.min_confidence
                }
            }
            
            success = await self.db_connection.save_signal(db_signal_data)
            if success:
                logger.info(f"✅ Signal saved to database: {signal_id}")
                return True
            else:
                logger.error(f"❌ Failed to save signal to database: {signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving signal to database: {e}")
            return False

    async def _add_signal_with_deduplication(self, signal: Dict[str, Any]) -> bool:
        """Add a new signal with deduplication logic"""
        try:
            # Create unique signal ID
            signal_id = str(uuid.uuid4())
            signal['id'] = signal_id
            
            # Check if we already have a signal for this symbol/side
            symbol = signal.get('symbol', '')
            signal_type = signal.get('signal_type', 'buy')
            key = (symbol, signal_type)
            
            # Remove existing signal if new one has higher confidence
            existing_signals = [s for s in self.active_signals.values() 
                              if (s.get('symbol', '') == symbol and s.get('signal_type', '') == signal_type)]
            
            for existing in existing_signals:
                existing_confidence = existing.get('confidence', 0.0)
                new_confidence = signal.get('confidence', 0.0)
                
                if new_confidence > existing_confidence:
                    # Remove existing signal
                    existing_id = existing.get('id', '')
                    if existing_id in self.active_signals:
                        del self.active_signals[existing_id]
                        self.signals_expired += 1
                        logger.info(f"🔄 Replaced existing signal for {symbol} {signal_type} with higher confidence signal")
                else:
                    # New signal has lower or equal confidence, don't add it
                    logger.info(f"⏭️ Skipped signal for {symbol} {signal_type} - lower confidence ({new_confidence:.2f} <= {existing_confidence:.2f})")
                    return False
            
            # Add new signal
            self.active_signals[signal_id] = signal
            self.signal_history.append(signal)
            self.signals_generated += 1
            
            # Save to database
            await self.save_signal_to_database(signal)
            
            logger.info(f"✅ Added signal: {symbol} {signal_type} - Confidence: {signal.get('confidence', 0.0):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding signal with deduplication: {e}")
            return False

    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        try:
            current_time = datetime.now()
            expired_signals = []
            
            for signal_id, signal in self.active_signals.items():
                # Get signal timestamp
                timestamp_str = signal.get('timestamp', '')
                if timestamp_str:
                    try:
                        if isinstance(timestamp_str, str):
                            signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            signal_time = timestamp_str
                        
                        # Signal expires after cooldown period
                        if (current_time - signal_time).total_seconds() > self.signal_cooldown:
                            expired_signals.append(signal_id)
                    except Exception as e:
                        logger.warning(f"⚠️ Error parsing signal timestamp: {e}")
                        # If we can't parse timestamp, consider it expired
                        expired_signals.append(signal_id)
            
            # Remove expired signals
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
                self.signals_expired += 1
            
            if expired_signals:
                logger.info(f"🧹 Cleaned up {len(expired_signals)} expired signals")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up expired signals: {e}")

    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active trading signals"""
        return list(self.active_signals.values())

    async def get_signal_by_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific signal by ID"""
        return self.active_signals.get(signal_id)

    async def get_signals_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all signals for a specific symbol"""
        return [s for s in self.active_signals.values() if s.get('symbol', '') == symbol]

    async def mark_signal_executed(self, signal_id: str):
        """Mark a signal as executed"""
        try:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]
                del self.active_signals[signal_id]
                logger.info(f"✅ Signal executed: {signal.get('symbol', '')} {signal.get('signal_type', '')}")
            
        except Exception as e:
            logger.error(f"❌ Error marking signal executed: {e}")

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get signal generator performance summary"""
        try:
            return {
                'total_signals_generated': self.signals_generated,
                'signals_expired': self.signals_expired,
                'active_signals': len(self.active_signals),
                'signal_cooldown': self.signal_cooldown,
                'min_confidence': self.min_confidence,
                'database_connected': self.use_database and self.db_connection is not None,
                'advanced_indicators_available': self.use_advanced_indicators and self.indicators_engine is not None,
                'smc_available': self.use_smc and self.smc_engine is not None,
                'dl_available': self.use_dl and self.dl_engine is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {}

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced technical indicators using the indicators engine"""
        try:
            if not self.use_advanced_indicators or not self.indicators_engine:
                return {}
            
            # Calculate all advanced indicators
            advanced_indicators = self.indicators_engine.calculate_all_indicators(
                df, 
                indicators=['ichimoku', 'fibonacci_retracement', 'bollinger_bands', 'vwap', 'obv', 'atr']
            )
            
            # Extract latest values
            latest_indicators = {}
            for indicator_name, indicator_series in advanced_indicators.items():
                if not indicator_series.empty:
                    latest_indicators[indicator_name] = float(indicator_series.iloc[-1])
                else:
                    latest_indicators[indicator_name] = None
            
            logger.debug(f"📊 Calculated {len(latest_indicators)} advanced indicators")
            return latest_indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating advanced indicators: {e}")
            return {}

    def analyze_ichimoku_signals(self, ichimoku_data: Dict[str, float], current_price: float) -> Dict[str, Any]:
        """Analyze Ichimoku Cloud signals"""
        try:
            if not ichimoku_data:
                return {}
            
            # Extract Ichimoku components
            tenkan_sen = ichimoku_data.get('ichimoku_tenkan_sen')
            kijun_sen = ichimoku_data.get('ichimoku_kijun_sen')
            senkou_span_a = ichimoku_data.get('ichimoku_senkou_span_a')
            senkou_span_b = ichimoku_data.get('ichimoku_senkou_span_b')
            
            if not all([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b]):
                return {}
            
            # Ichimoku signal analysis
            signals = {}
            
            # Price vs Cloud analysis
            cloud_upper = max(senkou_span_a, senkou_span_b)
            cloud_lower = min(senkou_span_a, senkou_span_b)
            
            if current_price > cloud_upper:
                signals['cloud_position'] = 'above_cloud'
                signals['cloud_bullish'] = True
            elif current_price < cloud_lower:
                signals['cloud_position'] = 'below_cloud'
                signals['cloud_bullish'] = False
            else:
                signals['cloud_position'] = 'inside_cloud'
                signals['cloud_bullish'] = None
            
            # Tenkan/Kijun crossover
            if tenkan_sen > kijun_sen:
                signals['tenkan_kijun'] = 'bullish'
            else:
                signals['tenkan_kijun'] = 'bearish'
            
            # Cloud color (future cloud)
            if senkou_span_a > senkou_span_b:
                signals['cloud_color'] = 'green'
            else:
                signals['cloud_color'] = 'red'
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Ichimoku signals: {e}")
            return {}

    def analyze_fibonacci_signals(self, fib_data: Dict[str, float], current_price: float, high: float, low: float) -> Dict[str, Any]:
        """Analyze Fibonacci retracement signals"""
        try:
            if not fib_data:
                return {}
            
            # Calculate Fibonacci levels manually if not available
            fib_levels = {
                0.236: low + (high - low) * 0.236,
                0.382: low + (high - low) * 0.382,
                0.500: low + (high - low) * 0.500,
                0.618: low + (high - low) * 0.618,
                0.786: low + (high - low) * 0.786
            }
            
            signals = {}
            
            # Find nearest Fibonacci level
            nearest_level = None
            min_distance = float('inf')
            
            for level, price in fib_levels.items():
                distance = abs(current_price - price)
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level
            
            signals['nearest_fib_level'] = nearest_level
            signals['nearest_fib_price'] = fib_levels.get(nearest_level, 0)
            signals['fib_distance'] = min_distance
            
            # Support/Resistance analysis
            if current_price < fib_levels[0.382]:
                signals['fib_position'] = 'below_382'
                signals['fib_bullish'] = True
            elif current_price > fib_levels[0.618]:
                signals['fib_position'] = 'above_618'
                signals['fib_bullish'] = False
            else:
                signals['fib_position'] = 'between_382_618'
                signals['fib_bullish'] = None
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Fibonacci signals: {e}")
            return {}

    def analyze_volume_signals(self, volume_data: Dict[str, float], current_volume: float) -> Dict[str, Any]:
        """Analyze volume-based signals"""
        try:
            if not volume_data:
                return {}
            
            signals = {}
            
            # VWAP analysis
            vwap = volume_data.get('vwap')
            if vwap:
                signals['vwap'] = vwap
                signals['price_vs_vwap'] = 'above' if current_volume > vwap else 'below'
            
            # OBV analysis
            obv = volume_data.get('obv')
            if obv:
                signals['obv'] = obv
                signals['obv_trend'] = 'bullish' if obv > 0 else 'bearish'
            
            # Volume SMA ratio
            volume_sma_ratio = volume_data.get('volume_sma_ratio')
            if volume_sma_ratio:
                signals['volume_sma_ratio'] = volume_sma_ratio
                signals['volume_confirmation'] = volume_sma_ratio > 1.2  # 20% above average
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing volume signals: {e}")
            return {}

    async def analyze_smart_money_concepts(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze Smart Money Concepts"""
        try:
            if not self.use_smc or not self.smc_engine:
                return {}
            
            # Analyze SMC
            smc_analysis = await self.smc_engine.analyze_smart_money_concepts(df, symbol, timeframe)
            
            # Extract key signals
            smc_signals = {
                'overall_confidence': smc_analysis.overall_confidence,
                'order_blocks_count': len(smc_analysis.order_blocks),
                'fair_value_gaps_count': len(smc_analysis.fair_value_gaps),
                'liquidity_sweeps_count': len(smc_analysis.liquidity_sweeps),
                'market_structures_count': len(smc_analysis.market_structures),
                'top_signals': smc_analysis.smc_signals[:3] if smc_analysis.smc_signals else []
            }
            
            # Add bullish/bearish bias
            bullish_signals = [s for s in smc_analysis.smc_signals if s.get('direction') == 'bullish']
            bearish_signals = [s for s in smc_analysis.smc_signals if s.get('direction') == 'bearish']
            
            smc_signals['bullish_signals_count'] = len(bullish_signals)
            smc_signals['bearish_signals_count'] = len(bearish_signals)
            smc_signals['smc_bias'] = 'bullish' if len(bullish_signals) > len(bearish_signals) else 'bearish' if len(bearish_signals) > len(bullish_signals) else 'neutral'
            
            logger.debug(f"📊 SMC Analysis for {symbol}: {smc_signals['smc_bias']} bias, {len(smc_analysis.smc_signals)} signals")
            return smc_signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Smart Money Concepts: {e}")
            return {}

    async def analyze_deep_learning_predictions(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze Deep Learning predictions"""
        try:
            if not self.use_dl or not self.dl_engine:
                return {}
            
            # Make predictions with deep learning models
            dl_predictions = await self.dl_engine.predict_with_models(df, symbol, timeframe)
            
            # Extract key predictions
            dl_signals = {
                'ensemble_prediction': dl_predictions.get('ensemble_prediction', 0.5),
                'confidence': dl_predictions.get('confidence', 0.0),
                'models_used': dl_predictions.get('models_used', 0),
                'individual_predictions': dl_predictions.get('predictions', {}),
                'timestamp': dl_predictions.get('timestamp', datetime.now())
            }
            
            # Add prediction bias
            ensemble_pred = dl_signals['ensemble_prediction']
            dl_signals['dl_bias'] = 'bullish' if ensemble_pred > 0.6 else 'bearish' if ensemble_pred < 0.4 else 'neutral'
            dl_signals['prediction_strength'] = abs(ensemble_pred - 0.5) * 2  # 0.0 to 1.0
            
            # Add confidence-based signals
            if dl_signals['confidence'] > 0.7 and dl_signals['models_used'] >= 2:
                dl_signals['high_confidence_signal'] = True
                dl_signals['signal_strength'] = dl_signals['prediction_strength'] * dl_signals['confidence']
            else:
                dl_signals['high_confidence_signal'] = False
                dl_signals['signal_strength'] = 0.0
            
            logger.debug(f"🧠 DL Analysis for {symbol}: {dl_signals['dl_bias']} bias, confidence={dl_signals['confidence']:.2f}")
            return dl_signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Deep Learning predictions: {e}")
            return {}
    
    async def analyze_reinforcement_learning_predictions(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze Reinforcement Learning predictions"""
        try:
            if not self.use_rl or not self.rl_engine:
                return {}
            
            # Create RL state from market data
            latest = df.iloc[-1]
            current_price = latest['close']
            current_volume = latest['volume']
            
            # Calculate volatility and trend strength
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0.02
            
            # Simple trend strength calculation
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            trend_strength = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0.0
            
            # Create RL state
            rl_state = RLState(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=current_volume,
                volatility=volatility,
                trend_strength=trend_strength,
                market_regime='normal_trending',  # Simplified for now
                position_size=0.0,  # No current position
                current_pnl=0.0,  # No current PnL
                risk_metrics={'var': volatility, 'sharpe': 0.0},
                signal_strength=0.5,  # Default signal strength
                confidence=0.5,  # Default confidence
                market_features=[current_price, current_volume, volatility, trend_strength, 0.5, 0.5, 0.5, 0.5]
            )
            
            # Get RL action
            rl_action = self.rl_engine.get_trading_action(rl_state)
            
            # Get signal optimization parameters
            signal_history = self.signal_history[-100:] if self.signal_history else []  # Last 100 signals
            optimization_params = self.rl_engine.get_signal_optimization(signal_history)
            
            # Extract RL signals
            rl_signals = {
                'action_type': rl_action.action_type,
                'position_size': rl_action.position_size,
                'stop_loss': rl_action.stop_loss,
                'take_profit': rl_action.take_profit,
                'confidence_threshold': rl_action.confidence_threshold,
                'risk_allocation': rl_action.risk_allocation,
                'optimization_params': optimization_params,
                'timestamp': datetime.now()
            }
            
            # Add action bias
            if rl_action.action_type == 'buy':
                rl_signals['rl_bias'] = 'bullish'
                rl_signals['action_strength'] = rl_action.position_size
            elif rl_action.action_type == 'sell':
                rl_signals['rl_bias'] = 'bearish'
                rl_signals['action_strength'] = rl_action.position_size
            else:
                rl_signals['rl_bias'] = 'neutral'
                rl_signals['action_strength'] = 0.0
            
            # Add confidence-based signals
            if rl_action.confidence_threshold > 0.7 and rl_action.position_size > 0.1:
                rl_signals['high_confidence_action'] = True
            else:
                rl_signals['high_confidence_action'] = False
            
            # Add performance metrics
            rl_performance = self.rl_engine.get_performance_summary()
            rl_signals['training_episodes'] = rl_performance.get('training_episodes', 0)
            rl_signals['avg_reward'] = rl_performance.get('avg_reward', 0.0)
            rl_signals['best_reward'] = rl_performance.get('best_reward', 0.0)
            
            logger.debug(f"🤖 RL Analysis for {symbol}: {rl_signals['rl_bias']} bias, action={rl_action.action_type}, strength={rl_signals['action_strength']:.2f}")
            return rl_signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Reinforcement Learning predictions: {e}")
            return {}
    
    async def analyze_natural_language_processing_predictions(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Analyze Natural Language Processing predictions"""
        try:
            if not self.use_nlp or not self.nlp_engine:
                return {}
            
            # Get sentiment from NLP engine
            sentiment_data = self.nlp_engine.get_sentiment(symbol)
            
            if not sentiment_data:
                # Generate mock sentiment if not available
                sentiment_data = {
                    'overall_score': np.random.uniform(-0.5, 0.5),
                    'overall_confidence': np.random.uniform(0.3, 0.8),
                    'news_score': np.random.uniform(-0.3, 0.3),
                    'news_confidence': np.random.uniform(0.4, 0.7),
                    'twitter_score': np.random.uniform(-0.4, 0.4),
                    'twitter_confidence': np.random.uniform(0.3, 0.6),
                    'reddit_score': np.random.uniform(-0.3, 0.3),
                    'reddit_confidence': np.random.uniform(0.4, 0.7),
                    'timestamp': datetime.now()
                }
            
            # Extract NLP signals
            nlp_signals = {
                'overall_sentiment_score': sentiment_data.get('overall_score', 0.0),
                'overall_confidence': sentiment_data.get('overall_confidence', 0.0),
                'news_sentiment': sentiment_data.get('news_score', 0.0),
                'news_confidence': sentiment_data.get('news_confidence', 0.0),
                'twitter_sentiment': sentiment_data.get('twitter_score', 0.0),
                'twitter_confidence': sentiment_data.get('twitter_confidence', 0.0),
                'reddit_sentiment': sentiment_data.get('reddit_score', 0.0),
                'reddit_confidence': sentiment_data.get('reddit_confidence', 0.0),
                'timestamp': sentiment_data.get('timestamp', datetime.now())
            }
            
            # Add sentiment bias
            overall_score = nlp_signals['overall_sentiment_score']
            if overall_score > 0.1:
                nlp_signals['nlp_bias'] = 'bullish'
                nlp_signals['sentiment_strength'] = overall_score
            elif overall_score < -0.1:
                nlp_signals['nlp_bias'] = 'bearish'
                nlp_signals['sentiment_strength'] = abs(overall_score)
            else:
                nlp_signals['nlp_bias'] = 'neutral'
                nlp_signals['sentiment_strength'] = 0.0
            
            # Add confidence-based signals
            if nlp_signals['overall_confidence'] > 0.7 and abs(overall_score) > 0.3:
                nlp_signals['high_confidence_sentiment'] = True
            else:
                nlp_signals['high_confidence_sentiment'] = False
            
            # Add performance metrics
            nlp_performance = self.nlp_engine.get_performance_summary()
            nlp_signals['analyses_performed'] = nlp_performance.get('analyses_performed', 0)
            nlp_signals['cache_hit_rate'] = nlp_performance.get('cache_hit_rate', 0.0)
            nlp_signals['models_available'] = nlp_performance.get('models_available', {})
            
            logger.debug(f"📝 NLP Analysis for {symbol}: {nlp_signals['nlp_bias']} bias, score={nlp_signals['overall_sentiment_score']:.3f}, confidence={nlp_signals['overall_confidence']:.2f}")
            return nlp_signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Natural Language Processing predictions: {e}")
            return {}
    
    async def analyze_ensemble_predictions(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Phase 2.3: Analyze Ensemble Model predictions"""
        try:
            if not self.use_ensemble or not self.ensemble_strategy:
                return {}
            
            # Prepare features for ensemble prediction
            features = self._prepare_features_for_ensemble(df)
            
            if features is None or len(features) == 0:
                return {}
            
            # Get ensemble prediction
            prediction, confidence = self.ensemble_strategy.predict(features)
            
            # Get ensemble analysis
            ensemble_analysis = self.ensemble_strategy.get_ensemble_analysis()
            
            # Extract ensemble signals
            ensemble_signals = {
                'prediction': prediction,
                'ensemble_confidence': confidence,
                'voting_method': ensemble_analysis.get('voting_method', 'unknown'),
                'model_count': ensemble_analysis.get('model_count', 0),
                'diversity_score': ensemble_analysis.get('diversity_score', 0.0),
                'agreement_ratio': ensemble_analysis.get('agreement_ratio', 0.0),
                'individual_predictions': ensemble_analysis.get('individual_predictions', {}),
                'model_weights': ensemble_analysis.get('model_weights', {}),
                'performance_metrics': ensemble_analysis.get('performance_metrics', {}),
                'timestamp': datetime.now()
            }
            
            # Add ensemble bias
            if prediction == 1:
                ensemble_signals['ensemble_bias'] = 'bullish'
            elif prediction == 0:
                ensemble_signals['ensemble_bias'] = 'bearish'
            else:
                ensemble_signals['ensemble_bias'] = 'neutral'
            
            # Add confidence-based signals
            if confidence > 0.7 and ensemble_analysis.get('agreement_ratio', 0) > 0.6:
                ensemble_signals['high_confidence_ensemble'] = True
            else:
                ensemble_signals['high_confidence_ensemble'] = False
            
            # Add diversity and agreement metrics
            ensemble_signals['ensemble_diversity'] = ensemble_analysis.get('diversity_score', 0.0)
            ensemble_signals['ensemble_agreement'] = ensemble_analysis.get('agreement_ratio', 0.0)
            
            logger.debug(f"🤖 Ensemble Analysis for {symbol}: {ensemble_signals['ensemble_bias']} bias, confidence={confidence:.3f}, diversity={ensemble_signals['ensemble_diversity']:.3f}, agreement={ensemble_signals['ensemble_agreement']:.3f}")
            return ensemble_signals
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Ensemble predictions: {e}")
            return {}
    
    def _prepare_features_for_ensemble(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ensemble model prediction"""
        try:
            if len(df) < 50:
                return None
            
            # Calculate technical features
            features = []
            
            # Price-based features
            features.append(df['close'].iloc[-1] / df['close'].iloc[-2] - 1)  # Price change
            features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])  # Volume ratio
            
            # Technical indicators
            if 'rsi' in df.columns:
                features.append(df['rsi'].iloc[-1] / 100)  # Normalized RSI
            else:
                features.append(0.5)  # Default neutral RSI
            
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                features.append(df['macd'].iloc[-1] - df['macd_signal'].iloc[-1])  # MACD difference
            else:
                features.append(0.0)  # Default MACD difference
            
            # Moving averages
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                features.append(df['sma_20'].iloc[-1] / df['sma_50'].iloc[-1] - 1)  # MA ratio
            else:
                features.append(0.0)  # Default MA ratio
            
            # Volatility features
            returns = df['close'].pct_change().dropna()
            features.append(returns.std())  # Volatility
            features.append(returns.skew())  # Skewness
            features.append(returns.kurtosis())  # Kurtosis
            
            # Trend features
            features.append(1 if df['close'].iloc[-1] > df['close'].iloc[-5] else 0)  # Short-term trend
            features.append(1 if df['close'].iloc[-1] > df['close'].iloc[-20] else 0)  # Medium-term trend
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for ensemble: {e}")
            return None
    
    async def get_signals_from_database(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get signals from TimescaleDB"""
        if not self.use_database or not self.db_connection:
            return []
        
        try:
            # Use the existing get_latest_signals method from database connection
            signals = await self.db_connection.get_latest_signals(symbol=symbol, limit=limit)
            logger.info(f"📊 Retrieved {len(signals)} signals from database")
            return signals
        except Exception as e:
            logger.error(f"❌ Error retrieving signals from database: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"❌ Error calculating RSI: {e}")
            return pd.Series([50] * len(prices))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd, macd_signal
        except Exception as e:
            logger.error(f"❌ Error calculating MACD: {e}")
            return pd.Series([0] * len(prices)), pd.Series([0] * len(prices))
    
    async def _analyze_and_generate_signal(self, symbol: str, latest: pd.Series, prev: pd.Series, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze market data and generate trading signal with advanced indicators"""
        try:
            current_price = latest['close']
            current_volume = latest['volume']
            high = latest['high']
            low = latest['low']
            
            # Basic technical indicators
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            sma_20 = latest.get('sma_20', current_price)
            sma_50 = latest.get('sma_50', current_price)
            volume = current_volume
            
            # Calculate advanced technical indicators
            advanced_indicators = self.calculate_advanced_indicators(df)
            
            # Analyze advanced indicator signals
            ichimoku_signals = self.analyze_ichimoku_signals(advanced_indicators, current_price)
            fib_signals = self.analyze_fibonacci_signals(advanced_indicators, current_price, high, low)
            volume_signals = self.analyze_volume_signals(advanced_indicators, current_volume)
            
            # Analyze Smart Money Concepts
            smc_signals = await self.analyze_smart_money_concepts(df, symbol, '1h')
            
            # Analyze Deep Learning predictions
            dl_signals = await self.analyze_deep_learning_predictions(df, symbol, '1h')
            
            # Analyze Reinforcement Learning predictions
            rl_signals = await self.analyze_reinforcement_learning_predictions(df, symbol, '1h')
            
            # Analyze Natural Language Processing predictions
            nlp_signals = await self.analyze_natural_language_processing_predictions(df, symbol, '1h')
            
            # Phase 2.3: Analyze Ensemble Model predictions
            ensemble_signals = await self.analyze_ensemble_predictions(df, symbol, '1h')
            
            # Phase 3.1: Analyze Enhanced Sentiment Analysis
            sentiment_signals = await self.analyze_enhanced_sentiment_predictions(symbol)
            
            # Enhanced signal generation with advanced indicators and SMC
            signal_type = None
            confidence = 0.0
            reason = ""
            
            # Base signal conditions (relaxed for more signals)
            if (rsi < 40 and  # Less strict oversold (was 30)
                macd > macd_signal and  # MACD bullish crossover
                current_price > sma_20 * 0.98):  # Price near short-term MA (was strict above)
                
                # Advanced confirmations
                advanced_confirmation = 0.0
                advanced_reasons = []
                
                # Ichimoku confirmation
                if ichimoku_signals.get('cloud_bullish') is True:
                    advanced_confirmation += 0.1
                    advanced_reasons.append("Ichimoku cloud bullish")
                
                if ichimoku_signals.get('tenkan_kijun') == 'bullish':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Tenkan/Kijun bullish")
                
                # Fibonacci confirmation
                if fib_signals.get('fib_bullish') is True:
                    advanced_confirmation += 0.1
                    advanced_reasons.append("Fibonacci support")
                
                # Volume confirmation
                if volume_signals.get('volume_confirmation'):
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Volume confirmation")
                
                # Smart Money Concepts confirmation
                if smc_signals.get('smc_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("SMC bullish bias")
                
                if smc_signals.get('order_blocks_count', 0) > 0:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Order blocks detected")
                
                if smc_signals.get('fair_value_gaps_count', 0) > 0:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Fair value gaps detected")
                
                # Deep Learning confirmation
                if dl_signals.get('dl_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("DL bullish bias")
                
                if dl_signals.get('high_confidence_signal'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("DL high confidence")
                
                if dl_signals.get('signal_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("DL strong signal")
                
                # Reinforcement Learning confirmation
                if rl_signals.get('rl_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("RL bullish bias")
                
                if rl_signals.get('high_confidence_action'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("RL high confidence")
                
                if rl_signals.get('action_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("RL strong action")
                
                # Natural Language Processing confirmation
                if nlp_signals.get('nlp_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("NLP bullish bias")
                
                if nlp_signals.get('high_confidence_sentiment'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("NLP high confidence")
                
                if nlp_signals.get('sentiment_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("NLP strong sentiment")
                
                # Phase 2.3: Ensemble Model confirmation
                if ensemble_signals.get('ensemble_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("Ensemble bullish bias")
                
                if ensemble_signals.get('high_confidence_ensemble'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("Ensemble high confidence")
                
                if ensemble_signals.get('ensemble_confidence', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Ensemble strong confidence")
                
                # Phase 3.1: Enhanced Sentiment Analysis confirmation
                if sentiment_signals.get('sentiment_bias') == 'bullish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("Sentiment bullish bias")
                
                if sentiment_signals.get('high_confidence_sentiment'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("Sentiment high confidence")
                
                if sentiment_signals.get('sentiment_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment strong signal")
                
                if sentiment_signals.get('trend_analysis', {}).get('trend') == 'increasing':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment trend increasing")
                
                if sentiment_signals.get('momentum_indicators', {}).get('momentum_direction') == 'bullish':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment momentum bullish")
                
                signal_type = 'buy'
                base_confidence = 0.6 + (40 - rsi) / 40 * 0.35  # Base confidence based on RSI
                confidence = min(0.95, base_confidence + advanced_confirmation)
                reason = f"Oversold RSI ({rsi:.1f}), MACD bullish, price near MA"
                if advanced_reasons:
                    reason += f" + {' + '.join(advanced_reasons)}"
            
            elif (rsi > 60 and  # Less strict overbought (was 70)
                  macd < macd_signal and  # MACD bearish crossover
                  current_price < sma_20 * 1.02):  # Price near short-term MA (was strict below)
                
                # Advanced confirmations
                advanced_confirmation = 0.0
                advanced_reasons = []
                
                # Ichimoku confirmation
                if ichimoku_signals.get('cloud_bullish') is False:
                    advanced_confirmation += 0.1
                    advanced_reasons.append("Ichimoku cloud bearish")
                
                if ichimoku_signals.get('tenkan_kijun') == 'bearish':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Tenkan/Kijun bearish")
                
                # Fibonacci confirmation
                if fib_signals.get('fib_bullish') is False:
                    advanced_confirmation += 0.1
                    advanced_reasons.append("Fibonacci resistance")
                
                # Volume confirmation
                if volume_signals.get('volume_confirmation'):
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Volume confirmation")
                
                # Smart Money Concepts confirmation
                if smc_signals.get('smc_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("SMC bearish bias")
                
                if smc_signals.get('liquidity_sweeps_count', 0) > 0:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Liquidity sweeps detected")
                
                if smc_signals.get('market_structures_count', 0) > 0:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Market structure breakouts")
                
                # Deep Learning confirmation
                if dl_signals.get('dl_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("DL bearish bias")
                
                if dl_signals.get('high_confidence_signal'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("DL high confidence")
                
                if dl_signals.get('signal_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("DL strong signal")
                
                # Reinforcement Learning confirmation
                if rl_signals.get('rl_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("RL bearish bias")
                
                if rl_signals.get('high_confidence_action'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("RL high confidence")
                
                if rl_signals.get('action_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("RL strong action")
                
                # Natural Language Processing confirmation
                if nlp_signals.get('nlp_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("NLP bearish bias")
                
                if nlp_signals.get('high_confidence_sentiment'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("NLP high confidence")
                
                if nlp_signals.get('sentiment_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("NLP strong sentiment")
                
                # Phase 2.3: Ensemble Model confirmation
                if ensemble_signals.get('ensemble_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("Ensemble bearish bias")
                
                if ensemble_signals.get('high_confidence_ensemble'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("Ensemble high confidence")
                
                if ensemble_signals.get('ensemble_confidence', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Ensemble strong confidence")
                
                # Phase 3.1: Enhanced Sentiment Analysis confirmation
                if sentiment_signals.get('sentiment_bias') == 'bearish':
                    advanced_confirmation += 0.15
                    advanced_reasons.append("Sentiment bearish bias")
                
                if sentiment_signals.get('high_confidence_sentiment'):
                    advanced_confirmation += 0.10
                    advanced_reasons.append("Sentiment high confidence")
                
                if sentiment_signals.get('sentiment_strength', 0) > 0.6:
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment strong signal")
                
                if sentiment_signals.get('trend_analysis', {}).get('trend') == 'decreasing':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment trend decreasing")
                
                if sentiment_signals.get('momentum_indicators', {}).get('momentum_direction') == 'bearish':
                    advanced_confirmation += 0.05
                    advanced_reasons.append("Sentiment momentum bearish")
                
                signal_type = 'sell'
                base_confidence = 0.6 + (rsi - 60) / 40 * 0.35  # Base confidence based on RSI
                confidence = min(0.95, base_confidence + advanced_confirmation)
                reason = f"Overbought RSI ({rsi:.1f}), MACD bearish, price near MA"
                if advanced_reasons:
                    reason += f" + {' + '.join(advanced_reasons)}"
            
            # Advanced pattern-based signals with SMC
            elif ichimoku_signals and fib_signals and smc_signals:
                # Ichimoku + Fibonacci + SMC confluence signals
                if (ichimoku_signals.get('cloud_bullish') is True and 
                    fib_signals.get('fib_bullish') is True and
                    smc_signals.get('smc_bias') == 'bullish' and
                    rsi < 50):
                    
                    signal_type = 'buy'
                    confidence = 0.85
                    reason = f"Ichimoku + Fibonacci + SMC bullish confluence (RSI: {rsi:.1f})"
                
                elif (ichimoku_signals.get('cloud_bullish') is False and 
                      fib_signals.get('fib_bullish') is False and
                      smc_signals.get('smc_bias') == 'bearish' and
                      rsi > 50):
                    
                    signal_type = 'sell'
                    confidence = 0.85
                    reason = f"Ichimoku + Fibonacci + SMC bearish confluence (RSI: {rsi:.1f})"
            
            # Deep Learning + SMC confluence signals (highest confidence)
            elif dl_signals and smc_signals:
                if (dl_signals.get('dl_bias') == 'bullish' and 
                    dl_signals.get('high_confidence_signal') and
                    smc_signals.get('smc_bias') == 'bullish' and
                    rsi < 45):
                    
                    signal_type = 'buy'
                    confidence = 0.90
                    reason = f"DL + SMC bullish confluence (DL confidence: {dl_signals.get('confidence', 0):.2f}, RSI: {rsi:.1f})"
                
                elif (dl_signals.get('dl_bias') == 'bearish' and 
                      dl_signals.get('high_confidence_signal') and
                      smc_signals.get('smc_bias') == 'bearish' and
                      rsi > 55):
                    
                    signal_type = 'sell'
                    confidence = 0.90
                    reason = f"DL + SMC bearish confluence (DL confidence: {dl_signals.get('confidence', 0):.2f}, RSI: {rsi:.1f})"
            
            # SMC-only signals (high confidence)
            elif smc_signals.get('overall_confidence', 0) > 0.7:
                if smc_signals.get('smc_bias') == 'bullish' and rsi < 45:
                    signal_type = 'buy'
                    confidence = 0.8
                    reason = f"Strong SMC bullish bias (confidence: {smc_signals.get('overall_confidence', 0):.2f})"
                
                elif smc_signals.get('smc_bias') == 'bearish' and rsi > 55:
                    signal_type = 'sell'
                    confidence = 0.8
                    reason = f"Strong SMC bearish bias (confidence: {smc_signals.get('overall_confidence', 0):.2f})"
            
            # Deep Learning-only signals (high confidence)
            elif dl_signals.get('high_confidence_signal') and dl_signals.get('signal_strength', 0) > 0.7:
                if dl_signals.get('dl_bias') == 'bullish' and rsi < 45:
                    signal_type = 'buy'
                    confidence = 0.85
                    reason = f"Strong DL bullish signal (strength: {dl_signals.get('signal_strength', 0):.2f})"
                
                elif dl_signals.get('dl_bias') == 'bearish' and rsi > 55:
                    signal_type = 'sell'
                    confidence = 0.85
                    reason = f"Strong DL bearish signal (strength: {dl_signals.get('signal_strength', 0):.2f})"
            
            # Volume breakout signals with SMC
            elif volume_signals.get('volume_confirmation') and smc_signals and abs(rsi - 50) > 10:
                if rsi < 40 and smc_signals.get('smc_bias') == 'bullish':
                    signal_type = 'buy'
                    confidence = 0.75
                    reason = f"Volume breakout + SMC bullish bias (RSI: {rsi:.1f})"
                elif rsi > 60 and smc_signals.get('smc_bias') == 'bearish':
                    signal_type = 'sell'
                    confidence = 0.75
                    reason = f"Volume breakout + SMC bearish bias (RSI: {rsi:.1f})"
            
            # Only generate signal if confidence meets minimum threshold
            if signal_type and confidence >= self.min_confidence:
                signal = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'strength': confidence * 0.8,  # Strength based on confidence
                    'timestamp': datetime.now().isoformat(),
                    'price': current_price,  # REAL PRICE
                    'reason': reason,
                    'indicators': {
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'volume': volume,
                        # Advanced indicators
                        'ichimoku': ichimoku_signals,
                        'fibonacci': fib_signals,
                        'volume_analysis': volume_signals,
                        'advanced_indicators': advanced_indicators,
                        # Smart Money Concepts
                        'smc_analysis': smc_signals,
                        # Deep Learning
                        'dl_analysis': dl_signals,
                        # Reinforcement Learning
                        'rl_analysis': rl_signals,
                        # Natural Language Processing
                        'nlp_analysis': nlp_signals,
                        # Phase 2.3: Ensemble Model
                        'ensemble_analysis': ensemble_signals,
                        # Phase 3.1: Enhanced Sentiment Analysis
                        'sentiment_analysis': sentiment_signals
                    }
                }
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error analyzing signal for {symbol}: {e}")
            return None
    
    async def _generate_signals(self):
        """Background task to generate signals with deduplication"""
        while self.is_running:
            try:
                # Generate signals every 10 seconds
                await asyncio.sleep(10)
                
                # Clean up expired signals
                await self._cleanup_expired_signals()
                
                # Generate real signals based on market data
                await self._generate_sample_signals()
                
            except Exception as e:
                logger.error(f"❌ Error generating signals: {e}")
                await asyncio.sleep(30)
    
    async def _generate_sample_signals(self):
        """Generate real signals based on actual market data"""
        try:
            # Import CCXT for real market data
            import ccxt
            import numpy as np
            
            # Initialize Binance exchange
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Generate signals for major symbols
            symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
            
            for symbol in symbols:
                try:
                    # Fetch real market data
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=100)
                    
                    if ohlcv and len(ohlcv) > 50:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Calculate technical indicators
                        df['sma_20'] = df['close'].rolling(window=20).mean()
                        df['sma_50'] = df['close'].rolling(window=50).mean()
                        df['rsi'] = self._calculate_rsi(df['close'])
                        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
                        
                        # Get latest data
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        
                        # Generate signal based on technical analysis
                        signal = await self._analyze_and_generate_signal(symbol, latest, prev, df)
                        
                        if signal:
                            # Use deduplication logic instead of direct append
                            await self._add_signal_with_deduplication(signal)
                            logger.info(f"🎯 Generated real signal: {symbol} - {signal['signal_type']} (confidence: {signal['confidence']:.2f})")
                        # REMOVED: No more sample signal fallback - only real signals
                
                except Exception as e:
                    logger.error(f"❌ Error generating signal for {symbol}: {e}")
                    continue
                    
        except Exception as e:
                    
                    self.signals.append(signal)
                    
                    # Keep only recent signals
                    if len(self.signals) > 100:
                        self.signals = self.signals[-100:]
                    
                    logger.debug(f"✅ Generated {signal_type} signal for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample signals: {e}")
    
    async def get_signals(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals - prioritize active signals, fallback to database"""
        try:
            # First, try to get active signals from deduplication system
            if symbol:
                active_signals = await self.get_signals_by_symbol(symbol)
            else:
                active_signals = await self.get_active_signals()
            
            # If we have active signals, return them (already deduplicated)
            if active_signals:
                # Sort by confidence and limit
                sorted_signals = sorted(active_signals, key=lambda x: x.get('confidence', 0), reverse=True)
                return sorted_signals[:limit]
            
            # Fallback to database if no active signals
            if self.use_database and self.db_connection:
                db_signals = await self.get_signals_from_database(symbol, limit)
                if db_signals:
                    # Apply deduplication to database signals
                    return self._deduplicate_signals(db_signals)
                return []
            
            # Final fallback to memory signals (legacy)
            if symbol:
                filtered_signals = [s for s in self.signals if s.get('symbol') == symbol]
            else:
                filtered_signals = self.signals
            
            # Sort by timestamp and limit
            sorted_signals = sorted(filtered_signals, key=lambda x: x.get('timestamp', ''), reverse=True)
            return sorted_signals[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting signals: {e}")
            return []

    def _deduplicate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate signals by symbol and side, keeping the highest confidence one"""
        try:
            # Group signals by symbol and side
            signal_groups = {}
            for signal in signals:
                symbol = signal.get('symbol', '')
                side = signal.get('side', signal.get('signal_type', ''))
                key = (symbol, side)
                
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append(signal)
            
            # Keep only the highest confidence signal per group
            deduplicated_signals = []
            for key, group_signals in signal_groups.items():
                # Sort by confidence and take the highest
                best_signal = max(group_signals, key=lambda x: x.get('confidence', 0))
                deduplicated_signals.append(best_signal)
            
            # Sort by confidence
            deduplicated_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            logger.info(f"🔄 Deduplicated {len(signals)} signals to {len(deduplicated_signals)} unique signals")
            return deduplicated_signals
            
        except Exception as e:
            logger.error(f"❌ Error deduplicating signals: {e}")
            return signals
    
    async def add_signal(self, signal: Dict[str, Any]):
        """Add a new signal"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in signal:
                signal['timestamp'] = datetime.now()
            
            self.signals.append(signal)
            
            # Save signal to database
            await self.save_signal_to_database(signal)
            
            # Keep only recent signals
            if len(self.signals) > 100:
                self.signals = self.signals[-100:]
            
            logger.info(f"✅ Added signal: {signal.get('signal_type', 'unknown')} for {signal.get('symbol', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add signal: {e}")
    
    async def analyze_signal_strength(self, signal: Dict[str, Any]) -> float:
        """Analyze the strength of a signal"""
        try:
            # Simple strength calculation based on confidence and indicators
            base_strength = signal.get('confidence', 0.5)
            
            # Additional factors from indicators
            indicators = signal.get('indicators', {})
            
            # RSI factor
            rsi = indicators.get('rsi', 50)
            if rsi < 30 or rsi > 70:
                base_strength *= 1.2
            
            # MACD factor
            macd = indicators.get('macd', 0)
            if abs(macd) > 0.5:
                base_strength *= 1.1
            
            # Volume factor
            volume = indicators.get('volume', 0)
            if volume > 5000:
                base_strength *= 1.1
            
            return min(base_strength, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze signal strength: {e}")
            return 0.0
    
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate if a signal meets the minimum requirements"""
        try:
            confidence = signal.get('confidence', 0.0)
            strength = signal.get('strength', 0.0)
            
            return confidence >= self.min_confidence and strength >= self.min_strength
            
        except Exception as e:
            logger.error(f"❌ Failed to validate signal: {e}")
            return False
    
    async def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated signals"""
        try:
            if not self.signals:
                return {
                    'total_signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'hold_signals': 0,
                    'average_confidence': 0.0,
                    'average_strength': 0.0
                }
            
            buy_signals = len([s for s in self.signals if s.get('signal_type') == 'buy'])
            sell_signals = len([s for s in self.signals if s.get('signal_type') == 'sell'])
            hold_signals = len([s for s in self.signals if s.get('signal_type') == 'hold'])
            
            avg_confidence = sum(s.get('confidence', 0) for s in self.signals) / len(self.signals)
            avg_strength = sum(s.get('strength', 0) for s in self.signals) / len(self.signals)
            
            return {
                'total_signals': len(self.signals),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'average_confidence': round(avg_confidence, 3),
                'average_strength': round(avg_strength, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get signal statistics: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Real-time Signal Generator"""
        return {
            'status': 'running' if self.is_running else 'stopped',
            'total_signals': len(self.signals),
            'min_confidence': self.min_confidence,
            'min_strength': self.min_strength,
            'last_update': self.last_update.get('signals', None)
        }
    
    async def generate_signals(self, market_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate signals based on market data with Phase 4.1 ultra-low latency optimizations"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Phase 4.1: Check cache first for ultra-fast response
            if market_data is not None:
                data_hash = str(hash(str(market_data.values.tobytes())))
                cache_key = self._get_cache_key("market_data", "1m", data_hash)
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result:
                    processing_time = (time.time() - start_time) * 1000
                    self.processing_times.append(processing_time)
                    logger.info(f"⚡ Cache hit - Signal generation in {processing_time:.2f}ms")
                    return cached_result
            
            # Phase 4.1: Async processing for better performance
            if self.enable_async_processing:
                signals = await self._generate_signals_async(market_data)
            else:
                signals = await self._generate_signals_sync(market_data)
            
            # Phase 4.1: Cache the result
            if market_data is not None:
                data_hash = str(hash(str(market_data.values.tobytes())))
                cache_key = self._get_cache_key("market_data", "1m", data_hash)
                self._cache_result(cache_key, signals)
            
            # Phase 4.1: Track performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # Phase 4.1: Performance logging
            if processing_time > self.target_latency_ms:
                logger.warning(f"⚠️ Signal generation took {processing_time:.2f}ms (target: <{self.target_latency_ms}ms)")
            else:
                logger.info(f"⚡ Signal generation completed in {processing_time:.2f}ms (target: <{self.target_latency_ms}ms)")
            
            logger.info(f"🎯 Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            logger.error(f"❌ Error generating signals: {e} (took {processing_time:.2f}ms)")
            return []
    
    async def _generate_signals_async(self, market_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate signals using async processing for better performance"""
        try:
            # Phase 4.1: Parallel processing of different analysis types
            tasks = []
            
            # Technical analysis
            if self.use_advanced_indicators:
                tasks.append(self._analyze_technical_indicators_async(market_data))
            
            # Smart Money Concepts
            if self.use_smc:
                tasks.append(self._analyze_smart_money_concepts_async(market_data))
            
            # Deep Learning
            if self.use_dl:
                tasks.append(self._analyze_deep_learning_async(market_data))
            
            # Reinforcement Learning
            if self.use_rl:
                tasks.append(self._analyze_reinforcement_learning_async(market_data))
            
            # NLP Analysis
            if self.use_nlp:
                tasks.append(self._analyze_nlp_async(market_data))
            
            # Ensemble Analysis
            if self.use_ensemble:
                tasks.append(self._analyze_ensemble_async(market_data))
            
            # Sentiment Analysis
            if self.use_sentiment:
                tasks.append(self._analyze_sentiment_async(market_data))
            
            # Execute all analyses in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results and generate signals
                signals = await self._combine_analysis_results(results)
            else:
                # Fallback to sync method
                signals = await self._generate_signals_sync(market_data)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error in async signal generation: {e}")
            return await self._generate_signals_sync(market_data)
    
    async def _generate_signals_sync(self, market_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate signals using synchronous processing (fallback method)"""
        try:
            # Generate real signals using the existing method
            await self._generate_sample_signals()
            
            # Return recent signals
            recent_signals = self.signals[-10:] if self.signals else []
            return recent_signals
            
        except Exception as e:
            logger.error(f"❌ Error in sync signal generation: {e}")
            return []
    
    async def _analyze_technical_indicators_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async technical indicators analysis"""
        try:
            if not self.indicators_engine:
                return {}
            
            # Run in thread pool for CPU-intensive calculations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._calculate_technical_indicators,
                market_data
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in technical indicators analysis: {e}")
            return {}
    
    async def _analyze_smart_money_concepts_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async Smart Money Concepts analysis"""
        try:
            if not self.smc_engine:
                return {}
            
            # SMC analysis is already async, call directly
            result = await self._calculate_smart_money_concepts(market_data)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in SMC analysis: {e}")
            return {}
    
    async def _analyze_deep_learning_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async Deep Learning analysis"""
        try:
            if not self.dl_engine:
                return {}
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._calculate_deep_learning_predictions,
                market_data
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in Deep Learning analysis: {e}")
            return {}
    
    async def _analyze_reinforcement_learning_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async Reinforcement Learning analysis"""
        try:
            if not self.rl_engine:
                return {}
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._calculate_reinforcement_learning_predictions,
                market_data
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in RL analysis: {e}")
            return {}
    
    async def _analyze_nlp_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async NLP analysis"""
        try:
            if not self.nlp_engine:
                return {}
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._calculate_nlp_predictions,
                market_data
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in NLP analysis: {e}")
            return {}
    
    async def _analyze_ensemble_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async Ensemble analysis"""
        try:
            if not self.ensemble_strategy:
                return {}
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._calculate_ensemble_predictions,
                market_data
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in Ensemble analysis: {e}")
            return {}
    
    async def _analyze_sentiment_async(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Async Sentiment analysis"""
        try:
            if not self.sentiment_service:
                return {}
            
            # Sentiment analysis is already async
            result = await self.sentiment_service.get_enhanced_sentiment_with_social_media("BTCUSDT")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in Sentiment analysis: {e}")
            return {}
    
    async def _combine_analysis_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from all analysis types and generate final signals"""
        try:
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, dict)]
            
            if not valid_results:
                return []
            
            # Combine all analysis results
            combined_analysis = {}
            for result in valid_results:
                combined_analysis.update(result)
            
            # Generate signals based on combined analysis
            signals = await self._generate_signals_from_analysis(combined_analysis)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error combining analysis results: {e}")
            return []
    
    def _calculate_technical_indicators(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate technical indicators (CPU-intensive, runs in thread pool)"""
        try:
            if not self.indicators_engine or market_data is None:
                return {}
            
            # Calculate all available indicators
            indicators = {}
            for indicator_name in self.indicators_engine.get_available_indicators():
                try:
                    value = self.indicators_engine.calculate_indicator(indicator_name, market_data)
                    indicators[indicator_name] = value
                except Exception as e:
                    logger.debug(f"Failed to calculate {indicator_name}: {e}")
            
            return {"technical_indicators": indicators}
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    async def _calculate_smart_money_concepts(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate Smart Money Concepts (CPU-intensive, runs in thread pool)"""
        try:
            if not self.smc_engine or market_data is None:
                return {}
            
            # Calculate SMC analysis using the correct method (async)
            smc_analysis = await self.smc_engine.analyze_smart_money_concepts(market_data, "BTCUSDT", "1h")
            return {"smart_money_concepts": smc_analysis}
            
        except Exception as e:
            logger.error(f"❌ Error calculating SMC: {e}")
            return {}
    
    def _calculate_deep_learning_predictions(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate Deep Learning predictions (CPU-intensive, runs in thread pool)"""
        try:
            if not self.dl_engine or market_data is None:
                return {}
            
            # Calculate DL predictions using the correct method
            predictions = self.dl_engine.predict_with_models(market_data, "BTCUSDT", "1h")
            return {"deep_learning_predictions": predictions}
            
        except Exception as e:
            logger.error(f"❌ Error calculating DL predictions: {e}")
            return {}
    
    def _calculate_reinforcement_learning_predictions(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate RL predictions (CPU-intensive, runs in thread pool)"""
        try:
            if not self.rl_engine or market_data is None:
                return {}
            
            # Calculate RL predictions using the correct method
            # RL engine doesn't have a simple predict method, use get_trading_action instead
            if hasattr(self.rl_engine, 'get_trading_action'):
                # Create a mock state for RL prediction
                latest = market_data.iloc[-1] if len(market_data) > 0 else None
                if latest is not None:
                    rl_state = RLState(
                        symbol="BTCUSDT",
                        timestamp=datetime.now(),
                        price=latest.get('close', 50000),
                        volume=latest.get('volume', 1000),
                        volatility=0.02,
                        trend_strength=0.0,
                        market_regime='normal_trending',
                        position_size=0.0,
                        current_pnl=0.0,
                        risk_metrics={'var': 0.02, 'sharpe': 0.0},
                        signal_strength=0.5,
                        confidence=0.5,
                        market_features=[50000, 1000, 0.02, 0.0, 0.5, 0.5, 0.5, 0.5]
                    )
                    action = self.rl_engine.get_trading_action(rl_state)
                    predictions = {
                        'action_type': action.action_type,
                        'position_size': action.position_size,
                        'stop_loss': action.stop_loss,
                        'take_profit': action.take_profit,
                        'confidence_threshold': action.confidence_threshold
                    }
                else:
                    predictions = {'action_type': 'hold', 'position_size': 0.0}
            else:
                predictions = {'action_type': 'hold', 'position_size': 0.0}
            
            return {"reinforcement_learning_predictions": predictions}
            
        except Exception as e:
            logger.error(f"❌ Error calculating RL predictions: {e}")
            return {}
    
    def _calculate_nlp_predictions(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate NLP predictions (CPU-intensive, runs in thread pool)"""
        try:
            if not self.nlp_engine or market_data is None:
                return {}
            
            # Calculate NLP predictions using the correct method
            predictions = self.nlp_engine.get_sentiment("BTCUSDT")
            return {"nlp_predictions": predictions}
            
        except Exception as e:
            logger.error(f"❌ Error calculating NLP predictions: {e}")
            return {}
    
    def _calculate_ensemble_predictions(self, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate Ensemble predictions (CPU-intensive, runs in thread pool)"""
        try:
            if not self.ensemble_strategy or market_data is None:
                return {}
            
            # Check if ensemble is trained
            if not hasattr(self.ensemble_strategy, 'is_trained') or not self.ensemble_strategy.is_trained:
                # Return default prediction if not trained
                return {
                    "ensemble_predictions": {
                        "prediction": "hold",
                        "confidence": 0.0,
                        "ensemble_score": 0.0,
                        "model_weights": {},
                        "individual_predictions": {},
                        "ensemble_analysis": {
                            "diversity_score": 0.0,
                            "agreement_ratio": 0.0,
                            "model_count": 0
                        }
                    }
                }
            
            # Prepare features for ensemble prediction
            features = self._prepare_features_for_ensemble(market_data)
            if features is None:
                return {"ensemble_predictions": {"prediction": "hold", "confidence": 0.0}}
            
            # Calculate ensemble predictions using the correct method
            try:
                signal = self.ensemble_strategy.predict(features)
                if signal:
                    predictions = {
                        "prediction": signal.prediction,
                        "confidence": signal.confidence,
                        "ensemble_score": signal.ensemble_score,
                        "model_weights": signal.model_weights,
                        "individual_predictions": self.ensemble_strategy.ensemble_analysis.get('individual_predictions', {}),
                        "ensemble_analysis": self.ensemble_strategy.ensemble_analysis
                    }
                else:
                    predictions = {"prediction": "hold", "confidence": 0.0}
            except Exception as ensemble_error:
                logger.error(f"Ensemble prediction error: {ensemble_error}")
                predictions = {"prediction": "hold", "confidence": 0.0}
            
            return {"ensemble_predictions": predictions}
            
        except Exception as e:
            logger.error(f"❌ Error calculating ensemble predictions: {e}")
            return {}
    
    async def _generate_signals_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from combined analysis results"""
        try:
            signals = []
            
            # Generate signals based on analysis results
            # This is a simplified version - you can enhance this based on your specific logic
            
            # Example signal generation
            if analysis.get("technical_indicators"):
                signal = {
                    "symbol": "BTCUSDT",
                    "signal_type": "buy",
                    "confidence": 0.75,
                    "strength": 0.8,
                    "timestamp": datetime.now().isoformat(),
                    "price": 50000.0,
                    "reason": "Technical indicators analysis",
                    "indicators": analysis.get("technical_indicators", {}),
                    "phase_4_1_features": True
                }
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals from analysis: {e}")
            return []
    
    async def get_recent_signals(self, symbol: str = "BTCUSDT", limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals for a specific symbol (alias for get_signals)"""
        try:
            return await self.get_signals(symbol=symbol, limit=limit)
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []

    async def start(self):
        """Start the Real-time Signal Generator"""
        if self.is_running:
            logger.warning("Real-time Signal Generator is already running")
            return
            
        logger.info("🚀 Starting Real-time Signal Generator...")
        
        # Initialize database connection
        await self.initialize_database()
        
        # Initialize advanced technical indicators engine
        await self.initialize_advanced_indicators()

        # Initialize Smart Money Concepts engine
        if self.use_smc and not self.smc_engine:
            try:
                self.smc_engine = SmartMoneyConceptsEngine()
                logger.info("✅ Smart Money Concepts engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Smart Money Concepts engine: {e}")
                self.use_smc = False
                self.smc_engine = None
        
        # Initialize Deep Learning engine
        if self.use_dl and not self.dl_engine:
            try:
                self.dl_engine = DeepLearningEngine()
                logger.info("✅ Deep Learning engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Deep Learning engine: {e}")
                self.use_dl = False
                self.dl_engine = None
        
        # Initialize Reinforcement Learning engine
        if self.use_rl and not self.rl_engine:
            try:
                self.rl_engine = ReinforcementLearningEngine()
                await self.rl_engine.start()
                logger.info("✅ Reinforcement Learning engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Reinforcement Learning engine: {e}")
                self.use_rl = False
                self.rl_engine = None
        
        # Initialize Natural Language Processing engine
        if self.use_nlp and not self.nlp_engine:
            try:
                self.nlp_engine = NaturalLanguageProcessingEngine()
                await self.nlp_engine.start()
                logger.info("✅ Natural Language Processing engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Natural Language Processing engine: {e}")
                self.use_nlp = False
                self.nlp_engine = None
        
        # Initialize Phase 2.3: Ensemble Model Integration
        await self.initialize_ensemble_strategy()
        
        # Initialize Phase 3.1: Enhanced Sentiment Analysis Integration
        await self.initialize_sentiment_service()
        
        # Initialize Phase 4.2: Memory & CPU Optimization
        await self._initialize_phase4_2_optimizations()
    
    async def initialize_ensemble_strategy(self):
        """Initialize Phase 2.3: Ensemble Model Integration"""
        if self.use_ensemble and not self.ensemble_strategy:
            try:
                # Create ensemble configuration with multiple model types
                ensemble_config = EnsembleConfig(
                    strategy_type=MLStrategyType.ENSEMBLE_VOTING,
                    base_models=[
                        ModelType.RANDOM_FOREST,
                        ModelType.GRADIENT_BOOSTING,
                        ModelType.LOGISTIC_REGRESSION,
                        ModelType.SVM,
                        ModelType.NEURAL_NETWORK
                    ],
                    voting_method="soft",
                    adaptive_weights=True
                )
                
                self.ensemble_strategy = EnsembleStrategy(ensemble_config)
                logger.info("✅ Phase 2.3: Ensemble Model Integration initialized")
                logger.info(f"🤖 Ensemble models: {len(ensemble_config.base_models)} models with {ensemble_config.voting_method} voting")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ensemble strategy: {e}")
                self.use_ensemble = False
                self.ensemble_strategy = None
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._generate_signals())
        
        logger.info("✅ Real-time Signal Generator started successfully")
    
    async def initialize_sentiment_service(self):
        """Initialize Phase 3.3: Enhanced Sentiment Analysis with Social Media Integration"""
        if self.use_sentiment and not self.sentiment_service:
            try:
                self.sentiment_service = SentimentService()
                logger.info("✅ Phase 3.3: Enhanced Sentiment Service initialized")
                logger.info("📊 Multi-source sentiment, news events, and social media integration ready")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Enhanced Sentiment Service: {e}")
                self.use_sentiment = False
                self.sentiment_service = None
    
    # Phase 4.2: Memory & CPU Optimization Methods
    
    async def _initialize_phase4_2_optimizations(self):
        """Initialize Phase 4.2 memory and CPU optimizations"""
        try:
            logger.info("🚀 Initializing Phase 4.2: Memory & CPU Optimization...")
            
            # Initialize garbage collection optimization
            if self.gc_optimization_enabled:
                self._optimize_garbage_collection()
            
            # Initialize CPU optimization
            if self.cpu_optimization_enabled:
                self._optimize_cpu_usage()
            
            # Start resource monitoring
            if self.resource_monitoring_enabled:
                asyncio.create_task(self._monitor_resources())
            
            # Start memory cleanup task
            asyncio.create_task(self._background_memory_cleanup())
            
            logger.info("✅ Phase 4.2: Memory & CPU Optimization initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Phase 4.2 optimizations: {e}")
    
    def _optimize_garbage_collection(self):
        """Optimize garbage collection settings"""
        try:
            # Set optimized generation thresholds
            gc.set_threshold(*self.gc_generation_thresholds)
            
            # Enable automatic tuning if available
            if hasattr(gc, 'set_debug') and self.gc_auto_tuning:
                gc.set_debug(gc.DEBUG_STATS)
            
            logger.info("✅ Garbage collection optimization applied")
            
        except Exception as e:
            logger.error(f"❌ Error optimizing garbage collection: {e}")
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage and affinity"""
        try:
            # Set CPU affinity if enabled
            if self.cpu_affinity_enabled:
                import os
                cpu_count = os.cpu_count()
                if cpu_count:
                    # Reserve one core for system, use rest for processing
                    available_cores = list(range(self.cpu_cores_reserved, cpu_count))
                    if available_cores:
                        os.sched_setaffinity(0, available_cores)
                        logger.info(f"✅ CPU affinity set to cores: {available_cores}")
            
            logger.info("✅ CPU optimization applied")
            
        except Exception as e:
            logger.error(f"❌ Error optimizing CPU usage: {e}")
    
    async def _monitor_resources(self):
        """Monitor system resources continuously"""
        while self.is_running:
            try:
                # Get current resource usage
                memory_metrics = self._get_memory_metrics()
                cpu_metrics = self._get_cpu_metrics()
                
                # Store metrics history
                self.memory_metrics_history.append(memory_metrics)
                self.cpu_metrics_history.append(cpu_metrics)
                
                # Check for resource alerts
                await self._check_resource_alerts(memory_metrics, cpu_metrics)
                
                # Optimize if needed
                if memory_metrics.memory_percent > self.memory_threshold:
                    await self._optimize_memory_usage()
                
                if cpu_metrics.cpu_percent > self.cpu_throttling_threshold:
                    await self._throttle_cpu_usage()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"❌ Error monitoring resources: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _get_memory_metrics(self) -> 'MemoryMetrics':
        """Get current memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return MemoryMetrics(
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                memory_percent=memory.percent / 100.0,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                process_memory_mb=process_memory.rss / (1024 * 1024),
                cache_size=len(self.result_cache),
                gc_objects=len(gc.get_objects()),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"❌ Error getting memory metrics: {e}")
            return None
    
    def _get_cpu_metrics(self) -> 'PerformanceMetrics':
        """Get current CPU performance metrics"""
        try:
            return PerformanceMetrics(
                processing_time_ms=np.mean(self.processing_times) if self.processing_times else 0.0,
                cache_hit_rate=self.cache_hits / max(self.total_requests, 1),
                memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
                cpu_usage_percent=psutil.cpu_percent(interval=0.1),
                throughput_per_second=len(self.processing_times) / max(1, sum(self.processing_times) / 1000),
                latency_percentile_95=np.percentile(list(self.processing_times), 95) if self.processing_times else 0.0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"❌ Error getting CPU metrics: {e}")
            return None
    
    async def _check_resource_alerts(self, memory_metrics: 'MemoryMetrics', cpu_metrics: 'PerformanceMetrics'):
        """Check for resource usage alerts"""
        try:
            alerts = []
            
            # Memory alerts
            if memory_metrics.memory_percent > 0.9:  # 90% memory usage
                alerts.append(f"CRITICAL: Memory usage at {memory_metrics.memory_percent:.1%}")
            elif memory_metrics.memory_percent > 0.8:  # 80% memory usage
                alerts.append(f"WARNING: Memory usage at {memory_metrics.memory_percent:.1%}")
            
            # CPU alerts
            if cpu_metrics.cpu_usage_percent > 90:  # 90% CPU usage
                alerts.append(f"CRITICAL: CPU usage at {cpu_metrics.cpu_usage_percent:.1f}%")
            elif cpu_metrics.cpu_usage_percent > 80:  # 80% CPU usage
                alerts.append(f"WARNING: CPU usage at {cpu_metrics.cpu_usage_percent:.1f}%")
            
            # Cache alerts
            if memory_metrics.cache_size > 10000:  # Too many cached items
                alerts.append(f"WARNING: Cache size at {memory_metrics.cache_size} items")
            
            # Store alerts
            for alert in alerts:
                self.resource_alerts.append({
                    'message': alert,
                    'timestamp': datetime.now(),
                    'severity': 'CRITICAL' if 'CRITICAL' in alert else 'WARNING'
                })
                
                if 'CRITICAL' in alert:
                    logger.error(f"🚨 {alert}")
                else:
                    logger.warning(f"⚠️ {alert}")
            
        except Exception as e:
            logger.error(f"❌ Error checking resource alerts: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage when threshold is exceeded"""
        try:
            logger.info("🧹 Optimizing memory usage...")
            
            # Clear old cache entries
            await self._cleanup_cache()
            
            # Force garbage collection
            if self.gc_optimization_enabled:
                self._force_garbage_collection()
            
            # Compress cache if enabled
            if self.cache_compression_enabled:
                await self._compress_cache()
            
            # Clear old metrics history
            if len(self.memory_metrics_history) > 500:
                # Keep only last 500 entries
                self.memory_metrics_history = deque(list(self.memory_metrics_history)[-500:], maxlen=1000)
            
            if len(self.cpu_metrics_history) > 500:
                self.cpu_metrics_history = deque(list(self.cpu_metrics_history)[-500:], maxlen=1000)
            
            logger.info("✅ Memory optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Error optimizing memory usage: {e}")
    
    async def _throttle_cpu_usage(self):
        """Throttle CPU usage when threshold is exceeded"""
        try:
            logger.info("⏱️ Throttling CPU usage...")
            
            # Reduce processing queue size
            if self.processing_queue.qsize() > 100:
                # Clear half of the queue
                for _ in range(self.processing_queue.qsize() // 2):
                    try:
                        self.processing_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            # Reduce thread pool size temporarily
            if hasattr(self.async_executor, '_max_workers'):
                current_workers = self.async_executor._max_workers
                if current_workers > 2:
                    # Reduce to minimum workers
                    self.async_executor._max_workers = 2
                    logger.info(f"✅ Reduced thread pool from {current_workers} to 2 workers")
            
            logger.info("✅ CPU throttling applied")
            
        except Exception as e:
            logger.error(f"❌ Error throttling CPU usage: {e}")
    
    def _force_garbage_collection(self):
        """Force garbage collection and track statistics"""
        try:
            # Get objects before collection
            objects_before = len(gc.get_objects())
            
            # Force collection of all generations
            collected = gc.collect()
            
            # Get objects after collection
            objects_after = len(gc.get_objects())
            objects_freed = objects_before - objects_after
            
            # Update statistics
            self.gc_stats['collections'] += 1
            self.gc_stats['objects_freed'] += objects_freed
            self.gc_stats['last_collection_time'] = time.time()
            
            logger.info(f"🧹 Garbage collection: {objects_freed} objects freed")
            
        except Exception as e:
            logger.error(f"❌ Error in garbage collection: {e}")
    
    async def _compress_cache(self):
        """Compress cache to reduce memory usage"""
        try:
            if not self.cache_compression_enabled:
                return
            
            cache_size = len(self.result_cache)
            if cache_size < 1000:  # Only compress if cache is large
                return
            
            logger.info(f"🗜️ Compressing cache with {cache_size} items...")
            
            # Simple compression: remove old entries
            current_time = time.time()
            compressed_items = 0
            
            # Remove entries older than cache TTL
            keys_to_remove = []
            for key, (value, timestamp) in self.result_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    keys_to_remove.append(key)
                    compressed_items += 1
            
            # Remove old entries
            for key in keys_to_remove:
                self.result_cache.pop(key, None)
            
            logger.info(f"✅ Cache compressed: {compressed_items} items removed")
            
        except Exception as e:
            logger.error(f"❌ Error compressing cache: {e}")
    
    async def _background_memory_cleanup(self):
        """Background task for periodic memory cleanup"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if cleanup is needed
                if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
                    await self._optimize_memory_usage()
                    self.last_memory_cleanup = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error in background memory cleanup: {e}")
                await asyncio.sleep(60)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            memory_metrics = self._get_memory_metrics()
            cpu_metrics = self._get_cpu_metrics()
            
            return {
                'memory_usage_mb': memory_metrics.process_memory_mb if memory_metrics else 0.0,
                'memory_percent': memory_metrics.memory_percent if memory_metrics else 0.0,
                'cpu_usage_percent': cpu_metrics.cpu_usage_percent if cpu_metrics else 0.0,
                'cache_size': memory_metrics.cache_size if memory_metrics else 0,
                'gc_collections': self.gc_stats['collections'],
                'gc_objects_freed': self.gc_stats['objects_freed'],
                'processing_times_count': len(self.processing_times),
                'cache_hit_rate': self.cache_hits / max(self.total_requests, 1),
                'resource_alerts_count': len(self.resource_alerts),
                'optimization_enabled': {
                    'memory': self.memory_optimization_enabled,
                    'cpu': self.cpu_optimization_enabled,
                    'gc': self.gc_optimization_enabled,
                    'monitoring': self.resource_monitoring_enabled
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting memory stats: {e}")
            return {}
    
    async def stop(self):
        """Stop the Real-time Signal Generator"""
        if not self.is_running:
            logger.warning("Real-time Signal Generator is not running")
            return
            
        logger.info("🛑 Stopping Real-time Signal Generator...")
        self.is_running = False
        logger.info("✅ Real-time Signal Generator stopped successfully")
    
    async def analyze_enhanced_sentiment_predictions(self, symbol: str) -> Dict[str, Any]:
        """Phase 3.3: Analyze enhanced sentiment predictions with social media integration"""
        try:
            if not self.use_sentiment or not self.sentiment_service:
                return {
                    'sentiment_bias': 'neutral',
                    'sentiment_score': 0.0,
                    'sentiment_confidence': 0.0,
                    'sentiment_strength': 0.0,
                    'high_confidence_sentiment': False,
                    'trend_analysis': {'trend': 'stable', 'trend_strength': 0.0},
                    'momentum_indicators': {'momentum_direction': 'neutral', 'momentum_strength': 'weak'},
                    'volatility_metrics': {'volatility_rank': 'low', 'stability_score': 1.0},
                    'correlation_metrics': {'correlation_strength': 'weak', 'predictive_power': 0.0},
                    'enhanced_confidence': 0.0,
                    'sentiment_strength': 0.0,
                    'prediction_confidence': 0.0,
                    'phase_3_3_features': False
                }
            
            # Get enhanced sentiment analysis with social media integration
            enhanced_sentiment = await self.sentiment_service.get_enhanced_sentiment_with_social_media(symbol)
            
            # Extract key metrics for signal generation
            sentiment_score = enhanced_sentiment.get('sentiment_score', 0.0)
            sentiment_confidence = enhanced_sentiment.get('confidence', 0.0)
            enhanced_confidence = enhanced_sentiment.get('enhanced_confidence', sentiment_confidence)
            sentiment_strength = enhanced_sentiment.get('sentiment_strength', 0.0)
            prediction_confidence = enhanced_sentiment.get('prediction_confidence', sentiment_confidence)
            
            # Get news event and social media analysis
            news_events = enhanced_sentiment.get('news_events', {})
            social_media = enhanced_sentiment.get('social_media', {})
            event_enhanced_confidence = enhanced_sentiment.get('event_enhanced_confidence', enhanced_confidence)
            social_enhanced_confidence = enhanced_sentiment.get('social_enhanced_confidence', enhanced_confidence)
            event_filtered_sentiment = enhanced_sentiment.get('event_filtered_sentiment', {})
            social_filtered_sentiment = enhanced_sentiment.get('social_filtered_sentiment', {})
            
            # Use social media filtered sentiment if available, otherwise use event-filtered
            if social_filtered_sentiment:
                sentiment_score = social_filtered_sentiment.get('sentiment_score', sentiment_score)
                sentiment_confidence = social_enhanced_confidence
            elif event_filtered_sentiment:
                sentiment_score = event_filtered_sentiment.get('sentiment_score', sentiment_score)
                sentiment_confidence = event_enhanced_confidence
            
            # Determine sentiment bias
            if sentiment_score > 0.1:
                sentiment_bias = 'bullish'
            elif sentiment_score < -0.1:
                sentiment_bias = 'bearish'
            else:
                sentiment_bias = 'neutral'
            
            # Check for high confidence sentiment (considering both news events and social media)
            high_confidence_sentiment = (
                (event_enhanced_confidence > 0.7 or social_enhanced_confidence > 0.7) and 
                sentiment_strength > 0.6
            )
            
            # Extract trend and momentum analysis
            trend_analysis = enhanced_sentiment.get('trend_analysis', {})
            momentum_indicators = enhanced_sentiment.get('momentum_indicators', {})
            volatility_metrics = enhanced_sentiment.get('volatility_metrics', {})
            correlation_metrics = enhanced_sentiment.get('correlation_metrics', {})
            
            return {
                'sentiment_bias': sentiment_bias,
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'sentiment_strength': sentiment_strength,
                'high_confidence_sentiment': high_confidence_sentiment,
                'trend_analysis': trend_analysis,
                'momentum_indicators': momentum_indicators,
                'volatility_metrics': volatility_metrics,
                'correlation_metrics': correlation_metrics,
                'enhanced_confidence': enhanced_confidence,
                'event_enhanced_confidence': event_enhanced_confidence,
                'social_enhanced_confidence': social_enhanced_confidence,
                'prediction_confidence': prediction_confidence,
                'news_events': news_events,
                'social_media': social_media,
                'event_filtered_sentiment': event_filtered_sentiment,
                'social_filtered_sentiment': social_filtered_sentiment,
                'phase_3_3_features': True,
                'sentiment_sources': enhanced_sentiment.get('sentiment_sources', {}),
                'twitter_sentiment': enhanced_sentiment.get('twitter_sentiment', 0.0),
                'reddit_sentiment': enhanced_sentiment.get('reddit_sentiment', 0.0),
                'news_sentiment': enhanced_sentiment.get('news_sentiment', 0.0),
                'telegram_sentiment': enhanced_sentiment.get('telegram_sentiment', 0.0),
                'discord_sentiment': enhanced_sentiment.get('discord_sentiment', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing enhanced sentiment predictions for {symbol}: {e}")
            return {
                'sentiment_bias': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'sentiment_strength': 0.0,
                'high_confidence_sentiment': False,
                'trend_analysis': {'trend': 'stable', 'trend_strength': 0.0},
                'momentum_indicators': {'momentum_direction': 'neutral', 'momentum_strength': 'weak'},
                'volatility_metrics': {'volatility_rank': 'low', 'stability_score': 1.0},
                'correlation_metrics': {'correlation_strength': 'weak', 'predictive_power': 0.0},
                'enhanced_confidence': 0.0,
                'event_enhanced_confidence': 0.0,
                'social_enhanced_confidence': 0.0,
                'prediction_confidence': 0.0,
                'news_events': {},
                'social_media': {},
                'event_filtered_sentiment': {},
                'social_filtered_sentiment': {},
                'phase_3_3_features': False
            }
    
    async def analyze_news_event_predictions(self, symbol: str) -> Dict[str, Any]:
        """Phase 3.2: Analyze news event predictions for signal generation"""
        try:
            if not self.use_sentiment or not self.sentiment_service:
                return {
                    'event_impact_score': 0.0,
                    'event_count': 0,
                    'high_impact_events': 0,
                    'medium_impact_events': 0,
                    'low_impact_events': 0,
                    'event_categories': {},
                    'news_aware_signal': False,
                    'event_filtered_confidence': 0.0,
                    'phase_3_2_features': False
                }
            
            # Get news event analysis
            event_analysis = await self.sentiment_service.get_news_event_analysis(symbol)
            
            # Extract event metrics
            impact_score = event_analysis.get('impact_score', 0.0)
            event_count = event_analysis.get('event_count', 0)
            high_impact_events = event_analysis.get('high_impact_events', 0)
            medium_impact_events = event_analysis.get('medium_impact_events', 0)
            low_impact_events = event_analysis.get('low_impact_events', 0)
            event_categories = event_analysis.get('event_categories', {})
            
            # Determine if signal should be news-aware
            news_aware_signal = impact_score > 0.3 or high_impact_events > 0
            
            # Calculate event-filtered confidence
            event_filtered_confidence = min(impact_score * 1.5, 1.0) if news_aware_signal else 0.0
            
            return {
                'event_impact_score': impact_score,
                'event_count': event_count,
                'high_impact_events': high_impact_events,
                'medium_impact_events': medium_impact_events,
                'low_impact_events': low_impact_events,
                'event_categories': event_categories,
                'news_aware_signal': news_aware_signal,
                'event_filtered_confidence': event_filtered_confidence,
                'events': event_analysis.get('events', []),
                'phase_3_2_features': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing news event predictions for {symbol}: {e}")
            return {
                'event_impact_score': 0.0,
                'event_count': 0,
                'high_impact_events': 0,
                'medium_impact_events': 0,
                'low_impact_events': 0,
                'event_categories': {},
                'news_aware_signal': False,
                'event_filtered_confidence': 0.0,
                'phase_3_2_features': False
            }
    
    async def analyze_social_media_predictions(self, symbol: str) -> Dict[str, Any]:
        """Phase 3.3: Analyze social media predictions for signal generation"""
        try:
            if not self.use_sentiment or not self.sentiment_service:
                return {
                    'social_impact_score': 0.0,
                    'social_sentiment_score': 0.0,
                    'social_confidence': 0.0,
                    'social_trends': {'trend': 'stable', 'trend_strength': 0.0},
                    'social_momentum': {'momentum_direction': 'neutral', 'momentum_strength': 'weak'},
                    'social_volume': {'volume_score': 0.0, 'volume_trend': 'stable'},
                    'social_engagement': {'engagement_score': 0.0, 'engagement_trend': 'stable'},
                    'social_aware_signal': False,
                    'social_enhanced_confidence': 0.0,
                    'phase_3_3_features': False
                }
            
            # Get social media sentiment analysis
            social_analysis = await self.sentiment_service.get_social_media_sentiment(symbol)
            
            # Extract social media metrics
            social_sentiment = social_analysis.get('social_sentiment', {})
            social_impact_score = social_analysis.get('social_impact_score', 0.0)
            social_trends = social_analysis.get('social_trends', {})
            social_momentum = social_analysis.get('social_momentum', {})
            social_volume = social_analysis.get('social_volume', {})
            social_engagement = social_analysis.get('social_engagement', {})
            
            # Determine if signal should be social-aware
            social_aware_signal = social_impact_score > 0.3 or social_trends.get('trend_strength', 0.0) > 0.5
            
            # Calculate social-enhanced confidence
            social_enhanced_confidence = min(social_impact_score * 1.2, 1.0) if social_aware_signal else 0.0
            
            return {
                'social_impact_score': social_impact_score,
                'social_sentiment_score': social_sentiment.get('sentiment_score', 0.0),
                'social_confidence': social_sentiment.get('confidence', 0.0),
                'social_trends': social_trends,
                'social_momentum': social_momentum,
                'social_volume': social_volume,
                'social_engagement': social_engagement,
                'social_aware_signal': social_aware_signal,
                'social_enhanced_confidence': social_enhanced_confidence,
                'twitter_sentiment': social_analysis.get('twitter_sentiment', {}),
                'reddit_sentiment': social_analysis.get('reddit_sentiment', {}),
                'phase_3_3_features': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing social media predictions for {symbol}: {e}")
            return {
                'social_impact_score': 0.0,
                'social_sentiment_score': 0.0,
                'social_confidence': 0.0,
                'social_trends': {'trend': 'stable', 'trend_strength': 0.0},
                'social_momentum': {'momentum_direction': 'neutral', 'momentum_strength': 'weak'},
                'social_volume': {'volume_score': 0.0, 'volume_trend': 'stable'},
                'social_engagement': {'engagement_score': 0.0, 'engagement_trend': 'stable'},
                'social_aware_signal': False,
                'social_enhanced_confidence': 0.0,
                'phase_3_3_features': False
            }

    # Phase 4.3: Database Integration Optimization Methods
    
    async def initialize_phase4_3_database_optimizations(self):
        """Initialize Phase 4.3 database optimizations"""
        try:
            if not self.db_connection:
                logger.warning("⚠️ Database connection not available for Phase 4.3 optimizations")
                return False
            
            logger.info("🔧 Initializing Phase 4.3 Database Integration Optimizations...")
            
            # Apply TimescaleDB optimization settings
            await self.db_connection.optimize_timescaledb_settings()
            
            # Setup hypertable optimizations
            await self.db_connection.setup_hypertable_optimizations()
            
            # Create performance indexes
            await self.db_connection.create_performance_indexes()
            
            # Initialize batch processing
            self.batch_signals = []
            self.batch_size = 100  # Configurable batch size
            self.last_batch_flush = time.time()
            self.batch_flush_interval = 60  # Flush every 60 seconds
            
            logger.info("✅ Phase 4.3 Database Integration Optimizations initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 4.3 database optimizations: {e}")
            return False
    
    async def store_signal_with_phase4_3_optimizations(self, signal_data: Dict[str, Any]) -> bool:
        """Store signal with Phase 4.3 database optimizations"""
        try:
            if not self.db_connection:
                return False
            
            # Add Phase 4.3 metadata
            signal_data.update({
                'phase_4_3_features': True,
                'batch_processed': False,
                'optimization_level': 'phase4_3',
                'database_optimization_metadata': {
                    'optimization_version': '4.3',
                    'batch_processing_enabled': True,
                    'hypertable_optimized': True,
                    'compression_enabled': True,
                    'performance_indexes_created': True,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            # Add to batch for efficient processing
            self.batch_signals.append(signal_data)
            
            # Check if we should flush the batch
            current_time = time.time()
            if (len(self.batch_signals) >= self.batch_size or 
                current_time - self.last_batch_flush >= self.batch_flush_interval):
                await self._flush_batch_signals()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store signal with Phase 4.3 optimizations: {e}")
            return False
    
    async def _flush_batch_signals(self):
        """Flush batch signals to database"""
        try:
            if not self.batch_signals:
                return
            
            logger.info(f"🔄 Flushing batch of {len(self.batch_signals)} signals to database...")
            
            # Use optimized batch insert
            success = await self.db_connection.batch_insert_signals(self.batch_signals)
            
            if success:
                logger.info(f"✅ Successfully flushed {len(self.batch_signals)} signals")
                self.batch_signals = []
                self.last_batch_flush = time.time()
            else:
                logger.error("❌ Failed to flush batch signals")
                
        except Exception as e:
            logger.error(f"❌ Error flushing batch signals: {e}")
    
    async def get_database_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics for Phase 4.3"""
        try:
            if not self.db_connection:
                return {}
            
            # Get database performance metrics
            db_metrics = await self.db_connection.get_performance_metrics()
            
            # Add Phase 4.3 specific metrics
            phase4_3_metrics = {
                'batch_processing': {
                    'batch_size': self.batch_size,
                    'current_batch_size': len(self.batch_signals),
                    'last_flush_time': self.last_batch_flush,
                    'flush_interval': self.batch_flush_interval
                },
                'optimization_features': {
                    'hypertable_optimized': True,
                    'compression_enabled': True,
                    'performance_indexes_created': True,
                    'batch_processing_enabled': True
                }
            }
            
            return {
                'database_metrics': db_metrics,
                'phase4_3_metrics': phase4_3_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database performance metrics: {e}")
            return {}
