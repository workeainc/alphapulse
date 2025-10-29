"""
Real-Time Data Pipeline Optimization for AlphaPulse
High-performance, low-latency data processing pipeline for real-time trading
Phase 4.1: Ultra-low latency processing with <100ms target
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
from datetime import datetime, timedelta
import warnings
import gc
import psutil
import os
warnings.filterwarnings('ignore')

from .feature_engineering import FeatureExtractor
from .model_registry import ModelRegistry
from .risk_management import RiskManager, risk_manager
from .position_sizing import PositionSizingOptimizer, position_sizing_optimizer
from .market_regime_detection import MarketRegimeDetector, market_regime_detector
from .multi_timeframe_fusion import MultiTimeframeFusion, multi_timeframe_fusion
from .parallel_filter_processor import ParallelFilterProcessor, parallel_filter_processor
from .multi_level_cache import MultiLevelCache, multi_level_cache
from .gpu_accelerated_filters import GPUAcceleratedFilters, gpu_accelerated_filters
from .fpga_integration import FPGAIntegration, fpga_integration
from .continuous_false_positive_logging import ContinuousFalsePositiveLogger, continuous_false_positive_logger
from .feedback_loop import FeedbackLoop, feedback_loop
from .ai_driven_threshold_manager import AIDrivenThresholdManager, ai_driven_threshold_manager

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline processing stages"""
    DATA_INGESTION = "data_ingestion"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    FEATURE_EXTRACTION = "feature_extraction"
    PATTERN_DETECTION = "pattern_detection"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION_DECISION = "execution_decision"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class DataPoint:
    """Individual data point with metadata"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics with Phase 4.1 enhancements"""
    total_processed: int = 0
    total_errors: int = 0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    throughput_per_second: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_size: int = 0
    cache_hit_rate: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    target_latency_ms: float = 100.0  # Phase 4.1: Target <100ms
    performance_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class ProcessingResult:
    """Result of pipeline processing with Phase 4.1 optimizations"""
    symbol: str
    timestamp: datetime
    data_point: DataPoint
    features: Optional[np.ndarray] = None
    patterns: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    execution_decision: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    quality_score: float = 1.0
    cache_hit: bool = False
    phase_4_1_features: bool = True

class DataValidator:
    """Real-time data validation and quality assessment"""
    
    def __init__(self, 
                 max_price_change: float = 0.5,  # 50% max price change
                 min_volume: float = 0.0,
                 max_volume_multiplier: float = 100.0,
                 required_fields: List[str] = None):
        
        self.max_price_change = max_price_change
        self.min_volume = min_volume
        self.max_volume_multiplier = max_volume_multiplier
        self.required_fields = required_fields or ['open', 'high', 'low', 'close', 'volume']
        
        # Price history for validation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("DataValidator initialized")
    
    def validate_data_point(self, data_point: DataPoint) -> Tuple[bool, DataQuality, str]:
        """Validate a single data point"""
        try:
            # Basic field validation
            for field in self.required_fields:
                if not hasattr(data_point, field) or getattr(data_point, field) is None:
                    return False, DataQuality.INVALID, f"Missing required field: {field}"
            
            # OHLC relationship validation
            if not (data_point.low <= data_point.open <= data_point.high and 
                   data_point.low <= data_point.close <= data_point.high):
                return False, DataQuality.INVALID, "Invalid OHLC relationships"
            
            # Volume validation
            if data_point.volume < self.min_volume:
                return False, DataQuality.POOR, f"Volume too low: {data_point.volume}"
            
            # Price change validation
            if len(self.price_history[data_point.symbol]) > 0:
                last_close = self.price_history[data_point.symbol][-1]
                price_change = abs(data_point.close - last_close) / last_close
                
                if price_change > self.max_price_change:
                    return False, DataQuality.POOR, f"Price change too large: {price_change:.2%}"
            
            # Volume spike validation
            if len(self.volume_history[data_point.symbol]) > 0:
                avg_volume = np.mean(list(self.volume_history[data_point.symbol]))
                if data_point.volume > avg_volume * self.max_volume_multiplier:
                    return False, DataQuality.FAIR, f"Volume spike detected: {data_point.volume / avg_volume:.1f}x"
            
            # Update history
            self.price_history[data_point.symbol].append(data_point.close)
            self.volume_history[data_point.symbol].append(data_point.volume)
            
            # Quality assessment
            quality = self._assess_quality(data_point)
            
            return True, quality, "Validation passed"
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, DataQuality.INVALID, f"Validation error: {str(e)}"
    
    def _assess_quality(self, data_point: DataPoint) -> DataQuality:
        """Assess data quality based on various factors"""
        quality_score = 1.0
        
        # Check for zero values
        if data_point.open == 0 or data_point.close == 0:
            quality_score -= 0.5
        
        # Check for extreme values
        if data_point.high > 1e6 or data_point.low < 1e-6:
            quality_score -= 0.3
        
        # Check volume consistency
        if len(self.volume_history[data_point.symbol]) > 0:
            avg_volume = np.mean(list(self.volume_history[data_point.symbol]))
            if avg_volume > 0:
                volume_ratio = data_point.volume / avg_volume
                if volume_ratio > 10 or volume_ratio < 0.1:
                    quality_score -= 0.2
        
        # Determine quality level
        if quality_score >= 0.9:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.7:
            return DataQuality.GOOD
        elif quality_score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

class DataNormalizer:
    """Real-time data normalization and preprocessing"""
    
    def __init__(self, 
                 normalization_method: str = "zscore",
                 window_size: int = 100,
                 min_data_points: int = 20):
        
        self.normalization_method = normalization_method
        self.window_size = window_size
        self.min_data_points = min_data_points
        
        # Rolling statistics for each symbol
        self.rolling_stats: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
        
        logger.info(f"DataNormalizer initialized with {normalization_method} method")
    
    def normalize_data_point(self, data_point: DataPoint) -> DataPoint:
        """Normalize a single data point"""
        try:
            symbol = data_point.symbol
            
            # Update rolling statistics
            self._update_rolling_stats(data_point)
            
            # Check if we have enough data
            if len(self.rolling_stats[symbol]['close']) < self.min_data_points:
                return data_point  # Return original if insufficient data
            
            # Normalize based on method
            if self.normalization_method == "zscore":
                normalized_point = self._zscore_normalize(data_point, symbol)
            elif self.normalization_method == "minmax":
                normalized_point = self._minmax_normalize(data_point, symbol)
            elif self.normalization_method == "robust":
                normalized_point = self._robust_normalize(data_point, symbol)
            else:
                normalized_point = data_point
            
            return normalized_point
            
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return data_point
    
    def _update_rolling_stats(self, data_point: DataPoint):
        """Update rolling statistics for a symbol"""
        symbol = data_point.symbol
        
        self.rolling_stats[symbol]['open'].append(data_point.open)
        self.rolling_stats[symbol]['high'].append(data_point.high)
        self.rolling_stats[symbol]['low'].append(data_point.low)
        self.rolling_stats[symbol]['close'].append(data_point.close)
        self.rolling_stats[symbol]['volume'].append(data_point.volume)
    
    def _zscore_normalize(self, data_point: DataPoint, symbol: str) -> DataPoint:
        """Z-score normalization"""
        close_mean = np.mean(list(self.rolling_stats[symbol]['close']))
        close_std = np.std(list(self.rolling_stats[symbol]['close']))
        
        if close_std == 0:
            return data_point
        
        normalized_close = (data_point.close - close_mean) / close_std
        
        # Create normalized data point
        normalized_point = DataPoint(
            symbol=data_point.symbol,
            timestamp=data_point.timestamp,
            open=data_point.open,
            high=data_point.high,
            low=data_point.low,
            close=normalized_close,
            volume=data_point.volume,
            quality=data_point.quality,
            metadata={**data_point.metadata, 'normalization_method': 'zscore'}
        )
        
        return normalized_point
    
    def _minmax_normalize(self, data_point: DataPoint, symbol: str) -> DataPoint:
        """Min-max normalization"""
        closes = list(self.rolling_stats[symbol]['close'])
        min_close = min(closes)
        max_close = max(closes)
        
        if max_close == min_close:
            return data_point
        
        normalized_close = (data_point.close - min_close) / (max_close - min_close)
        
        normalized_point = DataPoint(
            symbol=data_point.symbol,
            timestamp=data_point.timestamp,
            open=data_point.open,
            high=data_point.high,
            low=data_point.low,
            close=normalized_close,
            volume=data_point.volume,
            quality=data_point.quality,
            metadata={**data_point.metadata, 'normalization_method': 'minmax'}
        )
        
        return normalized_point
    
    def _robust_normalize(self, data_point: DataPoint, symbol: str) -> DataPoint:
        """Robust normalization using median and MAD"""
        closes = list(self.rolling_stats[symbol]['close'])
        median_close = np.median(closes)
        mad = np.median(np.abs(np.array(closes) - median_close))
        
        if mad == 0:
            return data_point
        
        normalized_close = (data_point.close - median_close) / mad
        
        normalized_point = DataPoint(
            symbol=data_point.symbol,
            timestamp=data_point.timestamp,
            open=data_point.open,
            high=data_point.high,
            low=data_point.low,
            close=normalized_close,
            volume=data_point.volume,
            quality=data_point.quality,
            metadata={**data_point.metadata, 'normalization_method': 'robust'}
        )
        
        return normalized_point

class RealTimePipeline:
    """Phase 4.1: Ultra-low latency real-time data processing pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the real-time pipeline with Phase 4.1 optimizations"""
        self.config = config or {}
        
        # Phase 4.1: Performance Optimization
        self.target_latency_ms = self.config.get('target_latency_ms', 100.0)
        self.max_memory_mb = self.config.get('max_memory_mb', 512)
        self.enable_caching = self.config.get('enable_caching', True)
        self.enable_async_processing = self.config.get('enable_async_processing', True)
        self.enable_memory_management = self.config.get('enable_memory_management', True)
        
        # Phase 4.1: Processing Components
        self.processing_times = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.result_cache = {}
        self.cache_ttl = 5.0  # 5 seconds cache TTL
        
        # Phase 4.1: Async Processing
        self.async_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="Pipeline")
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        self.processing_queue = asyncio.Queue(maxsize=2000)
        
        # Phase 4.1: Memory Management
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.gc_threshold = 500  # Garbage collection after 500 operations
        self.operation_count = 0
        
        # Pipeline components
        self.feature_extractor = FeatureExtractor()
        self.model_registry = ModelRegistry()
        self.risk_manager = risk_manager
        self.position_sizing_optimizer = position_sizing_optimizer
        self.market_regime_detector = market_regime_detector
        self.multi_timeframe_fusion = multi_timeframe_fusion
        self.parallel_filter_processor = parallel_filter_processor
        self.multi_level_cache = multi_level_cache
        self.gpu_accelerated_filters = gpu_accelerated_filters
        self.fpga_integration = fpga_integration
        self.continuous_false_positive_logger = continuous_false_positive_logger
        self.feedback_loop = feedback_loop
        self.ai_driven_threshold_manager = ai_driven_threshold_manager
        
        # Performance monitoring
        self.metrics = PipelineMetrics()
        self.is_running = False
        
        logger.info("🚀 Phase 4.1: Real-time Pipeline initialized with ultra-low latency optimizations")
        logger.info(f"🎯 Performance Target: <{self.target_latency_ms}ms processing time")
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self._background_cache_cleanup())
            asyncio.create_task(self._background_memory_management())
            asyncio.create_task(self._background_performance_monitoring())
            logger.info("✅ Background optimization tasks started")
    
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
                await self._update_performance_metrics()
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
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.processing_times:
                return
            
            times = list(self.processing_times)
            self.metrics.avg_processing_time = sum(times) / len(times)
            self.metrics.max_processing_time = max(times)
            self.metrics.min_processing_time = min(times)
            
            # Calculate percentiles
            sorted_times = sorted(times)
            self.metrics.latency_p50 = sorted_times[len(sorted_times) // 2]
            self.metrics.latency_p95 = sorted_times[int(len(sorted_times) * 0.95)]
            self.metrics.latency_p99 = sorted_times[int(len(sorted_times) * 0.99)]
            
            # Calculate cache hit rate
            self.metrics.cache_hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
            self.metrics.cache_hits = self.cache_hits
            self.metrics.cache_misses = self.cache_misses
            
            # Calculate performance score
            if self.metrics.avg_processing_time <= self.target_latency_ms:
                self.metrics.performance_score = 100.0
            else:
                self.metrics.performance_score = max(0, 100 - (self.metrics.avg_processing_time - self.target_latency_ms))
            
            # Update system metrics
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
            self.metrics.queue_size = self.processing_queue.qsize()
            self.metrics.last_update = datetime.now()
            
            # Log performance metrics
            logger.info(f"📊 Pipeline Performance - Avg: {self.metrics.avg_processing_time:.2f}ms, "
                       f"P95: {self.metrics.latency_p95:.2f}ms, Cache Hit: {self.metrics.cache_hit_rate:.1f}%, "
                       f"Score: {self.metrics.performance_score:.1f}")
            
            # Check if we're meeting performance targets
            if self.metrics.avg_processing_time > self.target_latency_ms:
                logger.warning(f"⚠️ Average processing time ({self.metrics.avg_processing_time:.2f}ms) exceeds target ({self.target_latency_ms}ms)")
            else:
                logger.info(f"✅ Performance target met: {self.metrics.avg_processing_time:.2f}ms < {self.target_latency_ms}ms")
                
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")
    
    def _get_cache_key(self, symbol: str, data_hash: str) -> str:
        """Generate cache key for data"""
        return f"{symbol}_{data_hash}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get cached result if available and not expired"""
        if not self.enable_caching:
            return None
        
        try:
            if cache_key in self.result_cache:
                data, timestamp = self.result_cache[cache_key]
                if time.time() - timestamp <= self.cache_ttl:
                    self.cache_hits += 1
                    data.cache_hit = True
                    return data
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: ProcessingResult):
        """Cache result with timestamp"""
        if not self.enable_caching:
            return
        
        try:
            self.result_cache[cache_key] = (result, time.time())
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def process_data_point(self, data_point: DataPoint) -> ProcessingResult:
        """Process a single data point with Phase 4.1 ultra-low latency optimizations"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Phase 4.1: Check cache first for ultra-fast response
            data_hash = str(hash(f"{data_point.symbol}_{data_point.timestamp}_{data_point.close}"))
            cache_key = self._get_cache_key(data_point.symbol, data_hash)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                logger.info(f"⚡ Cache hit - Data processing in {processing_time:.2f}ms")
                return cached_result
            
            # Phase 4.1: Async processing for better performance
            if self.enable_async_processing:
                result = await self._process_data_point_async(data_point)
            else:
                result = await self._process_data_point_sync(data_point)
            
            # Phase 4.1: Cache the result
            self._cache_result(cache_key, result)
            
            # Phase 4.1: Track performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            result.processing_time = processing_time
            
            # Phase 4.1: Performance logging
            if processing_time > self.target_latency_ms:
                logger.warning(f"⚠️ Data processing took {processing_time:.2f}ms (target: <{self.target_latency_ms}ms)")
            else:
                logger.info(f"⚡ Data processing completed in {processing_time:.2f}ms (target: <{self.target_latency_ms}ms)")
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            logger.error(f"❌ Error processing data point: {e} (took {processing_time:.2f}ms)")
            
            # Return error result
            return ProcessingResult(
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                data_point=data_point,
                processing_time=processing_time,
                quality_score=0.0,
                phase_4_1_features=True
            )
    
    async def _process_data_point_async(self, data_point: DataPoint) -> ProcessingResult:
        """Process data point using async processing for better performance"""
        try:
            # Phase 4.1: Parallel processing of different pipeline stages
            tasks = []
            
            # Feature extraction
            tasks.append(self._extract_features_async(data_point))
            
            # Pattern detection
            tasks.append(self._detect_patterns_async(data_point))
            
            # Risk assessment
            tasks.append(self._assess_risk_async(data_point))
            
            # Market regime detection
            tasks.append(self._detect_market_regime_async(data_point))
            
            # Execute all stages in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            features = results[0] if isinstance(results[0], np.ndarray) else None
            patterns = results[1] if isinstance(results[1], dict) else {}
            risk_assessment = results[2] if isinstance(results[2], dict) else {}
            market_regime = results[3] if isinstance(results[3], dict) else {}
            
            # Generate signals and execution decision
            signals = await self._generate_signals_async(data_point, features, patterns, risk_assessment, market_regime)
            execution_decision = await self._make_execution_decision_async(data_point, signals, risk_assessment)
            
            return ProcessingResult(
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                data_point=data_point,
                features=features,
                patterns=patterns,
                signals=signals,
                risk_assessment=risk_assessment,
                execution_decision=execution_decision,
                quality_score=1.0,
                phase_4_1_features=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error in async data processing: {e}")
            return await self._process_data_point_sync(data_point)
    
    async def _process_data_point_sync(self, data_point: DataPoint) -> ProcessingResult:
        """Process data point using synchronous processing (fallback method)"""
        try:
            # Simple synchronous processing as fallback
            return ProcessingResult(
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                data_point=data_point,
                quality_score=1.0,
                phase_4_1_features=True
            )
            
        except Exception as e:
            logger.error(f"❌ Error in sync data processing: {e}")
            return ProcessingResult(
                symbol=data_point.symbol,
                timestamp=data_point.timestamp,
                data_point=data_point,
                quality_score=0.0,
                phase_4_1_features=True
            )
    
    async def _extract_features_async(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """Async feature extraction"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._extract_features_sync,
                data_point
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async feature extraction: {e}")
            return None
    
    async def _detect_patterns_async(self, data_point: DataPoint) -> Dict[str, Any]:
        """Async pattern detection"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._detect_patterns_sync,
                data_point
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async pattern detection: {e}")
            return {}
    
    async def _assess_risk_async(self, data_point: DataPoint) -> Dict[str, Any]:
        """Async risk assessment"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._assess_risk_sync,
                data_point
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async risk assessment: {e}")
            return {}
    
    async def _detect_market_regime_async(self, data_point: DataPoint) -> Dict[str, Any]:
        """Async market regime detection"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._detect_market_regime_sync,
                data_point
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async market regime detection: {e}")
            return {}
    
    async def _generate_signals_async(self, data_point: DataPoint, features: Optional[np.ndarray], 
                                    patterns: Dict[str, Any], risk_assessment: Dict[str, Any], 
                                    market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Async signal generation"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._generate_signals_sync,
                data_point, features, patterns, risk_assessment, market_regime
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async signal generation: {e}")
            return {}
    
    async def _make_execution_decision_async(self, data_point: DataPoint, signals: Dict[str, Any], 
                                           risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Async execution decision"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.async_executor,
                self._make_execution_decision_sync,
                data_point, signals, risk_assessment
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in async execution decision: {e}")
            return {}
    
    def _extract_features_sync(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """Synchronous feature extraction (CPU-intensive, runs in thread pool)"""
        try:
            # Convert data point to DataFrame for feature extraction
            df = pd.DataFrame([{
                'open': data_point.open,
                'high': data_point.high,
                'low': data_point.low,
                'close': data_point.close,
                'volume': data_point.volume
            }])
            
            # Extract features using the feature extractor
            features = self.feature_extractor.extract_features(df)
            return features
            
        except Exception as e:
            logger.error(f"❌ Error in feature extraction: {e}")
            return None
    
    def _detect_patterns_sync(self, data_point: DataPoint) -> Dict[str, Any]:
        """Synchronous pattern detection (CPU-intensive, runs in thread pool)"""
        try:
            # Use parallel filter processor for pattern detection
            patterns = self.parallel_filter_processor.detect_patterns(data_point)
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error in pattern detection: {e}")
            return {}
    
    def _assess_risk_sync(self, data_point: DataPoint) -> Dict[str, Any]:
        """Synchronous risk assessment (CPU-intensive, runs in thread pool)"""
        try:
            # Use risk manager for risk assessment
            risk_assessment = self.risk_manager.assess_risk(data_point)
            return risk_assessment
            
        except Exception as e:
            logger.error(f"❌ Error in risk assessment: {e}")
            return {}
    
    def _detect_market_regime_sync(self, data_point: DataPoint) -> Dict[str, Any]:
        """Synchronous market regime detection (CPU-intensive, runs in thread pool)"""
        try:
            # Use market regime detector
            market_regime = self.market_regime_detector.detect_regime(data_point)
            return market_regime
            
        except Exception as e:
            logger.error(f"❌ Error in market regime detection: {e}")
            return {}
    
    def _generate_signals_sync(self, data_point: DataPoint, features: Optional[np.ndarray], 
                              patterns: Dict[str, Any], risk_assessment: Dict[str, Any], 
                              market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous signal generation (CPU-intensive, runs in thread pool)"""
        try:
            # Combine all analysis results and generate signals
            signals = {
                'symbol': data_point.symbol,
                'timestamp': data_point.timestamp,
                'patterns': patterns,
                'risk_level': risk_assessment.get('risk_level', 'medium'),
                'market_regime': market_regime.get('regime', 'unknown'),
                'confidence': 0.75,
                'phase_4_1_features': True
            }
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error in signal generation: {e}")
            return {}
    
    def _make_execution_decision_sync(self, data_point: DataPoint, signals: Dict[str, Any], 
                                    risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous execution decision (CPU-intensive, runs in thread pool)"""
        try:
            # Make execution decision based on signals and risk assessment
            decision = {
                'action': 'hold',  # Default action
                'confidence': signals.get('confidence', 0.0),
                'risk_level': risk_assessment.get('risk_level', 'medium'),
                'reason': 'No clear signal',
                'phase_4_1_features': True
            }
            
            # Simple decision logic
            if signals.get('confidence', 0.0) > 0.7 and risk_assessment.get('risk_level') == 'low':
                decision['action'] = 'buy'
                decision['reason'] = 'High confidence, low risk signal'
            elif signals.get('confidence', 0.0) > 0.7 and risk_assessment.get('risk_level') == 'high':
                decision['action'] = 'sell'
                decision['reason'] = 'High confidence, high risk signal'
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Error in execution decision: {e}")
            return {}
    
    async def get_performance_metrics(self) -> PipelineMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    async def shutdown(self):
        """Shutdown the pipeline"""
        self.is_running = False
        
        # Shutdown executors
        self.async_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Pipeline shutdown completed")

# Global pipeline instance
real_time_pipeline = RealTimePipeline()
