"""
GPU-Accelerated Filter Processing for AlphaPulse
Hardware acceleration for compute-intensive signal validation filters
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime
import threading

# GPU acceleration imports
try:
    import cupy as cp
    import numba
    from numba import cuda, jit
    GPU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("GPU acceleration libraries available")
except ImportError:
    GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("GPU acceleration libraries not available, falling back to CPU")

try:
    from .multi_timeframe_fusion import SignalDirection, SignalStrength
except ImportError:
    try:
        from multi_timeframe_fusion import SignalDirection, SignalStrength
    except ImportError:
        # Fallback definitions if imports fail
        from enum import Enum
        class SignalDirection(Enum):
            BULLISH = "bullish"
            BEARISH = "bearish"
            NEUTRAL = "neutral"
        
        class SignalStrength(Enum):
            WEAK = "weak"
            MODERATE = "moderate"
            STRONG = "strong"

class GPUFilterType(Enum):
    """Types of GPU-accelerated filters"""
    ADX_CALCULATION = "adx_calculation"
    VOLUME_PATTERN_ANALYSIS = "volume_pattern_analysis"
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_MATCHING = "pattern_matching"

@dataclass
class GPUFilterResult:
    """Result of GPU-accelerated filter processing"""
    filter_type: GPUFilterType
    passed: bool
    confidence: float
    processing_time: float
    gpu_memory_used: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GPUAcceleratedResult:
    """Combined result from GPU-accelerated processing"""
    symbol: str
    overall_passed: bool
    overall_confidence: float
    filter_results: Dict[GPUFilterType, GPUFilterResult]
    processing_time: float
    gpu_utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class GPUManager:
    """GPU memory and resource manager"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.gpu_memory_pool = {}
        self.gpu_streams = {}
        self.lock = threading.RLock()
        
        if self.gpu_available:
            try:
                # Initialize GPU memory pool
                self._init_gpu_memory_pool()
                logger.info("GPU Manager initialized successfully")
            except Exception as e:
                logger.error(f"GPU initialization failed: {e}")
                self.gpu_available = False
    
    def _init_gpu_memory_pool(self):
        """Initialize GPU memory pool"""
        if not self.gpu_available:
            return
        
        try:
            # Allocate memory pools for different data types
            self.gpu_memory_pool['float32'] = cp.cuda.MemoryPool()
            self.gpu_memory_pool['float64'] = cp.cuda.MemoryPool()
            self.gpu_memory_pool['int32'] = cp.cuda.MemoryPool()
            
            # Create CUDA streams for parallel processing
            for i in range(4):
                self.gpu_streams[f'stream_{i}'] = cp.cuda.Stream()
            
            logger.info("GPU memory pool initialized")
        except Exception as e:
            logger.error(f"GPU memory pool initialization failed: {e}")
            self.gpu_available = False
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            return {
                'available': True,
                'total_memory': meminfo[1],
                'free_memory': meminfo[0],
                'used_memory': meminfo[1] - meminfo[0],
                'utilization': (meminfo[1] - meminfo[0]) / meminfo[1] * 100
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {'available': False}
    
    def allocate_array(self, shape: Tuple[int, ...], dtype: str = 'float32') -> Optional[Any]:
        """Allocate GPU array with memory pooling"""
        if not self.gpu_available:
            return None
        
        try:
            with self.gpu_memory_pool[dtype]:
                return cp.zeros(shape, dtype=dtype)
        except Exception as e:
            logger.error(f"GPU allocation failed: {e}")
            return None
    
    def get_stream(self, stream_id: str = 'stream_0') -> Optional[Any]:
        """Get CUDA stream for parallel processing"""
        if not self.gpu_available:
            return None
        
        return self.gpu_streams.get(stream_id)

# GPU-accelerated ADX calculation
if GPU_AVAILABLE:
    @cuda.jit
    def _adx_kernel(high, low, close, tr, dx, adx, period):
        """CUDA kernel for ADX calculation"""
        idx = cuda.grid(1)
        
        if idx >= high.shape[0]:
            return
        
        # Calculate True Range
        if idx > 0:
            hl = high[idx] - low[idx]
            hc = abs(high[idx] - close[idx - 1])
            lc = abs(low[idx] - close[idx - 1])
            tr[idx] = max(hl, hc, lc)
        
        # Calculate Directional Movement
        if idx > 0:
            up_move = high[idx] - high[idx - 1]
            down_move = low[idx - 1] - low[idx]
            
            if up_move > down_move and up_move > 0:
                dx[idx] = up_move
            elif down_move > up_move and down_move > 0:
                dx[idx] = -down_move
            else:
                dx[idx] = 0.0
        
        # Calculate ADX using exponential moving average
        if idx >= period:
            # Simple moving average for ADX
            sum_dx = 0.0
            sum_tr = 0.0
            
            for i in range(period):
                sum_dx += abs(dx[idx - i])
                sum_tr += tr[idx - i]
            
            if sum_tr > 0:
                adx[idx] = (sum_dx / sum_tr) * 100.0

class GPUAcceleratedFilters:
    """GPU-accelerated signal validation filters"""
    
    def __init__(self, 
                enable_gpu: bool = True,
                batch_size: int = 1000,
                max_workers: int = 4):
        
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # GPU manager
        self.gpu_manager = GPUManager()
        
        # Data buffers
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.gpu_processing_times = deque(maxlen=1000)
        self.cpu_processing_times = deque(maxlen=1000)
        
        logger.info(f"GPU Accelerated Filters initialized (GPU: {self.enable_gpu})")
    
    async def process_batch_gpu(self, signals: List[Dict[str, Any]]) -> List[GPUAcceleratedResult]:
        """Process a batch of signals using GPU acceleration"""
        if not signals:
            return []
        
        start_time = time.time()
        results = []
        
        if self.enable_gpu:
            # GPU processing
            gpu_results = await self._process_gpu_batch(signals)
            results.extend(gpu_results)
            self.gpu_processing_times.append(time.time() - start_time)
        else:
            # CPU fallback
            cpu_results = await self._process_cpu_batch(signals)
            results.extend(cpu_results)
            self.cpu_processing_times.append(time.time() - start_time)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(signals)} signals in {processing_time:.3f}s (GPU: {self.enable_gpu})")
        
        return results
    
    async def _process_gpu_batch(self, signals: List[Dict[str, Any]]) -> List[GPUAcceleratedResult]:
        """Process batch using GPU acceleration"""
        results = []
        
        # Group signals by symbol for batch processing
        symbol_groups = defaultdict(list)
        for signal in signals:
            symbol_groups[signal['symbol']].append(signal)
        
        # Process each symbol group
        for symbol, symbol_signals in symbol_groups.items():
            try:
                # Update price and volume history
                for signal in symbol_signals:
                    self.price_history[symbol].append(signal['price'])
                    self.volume_history[symbol].append(signal['volume'])
                
                # Run GPU-accelerated filters
                filter_tasks = [
                    self._gpu_adx_filter(symbol),
                    self._gpu_volume_pattern_filter(symbol),
                    self._gpu_technical_indicators_filter(symbol)
                ]
                
                filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)
                
                # Process results
                valid_results = {}
                for i, result in enumerate(filter_results):
                    if isinstance(result, GPUFilterResult):
                        filter_type = list(GPUFilterType)[i]
                        valid_results[filter_type] = result
                
                # Calculate overall result
                if valid_results:
                    overall_passed = all(r.passed for r in valid_results.values())
                    overall_confidence = np.mean([r.confidence for r in valid_results.values()])
                    
                    # Get GPU utilization
                    gpu_info = self.gpu_manager.get_memory_info()
                    gpu_utilization = gpu_info.get('utilization', 0.0) if gpu_info.get('available', False) else 0.0
                    
                    result = GPUAcceleratedResult(
                        symbol=symbol,
                        overall_passed=overall_passed,
                        overall_confidence=overall_confidence,
                        filter_results=valid_results,
                        processing_time=sum(r.processing_time for r in valid_results.values()),
                        gpu_utilization=gpu_utilization
                    )
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"GPU processing error for {symbol}: {e}")
                continue
        
        return results
    
    async def _process_cpu_batch(self, signals: List[Dict[str, Any]]) -> List[GPUAcceleratedResult]:
        """Process batch using CPU fallback"""
        results = []
        
        for signal in signals:
            try:
                symbol = signal['symbol']
                
                # Update history
                self.price_history[symbol].append(signal['price'])
                self.volume_history[symbol].append(signal['volume'])
                
                # Run CPU filters
                filter_tasks = [
                    self._cpu_adx_filter(symbol),
                    self._cpu_volume_pattern_filter(symbol),
                    self._cpu_technical_indicators_filter(symbol)
                ]
                
                filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)
                
                # Process results
                valid_results = {}
                for i, result in enumerate(filter_results):
                    if isinstance(result, GPUFilterResult):
                        filter_type = list(GPUFilterType)[i]
                        valid_results[filter_type] = result
                
                if valid_results:
                    overall_passed = all(r.passed for r in valid_results.values())
                    overall_confidence = np.mean([r.confidence for r in valid_results.values()])
                    
                    result = GPUAcceleratedResult(
                        symbol=symbol,
                        overall_passed=overall_passed,
                        overall_confidence=overall_confidence,
                        filter_results=valid_results,
                        processing_time=sum(r.processing_time for r in valid_results.values()),
                        gpu_utilization=0.0
                    )
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"CPU processing error: {e}")
                continue
        
        return results
    
    async def _gpu_adx_filter(self, symbol: str) -> GPUFilterResult:
        """GPU-accelerated ADX filter"""
        start_time = time.time()
        
        try:
            prices = list(self.price_history[symbol])
            if len(prices) < 50:
                return GPUFilterResult(
                    filter_type=GPUFilterType.ADX_CALCULATION,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            if not self.enable_gpu:
                return await self._cpu_adx_filter(symbol)
            
            # Convert to GPU arrays
            high = cp.array(prices, dtype=cp.float32)
            low = cp.array(prices, dtype=cp.float32)
            close = cp.array(prices, dtype=cp.float32)
            
            # Allocate output arrays
            tr = cp.zeros_like(high)
            dx = cp.zeros_like(high)
            adx = cp.zeros_like(high)
            
            # Configure CUDA grid
            threadsperblock = 256
            blockspergrid = (high.size + (threadsperblock - 1)) // threadsperblock
            
            # Run ADX kernel
            _adx_kernel[blockspergrid, threadsperblock](high, low, close, tr, dx, adx, 14)
            
            # Get result
            current_adx = float(adx[-1])
            
            # Determine if trend is strong
            passed = current_adx > 25.0
            confidence = min(1.0, current_adx / 50.0)
            
            # Get GPU memory usage
            gpu_info = self.gpu_manager.get_memory_info()
            gpu_memory_used = gpu_info.get('used_memory', 0.0) if gpu_info.get('available', False) else 0.0
            
            return GPUFilterResult(
                filter_type=GPUFilterType.ADX_CALCULATION,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                gpu_memory_used=gpu_memory_used,
                metadata={'adx_value': current_adx}
            )
            
        except Exception as e:
            logger.error(f"GPU ADX filter error: {e}")
            return await self._cpu_adx_filter(symbol)
    
    async def _gpu_volume_pattern_filter(self, symbol: str) -> GPUFilterResult:
        """GPU-accelerated volume pattern analysis"""
        start_time = time.time()
        
        try:
            volumes = list(self.volume_history[symbol])
            if len(volumes) < 20:
                return GPUFilterResult(
                    filter_type=GPUFilterType.VOLUME_PATTERN_ANALYSIS,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            if not self.enable_gpu:
                return await self._cpu_volume_pattern_filter(symbol)
            
            # Convert to GPU array
            volume_array = cp.array(volumes, dtype=cp.float32)
            
            # Calculate volume statistics on GPU
            mean_volume = float(cp.mean(volume_array))
            std_volume = float(cp.std(volume_array))
            current_volume = float(volume_array[-1])
            
            # Volume pattern analysis
            volume_ratio = current_volume / mean_volume if mean_volume > 0 else 1.0
            volume_zscore = (current_volume - mean_volume) / std_volume if std_volume > 0 else 0.0
            
            # Determine if volume pattern is valid
            passed = 0.1 <= volume_ratio <= 10.0 and abs(volume_zscore) <= 3.0
            confidence = min(1.0, 1.0 - abs(volume_zscore) / 5.0)
            
            # Get GPU memory usage
            gpu_info = self.gpu_manager.get_memory_info()
            gpu_memory_used = gpu_info.get('used_memory', 0.0) if gpu_info.get('available', False) else 0.0
            
            return GPUFilterResult(
                filter_type=GPUFilterType.VOLUME_PATTERN_ANALYSIS,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                gpu_memory_used=gpu_memory_used,
                metadata={
                    'volume_ratio': volume_ratio,
                    'volume_zscore': volume_zscore,
                    'mean_volume': mean_volume
                }
            )
            
        except Exception as e:
            logger.error(f"GPU volume pattern filter error: {e}")
            return await self._cpu_volume_pattern_filter(symbol)
    
    async def _gpu_technical_indicators_filter(self, symbol: str) -> GPUFilterResult:
        """GPU-accelerated technical indicators filter"""
        start_time = time.time()
        
        try:
            prices = list(self.price_history[symbol])
            if len(prices) < 50:
                return GPUFilterResult(
                    filter_type=GPUFilterType.TECHNICAL_INDICATORS,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            if not self.enable_gpu:
                return await self._cpu_technical_indicators_filter(symbol)
            
            # Convert to GPU array
            price_array = cp.array(prices, dtype=cp.float32)
            
            # Calculate technical indicators on GPU
            # RSI calculation
            delta = cp.diff(price_array)
            gain = cp.where(delta > 0, delta, 0)
            loss = cp.where(delta < 0, -delta, 0)
            
            avg_gain = cp.mean(gain[-14:]) if len(gain) >= 14 else cp.mean(gain)
            avg_loss = cp.mean(loss[-14:]) if len(loss) >= 14 else cp.mean(loss)
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            sma_20 = float(cp.mean(price_array[-20:])) if len(price_array) >= 20 else float(cp.mean(price_array))
            sma_50 = float(cp.mean(price_array[-50:])) if len(price_array) >= 50 else float(cp.mean(price_array))
            
            current_price = float(price_array[-1])
            
            # Technical analysis
            rsi_value = float(rsi)
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            
            # Determine if technical indicators are favorable
            passed = 30 <= rsi_value <= 70 and price_above_sma20
            confidence = 0.5 + 0.5 * (1.0 - abs(rsi_value - 50) / 50.0)
            
            # Get GPU memory usage
            gpu_info = self.gpu_manager.get_memory_info()
            gpu_memory_used = gpu_info.get('used_memory', 0.0) if gpu_info.get('available', False) else 0.0
            
            return GPUFilterResult(
                filter_type=GPUFilterType.TECHNICAL_INDICATORS,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                gpu_memory_used=gpu_memory_used,
                metadata={
                    'rsi': rsi_value,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'price_above_sma20': price_above_sma20,
                    'price_above_sma50': price_above_sma50
                }
            )
            
        except Exception as e:
            logger.error(f"GPU technical indicators filter error: {e}")
            return await self._cpu_technical_indicators_filter(symbol)
    
    # CPU fallback methods
    async def _cpu_adx_filter(self, symbol: str) -> GPUFilterResult:
        """CPU fallback for ADX filter"""
        start_time = time.time()
        
        try:
            prices = list(self.price_history[symbol])
            if len(prices) < 50:
                return GPUFilterResult(
                    filter_type=GPUFilterType.ADX_CALCULATION,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            # Simple ADX calculation on CPU
            high = np.array(prices)
            low = np.array(prices)
            close = np.array(prices)
            
            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # Calculate Directional Movement
            dx = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    dx[i] = up_move
                elif down_move > up_move and down_move > 0:
                    dx[i] = -down_move
            
            # Calculate ADX
            period = 14
            adx = np.zeros_like(high)
            for i in range(period, len(high)):
                sum_dx = np.sum(np.abs(dx[i-period+1:i+1]))
                sum_tr = np.sum(tr[i-period+1:i+1])
                if sum_tr > 0:
                    adx[i] = (sum_dx / sum_tr) * 100.0
            
            current_adx = adx[-1]
            passed = current_adx > 25.0
            confidence = min(1.0, current_adx / 50.0)
            
            return GPUFilterResult(
                filter_type=GPUFilterType.ADX_CALCULATION,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                metadata={'adx_value': current_adx}
            )
            
        except Exception as e:
            logger.error(f"CPU ADX filter error: {e}")
            return GPUFilterResult(
                filter_type=GPUFilterType.ADX_CALCULATION,
                passed=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _cpu_volume_pattern_filter(self, symbol: str) -> GPUFilterResult:
        """CPU fallback for volume pattern filter"""
        start_time = time.time()
        
        try:
            volumes = list(self.volume_history[symbol])
            if len(volumes) < 20:
                return GPUFilterResult(
                    filter_type=GPUFilterType.VOLUME_PATTERN_ANALYSIS,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            volume_array = np.array(volumes)
            mean_volume = np.mean(volume_array)
            std_volume = np.std(volume_array)
            current_volume = volume_array[-1]
            
            volume_ratio = current_volume / mean_volume if mean_volume > 0 else 1.0
            volume_zscore = (current_volume - mean_volume) / std_volume if std_volume > 0 else 0.0
            
            passed = 0.1 <= volume_ratio <= 10.0 and abs(volume_zscore) <= 3.0
            confidence = min(1.0, 1.0 - abs(volume_zscore) / 5.0)
            
            return GPUFilterResult(
                filter_type=GPUFilterType.VOLUME_PATTERN_ANALYSIS,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                metadata={
                    'volume_ratio': volume_ratio,
                    'volume_zscore': volume_zscore,
                    'mean_volume': mean_volume
                }
            )
            
        except Exception as e:
            logger.error(f"CPU volume pattern filter error: {e}")
            return GPUFilterResult(
                filter_type=GPUFilterType.VOLUME_PATTERN_ANALYSIS,
                passed=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _cpu_technical_indicators_filter(self, symbol: str) -> GPUFilterResult:
        """CPU fallback for technical indicators filter"""
        start_time = time.time()
        
        try:
            prices = list(self.price_history[symbol])
            if len(prices) < 50:
                return GPUFilterResult(
                    filter_type=GPUFilterType.TECHNICAL_INDICATORS,
                    passed=True,
                    confidence=0.5,
                    processing_time=time.time() - start_time,
                    metadata={'reason': 'insufficient_data'}
                )
            
            price_array = np.array(prices)
            
            # Calculate RSI
            delta = np.diff(price_array)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            sma_20 = np.mean(price_array[-20:]) if len(price_array) >= 20 else np.mean(price_array)
            sma_50 = np.mean(price_array[-50:]) if len(price_array) >= 50 else np.mean(price_array)
            
            current_price = price_array[-1]
            
            rsi_value = rsi
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            
            passed = 30 <= rsi_value <= 70 and price_above_sma20
            confidence = 0.5 + 0.5 * (1.0 - abs(rsi_value - 50) / 50.0)
            
            return GPUFilterResult(
                filter_type=GPUFilterType.TECHNICAL_INDICATORS,
                passed=passed,
                confidence=confidence,
                processing_time=time.time() - start_time,
                metadata={
                    'rsi': rsi_value,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'price_above_sma20': price_above_sma20,
                    'price_above_sma50': price_above_sma50
                }
            )
            
        except Exception as e:
            logger.error(f"CPU technical indicators filter error: {e}")
            return GPUFilterResult(
                filter_type=GPUFilterType.TECHNICAL_INDICATORS,
                passed=False,
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        gpu_info = self.gpu_manager.get_memory_info()
        
        return {
            'gpu_available': self.enable_gpu,
            'gpu_memory_info': gpu_info,
            'avg_gpu_processing_time': np.mean(list(self.gpu_processing_times)) if self.gpu_processing_times else 0.0,
            'avg_cpu_processing_time': np.mean(list(self.cpu_processing_times)) if self.cpu_processing_times else 0.0,
            'gpu_processing_count': len(self.gpu_processing_times),
            'cpu_processing_count': len(self.cpu_processing_times)
        }

# Global GPU-accelerated filters instance
gpu_accelerated_filters = GPUAcceleratedFilters(
    enable_gpu=True,
    batch_size=1000,
    max_workers=4
)
