"""
Parallel Filter Processor for AlphaPulse
High-performance parallel processing of signal validation filters
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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

logger = logging.getLogger(__name__)

class FilterType(Enum):
    """Types of signal validation filters"""
    VOLUME_CONFIRMATION = "volume_confirmation"
    MULTI_TIMEFRAME_ALIGNMENT = "multi_timeframe_alignment"
    TREND_CHECK = "trend_check"

@dataclass
class FilterResult:
    """Result of a single filter validation"""
    filter_type: FilterType
    passed: bool
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParallelFilterResult:
    """Combined result from all parallel filters"""
    symbol: str
    overall_passed: bool
    overall_confidence: float
    filter_results: Dict[FilterType, FilterResult]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)

class ParallelFilterProcessor:
    """High-performance parallel filter processor"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.volume_history = defaultdict(lambda: deque(maxlen=50))
        self.price_history = defaultdict(lambda: deque(maxlen=50))
        
        logger.info(f"ParallelFilterProcessor initialized with {max_workers} workers")
    
    async def process_signal_batch(self, signals: List[Dict[str, Any]]) -> List[ParallelFilterResult]:
        """Process a batch of signals in parallel"""
        if not signals:
            return []
        
        start_time = time.time()
        results = []
        
        # Process signals in parallel batches
        batches = [signals[i:i + self.batch_size] for i in range(0, len(signals), self.batch_size)]
        
        for batch in batches:
            batch_tasks = [self._process_single_signal(signal) for signal in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, ParallelFilterResult):
                    results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(signals)} signals in {processing_time:.3f}s")
        
        return results
    
    async def _process_single_signal(self, signal_data: Dict[str, Any]) -> ParallelFilterResult:
        """Process a single signal with parallel filters"""
        symbol = signal_data.get('symbol', 'unknown')
        price = signal_data.get('price', 0.0)
        volume = signal_data.get('volume', 0.0)
        direction = signal_data.get('direction', SignalDirection.NEUTRAL)
        
        # Run filters in parallel
        filter_tasks = [
            self._volume_filter(symbol, volume),
            self._trend_filter(symbol, price, direction),
            self._timeframe_filter(signal_data)
        ]
        
        filter_results = await asyncio.gather(*filter_tasks)
        
        # Combine results
        overall_passed = all(r.passed for r in filter_results)
        overall_confidence = np.mean([r.confidence for r in filter_results])
        
        return ParallelFilterResult(
            symbol=symbol,
            overall_passed=overall_passed,
            overall_confidence=overall_confidence,
            filter_results={r.filter_type: r for r in filter_results},
            processing_time=sum(r.processing_time for r in filter_results)
        )
    
    async def _volume_filter(self, symbol: str, volume: float) -> FilterResult:
        """Volume confirmation filter"""
        start_time = time.time()
        
        self.volume_history[symbol].append(volume)
        
        if len(self.volume_history[symbol]) < 5:
            return FilterResult(
                filter_type=FilterType.VOLUME_CONFIRMATION,
                passed=volume > 1000,
                confidence=0.5,
                processing_time=time.time() - start_time
            )
        
        avg_volume = np.mean(list(self.volume_history[symbol]))
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        passed = volume > 1000 and volume_ratio < 5.0
        confidence = min(1.0, volume_ratio / 2.0)
        
        return FilterResult(
            filter_type=FilterType.VOLUME_CONFIRMATION,
            passed=passed,
            confidence=confidence,
            processing_time=time.time() - start_time
        )
    
    async def _trend_filter(self, symbol: str, price: float, direction: SignalDirection) -> FilterResult:
        """Trend validation filter"""
        start_time = time.time()
        
        self.price_history[symbol].append(price)
        
        if len(self.price_history[symbol]) < 20:
            return FilterResult(
                filter_type=FilterType.TREND_CHECK,
                passed=True,
                confidence=0.5,
                processing_time=time.time() - start_time
            )
        
        prices = list(self.price_history[symbol])
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        trend_direction = SignalDirection.BULLISH if slope > 0 else SignalDirection.BEARISH
        trend_aligned = direction == trend_direction
        
        confidence = 0.8 if trend_aligned else 0.3
        
        return FilterResult(
            filter_type=FilterType.TREND_CHECK,
            passed=trend_aligned,
            confidence=confidence,
            processing_time=time.time() - start_time
        )
    
    async def _timeframe_filter(self, signal_data: Dict[str, Any]) -> FilterResult:
        """Multi-timeframe alignment filter"""
        start_time = time.time()
        
        timeframe_signals = signal_data.get('timeframe_signals', {})
        
        if not timeframe_signals:
            return FilterResult(
                filter_type=FilterType.MULTI_TIMEFRAME_ALIGNMENT,
                passed=False,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Calculate agreement
        directions = [s.direction for s in timeframe_signals.values()]
        bullish_count = sum(1 for d in directions if d == SignalDirection.BULLISH)
        bearish_count = sum(1 for d in directions if d == SignalDirection.BEARISH)
        
        total_signals = len(directions)
        agreement = max(bullish_count, bearish_count) / total_signals if total_signals > 0 else 0
        
        passed = agreement >= 0.6
        confidence = agreement
        
        return FilterResult(
            filter_type=FilterType.MULTI_TIMEFRAME_ALIGNMENT,
            passed=passed,
            confidence=confidence,
            processing_time=time.time() - start_time
        )

# Global instance
parallel_filter_processor = ParallelFilterProcessor(max_workers=4, batch_size=100)
