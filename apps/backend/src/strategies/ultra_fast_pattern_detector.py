"""
Ultra-Fast Candlestick Pattern Detector for AlphaPlus
Implements vectorized detection, sliding windows, and async parallelization
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import numba
from numba import jit, prange
import talib

logger = logging.getLogger(__name__)

@dataclass
class PatternDetectionResult:
    """Result of pattern detection with enhanced metadata"""
    pattern_name: str
    confidence: float
    strength: str  # 'weak', 'moderate', 'strong'
    direction: str  # 'bullish', 'bearish', 'neutral'
    timestamp: datetime
    price_level: float
    volume_confirmation: bool
    volume_confidence: float
    trend_alignment: str
    metadata: Dict[str, Any]
    detection_latency_ms: float

class SlidingWindowBuffer:
    """Efficient sliding window buffer for candlestick data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = {}
        self.timestamps = {}
        
    def add_candle(self, symbol: str, timeframe: str, candle_data: Dict):
        """Add new candle to buffer, maintaining sliding window"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.buffer:
            self.buffer[key] = []
            self.timestamps[key] = []
        
        # Add new data
        self.buffer[key].append(candle_data)
        self.timestamps[key].append(candle_data['timestamp'])
        
        # Maintain sliding window size
        if len(self.buffer[key]) > self.max_size:
            self.buffer[key].pop(0)
            self.timestamps[key].pop(0)
    
    def get_recent_candles(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Get recent candles from buffer"""
        key = f"{symbol}_{timeframe}"
        if key in self.buffer:
            return self.buffer[key][-count:]
        return []
    
    def get_all_candles(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get all candles from buffer"""
        key = f"{symbol}_{timeframe}"
        return self.buffer.get(key, [])

@jit(nopython=True, parallel=True)
def vectorized_doji_detection(opens: np.ndarray, highs: np.ndarray, 
                             lows: np.ndarray, closes: np.ndarray, 
                             threshold: float = 0.1) -> np.ndarray:
    """Vectorized doji detection using Numba for ultra-fast performance"""
    n = len(opens)
    results = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):
        body_size = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i]
        
        if total_range > 0:
            body_ratio = body_size / total_range
            if body_ratio <= threshold:
                results[i] = 1
    
    return results

@jit(nopython=True, parallel=True)
def vectorized_hammer_detection(opens: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Vectorized hammer detection"""
    n = len(opens)
    results = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):
        body_size = abs(closes[i] - opens[i])
        lower_shadow = min(opens[i], closes[i]) - lows[i]
        upper_shadow = highs[i] - max(opens[i], closes[i])
        total_range = highs[i] - lows[i]
        
        if total_range > 0:
            # Hammer criteria: small body, long lower shadow, small upper shadow
            body_ratio = body_size / total_range
            lower_ratio = lower_shadow / total_range
            upper_ratio = upper_shadow / total_range
            
            if (body_ratio < 0.3 and lower_ratio > 0.6 and upper_ratio < 0.1):
                results[i] = 1 if closes[i] > opens[i] else -1  # Bullish/Bearish
    
    return results

@jit(nopython=True, parallel=True)
def vectorized_engulfing_detection(opens: np.ndarray, highs: np.ndarray, 
                                  lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Vectorized engulfing pattern detection"""
    n = len(opens)
    results = np.zeros(n, dtype=np.int8)
    
    for i in prange(1, n):  # Start from 1 to compare with previous candle
        prev_body_size = abs(closes[i-1] - opens[i-1])
        curr_body_size = abs(closes[i] - opens[i])
        
        if curr_body_size > prev_body_size * 1.5:  # Current body significantly larger
            # Bullish engulfing
            if (opens[i] < closes[i-1] and closes[i] > opens[i-1] and 
                closes[i] > opens[i]):
                results[i] = 1
            # Bearish engulfing
            elif (opens[i] > closes[i-1] and closes[i] < opens[i-1] and 
                  closes[i] < opens[i]):
                results[i] = -1
    
    return results

class UltraFastPatternDetector:
    """Ultra-fast candlestick pattern detector with vectorized operations"""
    
    def __init__(self, enable_gpu: bool = False):
        self.enable_gpu = enable_gpu
        self.sliding_buffers = {}
        self.pattern_cache = {}
        self.detection_stats = {
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Pre-compiled pattern detection functions
        self.vectorized_patterns = {
            'doji': vectorized_doji_detection,
            'hammer': vectorized_hammer_detection,
            'engulfing': vectorized_engulfing_detection
        }
        
        # TA-Lib patterns for complex patterns
        self.talib_patterns = {
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'hanging_man': talib.CDLHANGINGMAN,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'spinning_top': talib.CDLSPINNINGTOP,
            'marubozu': talib.CDLMARUBOZU,
            'tristar': talib.CDLTRISTAR,
            'three_inside': talib.CDL3INSIDE,
            'three_outside': talib.CDL3OUTSIDE,
            'breakaway': talib.CDLBREAKAWAY,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'dragonfly_doji': talib.CDLDRAGONFLYDOJI,
            'gravestone_doji': talib.CDLGRAVESTONEDOJI,
            'harami': talib.CDLHARAMI,
            'harami_cross': talib.CDLHARAMICROSS,
            'high_wave': talib.CDLHIGHWAVE,
            'identical_three_crows': talib.CDLIDENTICAL3CROWS,
            'kicking': talib.CDLKICKING,
            'ladder_bottom': talib.CDLLADDERBOTTOM,
            'long_legged_doji': talib.CDLLONGLEGGEDDOJI,
            'long_line': talib.CDLLONGLINE,
            'on_neck': talib.CDLONNECK,
            'piercing': talib.CDLPIERCING
        }
        
        logger.info(f"🚀 Ultra-fast pattern detector initialized with {len(self.vectorized_patterns)} vectorized patterns")
    
    def get_sliding_buffer(self, symbol: str, timeframe: str) -> SlidingWindowBuffer:
        """Get or create sliding buffer for symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        if key not in self.sliding_buffers:
            self.sliding_buffers[key] = SlidingWindowBuffer(max_size=1000)
        return self.sliding_buffers[key]
    
    async def detect_patterns_async(self, symbol: str, timeframe: str, 
                                  candles: List[Dict]) -> List[PatternDetectionResult]:
        """Async pattern detection with parallel processing"""
        start_time = datetime.now()
        
        # Update sliding buffer
        buffer = self.get_sliding_buffer(symbol, timeframe)
        for candle in candles:
            buffer.add_candle(symbol, timeframe, candle)
        
        # Extract OHLCV arrays
        if len(candles) < 5:
            return []
        
        opens = np.array([c['open'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        closes = np.array([c['close'] for c in candles])
        volumes = np.array([c.get('volume', 0) for c in candles])
        timestamps = [c['timestamp'] for c in candles]
        
        # Run vectorized patterns in parallel
        vectorized_tasks = []
        for pattern_name, pattern_func in self.vectorized_patterns.items():
            task = asyncio.create_task(
                self._detect_vectorized_pattern(pattern_name, pattern_func, 
                                              opens, highs, lows, closes, volumes, timestamps)
            )
            vectorized_tasks.append(task)
        
        # Run TA-Lib patterns in parallel
        talib_tasks = []
        for pattern_name, pattern_func in self.talib_patterns.items():
            task = asyncio.create_task(
                self._detect_talib_pattern(pattern_name, pattern_func,
                                         opens, highs, lows, closes, volumes, timestamps)
            )
            talib_tasks.append(task)
        
        # Wait for all detections to complete
        vectorized_results = await asyncio.gather(*vectorized_tasks)
        talib_results = await asyncio.gather(*talib_tasks)
        
        # Combine results
        all_results = []
        for result_list in vectorized_results + talib_results:
            all_results.extend(result_list)
        
        # Calculate detection latency
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update stats
        self.detection_stats['total_detections'] += len(all_results)
        self.detection_stats['avg_latency_ms'] = (
            (self.detection_stats['avg_latency_ms'] * (self.detection_stats['total_detections'] - len(all_results)) + 
             latency_ms * len(all_results)) / self.detection_stats['total_detections']
        )
        
        logger.info(f"⚡ Pattern detection completed: {len(all_results)} patterns in {latency_ms:.2f}ms")
        
        return all_results
    
    async def _detect_vectorized_pattern(self, pattern_name: str, pattern_func, 
                                       opens: np.ndarray, highs: np.ndarray, 
                                       lows: np.ndarray, closes: np.ndarray,
                                       volumes: np.ndarray, timestamps: List) -> List[PatternDetectionResult]:
        """Detect pattern using vectorized function"""
        try:
            # Run vectorized detection
            results = pattern_func(opens, highs, lows, closes)
            
            detected_patterns = []
            for i, result in enumerate(results):
                if result != 0:
                    confidence = self._calculate_pattern_confidence(
                        pattern_name, opens[i], highs[i], lows[i], closes[i], volumes[i]
                    )
                    
                    # Apply noise filtering
                    if self._passes_noise_filter(opens[i], highs[i], lows[i], closes[i], volumes[i]):
                        pattern_result = PatternDetectionResult(
                            pattern_name=pattern_name,
                            confidence=confidence,
                            strength=self._get_pattern_strength(confidence),
                            direction='bullish' if result > 0 else 'bearish',
                            timestamp=timestamps[i],
                            price_level=float(closes[i]),
                            volume_confirmation=self._check_volume_confirmation(volumes[i], i, volumes),
                            volume_confidence=self._calculate_volume_confidence(volumes[i], i, volumes),
                            trend_alignment=self._get_trend_alignment(closes[i], i, closes),
                            metadata={
                                'detection_method': 'vectorized',
                                'body_size': abs(closes[i] - opens[i]),
                                'total_range': highs[i] - lows[i]
                            },
                            detection_latency_ms=0.0  # Will be calculated at higher level
                        )
                        detected_patterns.append(pattern_result)
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error in vectorized {pattern_name} detection: {e}")
            return []
    
    async def _detect_talib_pattern(self, pattern_name: str, pattern_func, 
                                  opens: np.ndarray, highs: np.ndarray, 
                                  lows: np.ndarray, closes: np.ndarray,
                                  volumes: np.ndarray, timestamps: List) -> List[PatternDetectionResult]:
        """Detect pattern using TA-Lib"""
        try:
            # Run TA-Lib detection
            results = pattern_func(opens, highs, lows, closes)
            
            detected_patterns = []
            for i, result in enumerate(results):
                if result != 0:
                    confidence = self._calculate_pattern_confidence(
                        pattern_name, opens[i], highs[i], lows[i], closes[i], volumes[i]
                    )
                    
                    # Apply noise filtering
                    if self._passes_noise_filter(opens[i], highs[i], lows[i], closes[i], volumes[i]):
                        pattern_result = PatternDetectionResult(
                            pattern_name=pattern_name,
                            confidence=confidence,
                            strength=self._get_pattern_strength(confidence),
                            direction='bullish' if result > 0 else 'bearish',
                            timestamp=timestamps[i],
                            price_level=float(closes[i]),
                            volume_confirmation=self._check_volume_confirmation(volumes[i], i, volumes),
                            volume_confidence=self._calculate_volume_confidence(volumes[i], i, volumes),
                            trend_alignment=self._get_trend_alignment(closes[i], i, closes),
                            metadata={
                                'detection_method': 'talib',
                                'talib_result': int(result)
                            },
                            detection_latency_ms=0.0
                        )
                        detected_patterns.append(pattern_result)
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error in TA-Lib {pattern_name} detection: {e}")
            return []
    
    def _calculate_pattern_confidence(self, pattern_name: str, open_price: float, 
                                    high: float, low: float, close: float, 
                                    volume: float) -> float:
        """Calculate pattern confidence based on multiple factors"""
        try:
            body_size = abs(close - open_price)
            total_range = high - low
            
            if total_range == 0:
                return 0.0
            
            # Base confidence from pattern characteristics
            base_confidence = 0.5
            
            # Body ratio factor
            body_ratio = body_size / total_range
            if pattern_name == 'doji':
                base_confidence += (1.0 - body_ratio) * 0.3
            elif pattern_name in ['hammer', 'shooting_star']:
                base_confidence += body_ratio * 0.2
            
            # Volume factor
            if volume > 0:
                volume_factor = min(volume / 1000000, 1.0)  # Normalize volume
                base_confidence += volume_factor * 0.2
            
            # Range factor (larger ranges = higher confidence)
            range_factor = min(total_range / open_price, 0.1)  # Max 10% range
            base_confidence += range_factor * 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    def _passes_noise_filter(self, open_price: float, high: float, low: float, 
                           close: float, volume: float) -> bool:
        """Apply noise filtering to eliminate micro-patterns"""
        try:
            # Minimum ATR% move requirement
            total_range = high - low
            atr_percent = (total_range / open_price) * 100
            
            # Ignore patterns with very small ranges (noise)
            if atr_percent < 0.1:  # Less than 0.1% range
                return False
            
            # Minimum volume requirement
            if volume < 1000:  # Very low volume
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in noise filtering: {e}")
            return True
    
    def _check_volume_confirmation(self, current_volume: float, index: int, 
                                 volumes: np.ndarray) -> bool:
        """Check if volume confirms the pattern"""
        try:
            if index < 5:
                return False
            
            # Compare with recent average volume
            recent_avg = np.mean(volumes[max(0, index-5):index])
            return current_volume > recent_avg * 1.2  # 20% above average
            
        except Exception as e:
            logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _calculate_volume_confidence(self, current_volume: float, index: int, 
                                   volumes: np.ndarray) -> float:
        """Calculate volume confirmation confidence"""
        try:
            if index < 5:
                return 0.0
            
            recent_avg = np.mean(volumes[max(0, index-5):index])
            if recent_avg == 0:
                return 0.0
            
            volume_ratio = current_volume / recent_avg
            return min(volume_ratio / 2.0, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volume confidence: {e}")
            return 0.0
    
    def _get_trend_alignment(self, current_close: float, index: int, 
                           closes: np.ndarray) -> str:
        """Determine trend alignment"""
        try:
            if index < 10:
                return 'neutral'
            
            # Calculate short-term trend
            short_ma = np.mean(closes[max(0, index-5):index])
            long_ma = np.mean(closes[max(0, index-10):index])
            
            if current_close > short_ma > long_ma:
                return 'bullish'
            elif current_close < short_ma < long_ma:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining trend alignment: {e}")
            return 'neutral'
    
    def _get_pattern_strength(self, confidence: float) -> str:
        """Convert confidence to strength category"""
        if confidence >= 0.8:
            return 'strong'
        elif confidence >= 0.6:
            return 'moderate'
        else:
            return 'weak'
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_detections': self.detection_stats['total_detections'],
            'avg_latency_ms': self.detection_stats['avg_latency_ms'],
            'cache_hits': self.detection_stats['cache_hits'],
            'cache_misses': self.detection_stats['cache_misses'],
            'cache_hit_rate': (
                self.detection_stats['cache_hits'] / 
                (self.detection_stats['cache_hits'] + self.detection_stats['cache_misses'])
                if (self.detection_stats['cache_hits'] + self.detection_stats['cache_misses']) > 0 
                else 0.0
            ),
            'active_buffers': len(self.sliding_buffers)
        }
    
    def clear_cache(self):
        """Clear pattern cache"""
        self.pattern_cache.clear()
        logger.info("🧹 Pattern cache cleared")



