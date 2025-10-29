#!/usr/bin/env python3
"""
Ultra-Optimized Japanese Candlestick Pattern Detector
Implements vectorized operations, sliding window buffers, and async parallelization
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import hashlib
from numba import jit, prange

# Try to import TA-Lib, fallback to basic implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib successfully imported for ultra-optimized pattern detection")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic pattern detection implementations")

@dataclass
class UltraOptimizedPatternSignal:
    """Ultra-optimized pattern signal with performance metrics"""
    pattern: str
    index: int
    strength: float
    type: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float
    timestamp: Optional[str] = None
    processing_time_ms: float = 0.0
    volume_confirmation: bool = False
    trend_alignment: str = 'neutral'
    multi_timeframe_boost: float = 0.0
    metadata: Dict[str, Any] = None

class SlidingWindowBuffer:
    """Sliding window buffer for efficient candlestick data management"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = {}
        self.last_update = {}
        
    def update_buffer(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """Update sliding window buffer with new data"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.buffer:
            self.buffer[key] = new_data
            self.last_update[key] = time.time()
            return new_data
        
        # Append new data and maintain max size
        combined = pd.concat([self.buffer[key], new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp']).tail(self.max_size)
        
        self.buffer[key] = combined
        self.last_update[key] = time.time()
        
        return combined
    
    def get_buffer(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get current buffer for symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        return self.buffer.get(key)
    
    def cleanup_old_buffers(self, max_age_seconds: int = 3600):
        """Clean up old buffers to prevent memory bloat"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, last_update in self.last_update.items():
            if current_time - last_update > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.buffer[key]
            del self.last_update[key]
        
        if keys_to_remove:
            logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old buffers")

@jit(nopython=True, parallel=True)
def _vectorized_candlestick_properties(opens, highs, lows, closes):
    """Numba-optimized vectorized candlestick property calculation"""
    n = len(opens)
    
    # Pre-allocate arrays
    body = np.zeros(n)
    body_ratio = np.zeros(n)
    upper_shadow = np.zeros(n)
    lower_shadow = np.zeros(n)
    is_bullish = np.zeros(n, dtype=np.bool_)
    is_bearish = np.zeros(n, dtype=np.bool_)
    total_range = np.zeros(n)
    
    for i in prange(n):
        body[i] = abs(closes[i] - opens[i])
        total_range[i] = highs[i] - lows[i]
        
        if total_range[i] > 0:
            body_ratio[i] = body[i] / total_range[i]
        
        if closes[i] > opens[i]:
            is_bullish[i] = True
            upper_shadow[i] = highs[i] - closes[i]
            lower_shadow[i] = opens[i] - lows[i]
        else:
            is_bearish[i] = True
            upper_shadow[i] = highs[i] - opens[i]
            lower_shadow[i] = closes[i] - lows[i]
    
    return body, body_ratio, upper_shadow, lower_shadow, is_bullish, is_bearish, total_range

class UltraOptimizedPatternDetector:
    """
    Ultra-optimized pattern detector implementing the complete optimization playbook
    """
    
    def __init__(self, max_workers: int = 8, buffer_size: int = 1000):
        """Initialize ultra-optimized pattern detector"""
        self.max_workers = max_workers
        self.buffer_size = buffer_size
        self.sliding_buffer = SlidingWindowBuffer(buffer_size)
        
        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'vectorized_operations': 0,
            'parallel_operations': 0
        }
        
        # Pattern detection cache
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize TA-Lib patterns if available
        if TALIB_AVAILABLE:
            self.talib_patterns = {
                'hammer': talib.CDLHAMMER,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'engulfing': talib.CDLENGULFING,
                'doji': talib.CDLDOJI,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                'three_black_crows': talib.CDL3BLACKCROWS,
                'hanging_man': talib.CDLHANGINGMAN,
                'inverted_hammer': talib.CDLINVERTEDHAMMER,
                'spinning_top': talib.CDLSPINNINGTOP,
                'marubozu': talib.CDLMARUBOZU,
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
                'piercing': talib.CDLPIERCING,
                'rising_falling_three_methods': talib.CDLRISEFALL3METHODS,
                'separating_lines': talib.CDLSEPARATINGLINES,
                'short_line': talib.CDLSHORTLINE,
                'takuri': talib.CDLTAKURI,
                'tristar': talib.CDLTRISTAR,
                'three_inside_up': talib.CDL3INSIDE,
                'three_inside_down': talib.CDL3INSIDE,
                'three_outside_up': talib.CDL3OUTSIDE,
                'three_outside_down': talib.CDL3OUTSIDE,
                'breakaway': talib.CDLBREAKAWAY,
                'dark_cloud_cover': talib.CDLDARKCLOUDCOVER
            }
        
        logger.info(f"🚀 Ultra-Optimized Pattern Detector initialized with {max_workers} workers")
    
    def detect_patterns_ultra_optimized(self, df: pd.DataFrame, symbol: str = None, timeframe: str = None) -> List[UltraOptimizedPatternSignal]:
        """
        **ULTRA-OPTIMIZED PATTERN DETECTION**
        Combines vectorized operations, sliding windows, and parallel processing
        """
        start_time = time.time()
        
        if len(df) < 5:
            return []
        
        # Update sliding window buffer
        if symbol and timeframe:
            df = self.sliding_buffer.update_buffer(symbol, timeframe, df)
        
        # Create data hash for caching
        data_hash = self._create_data_hash(df)
        
        # Check cache first
        if data_hash in self.pattern_cache:
            self.stats['cache_hits'] += 1
            cached_patterns = self.pattern_cache[data_hash]
            cached_patterns[0].processing_time_ms = (time.time() - start_time) * 1000
            return cached_patterns
        
        # **1. VECTORIZED CANDLESTICK PROPERTIES**
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Use Numba-optimized vectorized calculation
        body, body_ratio, upper_shadow, lower_shadow, is_bullish, is_bearish, total_range = _vectorized_candlestick_properties(
            opens, highs, lows, closes
        )
        
        # **2. VECTORIZED PATTERN DETECTION**
        patterns = self._detect_patterns_vectorized(
            df, opens, highs, lows, closes, body, body_ratio, upper_shadow, lower_shadow, 
            is_bullish, is_bearish, total_range
        )
        
        # **3. VOLUME CONFIRMATION**
        if 'volume' in df.columns:
            patterns = self._add_volume_confirmation(patterns, df)
        
        # **4. TREND ALIGNMENT**
        patterns = self._add_trend_alignment(patterns, df)
        
        # **5. CONFIDENCE SCORING**
        patterns = self._calculate_confidence_scores(patterns, df)
        
        # Cache results
        self.pattern_cache[data_hash] = patterns
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self._update_stats(processing_time, len(patterns))
        
        # Add processing time to first pattern
        if patterns:
            patterns[0].processing_time_ms = processing_time
        
        logger.info(f"⚡ Ultra-optimized detection: {len(patterns)} patterns in {processing_time:.2f}ms")
        
        return patterns
    
    def _detect_patterns_vectorized(self, df: pd.DataFrame, opens, highs, lows, closes, body, body_ratio, upper_shadow, 
                                  lower_shadow, is_bullish, is_bearish, total_range) -> List[UltraOptimizedPatternSignal]:
        """Vectorized pattern detection using NumPy operations"""
        patterns = []
        
        # **HAMMER PATTERNS** (vectorized)
        hammer_mask = (
            (lower_shadow > body * 2) &
            (upper_shadow < body * 0.5) &
            (body_ratio > 0.1) &
            is_bullish
        )
        
        hanging_man_mask = (
            (lower_shadow > body * 2) &
            (upper_shadow < body * 0.5) &
            (body_ratio > 0.1) &
            is_bearish
        )
        
        inverted_hammer_mask = (
            (upper_shadow > body * 2) &
            (lower_shadow < body * 0.5) &
            (body_ratio > 0.1) &
            is_bullish
        )
        
        # **SHOOTING STAR** (vectorized)
        shooting_star_mask = (
            (upper_shadow > body * 2) &
            (lower_shadow < body * 0.5) &
            (body_ratio > 0.1) &
            is_bearish
        )
        
        # **DOJI PATTERNS** (vectorized)
        doji_mask = body_ratio < 0.1
        
        dragonfly_doji_mask = (
            doji_mask &
            (upper_shadow < body * 0.1) &
            (lower_shadow > body * 2)
        )
        
        gravestone_doji_mask = (
            doji_mask &
            (lower_shadow < body * 0.1) &
            (upper_shadow > body * 2)
        )
        
        # **ENGULFING PATTERNS** (vectorized)
        bullish_engulfing_mask = (
            is_bullish &
            np.roll(is_bearish, 1) &
            (opens < np.roll(closes, 1)) &
            (closes > np.roll(opens, 1)) &
            (body > np.roll(body, 1) * 1.2)
        )
        
        bearish_engulfing_mask = (
            is_bearish &
            np.roll(is_bullish, 1) &
            (opens > np.roll(closes, 1)) &
            (closes < np.roll(opens, 1)) &
            (body > np.roll(body, 1) * 1.2)
        )
        
        # **MORNING/EVENING STAR** (vectorized)
        morning_star_mask = (
            np.roll(is_bearish, 2) &
            (np.roll(body_ratio, 1) < 0.3) &
            is_bullish &
            (closes > (np.roll(opens, 2) + np.roll(closes, 2)) / 2)
        )
        
        evening_star_mask = (
            np.roll(is_bullish, 2) &
            (np.roll(body_ratio, 1) < 0.3) &
            is_bearish &
            (closes < (np.roll(opens, 2) + np.roll(closes, 2)) / 2)
        )
        
        # **MARUBOZU** (vectorized)
        marubozu_bullish_mask = (
            is_bullish &
            (upper_shadow < body * 0.1) &
            (lower_shadow < body * 0.1) &
            (body_ratio > 0.8)
        )
        
        marubozu_bearish_mask = (
            is_bearish &
            (upper_shadow < body * 0.1) &
            (lower_shadow < body * 0.1) &
            (body_ratio > 0.8)
        )
        
        # **SPINNING TOP** (vectorized)
        spinning_top_mask = (
            (upper_shadow > body * 1.5) &
            (lower_shadow > body * 1.5) &
            (body_ratio < 0.3)
        )
        
        # Convert masks to pattern signals
        pattern_masks = {
            'hammer': hammer_mask,
            'hanging_man': hanging_man_mask,
            'inverted_hammer': inverted_hammer_mask,
            'shooting_star': shooting_star_mask,
            'doji': doji_mask,
            'dragonfly_doji': dragonfly_doji_mask,
            'gravestone_doji': gravestone_doji_mask,
            'bullish_engulfing': bullish_engulfing_mask,
            'bearish_engulfing': bearish_engulfing_mask,
            'morning_star': morning_star_mask,
            'evening_star': evening_star_mask,
            'marubozu_bullish': marubozu_bullish_mask,
            'marubozu_bearish': marubozu_bearish_mask,
            'spinning_top': spinning_top_mask
        }
        
        # Generate pattern signals from masks
        for pattern_name, mask in pattern_masks.items():
            pattern_indices = np.where(mask)[0]
            
            for idx in pattern_indices:
                if idx < len(df):
                    pattern_type = self._get_pattern_type(pattern_name)
                    strength = self._calculate_pattern_strength(
                        df.iloc[idx], body[idx], body_ratio[idx], 
                        upper_shadow[idx], lower_shadow[idx]
                    )
                    
                    pattern = UltraOptimizedPatternSignal(
                        pattern=pattern_name,
                        index=idx,
                        strength=strength,
                        type=pattern_type,
                        confidence=0.0,  # Will be calculated later
                        timestamp=df.iloc[idx]['timestamp'] if 'timestamp' in df.columns else None,
                        metadata={
                            'body_ratio': body_ratio[idx],
                            'upper_shadow': upper_shadow[idx],
                            'lower_shadow': lower_shadow[idx],
                            'total_range': total_range[idx]
                        }
                    )
                    patterns.append(pattern)
        
        # **TA-Lib PATTERNS** (if available)
        if TALIB_AVAILABLE:
            talib_patterns = self._detect_talib_patterns_vectorized(df)
            patterns.extend(talib_patterns)
        
        return patterns
    
    def _detect_talib_patterns_vectorized(self, df: pd.DataFrame) -> List[UltraOptimizedPatternSignal]:
        """Detect TA-Lib patterns using vectorized operations"""
        patterns = []
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        for pattern_name, pattern_func in self.talib_patterns.items():
            try:
                result = pattern_func(opens, highs, lows, closes)
                pattern_indices = np.where(result != 0)[0]
                
                for idx in pattern_indices:
                    if idx < len(df):
                        pattern_type = 'bullish' if result[idx] > 0 else 'bearish'
                        strength = abs(result[idx]) / 100.0  # Normalize TA-Lib result
                        
                        pattern = UltraOptimizedPatternSignal(
                            pattern=f"talib_{pattern_name}",
                            index=idx,
                            strength=strength,
                            type=pattern_type,
                            confidence=0.0,
                            timestamp=df.iloc[idx]['timestamp'] if 'timestamp' in df.columns else None,
                            metadata={'talib_value': result[idx]}
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning(f"Error detecting TA-Lib pattern {pattern_name}: {e}")
                continue
        
        return patterns
    
    def _add_volume_confirmation(self, patterns: List[UltraOptimizedPatternSignal], df: pd.DataFrame) -> List[UltraOptimizedPatternSignal]:
        """Add volume confirmation to patterns"""
        if 'volume' not in df.columns:
            return patterns
        
        volumes = df['volume'].values
        
        for pattern in patterns:
            if pattern.index < len(volumes):
                current_volume = volumes[pattern.index]
                
                # Calculate average volume for comparison
                start_idx = max(0, pattern.index - 20)
                avg_volume = np.mean(volumes[start_idx:pattern.index])
                
                # Volume confirmation logic
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # 50% above average
                    pattern.volume_confirmation = True
                    pattern.confidence += 0.1  # Volume boost
                
                if pattern.metadata is None:
                    pattern.metadata = {}
                pattern.metadata['volume_ratio'] = volume_ratio
                pattern.metadata['volume_confirmation'] = pattern.volume_confirmation
        
        return patterns
    
    def _add_trend_alignment(self, patterns: List[UltraOptimizedPatternSignal], df: pd.DataFrame) -> List[UltraOptimizedPatternSignal]:
        """Add trend alignment analysis"""
        if len(df) < 20:
            return patterns
        
        # Calculate simple moving averages for trend
        closes = df['close'].values
        sma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')
        
        for pattern in patterns:
            if pattern.index >= 19:  # Need at least 20 candles for SMA
                current_price = closes[pattern.index]
                current_sma = sma_20[pattern.index - 19]
                
                if pattern.type == 'bullish' and current_price > current_sma:
                    pattern.trend_alignment = 'bullish'
                    pattern.confidence += 0.1
                elif pattern.type == 'bearish' and current_price < current_sma:
                    pattern.trend_alignment = 'bearish'
                    pattern.confidence += 0.1
                else:
                    pattern.trend_alignment = 'neutral'
                
                if pattern.metadata is None:
                    pattern.metadata = {}
                pattern.metadata['trend_alignment'] = pattern.trend_alignment
                pattern.metadata['price_vs_sma'] = current_price - current_sma
        
        return patterns
    
    def _calculate_confidence_scores(self, patterns: List[UltraOptimizedPatternSignal], df: pd.DataFrame) -> List[UltraOptimizedPatternSignal]:
        """Calculate comprehensive confidence scores"""
        for pattern in patterns:
            base_confidence = pattern.strength
            
            # Volume confirmation boost
            if pattern.volume_confirmation:
                base_confidence += 0.1
            
            # Trend alignment boost
            if pattern.trend_alignment in ['bullish', 'bearish']:
                base_confidence += 0.1
            
            # Multi-timeframe boost (placeholder for future implementation)
            base_confidence += pattern.multi_timeframe_boost
            
            # Cap confidence at 1.0
            pattern.confidence = min(base_confidence, 1.0)
        
        return patterns
    
    def _get_pattern_type(self, pattern_name: str) -> str:
        """Get pattern type (bullish/bearish/neutral)"""
        bullish_patterns = {
            'hammer', 'inverted_hammer', 'bullish_engulfing', 
            'morning_star', 'marubozu_bullish', 'dragonfly_doji'
        }
        
        bearish_patterns = {
            'hanging_man', 'shooting_star', 'bearish_engulfing',
            'evening_star', 'marubozu_bearish', 'gravestone_doji'
        }
        
        if pattern_name in bullish_patterns:
            return 'bullish'
        elif pattern_name in bearish_patterns:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_pattern_strength(self, candle: pd.Series, body: float, body_ratio: float, 
                                  upper_shadow: float, lower_shadow: float) -> float:
        """Calculate pattern strength based on candlestick properties"""
        # Base strength from body ratio
        strength = body_ratio
        
        # Shadow ratio boost
        if upper_shadow > 0 and lower_shadow > 0:
            shadow_ratio = min(upper_shadow, lower_shadow) / max(upper_shadow, lower_shadow)
            strength += shadow_ratio * 0.2
        
        # Cap strength at 1.0
        return min(strength, 1.0)
    
    def _create_data_hash(self, df: pd.DataFrame) -> str:
        """Create hash for data caching"""
        # Use last 10 rows for hash to avoid cache invalidation on every update
        hash_data = df.tail(10)[['open', 'high', 'low', 'close', 'volume']].values.tobytes()
        return hashlib.md5(hash_data).hexdigest()
    
    def _update_stats(self, processing_time: float, pattern_count: int):
        """Update performance statistics"""
        self.stats['total_detections'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_detections']
        )
        self.stats['vectorized_operations'] += 1
    
    async def detect_patterns_parallel(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[UltraOptimizedPatternSignal]]:
        """
        **PARALLEL PATTERN DETECTION**
        Detect patterns for multiple symbols/timeframes in parallel
        """
        logger.info(f"🔄 Starting parallel pattern detection for {len(data_dict)} datasets")
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all detection tasks
            future_to_key = {
                executor.submit(self.detect_patterns_ultra_optimized, df): key
                for key, df in data_dict.items()
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    patterns = future.result()
                    results[key] = patterns
                    logger.info(f"✅ Completed detection for {key}: {len(patterns)} patterns")
                except Exception as e:
                    logger.error(f"❌ Error detecting patterns for {key}: {e}")
                    results[key] = []
        
        self.stats['parallel_operations'] += 1
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.pattern_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_detections'], 1),
            'buffer_size': len(self.sliding_buffer.buffer),
            'avg_patterns_per_detection': self.stats['total_detections'] / max(self.stats['total_detections'], 1)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.pattern_cache.clear()
        self.sliding_buffer.cleanup_old_buffers()
        logger.info("🧹 All caches cleared")
    
    def cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        self.sliding_buffer.cleanup_old_buffers()
        
        # Clean up old cache entries
        current_time = time.time()
        keys_to_remove = []
        
        for key, patterns in self.pattern_cache.items():
            if patterns and hasattr(patterns[0], 'timestamp'):
                # Simple cleanup based on cache size
                if len(self.pattern_cache) > 1000:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove[:100]:  # Remove max 100 at a time
            del self.pattern_cache[key]
        
        if keys_to_remove:
            logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old cache entries")

# Example usage and performance testing
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 10000  # Large dataset for performance testing
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
    })
    
    # Test ultra-optimized detector
    detector = UltraOptimizedPatternDetector()
    
    # Performance test
    start_time = time.time()
    patterns = detector.detect_patterns_ultra_optimized(df, "BTCUSDT", "1m")
    optimized_time = (time.time() - start_time) * 1000
    
    print(f"🚀 Ultra-Optimized Detection Results:")
    print(f"   Patterns found: {len(patterns)}")
    print(f"   Processing time: {optimized_time:.2f}ms")
    print(f"   Patterns per second: {len(patterns) / (optimized_time / 1000):.0f}")
    print(f"   Performance stats: {detector.get_performance_stats()}")
    
    # Show pattern breakdown
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern.pattern] = pattern_counts.get(pattern.pattern, 0) + 1
    
    print(f"   Pattern breakdown: {pattern_counts}")
