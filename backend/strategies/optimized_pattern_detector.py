#!/usr/bin/env python3
"""
Optimized Pattern Detector for AlphaPulse
Implements the complete optimization playbook for maximum performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

@dataclass
class OptimizedPatternSignal:
    """Optimized pattern signal with performance metrics"""
    pattern: str
    index: int
    strength: float
    type: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float
    timestamp: Optional[str] = None
    processing_time_ms: float = 0.0
    cache_hit: bool = False

class OptimizedPatternDetector:
    """
    Ultra-optimized pattern detector implementing the complete optimization playbook
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize optimized pattern detector"""
        self.max_workers = max_workers
        self.indicator_cache = {}
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_cleanup = time.time()
        
        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        logger.info(f"Optimized Pattern Detector initialized with {max_workers} workers")
    
    def detect_patterns_vectorized(self, df: pd.DataFrame) -> List[OptimizedPatternSignal]:
        """
        **1. VECTORIZE PATTERN CALCULATIONS**
        Detect all patterns using vectorized operations instead of loops
        """
        start_time = time.time()
        
        # Pre-calculate all candlestick properties vectorized
        df = self._precompute_candlestick_properties(df)
        
        # Get cached indicators or calculate them
        indicators = self._get_cached_indicators(df)
        
        # **3. FILTER FIRST, DETECT LATER**
        # Apply fast preconditions to skip irrelevant rows
        filtered_mask = self._apply_pattern_preconditions(df)
        
        # **4. COMBINE RELATED PATTERNS INTO ONE PASS**
        # Detect all patterns in a single vectorized operation
        patterns = self._detect_all_patterns_vectorized(df, indicators, filtered_mask)
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self._update_stats(processing_time, len(patterns))
        
        logger.info(f"⚡ Vectorized detection: {len(patterns)} patterns in {processing_time:.2f}ms")
        
        return patterns
    
    def _precompute_candlestick_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute all candlestick properties using vectorized operations"""
        df = df.copy()
        
        # **VECTORIZED CALCULATIONS** - All in one pass
        df['body'] = (df['close'] - df['open']).abs()
        df['body_ratio'] = df['body'] / (df['high'] - df['low'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        df['price_change'] = df['close'].pct_change()
        
        # Volume properties (if available)
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_surge'] = df['volume_ratio'] > 1.5
        
        return df
    
    def _get_cached_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """**2. CACHE REPETITIVE INDICATORS**"""
        # Create a simple hash for caching
        df_hash = hash(str(df.shape) + str(df['close'].iloc[-1]))
        
        if df_hash in self.indicator_cache:
            self.stats['cache_hits'] += 1
            return self.indicator_cache[df_hash]
        
        # Calculate indicators
        indicators = {
            'sma_20': df['close'].rolling(20).mean().values,
            'sma_50': df['close'].rolling(50).mean().values,
            'atr_14': self._calculate_atr_vectorized(df).values,
            'rsi_14': self._calculate_rsi_vectorized(df).values
        }
        
        # Cache the results
        self.indicator_cache[df_hash] = indicators
        
        return indicators
    
    def _apply_pattern_preconditions(self, df: pd.DataFrame) -> np.ndarray:
        """**3. FILTER FIRST, DETECT LATER** - Fast preconditions to skip irrelevant rows"""
        
        # Create a boolean mask for rows that meet basic pattern criteria
        mask = np.ones(len(df), dtype=bool)
        
        # Skip first few rows (need history for patterns)
        mask[:3] = False
        
        # **Fast preconditions:**
        
        # Only check bullish engulfing if current close > open and previous close < open
        bullish_engulfing_candidates = (
            df['is_bullish'] & 
            df['is_bearish'].shift(1) &
            (df['body'] > df['body'].shift(1) * 1.2)
        )
        
        # Only check bearish engulfing if current close < open and previous close > open
        bearish_engulfing_candidates = (
            df['is_bearish'] & 
            df['is_bullish'].shift(1) &
            (df['body'] > df['body'].shift(1) * 1.2)
        )
        
        # Only check pin bars if shadow ratio is significant
        pin_bar_candidates = (
            (df['upper_shadow'] > df['body'] * 2) | 
            (df['lower_shadow'] > df['body'] * 2)
        )
        
        # Only check doji if body is very small
        doji_candidates = df['body_ratio'] < 0.1
        
        # Combine all candidates
        pattern_candidates = (
            bullish_engulfing_candidates |
            bearish_engulfing_candidates |
            pin_bar_candidates |
            doji_candidates
        )
        
        # Apply mask - only process rows that are pattern candidates
        mask = mask & pattern_candidates
        
        logger.info(f"🔍 Pre-filtering: {mask.sum()}/{len(df)} rows passed preconditions")
        
        return mask
    
    def _detect_all_patterns_vectorized(self, df: pd.DataFrame, indicators: Dict, filtered_mask: np.ndarray) -> List[OptimizedPatternSignal]:
        """**4. COMBINE RELATED PATTERNS INTO ONE PASS**"""
        
        patterns = []
        
        # Get indices where patterns are possible
        candidate_indices = np.where(filtered_mask)[0]
        
        if len(candidate_indices) == 0:
            return patterns
        
        # **VECTORIZED PATTERN DETECTION** - All patterns in one operation
        
        # Hammer/Shooting Star detection (vectorized)
        hammer_mask = (
            (df['lower_shadow'] > df['body'] * 2) &
            (df['upper_shadow'] < df['body'] * 0.5) &
            (df['body_ratio'] > 0.1)  # Not a doji
        )
        
        shooting_star_mask = (
            (df['upper_shadow'] > df['body'] * 2) &
            (df['lower_shadow'] < df['body'] * 0.5) &
            (df['body_ratio'] > 0.1)  # Not a doji
        )
        
        # Doji detection (vectorized)
        doji_mask = df['body_ratio'] < 0.1
        
        # Engulfing detection (vectorized)
        bullish_engulfing_mask = (
            df['is_bullish'] &
            df['is_bearish'].shift(1) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['body'] > df['body'].shift(1) * 1.2)
        )
        
        bearish_engulfing_mask = (
            df['is_bearish'] &
            df['is_bullish'].shift(1) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['body'] > df['body'].shift(1) * 1.2)
        )
        
        # Morning/Evening Star detection (vectorized)
        morning_star_mask = (
            df['is_bearish'].shift(2) &
            (df['body_ratio'].shift(1) < 0.3) &  # Small middle candle
            df['is_bullish'] &
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)
        )
        
        evening_star_mask = (
            df['is_bullish'].shift(2) &
            (df['body_ratio'].shift(1) < 0.3) &  # Small middle candle
            df['is_bearish'] &
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)
        )
        
        # **COMBINE ALL PATTERNS** - Create pattern signals for all detected patterns
        pattern_masks = {
            'hammer': hammer_mask,
            'shooting_star': shooting_star_mask,
            'doji': doji_mask,
            'bullish_engulfing': bullish_engulfing_mask,
            'bearish_engulfing': bearish_engulfing_mask,
            'morning_star': morning_star_mask,
            'evening_star': evening_star_mask
        }
        
        # Generate signals for all detected patterns
        for pattern_name, mask in pattern_masks.items():
            pattern_indices = np.where(mask & filtered_mask)[0]
            
            for idx in pattern_indices:
                confidence = self._calculate_pattern_confidence(df, pattern_name, idx, indicators)
                
                signal = OptimizedPatternSignal(
                    pattern=pattern_name,
                    index=idx,
                    strength=confidence,
                    type=self._get_pattern_type(pattern_name),
                    confidence=confidence,
                    timestamp=str(df.index[idx]) if hasattr(df.index[idx], 'isoformat') else str(df.index[idx]),
                    cache_hit=True  # Using cached indicators
                )
                patterns.append(signal)
        
        return patterns
    
    def _calculate_atr_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vectorized ATR calculation"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_rsi_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vectorized RSI calculation"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_pattern_confidence(self, df: pd.DataFrame, pattern: str, idx: int, indicators: Dict) -> float:
        """Calculate pattern confidence using cached indicators"""
        base_confidence = 0.7
        
        # Volume confirmation
        if 'volume' in df.columns and idx < len(df):
            volume_ratio = df.iloc[idx]['volume_ratio'] if 'volume_ratio' in df.columns else 1.0
            if volume_ratio > 1.5:
                base_confidence += 0.2
            elif volume_ratio > 1.2:
                base_confidence += 0.1
        
        # Technical indicator confirmation
        if idx < len(indicators.get('rsi_14', [])):
            rsi = indicators['rsi_14'][idx]
            if not np.isnan(rsi):
                if pattern in ['hammer', 'bullish_engulfing', 'morning_star'] and rsi < 30:
                    base_confidence += 0.1
                elif pattern in ['shooting_star', 'bearish_engulfing', 'evening_star'] and rsi > 70:
                    base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _get_pattern_type(self, pattern: str) -> str:
        """Get pattern type"""
        bullish_patterns = ['hammer', 'bullish_engulfing', 'morning_star']
        bearish_patterns = ['shooting_star', 'bearish_engulfing', 'evening_star']
        
        if pattern in bullish_patterns:
            return 'bullish'
        elif pattern in bearish_patterns:
            return 'bearish'
        else:
            return 'neutral'
    
    def _update_stats(self, processing_time: float, pattern_count: int):
        """Update performance statistics"""
        self.stats['total_detections'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_detections']
        )
    
    async def detect_patterns_parallel(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[OptimizedPatternSignal]]:
        """
        **5. PARALLELIZE ACROSS CONTRACTS & TIMEFRAMES**
        Detect patterns for multiple symbols/timeframes in parallel
        """
        logger.info(f"🔄 Starting parallel pattern detection for {len(data_dict)} datasets")
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all detection tasks
            future_to_key = {
                executor.submit(self.detect_patterns_vectorized, df): key
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
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.indicator_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_detections'], 1)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.indicator_cache.clear()
        self.pattern_cache.clear()
        # Note: _get_cached_indicators is not using lru_cache, so no cache_clear needed
        logger.info("🧹 Cache cleared")

# Example usage and performance comparison
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
        'volume': volumes
    })
    
    # Test optimized detector
    detector = OptimizedPatternDetector()
    
    # Performance test
    start_time = time.time()
    patterns = detector.detect_patterns_vectorized(df)
    optimized_time = (time.time() - start_time) * 1000
    
    print(f"🚀 Optimized Detection Results:")
    print(f"   Patterns found: {len(patterns)}")
    print(f"   Processing time: {optimized_time:.2f}ms")
    print(f"   Patterns per second: {len(patterns) / (optimized_time / 1000):.0f}")
    print(f"   Performance stats: {detector.get_performance_stats()}")
    
    # Show pattern breakdown
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern.pattern] = pattern_counts.get(pattern.pattern, 0) + 1
    
    print(f"   Pattern breakdown: {pattern_counts}")
