#!/usr/bin/env python3
"""
Phase 3: Noise Filtering Layer
Implements ATR-based minimum move requirements and volume-based noise filtering
to reduce false signals from micro-patterns and low-quality setups.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

@dataclass
class NoiseFilterResult:
    """Result from noise filtering analysis"""
    pattern_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    original_confidence: float
    filtered_confidence: float
    passed_filters: bool
    filter_reasons: List[str]
    atr_percentage: float
    volume_ratio: float
    price_movement: float
    noise_score: float

@dataclass
class NoiseFilterConfig:
    """Configuration for noise filtering parameters"""
    # ATR-based filtering
    min_atr_percentage: float = 0.3  # Minimum ATR% move required
    max_atr_percentage: float = 5.0  # Maximum ATR% to avoid extreme moves
    
    # Volume-based filtering
    min_volume_ratio: float = 1.2    # Minimum volume vs average
    max_volume_ratio: float = 10.0   # Maximum volume to avoid manipulation
    
    # Price movement filtering
    min_price_movement: float = 0.1  # Minimum price change %
    max_price_movement: float = 15.0 # Maximum price change %
    
    # Pattern-specific adjustments
    pattern_atr_multipliers: Dict[str, float] = None
    pattern_volume_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pattern_atr_multipliers is None:
            self.pattern_atr_multipliers = {
                'doji': 0.8,      # Doji can be smaller
                'hammer': 1.2,    # Hammer needs more movement
                'engulfing': 1.5, # Engulfing needs significant move
                'shooting_star': 1.3,
                'morning_star': 1.4,
                'evening_star': 1.4,
                'three_white_soldiers': 1.6,
                'three_black_crows': 1.6,
                'hanging_man': 1.1,
                'inverted_hammer': 1.2,
                'spinning_top': 0.9,
                'marubozu': 1.8,
                'tristar': 0.7,
                'breakaway': 2.0,
                'dark_cloud_cover': 1.7,
                'dragonfly_doji': 0.8,
                'gravestone_doji': 0.8,
                'harami': 1.0,
                'harami_cross': 0.9,
                'high_wave': 1.1,
                'identical_three_crows': 1.5,
                'kicking': 1.8,
                'ladder_bottom': 1.3,
                'long_legged_doji': 1.0,
                'long_line': 1.4,
                'on_neck': 1.2,
                'piercing': 1.6,
                'rising_three_methods': 1.3,
                'separating_lines': 1.5
            }
        
        if self.pattern_volume_multipliers is None:
            self.pattern_volume_multipliers = {
                'doji': 0.9,      # Doji can have lower volume
                'hammer': 1.3,    # Hammer needs volume confirmation
                'engulfing': 1.5, # Engulfing needs high volume
                'shooting_star': 1.2,
                'morning_star': 1.4,
                'evening_star': 1.4,
                'three_white_soldiers': 1.6,
                'three_black_crows': 1.6,
                'hanging_man': 1.1,
                'inverted_hammer': 1.2,
                'spinning_top': 1.0,
                'marubozu': 1.7,
                'tristar': 0.8,
                'breakaway': 2.0,
                'dark_cloud_cover': 1.6,
                'dragonfly_doji': 0.9,
                'gravestone_doji': 0.9,
                'harami': 1.1,
                'harami_cross': 1.0,
                'high_wave': 1.1,
                'identical_three_crows': 1.4,
                'kicking': 1.8,
                'ladder_bottom': 1.2,
                'long_legged_doji': 1.0,
                'long_line': 1.3,
                'on_neck': 1.1,
                'piercing': 1.5,
                'rising_three_methods': 1.2,
                'separating_lines': 1.4
            }

class ATRCalculator:
    """Calculates Average True Range for volatility measurement"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                     closes: np.ndarray) -> np.ndarray:
        """Calculate ATR values"""
        if len(highs) < self.period + 1:
            return np.full(len(highs), np.nan)
        
        # Calculate True Range
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using exponential moving average
        atr = np.full(len(highs), np.nan)
        atr[self.period] = np.mean(true_range[1:self.period + 1])
        
        for i in range(self.period + 1, len(highs)):
            atr[i] = (atr[i-1] * (self.period - 1) + true_range[i]) / self.period
        
        return atr
    
    def calculate_atr_percentage(self, atr: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate ATR as percentage of price"""
        return (atr / closes) * 100

class VolumeAnalyzer:
    """Analyzes volume patterns and ratios"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate_volume_ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate current volume vs average volume"""
        if len(volumes) < self.period:
            return np.full(len(volumes), 1.0)
        
        avg_volume = np.full(len(volumes), 1.0)
        for i in range(self.period, len(volumes)):
            avg_volume[i] = np.mean(volumes[i-self.period:i])
        
        return volumes / avg_volume
    
    def calculate_volume_trend(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume trend (increasing/decreasing)"""
        if len(volumes) < 3:
            return np.full(len(volumes), 0.0)
        
        volume_trend = np.full(len(volumes), 0.0)
        for i in range(2, len(volumes)):
            # Simple trend: positive if current > previous, negative otherwise
            if volumes[i] > volumes[i-1]:
                volume_trend[i] = 1.0
            elif volumes[i] < volumes[i-1]:
                volume_trend[i] = -1.0
        
        return volume_trend

class PriceMovementAnalyzer:
    """Analyzes price movements and patterns"""
    
    def calculate_price_movement(self, opens: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate percentage price movement"""
        return ((closes - opens) / opens) * 100
    
    def calculate_body_size(self, opens: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate candlestick body size as percentage"""
        return np.abs((closes - opens) / opens) * 100
    
    def calculate_wick_ratio(self, highs: np.ndarray, lows: np.ndarray, 
                           opens: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate wick to body ratio"""
        bodies = np.abs(closes - opens)
        upper_wicks = highs - np.maximum(opens, closes)
        lower_wicks = np.minimum(opens, closes) - lows
        
        # Avoid division by zero
        body_sizes = np.where(bodies > 0, bodies, 1.0)
        wick_ratios = (upper_wicks + lower_wicks) / body_sizes
        
        return wick_ratios

class NoiseFilteringLayer:
    """
    Implements comprehensive noise filtering for candlestick patterns
    to reduce false signals from micro-patterns and low-quality setups.
    """
    
    def __init__(self, config: Optional[NoiseFilterConfig] = None):
        self.config = config or NoiseFilterConfig()
        self.atr_calculator = ATRCalculator()
        self.volume_analyzer = VolumeAnalyzer()
        self.price_analyzer = PriceMovementAnalyzer()
        
        logger.info("Noise filtering layer initialized with config: %s", 
                   {k: v for k, v in self.config.__dict__.items() 
                    if not k.startswith('pattern_')})
    
    def filter_patterns(self, patterns: List[Dict], 
                       ohlcv_data: Dict[str, np.ndarray]) -> List[NoiseFilterResult]:
        """
        Filter patterns based on noise criteria
        
        Args:
            patterns: List of detected patterns
            ohlcv_data: OHLCV data arrays
            
        Returns:
            List of filtered pattern results
        """
        if not patterns:
            return []
        
        # Calculate technical indicators
        highs = ohlcv_data['high']
        lows = ohlcv_data['low']
        opens = ohlcv_data['open']
        closes = ohlcv_data['close']
        volumes = ohlcv_data['volume']
        
        atr = self.atr_calculator.calculate_atr(highs, lows, closes)
        atr_percentages = self.atr_calculator.calculate_atr_percentage(atr, closes)
        volume_ratios = self.volume_analyzer.calculate_volume_ratio(volumes)
        price_movements = self.price_analyzer.calculate_price_movement(opens, closes)
        
        filtered_results = []
        
        for pattern in patterns:
            result = self._filter_single_pattern(
                pattern, atr_percentages, volume_ratios, price_movements,
                highs, lows, opens, closes, volumes
            )
            filtered_results.append(result)
        
        return filtered_results
    
    def _filter_single_pattern(self, pattern: Dict, atr_percentages: np.ndarray,
                              volume_ratios: np.ndarray, price_movements: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray, 
                              opens: np.ndarray, closes: np.ndarray,
                              volumes: np.ndarray) -> NoiseFilterResult:
        """Filter a single pattern"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        original_confidence = pattern.get('confidence', 0.0)
        
        # Get pattern-specific multipliers
        atr_multiplier = self.config.pattern_atr_multipliers.get(pattern_type, 1.0)
        volume_multiplier = self.config.pattern_volume_multipliers.get(pattern_type, 1.0)
        
        # Calculate metrics at pattern index
        if pattern_index < len(atr_percentages):
            atr_percentage = atr_percentages[pattern_index]
            volume_ratio = volume_ratios[pattern_index]
            price_movement = price_movements[pattern_index]
        else:
            atr_percentage = 0.0
            volume_ratio = 1.0
            price_movement = 0.0
        
        # Apply pattern-specific adjustments
        adjusted_min_atr = self.config.min_atr_percentage * atr_multiplier
        adjusted_min_volume = self.config.min_volume_ratio * volume_multiplier
        
        # Check filters
        filter_reasons = []
        passed_filters = True
        
        # ATR filter
        if atr_percentage < adjusted_min_atr:
            filter_reasons.append(f"ATR too low: {atr_percentage:.2f}% < {adjusted_min_atr:.2f}%")
            passed_filters = False
        elif atr_percentage > self.config.max_atr_percentage:
            filter_reasons.append(f"ATR too high: {atr_percentage:.2f}% > {self.config.max_atr_percentage:.2f}%")
            passed_filters = False
        
        # Volume filter
        if volume_ratio < adjusted_min_volume:
            filter_reasons.append(f"Volume too low: {volume_ratio:.2f}x < {adjusted_min_volume:.2f}x")
            passed_filters = False
        elif volume_ratio > self.config.max_volume_ratio:
            filter_reasons.append(f"Volume too high: {volume_ratio:.2f}x > {self.config.max_volume_ratio:.2f}x")
            passed_filters = False
        
        # Price movement filter
        if abs(price_movement) < self.config.min_price_movement:
            filter_reasons.append(f"Price movement too small: {abs(price_movement):.2f}% < {self.config.min_price_movement:.2f}%")
            passed_filters = False
        elif abs(price_movement) > self.config.max_price_movement:
            filter_reasons.append(f"Price movement too large: {abs(price_movement):.2f}% > {self.config.max_price_movement:.2f}%")
            passed_filters = False
        
        # Calculate noise score (lower is better)
        noise_score = self._calculate_noise_score(
            atr_percentage, volume_ratio, price_movement, pattern_type
        )
        
        # Adjust confidence based on filtering
        filtered_confidence = original_confidence
        if passed_filters:
            # Boost confidence for high-quality patterns
            confidence_boost = min(0.2, noise_score * 0.1)
            filtered_confidence = min(1.0, original_confidence + confidence_boost)
        else:
            # Reduce confidence for filtered patterns
            filtered_confidence = original_confidence * 0.5
        
        return NoiseFilterResult(
            pattern_id=pattern.get('pattern_id', ''),
            symbol=pattern.get('symbol', ''),
            timeframe=pattern.get('timeframe', ''),
            timestamp=pattern.get('timestamp', datetime.now(timezone.utc)),
            original_confidence=original_confidence,
            filtered_confidence=filtered_confidence,
            passed_filters=passed_filters,
            filter_reasons=filter_reasons,
            atr_percentage=atr_percentage,
            volume_ratio=volume_ratio,
            price_movement=price_movement,
            noise_score=noise_score
        )
    
    def _calculate_noise_score(self, atr_percentage: float, volume_ratio: float,
                              price_movement: float, pattern_type: str) -> float:
        """Calculate noise score (lower is better)"""
        
        # Base score starts at 1.0
        score = 1.0
        
        # ATR contribution (optimal range: 0.5-2.0%)
        if 0.5 <= atr_percentage <= 2.0:
            score *= 0.8  # Good volatility
        elif atr_percentage < 0.3 or atr_percentage > 5.0:
            score *= 2.0  # Poor volatility
        
        # Volume contribution (optimal range: 1.2-3.0x)
        if 1.2 <= volume_ratio <= 3.0:
            score *= 0.7  # Good volume
        elif volume_ratio < 0.8 or volume_ratio > 8.0:
            score *= 1.8  # Poor volume
        
        # Price movement contribution (optimal range: 0.2-5.0%)
        if 0.2 <= abs(price_movement) <= 5.0:
            score *= 0.9  # Good movement
        elif abs(price_movement) < 0.1 or abs(price_movement) > 10.0:
            score *= 1.5  # Poor movement
        
        # Pattern-specific adjustments
        pattern_adjustments = {
            'doji': 0.9,      # Doji can be smaller
            'hammer': 1.1,    # Hammer needs more movement
            'engulfing': 1.2, # Engulfing needs significant move
            'shooting_star': 1.1,
            'morning_star': 1.0,
            'evening_star': 1.0,
        }
        
        pattern_multiplier = pattern_adjustments.get(pattern_type, 1.0)
        score *= pattern_multiplier
        
        return max(0.1, min(5.0, score))  # Clamp between 0.1 and 5.0
    
    def get_filtering_stats(self, results: List[NoiseFilterResult]) -> Dict:
        """Get statistics about filtering results"""
        if not results:
            return {}
        
        total_patterns = len(results)
        passed_patterns = sum(1 for r in results if r.passed_filters)
        failed_patterns = total_patterns - passed_patterns
        
        avg_atr = np.mean([r.atr_percentage for r in results])
        avg_volume_ratio = np.mean([r.volume_ratio for r in results])
        avg_noise_score = np.mean([r.noise_score for r in results])
        
        avg_confidence_before = np.mean([r.original_confidence for r in results])
        avg_confidence_after = np.mean([r.filtered_confidence for r in results])
        
        return {
            'total_patterns': total_patterns,
            'passed_patterns': passed_patterns,
            'failed_patterns': failed_patterns,
            'pass_rate': passed_patterns / total_patterns if total_patterns > 0 else 0.0,
            'avg_atr_percentage': avg_atr,
            'avg_volume_ratio': avg_volume_ratio,
            'avg_noise_score': avg_noise_score,
            'avg_confidence_before': avg_confidence_before,
            'avg_confidence_after': avg_confidence_after,
            'confidence_improvement': avg_confidence_after - avg_confidence_before
        }
