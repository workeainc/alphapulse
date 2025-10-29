#!/usr/bin/env python3
"""
Enhanced Pattern Detector with Direct Volume Integration for AlphaPulse
Integrates volume analysis directly into pattern detection functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

# Try to import TA-Lib, fallback to basic implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib successfully imported for enhanced pattern detection")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic pattern detection implementations")

@dataclass
class VolumeEnhancedPatternResult:
    """Volume-enhanced pattern detection result"""
    pattern: str
    confidence: float
    volume_confirmed: bool
    volume_factor: float  # relative to average
    signal_type: str  # "bullish", "bearish", "neutral"
    index: int
    strength: float
    volume_ratio: float
    volume_pattern: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class EnhancedPatternDetectorWithVolume:
    """
    Enhanced pattern detector that integrates volume analysis directly into detection
    """
    
    def __init__(self):
        """Initialize enhanced pattern detector"""
        # Pattern-specific volume rules
        self.volume_rules = {
            'hammer': {
                'required_ratio': 1.2,
                'preferred_ratio': 1.5,
                'description': 'Hammer requires above-average volume for confirmation'
            },
            'shooting_star': {
                'required_ratio': 1.3,
                'preferred_ratio': 1.8,
                'description': 'Shooting star needs high volume for bearish confirmation'
            },
            'bullish_engulfing': {
                'required_ratio': 1.5,
                'preferred_ratio': 2.0,
                'description': 'Bullish engulfing requires strong volume confirmation'
            },
            'bearish_engulfing': {
                'required_ratio': 1.5,
                'preferred_ratio': 2.0,
                'description': 'Bearish engulfing requires strong volume confirmation'
            },
            'doji': {
                'required_ratio': 1.1,
                'preferred_ratio': 1.3,
                'description': 'Doji benefits from moderate volume confirmation'
            },
            'morning_star': {
                'required_ratio': 1.4,
                'preferred_ratio': 1.8,
                'description': 'Morning star needs good volume on third candle'
            },
            'evening_star': {
                'required_ratio': 1.4,
                'preferred_ratio': 1.8,
                'description': 'Evening star needs good volume on third candle'
            },
            'breakout': {
                'required_ratio': 2.0,
                'preferred_ratio': 2.5,
                'description': 'Breakout patterns require very high volume confirmation'
            }
        }
        
        # Pattern metadata with volume importance
        self.pattern_metadata = {
            'hammer': {
                'type': 'bullish', 
                'reliability': 0.7, 
                'description': 'Potential reversal pattern',
                'volume_importance': 'important'
            },
            'shooting_star': {
                'type': 'bearish', 
                'reliability': 0.7, 
                'description': 'Potential reversal pattern',
                'volume_importance': 'critical'
            },
            'bullish_engulfing': {
                'type': 'bullish', 
                'reliability': 0.8, 
                'description': 'Strong bullish reversal signal',
                'volume_importance': 'critical'
            },
            'bearish_engulfing': {
                'type': 'bearish', 
                'reliability': 0.8, 
                'description': 'Strong bearish reversal signal',
                'volume_importance': 'critical'
            },
            'doji': {
                'type': 'neutral', 
                'reliability': 0.5, 
                'description': 'Indecision pattern',
                'volume_importance': 'moderate'
            },
            'morning_star': {
                'type': 'bullish', 
                'reliability': 0.8, 
                'description': 'Strong bullish reversal pattern',
                'volume_importance': 'important'
            },
            'evening_star': {
                'type': 'bearish', 
                'reliability': 0.8, 
                'description': 'Strong bearish reversal pattern',
                'volume_importance': 'important'
            },
            'breakout': {
                'type': 'both', 
                'reliability': 0.9, 
                'description': 'Strong continuation/breakout signal',
                'volume_importance': 'critical'
            }
        }
        
        # Define pattern functions with TA-Lib
        if TALIB_AVAILABLE:
            self.patterns = {
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
                'tristar': talib.CDLTRISTAR,
                'three_inside_up': talib.CDL3INSIDE,
                'three_inside_down': talib.CDL3INSIDE,
                'three_outside_up': talib.CDL3OUTSIDE,
                'three_outside_down': talib.CDL3OUTSIDE,
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
                'meeting_lines': talib.CDLSEPARATINGLINES,
                'on_neck': talib.CDLONNECK,
                'piercing': talib.CDLPIERCING,
                'rising_falling_three_methods': talib.CDLRISEFALL3METHODS,
                'separating_lines': talib.CDLSEPARATINGLINES,
                'short_line': talib.CDLSHORTLINE,
                'takuri': talib.CDLTAKURI,
                'thrusting': talib.CDLTHRUSTING,
                'unique_three_rivers': talib.CDLUNIQUE3RIVER,
                'upside_gap_two_crows': talib.CDLUPSIDEGAP2CROWS
            }
        else:
            # Basic patterns without TA-Lib
            self.patterns = {
                'hammer': self._detect_hammer_basic,
                'shooting_star': self._detect_shooting_star_basic,
                'engulfing': self._detect_engulfing_basic,
                'doji': self._detect_doji_basic,
                'morning_star': self._detect_morning_star_basic,
                'evening_star': self._detect_evening_star_basic,
                'three_white_soldiers': self._detect_three_white_soldiers_basic,
                'three_black_crows': self._detect_three_black_crows_basic
            }
        
        logger.info("🚀 Enhanced Pattern Detector with Volume Integration initialized")
    
    def detect_patterns_with_volume(
        self, 
        opens: np.ndarray, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        volumes: Optional[np.ndarray] = None
    ) -> List[VolumeEnhancedPatternResult]:
        """
        Detect patterns with integrated volume analysis
        
        Args:
            opens: Array of opening prices
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            volumes: Array of volumes for volume confirmation
            
        Returns:
            List of VolumeEnhancedPatternResult objects with volume analysis
        """
        if len(opens) < 20:
            logger.warning("Insufficient data for pattern detection (minimum 20 candles required)")
            return []
        
        if volumes is None:
            logger.warning("No volume data provided, using default volume analysis")
            volumes = np.full_like(closes, 1000)  # Default volume
        
        results = []
        
        # Calculate volume averages for comparison
        volume_avg_20 = self._calculate_rolling_average(volumes, 20)
        volume_avg_10 = self._calculate_rolling_average(volumes, 10)
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for pattern detection with volume integration
            for pattern_name, pattern_func in self.patterns.items():
                try:
                    # Detect pattern using TA-Lib
                    pattern_result = pattern_func(opens, highs, lows, closes)
                    
                    # Find pattern occurrences
                    pattern_indices = np.where(pattern_result != 0)[0]
                    
                    for idx in pattern_indices:
                        # Integrate volume analysis directly
                        volume_enhanced_result = self._enhance_pattern_with_volume(
                            pattern_name, idx, pattern_result[idx], 
                            volumes, volume_avg_20, volume_avg_10,
                            opens, highs, lows, closes
                        )
                        
                        if volume_enhanced_result:
                            results.append(volume_enhanced_result)
                            
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name} with volume integration: {e}")
                    continue
        else:
            # Use basic implementations with volume integration
            for pattern_name, pattern_func in self.patterns.items():
                try:
                    # Detect pattern using basic implementation
                    pattern_result = pattern_func(opens, highs, lows, closes, volumes)
                    
                    # Find pattern occurrences
                    pattern_indices = np.where(pattern_result != 0)[0]
                    
                    for idx in pattern_indices:
                        # Integrate volume analysis directly
                        volume_enhanced_result = self._enhance_pattern_with_volume(
                            pattern_name, idx, pattern_result[idx], 
                            volumes, volume_avg_20, volume_avg_10,
                            opens, highs, lows, closes
                        )
                        
                        if volume_enhanced_result:
                            results.append(volume_enhanced_result)
                            
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name} with basic implementation: {e}")
                    continue
        
        # Sort results by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"🔍 Detected {len(results)} volume-enhanced patterns")
        return results
    
    def _enhance_pattern_with_volume(
        self, 
        pattern_name: str, 
        index: int, 
        pattern_value: float,
        volumes: np.ndarray, 
        volume_avg_20: np.ndarray, 
        volume_avg_10: np.ndarray,
        opens: np.ndarray, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray
    ) -> Optional[VolumeEnhancedPatternResult]:
        """
        Enhance a pattern with integrated volume analysis
        """
        try:
            # Get pattern metadata
            metadata = self.pattern_metadata.get(pattern_name, {})
            volume_rules = self.volume_rules.get(pattern_name, {})
            
            # Calculate volume metrics
            current_volume = volumes[index]
            avg_volume_20 = volume_avg_20[index] if index < len(volume_avg_20) else current_volume
            avg_volume_10 = volume_avg_10[index] if index < len(volume_avg_10) else current_volume
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # Determine volume confirmation
            required_ratio = volume_rules.get('required_ratio', 1.2)
            preferred_ratio = volume_rules.get('preferred_ratio', 1.5)
            
            volume_confirmed = volume_ratio >= required_ratio
            
            # Calculate volume factor (0.0-1.0)
            if volume_ratio >= preferred_ratio:
                volume_factor = min(1.0, (volume_ratio - required_ratio) / (preferred_ratio - required_ratio))
            elif volume_ratio >= required_ratio:
                volume_factor = 0.5 + 0.5 * (volume_ratio - required_ratio) / (preferred_ratio - required_ratio)
            else:
                volume_factor = 0.0
            
            # Determine signal type
            if metadata.get('type') == 'both':
                signal_type = 'bullish' if pattern_value > 0 else 'bearish'
            else:
                signal_type = metadata.get('type', 'neutral')
            
            # Calculate base confidence
            base_confidence = metadata.get('reliability', 0.5)
            
            # Adjust confidence based on volume confirmation
            if volume_confirmed:
                confidence = base_confidence * (1.0 + volume_factor * 0.3)
            else:
                confidence = base_confidence * (1.0 - 0.2)  # Penalty for no volume confirmation
            
            # Determine volume pattern
            volume_pattern = self._classify_volume_pattern(volume_ratio, current_volume, avg_volume_20)
            
            # Create enhanced result
            result = VolumeEnhancedPatternResult(
                pattern=pattern_name,
                confidence=min(1.0, confidence),
                volume_confirmed=volume_confirmed,
                volume_factor=volume_factor,
                signal_type=signal_type,
                index=index,
                strength=abs(pattern_value) if TALIB_AVAILABLE else 1.0,
                volume_ratio=volume_ratio,
                volume_pattern=volume_pattern,
                additional_info={
                    'description': metadata.get('description', ''),
                    'reliability': metadata.get('reliability', 0.5),
                    'volume_importance': metadata.get('volume_importance', 'moderate'),
                    'required_volume_ratio': required_ratio,
                    'preferred_volume_ratio': preferred_ratio
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing pattern {pattern_name} with volume: {e}")
            return None
    
    def _calculate_rolling_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling average with proper handling of edge cases"""
        if len(data) < window:
            return data
        
        result = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = np.mean(data[start_idx:i+1])
        
        return result
    
    def _classify_volume_pattern(self, volume_ratio: float, current_volume: float, avg_volume: float) -> str:
        """Classify volume pattern based on ratio"""
        if volume_ratio > 2.0:
            return "extreme_spike"
        elif volume_ratio > 1.5:
            return "volume_spike"
        elif volume_ratio > 1.2:
            return "above_average"
        elif volume_ratio > 0.8:
            return "normal_volume"
        else:
            return "low_volume"
    
    def detect_patterns_from_dataframe_with_volume(self, df: pd.DataFrame) -> List[VolumeEnhancedPatternResult]:
        """
        Detect patterns with volume integration from DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of VolumeEnhancedPatternResult objects
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return []
        
        # Extract OHLCV data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Detect patterns with volume integration
        return self.detect_patterns_with_volume(opens, highs, lows, closes, volumes)
    
    def get_volume_enhanced_summary(self, results: List[VolumeEnhancedPatternResult]) -> Dict[str, Any]:
        """
        Get summary of volume-enhanced pattern detection
        
        Args:
            results: List of VolumeEnhancedPatternResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {'message': 'No patterns detected'}
        
        summary = {
            'total_patterns': len(results),
            'volume_confirmed_patterns': len([r for r in results if r.volume_confirmed]),
            'confirmation_rate': len([r for r in results if r.volume_confirmed]) / len(results),
            'average_confidence': np.mean([r.confidence for r in results]),
            'average_volume_factor': np.mean([r.volume_factor for r in results]),
            'patterns_by_type': {},
            'volume_patterns': {},
            'strongest_patterns': []
        }
        
        # Count patterns by type
        for result in results:
            if result.signal_type not in summary['patterns_by_type']:
                summary['patterns_by_type'][result.signal_type] = 0
            summary['patterns_by_type'][result.signal_type] += 1
            
            if result.volume_pattern not in summary['volume_patterns']:
                summary['volume_patterns'][result.volume_pattern] = 0
            summary['volume_patterns'][result.volume_pattern] += 1
        
        # Get strongest patterns (highest confidence)
        strong_patterns = sorted(results, key=lambda x: x.confidence, reverse=True)[:5]
        summary['strongest_patterns'] = [
            {
                'pattern': r.pattern,
                'confidence': r.confidence,
                'volume_confirmed': r.volume_confirmed,
                'volume_factor': r.volume_factor,
                'signal_type': r.signal_type
            }
            for r in strong_patterns
        ]
        
        return summary
    
    # Basic pattern detection implementations (fallback when TA-Lib is not available)
    def _detect_hammer_basic(self, opens, highs, lows, closes, volumes):
        """Basic hammer detection"""
        result = np.zeros_like(closes)
        for i in range(1, len(closes)):
            body = abs(closes[i] - opens[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            
            if (lower_shadow > 2 * body and upper_shadow < body and 
                closes[i] > opens[i]):  # Bullish hammer
                result[i] = 100
        return result
    
    def _detect_shooting_star_basic(self, opens, highs, lows, closes, volumes):
        """Basic shooting star detection"""
        result = np.zeros_like(closes)
        for i in range(1, len(closes)):
            body = abs(closes[i] - opens[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            
            if (upper_shadow > 2 * body and lower_shadow < body and 
                closes[i] < opens[i]):  # Bearish shooting star
                result[i] = -100
        return result
    
    def _detect_engulfing_basic(self, opens, highs, lows, closes, volumes):
        """Basic engulfing pattern detection"""
        result = np.zeros_like(closes)
        for i in range(1, len(closes)):
            prev_body = abs(closes[i-1] - opens[i-1])
            curr_body = abs(closes[i] - opens[i])
            
            if curr_body > prev_body:
                if (opens[i] < closes[i-1] and closes[i] > opens[i-1]):  # Bullish engulfing
                    result[i] = 100
                elif (opens[i] > closes[i-1] and closes[i] < opens[i-1]):  # Bearish engulfing
                    result[i] = -100
        return result
    
    def _detect_doji_basic(self, opens, highs, lows, closes, volumes):
        """Basic doji detection"""
        result = np.zeros_like(closes)
        for i in range(len(closes)):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            if body <= total_range * 0.1:  # Very small body
                result[i] = 100
        return result
    
    def _detect_morning_star_basic(self, opens, highs, lows, closes, volumes):
        """Basic morning star detection"""
        result = np.zeros_like(closes)
        for i in range(2, len(closes)):
            # First day: bearish
            # Second day: small body (doji-like)
            # Third day: bullish
            if (closes[i-2] < opens[i-2] and  # First day bearish
                abs(closes[i-1] - opens[i-1]) < abs(closes[i-2] - opens[i-2]) * 0.3 and  # Second day small
                closes[i] > opens[i]):  # Third day bullish
                result[i] = 100
        return result
    
    def _detect_evening_star_basic(self, opens, highs, lows, closes, volumes):
        """Basic evening star detection"""
        result = np.zeros_like(closes)
        for i in range(2, len(closes)):
            # First day: bullish
            # Second day: small body (doji-like)
            # Third day: bearish
            if (closes[i-2] > opens[i-2] and  # First day bullish
                abs(closes[i-1] - opens[i-1]) < abs(closes[i-2] - opens[i-2]) * 0.3 and  # Second day small
                closes[i] < opens[i]):  # Third day bearish
                result[i] = -100
        return result
    
    def _detect_three_white_soldiers_basic(self, opens, highs, lows, closes, volumes):
        """Basic three white soldiers detection"""
        result = np.zeros_like(closes)
        for i in range(2, len(closes)):
            if (closes[i] > opens[i] and closes[i-1] > opens[i-1] and closes[i-2] > opens[i-2] and
                closes[i] > closes[i-1] and closes[i-1] > closes[i-2]):
                result[i] = 100
        return result
    
    def _detect_three_black_crows_basic(self, opens, highs, lows, closes, volumes):
        """Basic three black crows detection"""
        result = np.zeros_like(closes)
        for i in range(2, len(closes)):
            if (closes[i] < opens[i] and closes[i-1] < opens[i-1] and closes[i-2] < opens[i-2] and
                closes[i] < closes[i-1] and closes[i-1] < closes[i-2]):
                result[i] = -100
        return result
