#!/usr/bin/env python3
"""
Dynamic Confidence Threshold Detector
Volatility-based threshold adjustment for adaptive pattern detection
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import talib
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class VolatilityMetrics:
    """Volatility metrics for threshold adjustment"""
    atr: float
    bollinger_width: float
    price_volatility: float
    volume_volatility: float
    market_regime: str
    volatility_score: float
    timestamp: datetime

@dataclass
class DynamicThresholdResult:
    """Result from dynamic threshold calculation"""
    pattern_name: str
    base_threshold: float
    adjusted_threshold: float
    volatility_factor: float
    market_regime: str
    confidence_boost: float
    timestamp: datetime
    metadata: Dict[str, Any]

class VolatilityAnalyzer:
    """Analyze market volatility for threshold adjustment"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.volatility_history = deque(maxlen=100)
        self.regime_thresholds = {
            'low_volatility': 0.02,    # < 2% ATR
            'medium_volatility': 0.05,  # 2-5% ATR
            'high_volatility': 0.10,    # 5-10% ATR
            'extreme_volatility': 0.15  # > 10% ATR
        }
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                     closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(highs, lows, closes, timeperiod=period)
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return np.zeros_like(closes)
    
    def calculate_bollinger_bands(self, closes: np.ndarray, 
                                period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(closes, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return np.zeros_like(closes), np.zeros_like(closes), np.zeros_like(closes)
    
    def calculate_price_volatility(self, closes: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate price volatility (rolling standard deviation of returns)"""
        try:
            returns = np.diff(closes) / closes[:-1]
            volatility = np.zeros_like(closes)
            
            for i in range(period, len(closes)):
                volatility[i] = np.std(returns[i-period:i])
            
            return volatility
        except Exception as e:
            logger.error(f"Error calculating price volatility: {e}")
            return np.zeros_like(closes)
    
    def calculate_volume_volatility(self, volumes: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate volume volatility"""
        try:
            volume_volatility = np.zeros_like(volumes)
            
            for i in range(period, len(volumes)):
                volume_volatility[i] = np.std(volumes[i-period:i]) / np.mean(volumes[i-period:i])
            
            return volume_volatility
        except Exception as e:
            logger.error(f"Error calculating volume volatility: {e}")
            return np.zeros_like(volumes)
    
    def analyze_volatility(self, highs: np.ndarray, lows: np.ndarray, 
                          closes: np.ndarray, volumes: np.ndarray) -> VolatilityMetrics:
        """Analyze comprehensive volatility metrics"""
        try:
            # Calculate ATR
            atr = self.calculate_atr(highs, lows, closes)
            current_atr = atr[-1] if len(atr) > 0 else 0.0
            
            # Calculate Bollinger Bands width
            upper, middle, lower = self.calculate_bollinger_bands(closes)
            bollinger_width = ((upper[-1] - lower[-1]) / middle[-1]) if len(middle) > 0 and middle[-1] != 0 else 0.0
            
            # Calculate price volatility
            price_vol = self.calculate_price_volatility(closes)
            current_price_vol = price_vol[-1] if len(price_vol) > 0 else 0.0
            
            # Calculate volume volatility
            volume_vol = self.calculate_volume_volatility(volumes)
            current_volume_vol = volume_vol[-1] if len(volume_vol) > 0 else 0.0
            
            # Determine market regime
            market_regime = self._determine_market_regime(current_atr, bollinger_width)
            
            # Calculate volatility score (0-1)
            volatility_score = self._calculate_volatility_score(
                current_atr, bollinger_width, current_price_vol, current_volume_vol
            )
            
            metrics = VolatilityMetrics(
                atr=current_atr,
                bollinger_width=bollinger_width,
                price_volatility=current_price_vol,
                volume_volatility=current_volume_vol,
                market_regime=market_regime,
                volatility_score=volatility_score,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Store in history
            self.volatility_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return self._create_default_volatility_metrics()
    
    def _determine_market_regime(self, atr: float, bollinger_width: float) -> str:
        """Determine market regime based on volatility metrics"""
        # Normalize ATR to percentage
        atr_percent = atr * 100
        
        if atr_percent < self.regime_thresholds['low_volatility']:
            return "low_volatility"
        elif atr_percent < self.regime_thresholds['medium_volatility']:
            return "medium_volatility"
        elif atr_percent < self.regime_thresholds['high_volatility']:
            return "high_volatility"
        else:
            return "extreme_volatility"
    
    def _calculate_volatility_score(self, atr: float, bollinger_width: float,
                                  price_vol: float, volume_vol: float) -> float:
        """Calculate normalized volatility score (0-1)"""
        # Normalize each component
        atr_norm = min(atr * 100 / self.regime_thresholds['extreme_volatility'], 1.0)
        bb_norm = min(bollinger_width / 0.2, 1.0)  # Normalize to 20% width
        price_norm = min(price_vol / 0.1, 1.0)  # Normalize to 10% volatility
        volume_norm = min(volume_vol / 2.0, 1.0)  # Normalize to 200% volume volatility
        
        # Weighted average
        volatility_score = (
            0.4 * atr_norm +
            0.3 * bb_norm +
            0.2 * price_norm +
            0.1 * volume_norm
        )
        
        return min(max(volatility_score, 0.0), 1.0)
    
    def _create_default_volatility_metrics(self) -> VolatilityMetrics:
        """Create default volatility metrics when analysis fails"""
        return VolatilityMetrics(
            atr=0.02,
            bollinger_width=0.05,
            price_volatility=0.02,
            volume_volatility=0.5,
            market_regime="medium_volatility",
            volatility_score=0.5,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_volatility_history(self) -> List[VolatilityMetrics]:
        """Get volatility history"""
        return list(self.volatility_history)
    
    def get_volatility_trend(self) -> str:
        """Get volatility trend (increasing/decreasing/stable)"""
        if len(self.volatility_history) < 5:
            return "stable"
        
        recent_scores = [m.volatility_score for m in list(self.volatility_history)[-5:]]
        
        if len(recent_scores) < 2:
            return "stable"
        
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"

class DynamicThresholdCalculator:
    """Calculate dynamic confidence thresholds based on volatility"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Base thresholds for different patterns
        self.base_thresholds = {
            'doji': 0.6,
            'hammer': 0.7,
            'engulfing': 0.8,
            'shooting_star': 0.7,
            'morning_star': 0.8,
            'evening_star': 0.8,
            'three_white_soldiers': 0.8,
            'three_black_crows': 0.8,
            'hanging_man': 0.7,
            'inverted_hammer': 0.7,
            'spinning_top': 0.6,
            'marubozu': 0.7,
            'tristar': 0.8,
            'three_inside': 0.8,
            'three_outside': 0.8,
            'breakaway': 0.8,
            'dark_cloud_cover': 0.8,
            'dragonfly_doji': 0.7,
            'gravestone_doji': 0.7,
            'harami': 0.7,
            'harami_cross': 0.7,
            'high_wave': 0.6,
            'identical_three_crows': 0.8,
            'kicking': 0.8,
            'ladder_bottom': 0.8,
            'long_legged_doji': 0.6,
            'long_line': 0.7,
            'on_neck': 0.7,
            'piercing': 0.8
        }
        
        # Volatility adjustment factors
        self.volatility_adjustments = {
            'low_volatility': {
                'factor': 0.8,      # Lower threshold in low volatility
                'boost': 0.1        # Confidence boost
            },
            'medium_volatility': {
                'factor': 1.0,      # Standard threshold
                'boost': 0.0        # No boost
            },
            'high_volatility': {
                'factor': 1.2,      # Higher threshold in high volatility
                'boost': -0.1       # Confidence penalty
            },
            'extreme_volatility': {
                'factor': 1.5,      # Much higher threshold
                'boost': -0.2       # Higher penalty
            }
        }
        
        # Pattern-specific adjustments
        self.pattern_adjustments = {
            'doji': {'volatility_sensitive': True, 'base_adjustment': 0.0},
            'hammer': {'volatility_sensitive': True, 'base_adjustment': 0.0},
            'engulfing': {'volatility_sensitive': True, 'base_adjustment': 0.0},
            'shooting_star': {'volatility_sensitive': True, 'base_adjustment': 0.0},
            'morning_star': {'volatility_sensitive': False, 'base_adjustment': 0.1},
            'evening_star': {'volatility_sensitive': False, 'base_adjustment': 0.1},
            'three_white_soldiers': {'volatility_sensitive': False, 'base_adjustment': 0.1},
            'three_black_crows': {'volatility_sensitive': False, 'base_adjustment': 0.1}
        }
    
    def calculate_dynamic_threshold(self, pattern_name: str, highs: np.ndarray,
                                  lows: np.ndarray, closes: np.ndarray,
                                  volumes: np.ndarray) -> DynamicThresholdResult:
        """Calculate dynamic threshold for a pattern"""
        try:
            # Analyze volatility
            volatility_metrics = self.volatility_analyzer.analyze_volatility(
                highs, lows, closes, volumes
            )
            
            # Get base threshold
            base_threshold = self.base_thresholds.get(pattern_name, 0.7)
            
            # Get volatility adjustment
            regime = volatility_metrics.market_regime
            adjustment = self.volatility_adjustments.get(regime, self.volatility_adjustments['medium_volatility'])
            
            # Calculate adjusted threshold
            volatility_factor = adjustment['factor']
            adjusted_threshold = base_threshold * volatility_factor
            
            # Apply pattern-specific adjustments
            pattern_config = self.pattern_adjustments.get(pattern_name, {
                'volatility_sensitive': True, 'base_adjustment': 0.0
            })
            
            if not pattern_config['volatility_sensitive']:
                # For patterns less sensitive to volatility, use smaller adjustment
                volatility_factor = 1.0 + (volatility_factor - 1.0) * 0.5
                adjusted_threshold = base_threshold * volatility_factor
            
            # Add base adjustment
            adjusted_threshold += pattern_config['base_adjustment']
            
            # Calculate confidence boost/penalty
            confidence_boost = adjustment['boost']
            
            # Additional boost based on volatility trend
            volatility_trend = self.volatility_analyzer.get_volatility_trend()
            if volatility_trend == "decreasing" and regime in ["high_volatility", "extreme_volatility"]:
                confidence_boost += 0.05  # Small boost when volatility is decreasing
            
            # Cap thresholds
            adjusted_threshold = max(min(adjusted_threshold, 0.95), 0.3)
            confidence_boost = max(min(confidence_boost, 0.3), -0.3)
            
            return DynamicThresholdResult(
                pattern_name=pattern_name,
                base_threshold=base_threshold,
                adjusted_threshold=adjusted_threshold,
                volatility_factor=volatility_factor,
                market_regime=regime,
                confidence_boost=confidence_boost,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'atr': volatility_metrics.atr,
                    'bollinger_width': volatility_metrics.bollinger_width,
                    'price_volatility': volatility_metrics.price_volatility,
                    'volume_volatility': volatility_metrics.volume_volatility,
                    'volatility_score': volatility_metrics.volatility_score,
                    'volatility_trend': volatility_trend,
                    'pattern_volatility_sensitive': pattern_config['volatility_sensitive']
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating dynamic threshold for {pattern_name}: {e}")
            return self._create_default_threshold_result(pattern_name)
    
    def _create_default_threshold_result(self, pattern_name: str) -> DynamicThresholdResult:
        """Create default threshold result when calculation fails"""
        base_threshold = self.base_thresholds.get(pattern_name, 0.7)
        
        return DynamicThresholdResult(
            pattern_name=pattern_name,
            base_threshold=base_threshold,
            adjusted_threshold=base_threshold,
            volatility_factor=1.0,
            market_regime="medium_volatility",
            confidence_boost=0.0,
            timestamp=datetime.now(timezone.utc),
            metadata={'error': 'Calculation failed'}
        )
    
    def get_threshold_stats(self) -> Dict[str, Any]:
        """Get threshold calculation statistics"""
        return {
            'base_thresholds': self.base_thresholds,
            'volatility_adjustments': self.volatility_adjustments,
            'pattern_adjustments': self.pattern_adjustments,
            'volatility_history_size': len(self.volatility_analyzer.volatility_history)
        }

class DynamicConfidenceThresholdDetector:
    """Dynamic confidence threshold detector for adaptive pattern detection"""
    
    def __init__(self):
        self.threshold_calculator = DynamicThresholdCalculator()
        self.threshold_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=2)  # Cache for 2 minutes
        
        logger.info("🚀 Dynamic Confidence Threshold Detector initialized")
    
    def get_dynamic_threshold(self, pattern_name: str, symbol: str, timeframe: str,
                            highs: np.ndarray, lows: np.ndarray, 
                            closes: np.ndarray, volumes: np.ndarray) -> DynamicThresholdResult:
        """Get dynamic threshold for pattern detection"""
        cache_key = f"{pattern_name}_{symbol}_{timeframe}"
        
        # Check cache
        if cache_key in self.threshold_cache:
            if datetime.now(timezone.utc) < self.cache_expiry[cache_key]:
                return self.threshold_cache[cache_key]
            else:
                # Remove expired cache
                del self.threshold_cache[cache_key]
                del self.cache_expiry[cache_key]
        
        # Calculate new threshold
        threshold_result = self.threshold_calculator.calculate_dynamic_threshold(
            pattern_name, highs, lows, closes, volumes
        )
        
        # Cache result
        self.threshold_cache[cache_key] = threshold_result
        self.cache_expiry[cache_key] = datetime.now(timezone.utc) + self.cache_duration
        
        return threshold_result
    
    def adjust_pattern_confidence(self, pattern_name: str, base_confidence: float,
                                symbol: str, timeframe: str, highs: np.ndarray,
                                lows: np.ndarray, closes: np.ndarray, 
                                volumes: np.ndarray) -> float:
        """Adjust pattern confidence based on dynamic thresholds"""
        try:
            # Get dynamic threshold
            threshold_result = self.get_dynamic_threshold(
                pattern_name, symbol, timeframe, highs, lows, closes, volumes
            )
            
            # Apply confidence boost/penalty
            adjusted_confidence = base_confidence + threshold_result.confidence_boost
            
            # Ensure confidence is within bounds
            adjusted_confidence = max(min(adjusted_confidence, 1.0), 0.0)
            
            return adjusted_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting pattern confidence: {e}")
            return base_confidence
    
    def get_volatility_metrics(self, highs: np.ndarray, lows: np.ndarray,
                              closes: np.ndarray, volumes: np.ndarray) -> VolatilityMetrics:
        """Get current volatility metrics"""
        return self.threshold_calculator.volatility_analyzer.analyze_volatility(
            highs, lows, closes, volumes
        )
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'cache_size': len(self.threshold_cache),
            'threshold_stats': self.threshold_calculator.get_threshold_stats(),
            'volatility_trend': self.threshold_calculator.volatility_analyzer.get_volatility_trend()
        }
