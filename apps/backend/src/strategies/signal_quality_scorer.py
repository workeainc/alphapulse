#!/usr/bin/env python3
"""
Phase 3: Signal Quality Scorer
Implements multi-factor signal ranking and quality scoring to prioritize
high-quality signals and provide comprehensive signal evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import logging
from enum import Enum
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import os

logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECT = "reject"

@dataclass
class SignalQualityResult:
    """Result from signal quality scoring"""
    pattern_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    original_confidence: float
    quality_score: float
    quality_level: SignalQuality
    quality_factors: Dict[str, float]
    quality_reasons: List[str]
    risk_score: float
    reward_potential: float
    signal_strength: float
    priority_rank: int
    
    # Phase 4A: Enhanced fields
    calibrated_confidence: Optional[float] = None
    calibration_confidence_interval: Optional[Dict[str, float]] = None
    multi_timeframe_alignment: Optional[float] = None
    timeframe_confirmation_count: Optional[int] = None
    market_regime: Optional[str] = None
    regime_adjusted_confidence: Optional[float] = None
    explanation_factors: Optional[Dict[str, Any]] = None

@dataclass
class QualityScoringConfig:
    """Configuration for signal quality scoring parameters"""
    # Quality thresholds
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    fair_threshold: float = 0.55
    poor_threshold: float = 0.40
    
    # Factor weights
    pattern_confidence_weight: float = 0.25
    volume_confirmation_weight: float = 0.20
    momentum_weight: float = 0.15
    volatility_weight: float = 0.15
    market_context_weight: float = 0.15
    risk_reward_weight: float = 0.10
    
    # Risk scoring
    max_risk_score: float = 1.0
    min_reward_potential: float = 0.1
    max_reward_potential: float = 5.0
    
    # Pattern-specific quality adjustments
    pattern_quality_multipliers: Dict[str, float] = None
    
    # Phase 4A: Confidence Calibration Settings
    enable_confidence_calibration: bool = True
    calibration_window_days: int = 30
    min_calibration_samples: int = 100
    calibration_method: str = 'platt_scaling'  # 'platt_scaling', 'isotonic_regression'
    calibration_update_frequency_hours: int = 24
    
    # Multi-timeframe integration settings
    enable_multi_timeframe_scoring: bool = True
    timeframe_hierarchy: Dict[str, List[str]] = None  # e.g., {'1h': ['5m', '15m'], '4h': ['1h', '30m']}
    timeframe_confirmation_weight: float = 0.15
    multi_timeframe_alignment_threshold: float = 0.6
    
    # Market regime settings
    enable_regime_scoring: bool = True
    regime_weights: Dict[str, float] = None  # bull, bear, sideways, crash
    
    def __post_init__(self):
        if self.pattern_quality_multipliers is None:
            self.pattern_quality_multipliers = {
                'doji': 0.9,      # Doji is less reliable
                'hammer': 1.1,    # Hammer is more reliable
                'engulfing': 1.2, # Engulfing is very reliable
                'shooting_star': 1.0,
                'morning_star': 1.3,
                'evening_star': 1.3,
                'three_white_soldiers': 1.4,
                'three_black_crows': 1.4,
                'hanging_man': 1.0,
                'inverted_hammer': 1.1,
                'spinning_top': 0.8,
                'marubozu': 1.5,
                'tristar': 0.7,
                'breakaway': 1.6,
                'dark_cloud_cover': 1.3,
                'dragonfly_doji': 0.9,
                'gravestone_doji': 0.9,
                'harami': 1.0,
                'harami_cross': 0.9,
                'high_wave': 0.9,
                'identical_three_crows': 1.2,
                'kicking': 1.5,
                'ladder_bottom': 1.1,
                'long_legged_doji': 0.9,
                'long_line': 1.2,
                'on_neck': 1.0,
                'piercing': 1.3,
                'rising_three_methods': 1.1,
                'separating_lines': 1.2
            }
        
        if self.timeframe_hierarchy is None:
            self.timeframe_hierarchy = {
                '1m': ['1m'],
                '5m': ['1m', '5m'],
                '15m': ['5m', '15m'],
                '30m': ['15m', '30m'],
                '1h': ['30m', '1h'],
                '4h': ['1h', '4h'],
                '1d': ['4h', '1d']
            }
        
        if self.regime_weights is None:
            self.regime_weights = {
                'bull': 1.2,      # Higher confidence in bull markets
                'bear': 0.8,      # Lower confidence in bear markets
                'sideways': 1.0,  # Neutral in sideways markets
                'crash': 0.5      # Much lower confidence in crash markets
            }

class RiskRewardAnalyzer:
    """Analyzes risk-reward characteristics of signals"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
    
    def calculate_risk_reward_ratio(self, pattern: Dict, ohlcv_data: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """Calculate risk-reward ratio and potential"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        pattern_direction = pattern.get('direction', 'neutral')
        
        # Get recent price data for analysis
        start_idx = max(0, pattern_index - 20)
        end_idx = min(pattern_index + 10, len(ohlcv_data['close']))
        
        if start_idx >= end_idx:
            return 0.0, 0.0, 0.0
        
        highs = ohlcv_data['high'][start_idx:end_idx]
        lows = ohlcv_data['low'][start_idx:end_idx]
        closes = ohlcv_data['close'][start_idx:end_idx]
        
        # Calculate support and resistance levels
        recent_high = np.max(highs)
        recent_low = np.min(lows)
        current_price = closes[-1]
        
        # Calculate potential reward and risk
        if pattern_direction == 'bullish':
            # For bullish patterns, reward is distance to resistance, risk is distance to support
            potential_reward = (recent_high - current_price) / current_price * 100
            potential_risk = (current_price - recent_low) / current_price * 100
        elif pattern_direction == 'bearish':
            # For bearish patterns, reward is distance to support, risk is distance to resistance
            potential_reward = (current_price - recent_low) / current_price * 100
            potential_risk = (recent_high - current_price) / current_price * 100
        else:
            # For neutral patterns, use average movement
            avg_range = (recent_high - recent_low) / current_price * 100
            potential_reward = avg_range * 0.5
            potential_risk = avg_range * 0.5
        
        # Clamp values
        potential_reward = max(self.config.min_reward_potential, 
                             min(self.config.max_reward_potential, potential_reward))
        potential_risk = max(0.1, min(10.0, potential_risk))
        
        # Calculate risk-reward ratio
        risk_reward_ratio = potential_reward / potential_risk if potential_risk > 0 else 0.0
        
        return risk_reward_ratio, potential_reward, potential_risk

class MarketContextAnalyzer:
    """Analyzes market context for signal quality"""
    
    def __init__(self, config: QualityScoringConfig):
        self.config = config
    
    def analyze_market_context(self, pattern: Dict, ohlcv_data: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
        """Analyze market context for signal quality"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        pattern_direction = pattern.get('direction', 'neutral')
        
        # Get recent data for context analysis
        start_idx = max(0, pattern_index - 50)
        end_idx = pattern_index + 1
        
        if start_idx >= end_idx:
            return 0.0, ["Insufficient market context data"]
        
        closes = ohlcv_data['close'][start_idx:end_idx]
        volumes = ohlcv_data['volume'][start_idx:end_idx]
        
        context_score = 0.0
        context_reasons = []
        
        # Trend analysis
        if len(closes) >= 20:
            short_trend = np.polyfit(range(10), closes[-10:], 1)[0]
            long_trend = np.polyfit(range(20), closes[-20:], 1)[0]
            
            # Check trend alignment with pattern direction
            if pattern_direction == 'bullish' and short_trend > 0 and long_trend > 0:
                context_score += 0.3
                context_reasons.append("Strong bullish trend alignment")
            elif pattern_direction == 'bearish' and short_trend < 0 and long_trend < 0:
                context_score += 0.3
                context_reasons.append("Strong bearish trend alignment")
            elif pattern_direction == 'neutral':
                context_score += 0.2
                context_reasons.append("Neutral pattern in mixed trend")
            else:
                context_reasons.append("Pattern direction conflicts with trend")
        
        # Volume trend analysis
        if len(volumes) >= 10:
            recent_volume_trend = np.polyfit(range(10), volumes[-10:], 1)[0]
            avg_volume = np.mean(volumes[-10:])
            current_volume = volumes[-1]
            
            if current_volume > avg_volume * 1.2:
                context_score += 0.2
                context_reasons.append("Above-average volume")
            elif current_volume < avg_volume * 0.8:
                context_score -= 0.1
                context_reasons.append("Below-average volume")
            
            if recent_volume_trend > 0:
                context_score += 0.1
                context_reasons.append("Increasing volume trend")
        
        # Volatility analysis
        if len(closes) >= 20:
            returns = np.diff(closes) / closes[:-1] * 100
            volatility = np.std(returns)
            
            if 0.5 <= volatility <= 3.0:  # Optimal volatility range
                context_score += 0.2
                context_reasons.append("Optimal volatility range")
            elif volatility < 0.3:
                context_score -= 0.1
                context_reasons.append("Low volatility environment")
            elif volatility > 5.0:
                context_score -= 0.1
                context_reasons.append("High volatility environment")
        
        # Price level analysis
        if len(closes) >= 50:
            price_percentile = np.percentile(closes, 50)
            current_price = closes[-1]
            
            if pattern_direction == 'bullish' and current_price < price_percentile:
                context_score += 0.1
                context_reasons.append("Bullish pattern at lower price level")
            elif pattern_direction == 'bearish' and current_price > price_percentile:
                context_score += 0.1
                context_reasons.append("Bearish pattern at higher price level")
        
        return max(0.0, min(1.0, context_score)), context_reasons

class SignalQualityScorer:
    """
    Implements comprehensive signal quality scoring for candlestick patterns
    to prioritize high-quality signals and provide detailed evaluation.
    """
    
    def __init__(self, config: Optional[QualityScoringConfig] = None):
        self.config = config or QualityScoringConfig()
        self.risk_reward_analyzer = RiskRewardAnalyzer(self.config)
        self.market_context_analyzer = MarketContextAnalyzer(self.config)
        
        # Phase 4A: Confidence calibration components
        self.calibration_models = {}
        self.calibration_data = {}
        self.last_calibration_update = {}
        
        # Phase 4A: Multi-timeframe analysis components
        self.timeframe_analyzers = {}
        
        logger.info("🎯 Signal quality scorer initialized with Phase 4A enhancements")
    
    def score_signals(self, patterns: List[Dict], 
                     noise_filter_results: List[Dict],
                     validation_results: List[Dict],
                     ohlcv_data: Dict[str, np.ndarray]) -> List[SignalQualityResult]:
        """
        Score signal quality for patterns
        
        Args:
            patterns: List of detected patterns
            noise_filter_results: Results from noise filtering
            validation_results: Results from post-detection validation
            ohlcv_data: OHLCV data arrays
            
        Returns:
            List of signal quality results
        """
        if not patterns:
            return []
        
        # Create lookup dictionaries for noise filter and validation results
        noise_lookup = {r.get('pattern_id', ''): r for r in noise_filter_results}
        validation_lookup = {r.get('pattern_id', ''): r for r in validation_results}
        
        quality_results = []
        
        for pattern in patterns:
            pattern_id = pattern.get('pattern_id', '')
            noise_result = noise_lookup.get(pattern_id, {})
            validation_result = validation_lookup.get(pattern_id, {})
            
            result = self._score_single_signal(
                pattern, noise_result, validation_result, ohlcv_data
            )
            quality_results.append(result)
        
        # Sort by quality score and assign priority ranks
        quality_results.sort(key=lambda x: x.quality_score, reverse=True)
        for i, result in enumerate(quality_results):
            result.priority_rank = i + 1
        
        return quality_results
    
    def _score_single_signal(self, pattern: Dict, noise_result: Dict, 
                           validation_result: Dict, ohlcv_data: Dict[str, np.ndarray]) -> SignalQualityResult:
        """Score quality for a single signal"""
        
        pattern_id = pattern.get('pattern_id', '')
        symbol = pattern.get('symbol', '')
        timeframe = pattern.get('timeframe', '')
        timestamp = pattern.get('timestamp', datetime.now(timezone.utc))
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        original_confidence = pattern.get('confidence', 0.0)
        
        # Get pattern-specific quality multiplier
        quality_multiplier = self.config.pattern_quality_multipliers.get(pattern_type, 1.0)
        
        # Calculate quality factors
        quality_factors = {}
        quality_reasons = []
        
        # 1. Pattern confidence factor
        confidence_factor = original_confidence
        quality_factors['pattern_confidence'] = confidence_factor
        quality_reasons.append(f"Pattern confidence: {confidence_factor:.2f}")
        
        # 2. Volume confirmation factor
        volume_factor = validation_result.get('volume_confirmation_score', 0.0)
        quality_factors['volume_confirmation'] = volume_factor
        if volume_factor > 0.7:
            quality_reasons.append(f"Strong volume confirmation: {volume_factor:.2f}")
        elif volume_factor < 0.3:
            quality_reasons.append(f"Weak volume confirmation: {volume_factor:.2f}")
        
        # 3. Momentum factor
        momentum_factor = validation_result.get('momentum_score', 0.0)
        quality_factors['momentum'] = momentum_factor
        if momentum_factor > 0.7:
            quality_reasons.append(f"Strong momentum: {momentum_factor:.2f}")
        elif momentum_factor < 0.3:
            quality_reasons.append(f"Weak momentum: {momentum_factor:.2f}")
        
        # 4. Volatility factor (from noise filtering)
        volatility_factor = 1.0 - (noise_result.get('noise_score', 1.0) - 1.0) / 4.0  # Normalize to 0-1
        volatility_factor = max(0.0, min(1.0, volatility_factor))
        quality_factors['volatility'] = volatility_factor
        if volatility_factor > 0.7:
            quality_reasons.append(f"Good volatility conditions: {volatility_factor:.2f}")
        elif volatility_factor < 0.3:
            quality_reasons.append(f"Poor volatility conditions: {volatility_factor:.2f}")
        
        # 5. Market context factor
        market_context_factor, context_reasons = self.market_context_analyzer.analyze_market_context(
            pattern, ohlcv_data
        )
        quality_factors['market_context'] = market_context_factor
        quality_reasons.extend(context_reasons)
        
        # 6. Risk-reward factor
        risk_reward_ratio, reward_potential, risk_score = self.risk_reward_analyzer.calculate_risk_reward_ratio(
            pattern, ohlcv_data
        )
        risk_reward_factor = min(1.0, risk_reward_ratio / 3.0)  # Normalize to 0-1
        quality_factors['risk_reward'] = risk_reward_factor
        quality_reasons.append(f"Risk-reward ratio: {risk_reward_ratio:.2f}")
        
        # Calculate weighted quality score
        quality_score = (
            confidence_factor * self.config.pattern_confidence_weight +
            volume_factor * self.config.volume_confirmation_weight +
            momentum_factor * self.config.momentum_weight +
            volatility_factor * self.config.volatility_weight +
            market_context_factor * self.config.market_context_weight +
            risk_reward_factor * self.config.risk_reward_weight
        ) * quality_multiplier
        
        # Clamp quality score
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Determine quality level
        if quality_score >= self.config.excellent_threshold:
            quality_level = SignalQuality.EXCELLENT
        elif quality_score >= self.config.good_threshold:
            quality_level = SignalQuality.GOOD
        elif quality_score >= self.config.fair_threshold:
            quality_level = SignalQuality.FAIR
        elif quality_score >= self.config.poor_threshold:
            quality_level = SignalQuality.POOR
        else:
            quality_level = SignalQuality.REJECT
        
        # Calculate signal strength (combination of quality and confidence)
        signal_strength = (quality_score + original_confidence) / 2
        
        return SignalQualityResult(
            pattern_id=pattern_id,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            original_confidence=original_confidence,
            quality_score=quality_score,
            quality_level=quality_level,
            quality_factors=quality_factors,
            quality_reasons=quality_reasons,
            risk_score=risk_score,
            reward_potential=reward_potential,
            signal_strength=signal_strength,
            priority_rank=0  # Will be set after sorting
        )
    
    def get_quality_stats(self, results: List[SignalQualityResult]) -> Dict:
        """Get statistics about quality scoring results"""
        if not results:
            return {}
        
        total_signals = len(results)
        quality_counts = {
            SignalQuality.EXCELLENT: 0,
            SignalQuality.GOOD: 0,
            SignalQuality.FAIR: 0,
            SignalQuality.POOR: 0,
            SignalQuality.REJECT: 0
        }
        
        for result in results:
            quality_counts[result.quality_level] += 1
        
        avg_quality_score = np.mean([r.quality_score for r in results])
        avg_signal_strength = np.mean([r.signal_strength for r in results])
        avg_risk_reward = np.mean([r.reward_potential / r.risk_score for r in results if r.risk_score > 0])
        
        return {
            'total_signals': total_signals,
            'quality_distribution': {level.value: count for level, count in quality_counts.items()},
            'avg_quality_score': avg_quality_score,
            'avg_signal_strength': avg_signal_strength,
            'avg_risk_reward_ratio': avg_risk_reward,
            'excellent_signals': quality_counts[SignalQuality.EXCELLENT],
            'good_signals': quality_counts[SignalQuality.GOOD],
            'fair_signals': quality_counts[SignalQuality.FAIR],
            'poor_signals': quality_counts[SignalQuality.POOR],
            'rejected_signals': quality_counts[SignalQuality.REJECT]
        }
    
    def filter_by_quality(self, results: List[SignalQualityResult], 
                         min_quality: SignalQuality = SignalQuality.FAIR) -> List[SignalQualityResult]:
        """Filter results by minimum quality level"""
        quality_order = {
            SignalQuality.EXCELLENT: 4,
            SignalQuality.GOOD: 3,
            SignalQuality.FAIR: 2,
            SignalQuality.POOR: 1,
            SignalQuality.REJECT: 0
        }
        
        min_quality_level = quality_order[min_quality]
        
        filtered_results = [
            result for result in results
            if quality_order[result.quality_level] >= min_quality_level
        ]
        
        return filtered_results
    
    # Phase 4A: Confidence Calibration Methods
    
    def calibrate_confidence(self, pattern_type: str, confidence_scores: List[float], 
                           actual_outcomes: List[bool]) -> Optional[float]:
        """Calibrate confidence scores using historical outcomes"""
        if not self.config.enable_confidence_calibration:
            return None
        
        if len(confidence_scores) < self.config.min_calibration_samples:
            logger.warning(f"Insufficient calibration data for {pattern_type}: {len(confidence_scores)} samples")
            return None
        
        try:
            if self.config.calibration_method == 'platt_scaling':
                return self._platt_scaling_calibration(confidence_scores, actual_outcomes)
            elif self.config.calibration_method == 'isotonic_regression':
                return self._isotonic_calibration(confidence_scores, actual_outcomes)
            else:
                logger.warning(f"Unknown calibration method: {self.config.calibration_method}")
                return None
        except Exception as e:
            logger.error(f"Error in confidence calibration for {pattern_type}: {e}")
            return None
    
    def _platt_scaling_calibration(self, confidence_scores: List[float], 
                                 actual_outcomes: List[bool]) -> float:
        """Apply Platt scaling for confidence calibration"""
        try:
            # Convert to numpy arrays
            X = np.array(confidence_scores).reshape(-1, 1)
            y = np.array(actual_outcomes, dtype=int)
            
            # Fit logistic regression for calibration
            calibrator = CalibratedClassifierCV(
                LogisticRegression(), 
                cv='prefit', 
                method='sigmoid'
            )
            
            # Create a dummy classifier with the confidence scores
            from sklearn.base import BaseEstimator, ClassifierMixin
            class ConfidenceClassifier(BaseEstimator, ClassifierMixin):
                def fit(self, X, y):
                    self.classes_ = np.array([0, 1])
                    return self
                def predict_proba(self, X):
                    return np.column_stack([1 - X.flatten(), X.flatten()])
            
            dummy_classifier = ConfidenceClassifier()
            dummy_classifier.fit(X, y)
            
            # Calibrate
            calibrator.fit(dummy_classifier, X, y)
            
            # Return calibrated probability for the latest confidence score
            latest_confidence = confidence_scores[-1]
            calibrated_prob = calibrator.predict_proba([[latest_confidence]])[0][1]
            
            return calibrated_prob
            
        except Exception as e:
            logger.error(f"Error in Platt scaling calibration: {e}")
            return confidence_scores[-1] if confidence_scores else 0.5
    
    def _isotonic_calibration(self, confidence_scores: List[float], 
                            actual_outcomes: List[bool]) -> float:
        """Apply isotonic regression for confidence calibration"""
        try:
            # Convert to numpy arrays
            X = np.array(confidence_scores)
            y = np.array(actual_outcomes, dtype=int)
            
            # Fit isotonic regression
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(X, y)
            
            # Return calibrated probability for the latest confidence score
            latest_confidence = confidence_scores[-1]
            calibrated_prob = isotonic.predict([latest_confidence])[0]
            
            return calibrated_prob
            
        except Exception as e:
            logger.error(f"Error in isotonic calibration: {e}")
            return confidence_scores[-1] if confidence_scores else 0.5
    
    def get_calibration_confidence_interval(self, calibrated_confidence: float, 
                                          sample_size: int) -> Dict[str, float]:
        """Calculate confidence interval for calibrated confidence"""
        if sample_size < 30:
            return {'lower': calibrated_confidence * 0.8, 'upper': calibrated_confidence * 1.2}
        
        # Wilson score interval for binomial proportion
        z = 1.96  # 95% confidence level
        n = sample_size
        p = calibrated_confidence
        
        denominator = 1 + z**2/n
        centre_adjusted_probability = (p + z*z/(2*n))/denominator
        adjusted_standard_error = z * np.sqrt((p*(1-p))/n + z*z/(4*n*n))/denominator
        
        lower_bound = max(0, centre_adjusted_probability - adjusted_standard_error)
        upper_bound = min(1, centre_adjusted_probability + adjusted_standard_error)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence_level': 0.95
        }
    
    # Phase 4A: Multi-Timeframe Analysis Methods
    
    def analyze_multi_timeframe_alignment(self, pattern: Dict, 
                                        multi_timeframe_patterns: Dict[str, List[Dict]]) -> Tuple[float, int, Dict]:
        """Analyze multi-timeframe alignment for a pattern"""
        if not self.config.enable_multi_timeframe_scoring:
            return 0.0, 0, {}
        
        if not pattern or not multi_timeframe_patterns:
            return 0.0, 0, {}
        
        pattern_timeframe = pattern.get('timeframe', '1h')
        pattern_direction = pattern.get('direction', 'neutral')
        pattern_type = pattern.get('pattern_type', 'unknown')
        
        # Get relevant timeframes for hierarchy
        if self.config.timeframe_hierarchy is None:
            hierarchy_timeframes = [pattern_timeframe]
        else:
            hierarchy_timeframes = self.config.timeframe_hierarchy.get(pattern_timeframe, [pattern_timeframe])
        
        alignment_score = 0.0
        confirmation_count = 0
        alignment_details = {}
        
        for tf in hierarchy_timeframes:
            if tf == pattern_timeframe:
                continue  # Skip the current timeframe
                
            tf_patterns = multi_timeframe_patterns.get(tf, [])
            if not tf_patterns:
                continue
            
            # Find patterns in the same time window
            pattern_timestamp = pattern.get('timestamp')
            if not pattern_timestamp:
                continue
            
            # Look for confirming patterns within ±2 candles
            for tf_pattern in tf_patterns:
                tf_timestamp = tf_pattern.get('timestamp')
                if not tf_timestamp:
                    continue
                
                # Check if patterns are in the same time window
                time_diff = abs((pattern_timestamp - tf_timestamp).total_seconds())
                max_time_diff = self._get_timeframe_seconds(tf) * 2
                
                if time_diff <= max_time_diff:
                    tf_direction = tf_pattern.get('direction', 'neutral')
                    tf_type = tf_pattern.get('pattern_type', 'unknown')
                    
                    # Check alignment
                    if tf_direction == pattern_direction:
                        alignment_score += 0.3
                        confirmation_count += 1
                        alignment_details[tf] = {
                            'direction_match': True,
                            'pattern_type': tf_type,
                            'confidence': tf_pattern.get('confidence', 0.0)
                        }
                    elif tf_direction != 'neutral':
                        # Conflicting direction
                        alignment_score -= 0.1
                        alignment_details[tf] = {
                            'direction_match': False,
                            'pattern_type': tf_type,
                            'confidence': tf_pattern.get('confidence', 0.0)
                        }
        
        # Normalize alignment score
        if hierarchy_timeframes:
            alignment_score = max(0.0, min(1.0, alignment_score / len(hierarchy_timeframes)))
        
        return alignment_score, confirmation_count, alignment_details
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 3600)
    
    def detect_market_regime(self, ohlcv_data: Dict[str, np.ndarray]) -> str:
        """Detect current market regime based on price action"""
        if not self.config.enable_regime_scoring:
            return 'neutral'
        
        try:
            closes = ohlcv_data.get('close', [])
            if len(closes) < 50:
                return 'neutral'
            
            # Calculate trend indicators
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:])
            current_price = closes[-1]
            
            # Calculate volatility
            returns = np.diff(closes[-20:]) / closes[-20:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate momentum
            momentum_5 = (closes[-1] - closes[-5]) / closes[-5]
            momentum_20 = (closes[-1] - closes[-20]) / closes[-20]
            
            # Determine regime
            if volatility > 0.8:  # High volatility
                if momentum_5 < -0.1 and momentum_20 < -0.2:
                    return 'crash'
                elif momentum_5 > 0.1 and momentum_20 > 0.2:
                    return 'bull'
                else:
                    return 'sideways'
            else:  # Normal volatility
                if current_price > sma_20 > sma_50 and momentum_20 > 0.05:
                    return 'bull'
                elif current_price < sma_20 < sma_50 and momentum_20 < -0.05:
                    return 'bear'
                else:
                    return 'sideways'
                    
        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return 'neutral'
    
    def apply_regime_adjustment(self, confidence: float, regime: str) -> float:
        """Apply regime-specific adjustment to confidence"""
        if not self.config.enable_regime_scoring:
            return confidence
        
        if self.config.regime_weights is None:
            return confidence
        
        regime_weight = self.config.regime_weights.get(regime, 1.0)
        adjusted_confidence = confidence * regime_weight
        
        return max(0.0, min(1.0, adjusted_confidence))
