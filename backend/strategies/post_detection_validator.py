#!/usr/bin/env python3
"""
Phase 3: Post-Detection Validator
Implements follow-through confirmation and volume analysis to validate
patterns after detection and separate raw detections from trade-worthy signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result from post-detection validation"""
    pattern_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    original_confidence: float
    validated_confidence: float
    validation_passed: bool
    validation_reasons: List[str]
    follow_through_score: float
    volume_confirmation_score: float
    momentum_score: float
    overall_validation_score: float
    validation_details: Dict

@dataclass
class ValidationConfig:
    """Configuration for post-detection validation parameters"""
    # Follow-through validation
    follow_through_periods: int = 3  # Number of candles to check for follow-through
    min_follow_through_percentage: float = 0.5  # Minimum follow-through required
    max_follow_through_percentage: float = 5.0  # Maximum follow-through before reversal
    
    # Volume confirmation
    volume_confirmation_periods: int = 2  # Periods to check volume confirmation
    min_volume_expansion: float = 1.3  # Minimum volume expansion required
    max_volume_expansion: float = 8.0  # Maximum volume to avoid manipulation
    
    # Momentum validation
    momentum_periods: int = 5  # Periods to check momentum
    min_momentum_threshold: float = 0.2  # Minimum momentum required
    max_momentum_threshold: float = 3.0  # Maximum momentum before overextension
    
    # Pattern-specific adjustments
    pattern_follow_through_multipliers: Dict[str, float] = None
    pattern_volume_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pattern_follow_through_multipliers is None:
            self.pattern_follow_through_multipliers = {
                'doji': 0.7,      # Doji needs less follow-through
                'hammer': 1.2,    # Hammer needs strong follow-through
                'engulfing': 1.5, # Engulfing needs significant follow-through
                'shooting_star': 1.3,
                'morning_star': 1.4,
                'evening_star': 1.4,
                'three_white_soldiers': 1.6,
                'three_black_crows': 1.6,
                'hanging_man': 1.1,
                'inverted_hammer': 1.2,
                'spinning_top': 0.8,
                'marubozu': 1.8,
                'tristar': 0.6,
                'breakaway': 2.0,
                'dark_cloud_cover': 1.7,
                'dragonfly_doji': 0.7,
                'gravestone_doji': 0.7,
                'harami': 1.0,
                'harami_cross': 0.9,
                'high_wave': 1.0,
                'identical_three_crows': 1.5,
                'kicking': 1.8,
                'ladder_bottom': 1.3,
                'long_legged_doji': 0.8,
                'long_line': 1.4,
                'on_neck': 1.2,
                'piercing': 1.6,
                'rising_three_methods': 1.3,
                'separating_lines': 1.5
            }
        
        if self.pattern_volume_multipliers is None:
            self.pattern_volume_multipliers = {
                'doji': 0.8,      # Doji can have lower volume
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
                'tristar': 0.7,
                'breakaway': 2.0,
                'dark_cloud_cover': 1.6,
                'dragonfly_doji': 0.8,
                'gravestone_doji': 0.8,
                'harami': 1.1,
                'harami_cross': 1.0,
                'high_wave': 1.0,
                'identical_three_crows': 1.4,
                'kicking': 1.8,
                'ladder_bottom': 1.2,
                'long_legged_doji': 0.9,
                'long_line': 1.3,
                'on_neck': 1.1,
                'piercing': 1.5,
                'rising_three_methods': 1.2,
                'separating_lines': 1.4
            }

class FollowThroughAnalyzer:
    """Analyzes pattern follow-through after detection"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def analyze_follow_through(self, pattern: Dict, ohlcv_data: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
        """Analyze follow-through after pattern detection"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        pattern_direction = pattern.get('direction', 'neutral')
        
        # Get pattern-specific multiplier
        follow_through_multiplier = self.config.pattern_follow_through_multipliers.get(pattern_type, 1.0)
        
        # Calculate required follow-through
        required_follow_through = self.config.min_follow_through_percentage * follow_through_multiplier
        
        # Get data after pattern
        start_idx = pattern_index + 1
        end_idx = min(start_idx + self.config.follow_through_periods, len(ohlcv_data['close']))
        
        if start_idx >= len(ohlcv_data['close']):
            return 0.0, ["No follow-through data available"]
        
        # Extract follow-through data
        follow_opens = ohlcv_data['open'][start_idx:end_idx]
        follow_closes = ohlcv_data['close'][start_idx:end_idx]
        follow_highs = ohlcv_data['high'][start_idx:end_idx]
        follow_lows = ohlcv_data['low'][start_idx:end_idx]
        
        # Calculate follow-through metrics
        follow_through_score = 0.0
        validation_reasons = []
        
        if pattern_direction == 'bullish':
            # For bullish patterns, check upward movement
            price_changes = ((follow_closes - follow_opens) / follow_opens) * 100
            positive_moves = np.sum(price_changes > 0)
            avg_positive_move = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
            
            # Check if we have enough positive moves
            if positive_moves >= len(price_changes) * 0.6:  # At least 60% positive moves
                follow_through_score = min(1.0, avg_positive_move / required_follow_through)
                validation_reasons.append(f"Bullish follow-through: {positive_moves}/{len(price_changes)} positive moves")
            else:
                validation_reasons.append(f"Insufficient bullish follow-through: {positive_moves}/{len(price_changes)} positive moves")
        
        elif pattern_direction == 'bearish':
            # For bearish patterns, check downward movement
            price_changes = ((follow_opens - follow_closes) / follow_opens) * 100
            negative_moves = np.sum(price_changes > 0)
            avg_negative_move = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
            
            # Check if we have enough negative moves
            if negative_moves >= len(price_changes) * 0.6:  # At least 60% negative moves
                follow_through_score = min(1.0, avg_negative_move / required_follow_through)
                validation_reasons.append(f"Bearish follow-through: {negative_moves}/{len(price_changes)} negative moves")
            else:
                validation_reasons.append(f"Insufficient bearish follow-through: {negative_moves}/{len(price_changes)} negative moves")
        
        else:
            # For neutral patterns, check for any significant movement
            price_changes = np.abs((follow_closes - follow_opens) / follow_opens) * 100
            avg_move = np.mean(price_changes)
            follow_through_score = min(1.0, avg_move / required_follow_through)
            validation_reasons.append(f"Neutral follow-through: {avg_move:.2f}% average move")
        
        return follow_through_score, validation_reasons

class VolumeConfirmationAnalyzer:
    """Analyzes volume confirmation after pattern detection"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def analyze_volume_confirmation(self, pattern: Dict, ohlcv_data: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
        """Analyze volume confirmation after pattern detection"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        
        # Get pattern-specific multiplier
        volume_multiplier = self.config.pattern_volume_multipliers.get(pattern_type, 1.0)
        
        # Calculate required volume expansion
        required_volume_expansion = self.config.min_volume_expansion * volume_multiplier
        
        # Get data after pattern
        start_idx = pattern_index + 1
        end_idx = min(start_idx + self.config.volume_confirmation_periods, len(ohlcv_data['volume']))
        
        if start_idx >= len(ohlcv_data['volume']):
            return 0.0, ["No volume confirmation data available"]
        
        # Get volume data
        pattern_volume = ohlcv_data['volume'][pattern_index]
        follow_volumes = ohlcv_data['volume'][start_idx:end_idx]
        
        # Calculate average volume before pattern (for comparison)
        pre_start = max(0, pattern_index - 10)
        pre_volumes = ohlcv_data['volume'][pre_start:pattern_index]
        avg_pre_volume = np.mean(pre_volumes) if len(pre_volumes) > 0 else pattern_volume
        
        # Calculate volume metrics
        avg_follow_volume = np.mean(follow_volumes)
        volume_expansion = avg_follow_volume / avg_pre_volume
        
        # Calculate volume confirmation score
        volume_confirmation_score = 0.0
        validation_reasons = []
        
        if volume_expansion >= required_volume_expansion:
            # Good volume expansion
            volume_confirmation_score = min(1.0, volume_expansion / (required_volume_expansion * 2))
            validation_reasons.append(f"Good volume expansion: {volume_expansion:.2f}x")
        elif volume_expansion >= 1.0:
            # Moderate volume expansion
            volume_confirmation_score = 0.5
            validation_reasons.append(f"Moderate volume expansion: {volume_expansion:.2f}x")
        else:
            # Poor volume expansion
            validation_reasons.append(f"Poor volume expansion: {volume_expansion:.2f}x < {required_volume_expansion:.2f}x")
        
        # Check for excessive volume (manipulation)
        if volume_expansion > self.config.max_volume_expansion:
            volume_confirmation_score *= 0.5
            validation_reasons.append(f"Excessive volume: {volume_expansion:.2f}x > {self.config.max_volume_expansion:.2f}x")
        
        return volume_confirmation_score, validation_reasons

class MomentumAnalyzer:
    """Analyzes momentum after pattern detection"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def analyze_momentum(self, pattern: Dict, ohlcv_data: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
        """Analyze momentum after pattern detection"""
        
        pattern_type = pattern.get('pattern_type', 'unknown').lower()
        pattern_index = pattern.get('index', 0)
        pattern_direction = pattern.get('direction', 'neutral')
        
        # Get data after pattern
        start_idx = pattern_index + 1
        end_idx = min(start_idx + self.config.momentum_periods, len(ohlcv_data['close']))
        
        if start_idx >= len(ohlcv_data['close']):
            return 0.0, ["No momentum data available"]
        
        # Calculate momentum indicators
        closes = ohlcv_data['close'][start_idx:end_idx]
        volumes = ohlcv_data['volume'][start_idx:end_idx]
        
        # Price momentum (rate of change)
        if len(closes) > 1:
            price_momentum = ((closes[-1] - closes[0]) / closes[0]) * 100
        else:
            price_momentum = 0.0
        
        # Volume momentum
        if len(volumes) > 1:
            volume_momentum = ((volumes[-1] - volumes[0]) / volumes[0]) * 100
        else:
            volume_momentum = 0.0
        
        # Calculate momentum score
        momentum_score = 0.0
        validation_reasons = []
        
        # Check if momentum aligns with pattern direction
        if pattern_direction == 'bullish' and price_momentum > 0:
            momentum_score = min(1.0, abs(price_momentum) / self.config.min_momentum_threshold)
            validation_reasons.append(f"Bullish momentum: {price_momentum:.2f}%")
        elif pattern_direction == 'bearish' and price_momentum < 0:
            momentum_score = min(1.0, abs(price_momentum) / self.config.min_momentum_threshold)
            validation_reasons.append(f"Bearish momentum: {price_momentum:.2f}%")
        else:
            # Check for any significant momentum
            if abs(price_momentum) >= self.config.min_momentum_threshold:
                momentum_score = min(1.0, abs(price_momentum) / self.config.max_momentum_threshold)
                validation_reasons.append(f"Neutral momentum: {price_momentum:.2f}%")
            else:
                validation_reasons.append(f"Weak momentum: {price_momentum:.2f}% < {self.config.min_momentum_threshold:.2f}%")
        
        # Check for overextension
        if abs(price_momentum) > self.config.max_momentum_threshold:
            momentum_score *= 0.7
            validation_reasons.append(f"Momentum overextension: {price_momentum:.2f}% > {self.config.max_momentum_threshold:.2f}%")
        
        return momentum_score, validation_reasons

class PostDetectionValidator:
    """
    Implements comprehensive post-detection validation for candlestick patterns
    to separate raw detections from trade-worthy signals.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.follow_through_analyzer = FollowThroughAnalyzer(self.config)
        self.volume_analyzer = VolumeConfirmationAnalyzer(self.config)
        self.momentum_analyzer = MomentumAnalyzer(self.config)
        
        logger.info("Post-detection validator initialized with config: %s", 
                   {k: v for k, v in self.config.__dict__.items() 
                    if not k.startswith('pattern_')})
    
    def validate_patterns(self, patterns: List[Dict], 
                         ohlcv_data: Dict[str, np.ndarray]) -> List[ValidationResult]:
        """
        Validate patterns through post-detection analysis
        
        Args:
            patterns: List of detected patterns
            ohlcv_data: OHLCV data arrays
            
        Returns:
            List of validation results
        """
        if not patterns:
            return []
        
        validation_results = []
        
        for pattern in patterns:
            result = self._validate_single_pattern(pattern, ohlcv_data)
            validation_results.append(result)
        
        return validation_results
    
    def _validate_single_pattern(self, pattern: Dict, 
                                ohlcv_data: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate a single pattern"""
        
        pattern_id = pattern.get('pattern_id', '')
        symbol = pattern.get('symbol', '')
        timeframe = pattern.get('timeframe', '')
        timestamp = pattern.get('timestamp', datetime.now(timezone.utc))
        original_confidence = pattern.get('confidence', 0.0)
        
        # Perform validation analyses
        follow_through_score, follow_through_reasons = self.follow_through_analyzer.analyze_follow_through(
            pattern, ohlcv_data
        )
        
        volume_confirmation_score, volume_reasons = self.volume_analyzer.analyze_volume_confirmation(
            pattern, ohlcv_data
        )
        
        momentum_score, momentum_reasons = self.momentum_analyzer.analyze_momentum(
            pattern, ohlcv_data
        )
        
        # Combine all validation reasons
        all_reasons = follow_through_reasons + volume_reasons + momentum_reasons
        
        # Calculate overall validation score
        overall_validation_score = (
            follow_through_score * 0.4 +
            volume_confirmation_score * 0.3 +
            momentum_score * 0.3
        )
        
        # Determine if validation passed
        validation_passed = overall_validation_score >= 0.6
        
        # Adjust confidence based on validation
        validated_confidence = original_confidence
        if validation_passed:
            # Boost confidence for validated patterns
            confidence_boost = min(0.3, overall_validation_score * 0.2)
            validated_confidence = min(1.0, original_confidence + confidence_boost)
        else:
            # Reduce confidence for failed validation
            validated_confidence = original_confidence * 0.6
        
        # Create validation details
        validation_details = {
            'follow_through_analysis': {
                'score': follow_through_score,
                'reasons': follow_through_reasons
            },
            'volume_confirmation_analysis': {
                'score': volume_confirmation_score,
                'reasons': volume_reasons
            },
            'momentum_analysis': {
                'score': momentum_score,
                'reasons': momentum_reasons
            }
        }
        
        return ValidationResult(
            pattern_id=pattern_id,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            original_confidence=original_confidence,
            validated_confidence=validated_confidence,
            validation_passed=validation_passed,
            validation_reasons=all_reasons,
            follow_through_score=follow_through_score,
            volume_confirmation_score=volume_confirmation_score,
            momentum_score=momentum_score,
            overall_validation_score=overall_validation_score,
            validation_details=validation_details
        )
    
    def get_validation_stats(self, results: List[ValidationResult]) -> Dict:
        """Get statistics about validation results"""
        if not results:
            return {}
        
        total_patterns = len(results)
        passed_patterns = sum(1 for r in results if r.validation_passed)
        failed_patterns = total_patterns - passed_patterns
        
        avg_follow_through = np.mean([r.follow_through_score for r in results])
        avg_volume_confirmation = np.mean([r.volume_confirmation_score for r in results])
        avg_momentum = np.mean([r.momentum_score for r in results])
        avg_validation_score = np.mean([r.overall_validation_score for r in results])
        
        avg_confidence_before = np.mean([r.original_confidence for r in results])
        avg_confidence_after = np.mean([r.validated_confidence for r in results])
        
        return {
            'total_patterns': total_patterns,
            'passed_patterns': passed_patterns,
            'failed_patterns': failed_patterns,
            'pass_rate': passed_patterns / total_patterns if total_patterns > 0 else 0.0,
            'avg_follow_through_score': avg_follow_through,
            'avg_volume_confirmation_score': avg_volume_confirmation,
            'avg_momentum_score': avg_momentum,
            'avg_validation_score': avg_validation_score,
            'avg_confidence_before': avg_confidence_before,
            'avg_confidence_after': avg_confidence_after,
            'confidence_improvement': avg_confidence_after - avg_confidence_before
        }
