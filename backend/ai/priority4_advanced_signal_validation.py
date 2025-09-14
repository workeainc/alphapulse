#!/usr/bin/env python3
"""Priority 4: Advanced Signal Validation System for AlphaPulse"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class SignalQualityLevel(Enum):
    """Signal quality classification levels"""
    EXCELLENT = "excellent"
    HIGH = "high"
    GOOD = "good"
    MEDIUM = "medium"
    LOW = "low"
    POOR = "poor"
    REJECT = "reject"

class ValidationResult(Enum):
    """Signal validation results"""
    APPROVED = "approved"
    CONDITIONAL_APPROVAL = "conditional_approval"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

@dataclass
class SignalMetrics:
    """Signal quality metrics"""
    confidence_score: float
    volatility_score: float
    trend_strength: float
    volume_confirmation: float
    market_regime_score: float
    overall_quality: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationMetrics:
    """Validation performance metrics"""
    total_signals: int = 0
    approved_signals: int = 0
    rejected_signals: int = 0
    false_positives: int = 0
    true_positives: int = 0
    accuracy_rate: float = 0.0
    precision_rate: float = 0.0
    recall_rate: float = 0.0
    f1_score: float = 0.0

class Priority4AdvancedSignalValidation:
    """Priority 4 Advanced Signal Validation System"""
    
    def __init__(self, enable_adaptive_thresholds=True, db_connection=None):
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.db_connection = db_connection
        self.validation_metrics = ValidationMetrics()
        self.false_positive_log = []
        self.signal_quality_history = []
        self.adaptive_thresholds = {
            'min_quality': 0.6,
            'min_confidence': 0.5,
            'min_volatility': 0.3,
            'min_trend_strength': 0.4,
            'min_volume_confirmation': 0.5,
            'min_market_regime_score': 0.4
        }
        self.market_regime_weights = {
            'bull': 1.2,
            'bear': 0.8,
            'sideways': 1.0,
            'volatile': 1.1
        }
        logger.info("Priority 4 Advanced Signal Validation System initialized")
    
    async def validate_signal(self, signal_data: Dict[str, Any], market_data: pd.DataFrame) -> Tuple[ValidationResult, SignalMetrics]:
        """Validate a trading signal using advanced criteria"""
        try:
            # Extract signal quality metrics
            quality_metrics = await self._calculate_signal_quality(signal_data, market_data)
            
            # Determine validation result
            validation_result = self._determine_validation_result(quality_metrics)
            
            # Update metrics
            self._update_validation_metrics(validation_result, quality_metrics)
            
            # Log false positives for analysis
            if validation_result == ValidationResult.REJECTED and quality_metrics.overall_quality > 0.5:
                await self._log_false_positive(signal_data, quality_metrics)
            
            return validation_result, quality_metrics
            
        except ValueError as e:
            logger.error(f"Signal validation data error: {e}", exc_info=True)
            return ValidationResult.NEEDS_REVIEW, None
        except ImportError as e:
            logger.error(f"Signal validation module import error: {e}", exc_info=True)
            return ValidationResult.NEEDS_REVIEW, None
        except Exception as e:
            logger.error(f"Signal validation failed: {e}", exc_info=True)
            return ValidationResult.NEEDS_REVIEW, None
    
    async def _calculate_signal_quality(self, signal_data: Dict[str, Any], market_data: pd.DataFrame) -> SignalMetrics:
        """Calculate comprehensive signal quality metrics"""
        try:
            # Confidence score based on signal strength and historical accuracy
            confidence_score = self._calculate_confidence_score(signal_data)
            
            # Volatility score based on market conditions
            volatility_score = self._calculate_volatility_score(market_data)
            
            # Trend strength score
            trend_strength = self._calculate_trend_strength(market_data)
            
            # Volume confirmation score
            volume_confirmation = self._calculate_volume_confirmation(signal_data, market_data)
            
            # Market regime score
            market_regime_score = self._calculate_market_regime_score(market_data)
            
            # Calculate overall quality with weighted components
            overall_quality = self._calculate_overall_quality(
                confidence_score, volatility_score, trend_strength, 
                volume_confirmation, market_regime_score
            )
            
            return SignalMetrics(
                confidence_score=confidence_score,
                volatility_score=volatility_score,
                trend_strength=trend_strength,
                volume_confirmation=volume_confirmation,
                market_regime_score=market_regime_score,
                overall_quality=overall_quality
            )
            
        except ValueError as e:
            logger.error(f"Signal quality calculation data error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Signal quality calculation failed: {e}", exc_info=True)
            raise
    
    def _calculate_confidence_score(self, signal_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on signal characteristics"""
        try:
            # Base confidence from signal strength
            base_confidence = signal_data.get('signal_strength', 0.5)
            
            # Historical accuracy bonus
            historical_accuracy = signal_data.get('historical_accuracy', 0.5)
            
            # Pattern recognition bonus
            pattern_confidence = signal_data.get('pattern_confidence', 0.5)
            
            # Combine scores with weights
            confidence_score = (
                base_confidence * 0.4 +
                historical_accuracy * 0.4 +
                pattern_confidence * 0.2
            )
            
            return min(max(confidence_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.5
    
    def _calculate_volatility_score(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility score based on market data"""
        try:
            if market_data.empty or len(market_data) < 20:
                return 0.5
            
            # Calculate rolling volatility
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 20:
                return 0.5
            
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Normalize volatility score (0.1 to 0.5 is optimal for trading)
            if volatility < 0.1:
                volatility_score = 0.3  # Too low volatility
            elif volatility > 0.5:
                volatility_score = 0.2  # Too high volatility
            else:
                volatility_score = 1.0 - (volatility - 0.1) / 0.4  # Optimal range
            
            return max(volatility_score, 0.0)
            
        except Exception as e:
            logger.error(f"Volatility score calculation failed: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength score"""
        try:
            if market_data.empty or len(market_data) < 50:
                return 0.5
            
            # Calculate multiple trend indicators
            close_prices = market_data['close'].values
            
            # Linear regression trend
            x = np.arange(len(close_prices))
            slope, _ = np.polyfit(x, close_prices, 1)
            trend_direction = np.sign(slope)
            
            # Moving average trend
            ma_short = market_data['close'].rolling(window=10).mean()
            ma_long = market_data['close'].rolling(window=50).mean()
            ma_trend = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else 0.0
            
            # Price momentum
            momentum = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if len(close_prices) >= 20 else 0
            
            # Combine trend indicators
            trend_strength = (
                abs(slope) * 0.4 +
                ma_trend * 0.3 +
                abs(momentum) * 0.3
            )
            
            return min(max(trend_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.5
    
    def _calculate_volume_confirmation(self, signal_data: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        try:
            if market_data.empty or len(market_data) < 20:
                return 0.5
            
            # Volume trend analysis
            volume_data = market_data.get('volume', pd.Series([1.0] * len(market_data)))
            avg_volume = volume_data.rolling(window=20).mean()
            current_volume = volume_data.iloc[-1]
            
            # Volume confirmation score
            if current_volume > avg_volume.iloc[-1] * 1.5:
                volume_score = 1.0  # High volume confirmation
            elif current_volume > avg_volume.iloc[-1] * 1.2:
                volume_score = 0.8  # Good volume confirmation
            elif current_volume > avg_volume.iloc[-1]:
                volume_score = 0.6  # Moderate volume confirmation
            else:
                volume_score = 0.3  # Low volume confirmation
            
            # Signal-specific volume requirements
            signal_type = signal_data.get('signal_type', 'unknown')
            if signal_type in ['breakout', 'reversal']:
                # These signals need higher volume confirmation
                volume_score *= 1.2
            
            return min(max(volume_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Volume confirmation calculation failed: {e}")
            return 0.5
    
    def _calculate_market_regime_score(self, market_data: pd.DataFrame) -> float:
        """Calculate market regime score"""
        try:
            if market_data.empty or len(market_data) < 50:
                return 0.5
            
            # Determine market regime
            regime = self._classify_market_regime(market_data)
            
            # Get regime weight
            regime_weight = self.market_regime_weights.get(regime, 1.0)
            
            # Base regime score
            if regime == 'bull':
                regime_score = 0.8
            elif regime == 'bear':
                regime_score = 0.6
            elif regime == 'sideways':
                regime_score = 0.7
            elif regime == 'volatile':
                regime_score = 0.5
            else:
                regime_score = 0.5
            
            # Apply regime weight
            final_score = regime_score * regime_weight
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Market regime score calculation failed: {e}")
            return 0.5
    
    def _classify_market_regime(self, market_data: pd.DataFrame) -> str:
        """Classify current market regime"""
        try:
            if market_data.empty or len(market_data) < 50:
                return 'unknown'
            
            close_prices = market_data['close'].values
            
            # Calculate trend indicators
            ma_short = market_data['close'].rolling(window=20).mean()
            ma_long = market_data['close'].rolling(window=50).mean()
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0.1
            
            # Determine regime
            if volatility > 0.4:
                return 'volatile'
            elif ma_short.iloc[-1] > ma_long.iloc[-1] * 1.05:
                return 'bull'
            elif ma_short.iloc[-1] < ma_long.iloc[-1] * 0.95:
                return 'bear'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Market regime classification failed: {e}")
            return 'unknown'
    
    def _calculate_overall_quality(self, confidence: float, volatility: float, 
                                 trend: float, volume: float, regime: float) -> float:
        """Calculate overall signal quality score"""
        try:
            # Weighted combination of all metrics
            overall_quality = (
                confidence * 0.3 +
                volatility * 0.2 +
                trend * 0.2 +
                volume * 0.2 +
                regime * 0.1
            )
            
            return min(max(overall_quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Overall quality calculation failed: {e}")
            return 0.5
    
    def _determine_validation_result(self, quality_metrics: SignalMetrics) -> ValidationResult:
        """Determine validation result based on quality metrics"""
        try:
            overall_quality = quality_metrics.overall_quality
            
            # Get current thresholds
            thresholds = self._get_current_thresholds()
            
            # Determine result based on quality and individual metrics
            if overall_quality >= thresholds['min_quality']:
                # Check if most critical metrics pass
                critical_metrics_passed = sum([
                    quality_metrics.confidence_score >= thresholds['min_confidence'],
                    quality_metrics.volatility_score >= thresholds['min_volatility'],
                    quality_metrics.trend_strength >= thresholds['min_trend_strength']
                ])
                
                if critical_metrics_passed >= 2:  # At least 2 out of 3 critical metrics
                    return ValidationResult.APPROVED
                else:
                    return ValidationResult.CONDITIONAL_APPROVAL
            elif overall_quality >= thresholds['min_quality'] * 0.7:  # Lower threshold for review
                return ValidationResult.NEEDS_REVIEW
            else:
                return ValidationResult.REJECTED
                
        except Exception as e:
            logger.error(f"Validation result determination failed: {e}")
            return ValidationResult.NEEDS_REVIEW
    
    def _get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        if not self.enable_adaptive_thresholds:
            return {
                'min_quality': 0.6,
                'min_confidence': 0.5,
                'min_volatility': 0.3,
                'min_trend_strength': 0.4,
                'min_volume_confirmation': 0.5,
                'min_market_regime_score': 0.4
            }
        
        # Adaptive thresholds based on recent performance
        return self.adaptive_thresholds
    
    def _update_validation_metrics(self, result: ValidationResult, metrics: SignalMetrics):
        """Update validation performance metrics"""
        try:
            self.validation_metrics.total_signals += 1
            
            if result == ValidationResult.APPROVED:
                self.validation_metrics.approved_signals += 1
            elif result == ValidationResult.REJECTED:
                self.validation_metrics.rejected_signals += 1
            
            # Store quality metrics for analysis
            self.signal_quality_history.append(metrics)
            
            # Keep only recent history
            if len(self.signal_quality_history) > 1000:
                self.signal_quality_history = self.signal_quality_history[-1000:]
            
            # Update accuracy metrics
            self._recalculate_accuracy_metrics()
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    def _recalculate_accuracy_metrics(self):
        """Recalculate accuracy metrics"""
        try:
            total = self.validation_metrics.total_signals
            if total == 0:
                return
            
            approved = self.validation_metrics.approved_signals
            rejected = self.validation_metrics.rejected_signals
            
            # Basic accuracy
            self.validation_metrics.accuracy_rate = approved / total if total > 0 else 0.0
            
            # Update adaptive thresholds based on performance
            if self.enable_adaptive_thresholds:
                self._adjust_adaptive_thresholds()
                
        except Exception as e:
            logger.error(f"Accuracy metrics recalculation failed: {e}")
    
    def _adjust_adaptive_thresholds(self):
        """Adjust adaptive thresholds based on performance"""
        try:
            if len(self.signal_quality_history) < 50:
                return
            
            recent_quality = [m.overall_quality for m in self.signal_quality_history[-50:]]
            avg_quality = np.mean(recent_quality)
            
            # Adjust thresholds based on recent performance
            if avg_quality > 0.8:
                # High performance - tighten thresholds
                adjustment = 0.05
            elif avg_quality < 0.6:
                # Low performance - loosen thresholds
                adjustment = -0.05
            else:
                # Stable performance - minor adjustment
                adjustment = 0.01
            
            # Apply adjustments
            for key in self.adaptive_thresholds:
                self.adaptive_thresholds[key] = max(0.1, min(0.9, 
                    self.adaptive_thresholds[key] + adjustment))
                    
        except Exception as e:
            logger.error(f"Adaptive threshold adjustment failed: {e}")
    
    async def _log_false_positive(self, signal_data: Dict[str, Any], metrics: SignalMetrics):
        """Log potential false positive for analysis"""
        try:
            false_positive_entry = {
                'timestamp': datetime.now(),
                'signal_data': signal_data,
                'quality_metrics': {
                    'confidence_score': metrics.confidence_score,
                    'volatility_score': metrics.volatility_score,
                    'trend_strength': metrics.trend_strength,
                    'volume_confirmation': metrics.volume_confirmation,
                    'market_regime_score': metrics.market_regime_score,
                    'overall_quality': metrics.overall_quality
                },
                'market_conditions': {
                    'regime': self._classify_market_regime(pd.DataFrame()),  # Placeholder
                    'thresholds': self._get_current_thresholds()
                }
            }
            
            self.false_positive_log.append(false_positive_entry)
            
            # Keep only recent false positives
            if len(self.false_positive_log) > 500:
                self.false_positive_log = self.false_positive_log[-500:]
            
            # Log to database if available
            if self.db_connection:
                await self._save_false_positive_to_db(false_positive_entry)
                
        except Exception as e:
            logger.error(f"False positive logging failed: {e}")
    
    async def _save_false_positive_to_db(self, false_positive_entry: Dict[str, Any]):
        """Save false positive to database"""
        try:
            # This would be implemented based on your database schema
            # For now, just log the attempt
            logger.info(f"False positive logged to database: {false_positive_entry['timestamp']}")
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
    
    def get_validation_performance(self) -> Dict[str, Any]:
        """Get current validation performance metrics"""
        try:
            return {
                'total_signals': self.validation_metrics.total_signals,
                'approved_signals': self.validation_metrics.approved_signals,
                'rejected_signals': self.validation_metrics.rejected_signals,
                'accuracy_rate': self.validation_metrics.accuracy_rate,
                'current_thresholds': self._get_current_thresholds(),
                'recent_quality_trend': self._get_recent_quality_trend(),
                'false_positive_count': len(self.false_positive_log)
            }
        except Exception as e:
            logger.error(f"Performance metrics retrieval failed: {e}")
            return {}
    
    def _get_recent_quality_trend(self) -> List[float]:
        """Get recent quality trend for analysis"""
        try:
            if len(self.signal_quality_history) < 10:
                return []
            
            recent_metrics = self.signal_quality_history[-10:]
            return [m.overall_quality for m in recent_metrics]
            
        except Exception as e:
            logger.error(f"Quality trend calculation failed: {e}")
            return []
    
    async def batch_validate_signals(self, signals: List[Dict[str, Any]], 
                                   market_data: pd.DataFrame) -> List[Tuple[ValidationResult, SignalMetrics]]:
        """Validate multiple signals in batch"""
        try:
            results = []
            for signal in signals:
                result = await self.validate_signal(signal, market_data)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch validation failed: {e}")
            return []
    
    def export_validation_data(self) -> Dict[str, Any]:
        """Export validation data for analysis"""
        try:
            return {
                'validation_metrics': {
                    'total_signals': self.validation_metrics.total_signals,
                    'approved_signals': self.validation_metrics.approved_signals,
                    'rejected_signals': self.validation_metrics.rejected_signals,
                    'accuracy_rate': self.validation_metrics.accuracy_rate
                },
                'adaptive_thresholds': self.adaptive_thresholds,
                'market_regime_weights': self.market_regime_weights,
                'recent_quality_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'overall_quality': m.overall_quality,
                        'confidence_score': m.confidence_score,
                        'volatility_score': m.volatility_score,
                        'trend_strength': m.trend_strength,
                        'volume_confirmation': m.volume_confirmation,
                        'market_regime_score': m.market_regime_score
                    }
                    for m in self.signal_quality_history[-100:]  # Last 100 entries
                ],
                'false_positive_summary': {
                    'total_count': len(self.false_positive_log),
                    'recent_entries': len([fp for fp in self.false_positive_log 
                                        if (datetime.now() - fp['timestamp']).days <= 7])
                }
            }
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return {}
