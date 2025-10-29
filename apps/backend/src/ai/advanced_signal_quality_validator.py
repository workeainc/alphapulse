"""
Advanced Signal Quality & Validation System (Phase 9)
Comprehensive signal validation with quality scoring, market regime filtering, and adaptive thresholds
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
from collections import deque

logger = logging.getLogger(__name__)

class SignalQualityLevel(Enum):
    """Signal quality classification levels"""
    EXCELLENT = "excellent"  # 90-100%
    HIGH = "high"           # 80-89%
    GOOD = "good"           # 70-79%
    MEDIUM = "medium"       # 60-69%
    LOW = "low"             # 50-59%
    POOR = "poor"           # 40-49%
    REJECT = "reject"       # <40%

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class SignalQualityMetrics:
    """Comprehensive signal quality metrics"""
    confidence_score: float = 0.0
    volatility_score: float = 0.0
    trend_strength_score: float = 0.0
    volume_confirmation_score: float = 0.0
    market_regime_score: float = 0.0
    overall_quality_score: float = 0.0
    quality_level: SignalQualityLevel = SignalQualityLevel.REJECT
    validation_passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    market_regime: str = "unknown"
    
    @property
    def quality_score(self) -> float:
        """Alias for overall_quality_score for backward compatibility"""
        return self.overall_quality_score
    
    @property
    def is_valid(self) -> bool:
        """Alias for validation_passed for backward compatibility"""
        return self.validation_passed

@dataclass
class ValidationThresholds:
    """Dynamic validation thresholds"""
    min_confidence: float = 0.85
    min_quality_score: float = 0.70
    min_volume_confirmation: float = 0.60
    min_trend_strength: float = 0.65
    max_volatility: float = 0.80
    regime_weights: Dict[str, float] = field(default_factory=lambda: {
        'bull': 1.0,
        'bear': 0.9,
        'sideways': 0.8,
        'volatile': 0.7
    })

@dataclass
class FalsePositiveAnalysis:
    """False positive analysis results"""
    total_signals: int = 0
    rejected_signals: int = 0
    false_positives: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    analysis_period: timedelta = field(default_factory=lambda: timedelta(days=7))

class AdvancedSignalQualityValidator:
    """
    Advanced signal quality validation system
    Provides comprehensive validation with adaptive thresholds and market regime awareness
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.thresholds = ValidationThresholds()
        self.quality_history = deque(maxlen=1000)
        self.false_positive_history = deque(maxlen=500)
        self.performance_metrics = {
            'total_validated': 0,
            'total_passed': 0,
            'total_rejected': 0,
            'avg_quality_score': 0.0,
            'avg_processing_time_ms': 0.0
        }
        
        # Market regime detection
        self.regime_detection_window = 50  # candles for regime detection
        self.regime_history = deque(maxlen=self.regime_detection_window)
        
        # Adaptive threshold management
        self.adaptation_window = 100  # signals for threshold adaptation
        self.performance_history = deque(maxlen=self.adaptation_window)
        
        logger.info("🚀 Advanced Signal Quality Validator initialized")
    
    async def validate_signal_quality(self, 
                                    market_data: Dict[str, Any],
                                    historical_data: pd.DataFrame,
                                    signal_data: Dict[str, Any] = None) -> SignalQualityMetrics:
        """
        Comprehensive signal quality validation
        
        Args:
            signal_data: Signal information (confidence, direction, etc.)
            market_data: Current market conditions
            historical_data: Historical price/volume data
            
        Returns:
            SignalQualityMetrics with validation results
        """
        start_time = datetime.now()
        
        try:
            # 1. Calculate individual quality scores
            confidence_score = self._calculate_confidence_score(signal_data)
            volatility_score = self._calculate_volatility_score(market_data, historical_data)
            trend_strength_score = self._calculate_trend_strength_score(historical_data)
            volume_confirmation_score = self._calculate_volume_confirmation_score(market_data, historical_data)
            market_regime_score = self._calculate_market_regime_score(historical_data)
            
            # 2. Calculate overall quality score
            overall_quality_score = self._calculate_overall_quality_score(
                confidence_score, volatility_score, trend_strength_score,
                volume_confirmation_score, market_regime_score
            )
            
            # 3. Determine quality level
            quality_level = self._determine_quality_level(overall_quality_score)
            
            # 4. Apply validation rules
            validation_passed, rejection_reasons = self._apply_validation_rules(
                confidence_score, volatility_score, trend_strength_score,
                volume_confirmation_score, market_regime_score, overall_quality_score
            )
            
            # 5. Detect market regime
            if not historical_data.empty:
                close_prices = historical_data['close'].values
                detected_regime = self._detect_market_regime(close_prices)
                market_regime_str = detected_regime.value
            else:
                market_regime_str = "unknown"
            
            # 6. Create quality metrics
            quality_metrics = SignalQualityMetrics(
                confidence_score=confidence_score,
                volatility_score=volatility_score,
                trend_strength_score=trend_strength_score,
                volume_confirmation_score=volume_confirmation_score,
                market_regime_score=market_regime_score,
                overall_quality_score=overall_quality_score,
                quality_level=quality_level,
                validation_passed=validation_passed,
                rejection_reasons=rejection_reasons,
                market_regime=market_regime_str
            )
            
            # 6. Update performance tracking
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(quality_metrics, processing_time)
            
            # 7. Store quality metrics
            await self._store_quality_metrics(quality_metrics, signal_data)
            
            # 8. Adaptive threshold management
            await self._update_adaptive_thresholds(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Signal quality validation failed: {e}")
            return SignalQualityMetrics(
                validation_passed=False,
                rejection_reasons=[f"Validation error: {str(e)}"],
                market_regime="unknown"
            )
    
    def _calculate_confidence_score(self, signal_data: Dict[str, Any]) -> float:
        """Calculate confidence score from signal data"""
        try:
            # Handle None signal_data
            if signal_data is None:
                return 0.5
            
            # Base confidence from signal
            base_confidence = signal_data.get('confidence', 0.0)
            
            # Additional confidence factors
            model_agreement = signal_data.get('model_agreement', 0.0)
            feature_importance = signal_data.get('feature_importance', 0.0)
            historical_accuracy = signal_data.get('historical_accuracy', 0.0)
            
            # Weighted confidence calculation
            confidence_score = (
                0.5 * base_confidence +
                0.2 * model_agreement +
                0.2 * feature_importance +
                0.1 * historical_accuracy
            )
            
            return min(max(confidence_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Confidence score calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, market_data: Dict[str, Any], historical_data: pd.DataFrame) -> float:
        """Calculate volatility score based on market conditions"""
        try:
            if historical_data.empty:
                return 0.5
            
            # Calculate ATR-based volatility
            high = historical_data['high'].values
            low = historical_data['low'].values
            close = historical_data['close'].values
            
            # True Range calculation
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # ATR (14-period)
            atr = np.mean(true_range[-14:])
            avg_price = np.mean(close[-14:])
            volatility_ratio = atr / avg_price if avg_price > 0 else 0
            
            # Volatility score (lower is better for signal quality)
            volatility_score = max(0.0, 1.0 - volatility_ratio)
            
            return min(max(volatility_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Volatility score calculation failed: {e}")
            return 0.5
    
    def _calculate_trend_strength_score(self, historical_data: pd.DataFrame) -> float:
        """Calculate trend strength score"""
        try:
            if historical_data.empty or len(historical_data) < 20:
                return 0.5
            
            close = historical_data['close'].values
            
            # Linear regression slope
            x = np.arange(len(close))
            slope, intercept = np.polyfit(x, close, 1)
            
            # R-squared calculation
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Moving average alignment
            ma_short = np.mean(close[-5:])
            ma_long = np.mean(close[-20:])
            ma_alignment = 1.0 if (ma_short > ma_long and slope > 0) or (ma_short < ma_long and slope < 0) else 0.5
            
            # Trend strength score
            trend_strength = (0.6 * abs(slope) + 0.4 * r_squared) * ma_alignment
            
            return min(max(trend_strength, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Trend strength calculation failed: {e}")
            return 0.5
    
    def _calculate_volume_confirmation_score(self, market_data: Dict[str, Any], historical_data: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        try:
            if historical_data.empty or 'volume' not in historical_data.columns:
                return 0.5
            
            volume = historical_data['volume'].values
            close = historical_data['close'].values
            
            # Volume trend analysis
            recent_volume = np.mean(volume[-5:])
            avg_volume = np.mean(volume[-20:])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price-volume correlation
            if len(close) >= 10:
                price_changes = np.diff(close[-10:])
                volume_changes = np.diff(volume[-10:])
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                correlation = 0.0 if np.isnan(correlation) else correlation
            else:
                correlation = 0.0
            
            # Volume confirmation score
            volume_score = (0.7 * min(volume_ratio, 2.0) + 0.3 * abs(correlation)) / 2.0
            
            return min(max(volume_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Volume confirmation calculation failed: {e}")
            return 0.5
    
    def _calculate_market_regime_score(self, historical_data: pd.DataFrame) -> float:
        """Calculate market regime score"""
        try:
            if historical_data.empty or len(historical_data) < 20:
                return 0.5
            
            close = historical_data['close'].values
            
            # Detect market regime
            regime = self._detect_market_regime(close)
            
            # Regime-specific scoring
            if regime == MarketRegime.BULL:
                # Bull market: prefer long signals
                return 0.9
            elif regime == MarketRegime.BEAR:
                # Bear market: prefer short signals
                return 0.8
            elif regime == MarketRegime.SIDEWAYS:
                # Sideways: neutral
                return 0.6
            else:  # VOLATILE
                # Volatile: lower score due to uncertainty
                return 0.4
                
        except Exception as e:
            logger.error(f"❌ Market regime score calculation failed: {e}")
            return 0.5
    
    def _detect_market_regime(self, prices: np.ndarray) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(prices) < 20:
                return MarketRegime.SIDEWAYS
            
            # Calculate moving averages
            ma_short = np.mean(prices[-5:])
            ma_medium = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns[-20:])
            
            # Regime detection logic
            if volatility > 0.03:  # High volatility
                return MarketRegime.VOLATILE
            elif ma_short > ma_medium > ma_long:  # Uptrend
                return MarketRegime.BULL
            elif ma_short < ma_medium < ma_long:  # Downtrend
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def _calculate_overall_quality_score(self, 
                                       confidence_score: float,
                                       volatility_score: float,
                                       trend_strength_score: float,
                                       volume_confirmation_score: float,
                                       market_regime_score: float) -> float:
        """Calculate overall quality score with weighted combination"""
        try:
            # Weighted combination
            overall_score = (
                0.30 * confidence_score +
                0.20 * volatility_score +
                0.20 * trend_strength_score +
                0.20 * volume_confirmation_score +
                0.10 * market_regime_score
            )
            
            return min(max(overall_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"❌ Overall quality score calculation failed: {e}")
            return 0.0
    
    def _determine_quality_level(self, overall_score: float) -> SignalQualityLevel:
        """Determine quality level based on overall score"""
        if overall_score >= 0.90:
            return SignalQualityLevel.EXCELLENT
        elif overall_score >= 0.80:
            return SignalQualityLevel.HIGH
        elif overall_score >= 0.70:
            return SignalQualityLevel.GOOD
        elif overall_score >= 0.60:
            return SignalQualityLevel.MEDIUM
        elif overall_score >= 0.50:
            return SignalQualityLevel.LOW
        elif overall_score >= 0.40:
            return SignalQualityLevel.POOR
        else:
            return SignalQualityLevel.REJECT
    
    def _apply_validation_rules(self,
                               confidence_score: float,
                               volatility_score: float,
                               trend_strength_score: float,
                               volume_confirmation_score: float,
                               market_regime_score: float,
                               overall_score: float) -> Tuple[bool, List[str]]:
        """Apply validation rules and return pass/fail with reasons"""
        rejection_reasons = []
        
        # Check individual thresholds
        if confidence_score < self.thresholds.min_confidence:
            rejection_reasons.append(f"Confidence too low: {confidence_score:.3f} < {self.thresholds.min_confidence}")
        
        if overall_score < self.thresholds.min_quality_score:
            rejection_reasons.append(f"Quality score too low: {overall_score:.3f} < {self.thresholds.min_quality_score}")
        
        if volume_confirmation_score < self.thresholds.min_volume_confirmation:
            rejection_reasons.append(f"Volume confirmation too low: {volume_confirmation_score:.3f} < {self.thresholds.min_volume_confirmation}")
        
        if trend_strength_score < self.thresholds.min_trend_strength:
            rejection_reasons.append(f"Trend strength too low: {trend_strength_score:.3f} < {self.thresholds.min_trend_strength}")
        
        if volatility_score < (1.0 - self.thresholds.max_volatility):
            rejection_reasons.append(f"Volatility too high: {1.0 - volatility_score:.3f} > {self.thresholds.max_volatility}")
        
        validation_passed = len(rejection_reasons) == 0
        return validation_passed, rejection_reasons
    
    def _update_performance_metrics(self, quality_metrics: SignalQualityMetrics, processing_time_ms: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_validated'] += 1
        
        if quality_metrics.validation_passed:
            self.performance_metrics['total_passed'] += 1
        else:
            self.performance_metrics['total_rejected'] += 1
        
        # Update averages
        total = self.performance_metrics['total_validated']
        current_avg = self.performance_metrics['avg_quality_score']
        self.performance_metrics['avg_quality_score'] = (current_avg * (total - 1) + quality_metrics.overall_quality_score) / total
        
        current_time_avg = self.performance_metrics['avg_processing_time_ms']
        self.performance_metrics['avg_processing_time_ms'] = (current_time_avg * (total - 1) + processing_time_ms) / total
        
        # Store in history
        self.quality_history.append(quality_metrics)
    
    async def _store_quality_metrics(self, quality_metrics: SignalQualityMetrics, signal_data: Dict[str, Any]):
        """Store quality metrics in database"""
        try:
            # Handle None signal_data
            if signal_data is None:
                signal_data = {}
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_signal_quality_metrics 
                    (symbol, timeframe, signal_id, confidence_score, volatility_score, 
                     trend_strength_score, volume_confirmation_score, market_regime_score,
                     overall_quality_score, quality_level, validation_passed, rejection_reasons,
                     created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, 
                signal_data.get('symbol', 'UNKNOWN'),
                signal_data.get('timeframe', '1h'),
                signal_data.get('signal_id', ''),
                quality_metrics.confidence_score,
                quality_metrics.volatility_score,
                quality_metrics.trend_strength_score,
                quality_metrics.volume_confirmation_score,
                quality_metrics.market_regime_score,
                quality_metrics.overall_quality_score,
                quality_metrics.quality_level.value,
                quality_metrics.validation_passed,
                quality_metrics.rejection_reasons,
                quality_metrics.timestamp
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to store quality metrics: {e}")
    
    async def _update_adaptive_thresholds(self, quality_metrics: SignalQualityMetrics):
        """Update adaptive thresholds based on recent performance"""
        try:
            self.performance_history.append(quality_metrics)
            
            if len(self.performance_history) >= self.adaptation_window:
                # Calculate recent performance
                recent_metrics = list(self.performance_history)[-self.adaptation_window:]
                pass_rate = sum(1 for m in recent_metrics if m.validation_passed) / len(recent_metrics)
                avg_quality = np.mean([m.overall_quality_score for m in recent_metrics])
                
                # Adjust thresholds based on performance
                if pass_rate < 0.3:  # Too strict
                    self.thresholds.min_confidence *= 0.95
                    self.thresholds.min_quality_score *= 0.95
                    logger.info(f"📉 Relaxing thresholds - Pass rate: {pass_rate:.2f}")
                elif pass_rate > 0.7:  # Too lenient
                    self.thresholds.min_confidence *= 1.05
                    self.thresholds.min_quality_score *= 1.05
                    logger.info(f"📈 Tightening thresholds - Pass rate: {pass_rate:.2f}")
                
                # Ensure thresholds stay within reasonable bounds
                self.thresholds.min_confidence = max(0.70, min(0.95, self.thresholds.min_confidence))
                self.thresholds.min_quality_score = max(0.60, min(0.90, self.thresholds.min_quality_score))
                
                # Clear history for next adaptation cycle
                self.performance_history.clear()
                
        except Exception as e:
            logger.error(f"❌ Adaptive threshold update failed: {e}")
    
    async def get_false_positive_analysis(self, symbol: str, timeframe: str, 
                                        days: int = 7) -> FalsePositiveAnalysis:
        """Analyze false positive signals"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent signals
                start_date = datetime.now() - timedelta(days=days)
                
                signals = await conn.fetch("""
                    SELECT validation_passed, overall_quality_score, quality_level
                    FROM sde_signal_quality_metrics
                    WHERE symbol = $1 AND timeframe = $2 AND created_at >= $3
                    ORDER BY created_at DESC
                """, symbol, timeframe, start_date)
                
                if not signals:
                    return FalsePositiveAnalysis()
                
                total_signals = len(signals)
                rejected_signals = sum(1 for s in signals if not s['validation_passed'])
                passed_signals = total_signals - rejected_signals
                
                # Calculate metrics (simplified - in real implementation you'd track actual outcomes)
                accuracy = passed_signals / total_signals if total_signals > 0 else 0.0
                precision = passed_signals / total_signals if total_signals > 0 else 0.0
                recall = passed_signals / total_signals if total_signals > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                return FalsePositiveAnalysis(
                    total_signals=total_signals,
                    rejected_signals=rejected_signals,
                    false_positives=0,  # Would need outcome tracking
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    analysis_period=timedelta(days=days)
                )
                
        except Exception as e:
            logger.error(f"❌ False positive analysis failed: {e}")
            return FalsePositiveAnalysis()
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'current_thresholds': {
                'min_confidence': self.thresholds.min_confidence,
                'min_quality_score': self.thresholds.min_quality_score,
                'min_volume_confirmation': self.thresholds.min_volume_confirmation,
                'min_trend_strength': self.thresholds.min_trend_strength,
                'max_volatility': self.thresholds.max_volatility
            },
            'quality_distribution': self._get_quality_distribution(),
            'recent_performance': self._get_recent_performance()
        }
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality levels"""
        distribution = {level.value: 0 for level in SignalQualityLevel}
        
        for metric in self.quality_history:
            distribution[metric.quality_level.value] += 1
        
        return distribution
    
    def _get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics"""
        if not self.quality_history:
            return {}
        
        recent_metrics = list(self.quality_history)[-100:]  # Last 100 signals
        
        return {
            'avg_quality_score': np.mean([m.overall_quality_score for m in recent_metrics]),
            'pass_rate': sum(1 for m in recent_metrics if m.validation_passed) / len(recent_metrics),
            'avg_confidence': np.mean([m.confidence_score for m in recent_metrics]),
            'avg_volatility_score': np.mean([m.volatility_score for m in recent_metrics])
        }
