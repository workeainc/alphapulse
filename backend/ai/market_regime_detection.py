"""
Market Regime Detection & Adaptation for AlphaPulse
Smart market regime classification and adaptive strategy selection
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .risk_management import RiskManager, risk_manager
from .position_sizing import PositionSizingOptimizer, position_sizing_optimizer

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    SIDEWAYS = "sideways"

class RegimeConfidence(Enum):
    """Confidence levels for regime detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class RegimeMetrics:
    """Metrics for market regime analysis"""
    trend_strength: float
    volatility: float
    momentum: float
    volume_trend: float
    price_range: float
    support_resistance_ratio: float
    breakout_probability: float
    consolidation_score: float

@dataclass
class MarketRegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: RegimeConfidence
    confidence_score: float
    metrics: RegimeMetrics
    duration: timedelta
    transition_probability: float
    recommended_strategy: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class RegimeTransition:
    """Market regime transition event"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    trigger_metrics: Dict[str, float]
    confidence: float

class RegimeDetectionMethod(Enum):
    """Methods for regime detection"""
    TECHNICAL_INDICATORS = "technical_indicators"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"

class StrategyAdaptation(Enum):
    """Strategy adaptation types"""
    POSITION_SIZING = "position_sizing"
    ENTRY_EXIT_RULES = "entry_exit_rules"
    RISK_PARAMETERS = "risk_parameters"
    TIME_FRAME_SELECTION = "time_frame_selection"
    FEATURE_WEIGHTING = "feature_weighting"

class MarketRegimeDetector:
    """
    Advanced market regime detection and adaptation system
    """

    def __init__(self,
                 risk_manager: RiskManager = None,
                 position_sizing_optimizer: PositionSizingOptimizer = None,
                 lookback_period: int = 50,
                 volatility_threshold: float = 0.15,  # Lower threshold for volatility detection
                 trend_threshold: float = 0.01,  # More sensitive threshold
                 volume_threshold: float = 1.5,
                 min_regime_duration: int = 10,
                 enable_ml_detection: bool = True,
                 update_interval: int = 60):  # seconds

        # Dependencies
        self.risk_manager = risk_manager or risk_manager
        self.position_sizing_optimizer = position_sizing_optimizer or position_sizing_optimizer

        # Parameters
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.volume_threshold = volume_threshold
        self.min_regime_duration = min_regime_duration
        self.enable_ml_detection = enable_ml_detection
        self.update_interval = update_interval

        # Data storage
        self.regime_history: Dict[str, List[MarketRegimeState]] = {}
        self.transition_history: Dict[str, List[RegimeTransition]] = {}
        self.performance_by_regime: Dict[MarketRegime, Dict[str, float]] = {}
        
        # Current states
        self.current_regimes: Dict[str, MarketRegimeState] = {}
        self.regime_models: Dict[str, Any] = {}
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task = None

        # Strategy adaptation rules
        self.adaptation_rules = self._initialize_adaptation_rules()

        logger.info("MarketRegimeDetector initialized")

    def _initialize_adaptation_rules(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """Initialize strategy adaptation rules for each regime"""
        return {
            MarketRegime.TRENDING_UP: {
                'position_sizing_multiplier': 1.2,
                'risk_tolerance': 'moderate',
                'timeframe_preference': 'medium',
                'entry_strategy': 'momentum_following',
                'exit_strategy': 'trailing_stop',
                'feature_weights': {'momentum': 0.4, 'trend': 0.4, 'volatility': 0.2}
            },
            MarketRegime.TRENDING_DOWN: {
                'position_sizing_multiplier': 0.8,
                'risk_tolerance': 'conservative',
                'timeframe_preference': 'medium',
                'entry_strategy': 'counter_trend',
                'exit_strategy': 'tight_stop',
                'feature_weights': {'momentum': 0.3, 'trend': 0.3, 'volatility': 0.4}
            },
            MarketRegime.RANGING: {
                'position_sizing_multiplier': 1.0,
                'risk_tolerance': 'moderate',
                'timeframe_preference': 'short',
                'entry_strategy': 'support_resistance',
                'exit_strategy': 'fixed_targets',
                'feature_weights': {'support_resistance': 0.5, 'volatility': 0.3, 'momentum': 0.2}
            },
            MarketRegime.VOLATILE: {
                'position_sizing_multiplier': 0.6,
                'risk_tolerance': 'conservative',
                'timeframe_preference': 'short',
                'entry_strategy': 'volatility_breakout',
                'exit_strategy': 'quick_exit',
                'feature_weights': {'volatility': 0.6, 'momentum': 0.3, 'trend': 0.1}
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_sizing_multiplier': 1.3,
                'risk_tolerance': 'aggressive',
                'timeframe_preference': 'long',
                'entry_strategy': 'breakout_anticipation',
                'exit_strategy': 'wide_targets',
                'feature_weights': {'trend': 0.4, 'momentum': 0.4, 'volatility': 0.2}
            },
            MarketRegime.BREAKOUT: {
                'position_sizing_multiplier': 1.5,
                'risk_tolerance': 'aggressive',
                'timeframe_preference': 'medium',
                'entry_strategy': 'breakout_following',
                'exit_strategy': 'momentum_trailing',
                'feature_weights': {'momentum': 0.5, 'trend': 0.3, 'volatility': 0.2}
            },
            MarketRegime.BREAKDOWN: {
                'position_sizing_multiplier': 0.5,
                'risk_tolerance': 'very_conservative',
                'timeframe_preference': 'short',
                'entry_strategy': 'short_only',
                'exit_strategy': 'aggressive_stop',
                'feature_weights': {'momentum': 0.4, 'trend': 0.4, 'volatility': 0.2}
            },
            MarketRegime.SIDEWAYS: {
                'position_sizing_multiplier': 0.9,
                'risk_tolerance': 'moderate',
                'timeframe_preference': 'short',
                'entry_strategy': 'mean_reversion',
                'exit_strategy': 'quick_profit',
                'feature_weights': {'mean_reversion': 0.5, 'volatility': 0.3, 'momentum': 0.2}
            }
        }

    def calculate_regime_metrics(self, prices: List[float], volumes: List[float] = None) -> RegimeMetrics:
        """Calculate comprehensive regime metrics"""
        if len(prices) < self.lookback_period:
            return None

        prices = np.array(prices[-self.lookback_period:])
        volumes = np.array(volumes[-self.lookback_period:]) if volumes else np.ones_like(prices)

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Trend strength (using linear regression)
        x = np.arange(len(prices))
        trend_slope, _ = np.polyfit(x, prices, 1)
        trend_strength = abs(trend_slope) / np.mean(prices)
        
        # Volatility
        volatility = np.std(returns)
        
        # Momentum (price change over lookback period)
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # Volume trend
        volume_trend = np.corrcoef(x, volumes)[0, 1] if len(volumes) > 1 else 0
        
        # Price range
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
        
        # Support/Resistance ratio (simplified)
        support_resistance_ratio = self._calculate_support_resistance_ratio(prices)
        
        # Breakout probability
        breakout_probability = self._calculate_breakout_probability(prices, volumes)
        
        # Consolidation score
        consolidation_score = self._calculate_consolidation_score(prices)

        return RegimeMetrics(
            trend_strength=trend_strength,
            volatility=volatility,
            momentum=momentum,
            volume_trend=volume_trend,
            price_range=price_range,
            support_resistance_ratio=support_resistance_ratio,
            breakout_probability=breakout_probability,
            consolidation_score=consolidation_score
        )

    def _calculate_support_resistance_ratio(self, prices: np.ndarray) -> float:
        """Calculate support/resistance ratio"""
        # Simplified implementation - ratio of local minima to local maxima
        from scipy.signal import argrelextrema
        
        try:
            minima = argrelextrema(prices, np.less, order=3)[0]
            maxima = argrelextrema(prices, np.greater, order=3)[0]
            
            if len(maxima) == 0:
                return 1.0
            
            return len(minima) / len(maxima)
        except:
            return 1.0

    def _calculate_breakout_probability(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate breakout probability"""
        # Simplified implementation based on price and volume patterns
        recent_prices = prices[-10:]
        recent_volumes = volumes[-10:]
        
        # Check for increasing volume and price acceleration
        volume_increasing = np.corrcoef(range(len(recent_volumes)), recent_volumes)[0, 1] > 0.3
        price_acceleration = np.diff(np.diff(recent_prices)).mean() > 0
        
        if volume_increasing and price_acceleration:
            return 0.8
        elif volume_increasing or price_acceleration:
            return 0.5
        else:
            return 0.2

    def _calculate_consolidation_score(self, prices: np.ndarray) -> float:
        """Calculate consolidation score"""
        # Higher score means more consolidation
        returns = np.diff(prices) / prices[:-1]
        return 1.0 - np.std(returns)  # Lower volatility = higher consolidation

    def detect_regime(self, symbol: str, prices: List[float], volumes: List[float] = None) -> MarketRegimeState:
        """Detect current market regime for a symbol"""
        metrics = self.calculate_regime_metrics(prices, volumes)
        if not metrics:
            return None

        # Determine regime based on metrics
        regime = self._classify_regime(metrics)
        confidence_score = self._calculate_confidence(metrics, regime)
        confidence = self._get_confidence_level(confidence_score)
        
        # Check for regime transition
        current_regime = self.current_regimes.get(symbol)
        transition_probability = 0.0
        
        if current_regime and current_regime.regime != regime:
            transition_probability = self._calculate_transition_probability(
                current_regime, metrics, regime
            )
        
        # Calculate regime duration
        duration = timedelta(0)
        if current_regime and current_regime.regime == regime:
            duration = datetime.now() - current_regime.timestamp
        
        # Get recommended strategy
        recommended_strategy = self.adaptation_rules[regime]['entry_strategy']
        
        regime_state = MarketRegimeState(
            regime=regime,
            confidence=confidence,
            confidence_score=confidence_score,
            metrics=metrics,
            duration=duration,
            transition_probability=transition_probability,
            recommended_strategy=recommended_strategy,
            timestamp=datetime.now(),
            metadata={'symbol': symbol}
        )
        
        # Update current regime
        self.current_regimes[symbol] = regime_state
        
        # Store in history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append(regime_state)
        
        # Limit history size
        if len(self.regime_history[symbol]) > 1000:
            self.regime_history[symbol] = self.regime_history[symbol][-500:]
        
        return regime_state

    def _classify_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Classify regime based on metrics"""
        # Strong trends (check first to prioritize trend detection)
        if metrics.trend_strength > self.trend_threshold:
            if metrics.momentum > 0.02:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # High volatility
        if metrics.volatility > self.volatility_threshold:
            if metrics.momentum > 0.05:
                return MarketRegime.BREAKOUT
            elif metrics.momentum < -0.05:
                return MarketRegime.BREAKDOWN
            else:
                return MarketRegime.VOLATILE
        
        # Low volatility (but only if not trending)
        if metrics.volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        
        # Consolidation patterns
        if metrics.consolidation_score > 0.8:  # Higher threshold for consolidation
            return MarketRegime.SIDEWAYS
        
        # Ranging market
        return MarketRegime.RANGING

    def _calculate_confidence(self, metrics: RegimeMetrics, regime: MarketRegime) -> float:
        """Calculate confidence score for regime detection"""
        confidence_factors = []
        
        # Volatility confidence
        if regime in [MarketRegime.VOLATILE, MarketRegime.BREAKOUT, MarketRegime.BREAKDOWN]:
            vol_confidence = min(metrics.volatility / self.volatility_threshold, 1.0)
            confidence_factors.append(vol_confidence)
        
        # Trend confidence
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            trend_confidence = min(metrics.trend_strength / self.trend_threshold, 1.0)
            confidence_factors.append(trend_confidence)
        
        # Momentum confidence
        if regime in [MarketRegime.BREAKOUT, MarketRegime.BREAKDOWN]:
            momentum_confidence = min(abs(metrics.momentum) / 0.05, 1.0)
            confidence_factors.append(momentum_confidence)
        
        # Volume confidence
        if abs(metrics.volume_trend) > 0.3:
            confidence_factors.append(0.8)
        
        # Consolidation confidence
        if regime == MarketRegime.SIDEWAYS:
            confidence_factors.append(metrics.consolidation_score)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _get_confidence_level(self, confidence_score: float) -> RegimeConfidence:
        """Convert confidence score to confidence level"""
        if confidence_score >= 0.8:
            return RegimeConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return RegimeConfidence.HIGH
        elif confidence_score >= 0.4:
            return RegimeConfidence.MEDIUM
        else:
            return RegimeConfidence.LOW

    def _calculate_transition_probability(self, 
                                        current_regime: MarketRegimeState,
                                        metrics: RegimeMetrics,
                                        new_regime: MarketRegime) -> float:
        """Calculate probability of regime transition"""
        # Base transition probability
        base_prob = 0.3
        
        # Adjust based on confidence
        confidence_adjustment = current_regime.confidence_score * 0.4
        
        # Adjust based on duration
        duration_days = current_regime.duration.days
        duration_adjustment = min(duration_days / 30, 1.0) * 0.3
        
        # Adjust based on metric changes
        metric_adjustment = 0.0
        if new_regime in [MarketRegime.BREAKOUT, MarketRegime.BREAKDOWN]:
            metric_adjustment = metrics.breakout_probability * 0.4
        elif new_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            metric_adjustment = metrics.trend_strength * 0.3
        
        return min(base_prob + confidence_adjustment + duration_adjustment + metric_adjustment, 1.0)

    def get_adaptive_parameters(self, symbol: str, regime: MarketRegime) -> Dict[str, Any]:
        """Get adaptive parameters for current regime"""
        if regime not in self.adaptation_rules:
            return {}
        
        base_rules = self.adaptation_rules[regime].copy()
        
        # Adjust based on confidence
        current_regime = self.current_regimes.get(symbol)
        if current_regime:
            confidence_multiplier = current_regime.confidence_score
            base_rules['position_sizing_multiplier'] *= confidence_multiplier
        
        return base_rules

    def adapt_strategy(self, symbol: str, strategy_type: StrategyAdaptation) -> Dict[str, Any]:
        """Adapt strategy based on current regime"""
        current_regime = self.current_regimes.get(symbol)
        if not current_regime:
            return {}
        
        adaptive_params = self.get_adaptive_parameters(symbol, current_regime.regime)
        
        if strategy_type == StrategyAdaptation.POSITION_SIZING:
            return {
                'multiplier': adaptive_params.get('position_sizing_multiplier', 1.0),
                'risk_tolerance': adaptive_params.get('risk_tolerance', 'moderate')
            }
        elif strategy_type == StrategyAdaptation.ENTRY_EXIT_RULES:
            return {
                'entry_strategy': adaptive_params.get('entry_strategy', 'default'),
                'exit_strategy': adaptive_params.get('exit_strategy', 'default')
            }
        elif strategy_type == StrategyAdaptation.RISK_PARAMETERS:
            return {
                'risk_multiplier': adaptive_params.get('position_sizing_multiplier', 1.0),
                'stop_loss_adjustment': self._get_stop_loss_adjustment(current_regime.regime)
            }
        elif strategy_type == StrategyAdaptation.TIME_FRAME_SELECTION:
            return {
                'preferred_timeframe': adaptive_params.get('timeframe_preference', 'medium'),
                'backup_timeframes': self._get_backup_timeframes(current_regime.regime)
            }
        elif strategy_type == StrategyAdaptation.FEATURE_WEIGHTING:
            return adaptive_params.get('feature_weights', {})
        
        return {}

    def _get_stop_loss_adjustment(self, regime: MarketRegime) -> float:
        """Get stop loss adjustment for regime"""
        adjustments = {
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.BREAKDOWN: 1.3,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 1.0,
            MarketRegime.SIDEWAYS: 0.9,
            MarketRegime.TRENDING_UP: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.7,
            MarketRegime.BREAKOUT: 0.6
        }
        return adjustments.get(regime, 1.0)

    def _get_backup_timeframes(self, regime: MarketRegime) -> List[str]:
        """Get backup timeframes for regime"""
        timeframe_mapping = {
            MarketRegime.VOLATILE: ['5m', '15m'],
            MarketRegime.BREAKOUT: ['15m', '1h'],
            MarketRegime.BREAKDOWN: ['5m', '15m'],
            MarketRegime.TRENDING_UP: ['1h', '4h'],
            MarketRegime.TRENDING_DOWN: ['15m', '1h'],
            MarketRegime.RANGING: ['5m', '15m'],
            MarketRegime.SIDEWAYS: ['15m', '1h'],
            MarketRegime.LOW_VOLATILITY: ['4h', '1d']
        }
        return timeframe_mapping.get(regime, ['15m', '1h'])

    async def start_monitoring(self):
        """Start continuous regime monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Market regime monitoring started")

    async def stop_monitoring(self):
        """Stop continuous regime monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Market regime monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for regime transitions
                for symbol, current_regime in self.current_regimes.items():
                    if current_regime.transition_probability > 0.7:
                        logger.warning(f"High regime transition probability for {symbol}: "
                                     f"{current_regime.regime.value} -> {current_regime.transition_probability:.2f}")
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in regime monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _update_performance_metrics(self):
        """Update performance metrics by regime"""
        # This would integrate with actual trading performance data
        # For now, we'll use placeholder logic
        pass

    def get_regime_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive regime summary"""
        if symbol:
            current_regime = self.current_regimes.get(symbol)
            if not current_regime:
                return {}
            
            return {
                'symbol': symbol,
                'current_regime': current_regime.regime.value,
                'confidence': current_regime.confidence.value,
                'confidence_score': current_regime.confidence_score,
                'duration': str(current_regime.duration),
                'transition_probability': current_regime.transition_probability,
                'recommended_strategy': current_regime.recommended_strategy,
                'metrics': {
                    'trend_strength': current_regime.metrics.trend_strength,
                    'volatility': current_regime.metrics.volatility,
                    'momentum': current_regime.metrics.momentum,
                    'volume_trend': current_regime.metrics.volume_trend
                }
            }
        else:
            return {
                'total_symbols': len(self.current_regimes),
                'regime_distribution': self._get_regime_distribution(),
                'average_confidence': np.mean([r.confidence_score for r in self.current_regimes.values()]),
                'high_transition_risk': len([r for r in self.current_regimes.values() 
                                           if r.transition_probability > 0.7])
            }

    def _get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of current regimes"""
        distribution = {}
        for regime in MarketRegime:
            distribution[regime.value] = 0
        
        for current_regime in self.current_regimes.values():
            distribution[current_regime.regime.value] += 1
        
        return distribution

    def reset_regime_tracking(self, symbol: str = None):
        """Reset regime tracking for symbol or all symbols"""
        if symbol:
            if symbol in self.current_regimes:
                del self.current_regimes[symbol]
            if symbol in self.regime_history:
                del self.regime_history[symbol]
            if symbol in self.transition_history:
                del self.transition_history[symbol]
        else:
            self.current_regimes.clear()
            self.regime_history.clear()
            self.transition_history.clear()

# Global market regime detector instance
market_regime_detector = MarketRegimeDetector(
    risk_manager=risk_manager,
    position_sizing_optimizer=position_sizing_optimizer
)
