"""
Multi-Timeframe Signal Fusion for AlphaPulse
Combine signals from different timeframes with intelligent weighting and confirmation logic
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
from .position_sizing import PositionSizingOptimizer, position_sizing_optimizer, MarketCondition

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class SignalDirection(Enum):
    """Signal directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: Timeframe
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    pattern_type: str
    price_level: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class FusedSignal:
    """Fused signal from multiple timeframes"""
    symbol: str
    primary_direction: SignalDirection
    overall_strength: SignalStrength
    confidence_score: float
    timeframe_agreement: float
    signal_consistency: float
    recommended_action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe_signals: Dict[Timeframe, TimeframeSignal]
    fusion_metadata: Dict[str, Any]
    timestamp: datetime

class TimeframeWeighting:
    """Dynamic weighting system for different timeframes"""
    
    def __init__(self):
        # Base weights for different timeframes
        self.base_weights = {
            Timeframe.M1: 0.05,   # 5% - noise prone
            Timeframe.M5: 0.10,   # 10% - short-term
            Timeframe.M15: 0.15,  # 15% - intraday
            Timeframe.H1: 0.25,   # 25% - swing
            Timeframe.H4: 0.25,   # 25% - trend
            Timeframe.D1: 0.20    # 20% - long-term
        }
        
        # Market condition adjustments
        self.market_adjustments = {
            'trending': {
                Timeframe.H1: 1.2,
                Timeframe.H4: 1.3,
                Timeframe.D1: 1.4
            },
            'ranging': {
                Timeframe.M15: 1.2,
                Timeframe.H1: 1.1,
                Timeframe.M5: 1.1
            },
            'volatile': {
                Timeframe.M1: 0.5,
                Timeframe.M5: 0.8,
                Timeframe.D1: 1.2
            }
        }
    
    def get_adjusted_weights(self, market_condition: str) -> Dict[Timeframe, float]:
        """Get adjusted weights based on market condition"""
        weights = self.base_weights.copy()
        
        if market_condition in self.market_adjustments:
            adjustments = self.market_adjustments[market_condition]
            for tf, adj in adjustments.items():
                weights[tf] *= adj
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {tf: w / total_weight for tf, w in weights.items()}
        
        return weights

class MultiTimeframeFusion:
    """
    Multi-timeframe signal fusion system for AlphaPulse
    """
    
    def __init__(self, 
                 risk_manager: RiskManager = None,
                 position_sizing_optimizer: PositionSizingOptimizer = None,
                 min_confidence_threshold: float = 0.6,
                 min_timeframe_agreement: float = 0.5,
                 enable_dynamic_weighting: bool = True):
        
        self.risk_manager = risk_manager or risk_manager
        self.position_sizing_optimizer = position_sizing_optimizer or position_sizing_optimizer
        self.min_confidence_threshold = min_confidence_threshold
        self.min_timeframe_agreement = min_timeframe_agreement
        self.enable_dynamic_weighting = enable_dynamic_weighting
        
        # Initialize components
        self.timeframe_weighting = TimeframeWeighting()
        self.signal_history: List[FusedSignal] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'accuracy': [],
            'profit_factor': [],
            'max_drawdown': []
        }
        
        # Signal strength mappings
        self.strength_values = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0
        }
        
        # Direction values
        self.direction_values = {
            SignalDirection.BULLISH: 1.0,
            SignalDirection.BEARISH: -1.0,
            SignalDirection.NEUTRAL: 0.0
        }
        
        logger.info("MultiTimeframeFusion initialized")
    
    def detect_market_condition(self, signals: Dict[Timeframe, TimeframeSignal]) -> str:
        """Detect current market condition based on timeframe signals"""
        if not signals:
            return 'ranging'
        
        # Analyze signal patterns across timeframes
        short_term_signals = []
        long_term_signals = []
        
        for tf, signal in signals.items():
            if tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
                short_term_signals.append(signal)
            elif tf in [Timeframe.H1, Timeframe.H4, Timeframe.D1]:
                long_term_signals.append(signal)
        
        # Check for trending conditions
        if long_term_signals:
            long_term_directions = [s.direction for s in long_term_signals]
            if all(d == SignalDirection.BULLISH for d in long_term_directions):
                return 'trending'
            elif all(d == SignalDirection.BEARISH for d in long_term_directions):
                return 'trending'
        
        # Check for volatile conditions
        if short_term_signals:
            short_term_strengths = [self.strength_values[s.strength] for s in short_term_signals]
            avg_short_strength = np.mean(short_term_strengths)
            if avg_short_strength > 0.7:
                return 'volatile'
        
        return 'ranging'
    
    def calculate_timeframe_agreement(self, signals: Dict[Timeframe, TimeframeSignal]) -> float:
        """Calculate agreement level across timeframes"""
        if len(signals) < 2:
            return 1.0
        
        directions = [signal.direction for signal in signals.values()]
        
        # Count bullish and bearish signals
        bullish_count = sum(1 for d in directions if d == SignalDirection.BULLISH)
        bearish_count = sum(1 for d in directions if d == SignalDirection.BEARISH)
        neutral_count = sum(1 for d in directions if d == SignalDirection.NEUTRAL)
        
        total_signals = len(directions)
        
        # Calculate agreement as percentage of dominant direction
        if bullish_count > bearish_count:
            agreement = bullish_count / total_signals
        elif bearish_count > bullish_count:
            agreement = bearish_count / total_signals
        else:
            agreement = 0.5  # Split or neutral
        
        return agreement
    
    def calculate_signal_consistency(self, signals: Dict[Timeframe, TimeframeSignal]) -> float:
        """Calculate consistency of signal strengths across timeframes"""
        if len(signals) < 2:
            return 1.0
        
        strengths = [self.strength_values[signal.strength] for signal in signals.values()]
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)
        
        if mean_strength == 0:
            return 1.0
        
        cv = std_strength / mean_strength
        consistency = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
        
        return consistency
    
    def fuse_signals(self, 
                     symbol: str,
                     signals: Dict[Timeframe, TimeframeSignal],
                     current_price: float) -> Optional[FusedSignal]:
        """
        Fuse signals from multiple timeframes into a single signal
        
        Args:
            symbol: Trading symbol
            signals: Dictionary of timeframe signals
            current_price: Current market price
        
        Returns:
            FusedSignal if fusion criteria are met, None otherwise
        """
        if not signals:
            return None
        
        # Detect market condition
        market_condition = self.detect_market_condition(signals)
        
        # Get adjusted weights
        if self.enable_dynamic_weighting:
            weights = self.timeframe_weighting.get_adjusted_weights(market_condition)
        else:
            weights = self.timeframe_weighting.base_weights
        
        # Calculate weighted signal
        weighted_direction = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for tf, signal in signals.items():
            if tf in weights:
                weight = weights[tf]
                weighted_direction += self.direction_values[signal.direction] * weight
                weighted_strength += self.strength_values[signal.strength] * weight
                weighted_confidence += signal.confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Normalize weighted values
        weighted_direction /= total_weight
        weighted_strength /= total_weight
        weighted_confidence /= total_weight
        
        # Determine primary direction
        if weighted_direction > 0.2:
            primary_direction = SignalDirection.BULLISH
        elif weighted_direction < -0.2:
            primary_direction = SignalDirection.BEARISH
        else:
            primary_direction = SignalDirection.NEUTRAL
        
        # Determine overall strength
        if weighted_strength > 0.75:
            overall_strength = SignalStrength.VERY_STRONG
        elif weighted_strength > 0.5:
            overall_strength = SignalStrength.STRONG
        elif weighted_strength > 0.25:
            overall_strength = SignalStrength.MODERATE
        else:
            overall_strength = SignalStrength.WEAK
        
        # Calculate agreement and consistency
        timeframe_agreement = self.calculate_timeframe_agreement(signals)
        signal_consistency = self.calculate_signal_consistency(signals)
        
        # Check fusion criteria
        if (weighted_confidence < self.min_confidence_threshold or 
            timeframe_agreement < self.min_timeframe_agreement):
            return None
        
        # Generate trading recommendations
        recommended_action = self._generate_recommendation(
            primary_direction, overall_strength, weighted_confidence, timeframe_agreement
        )
        
        # Calculate entry, stop loss, and take profit levels
        entry_price, stop_loss, take_profit = self._calculate_price_levels(
            current_price, primary_direction, overall_strength, signals
        )
        
        # Create fused signal
        fused_signal = FusedSignal(
            symbol=symbol,
            primary_direction=primary_direction,
            overall_strength=overall_strength,
            confidence_score=weighted_confidence,
            timeframe_agreement=timeframe_agreement,
            signal_consistency=signal_consistency,
            recommended_action=recommended_action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe_signals=signals,
            fusion_metadata={
                'market_condition': market_condition,
                'weighted_direction': weighted_direction,
                'weighted_strength': weighted_strength,
                'weights_used': weights
            },
            timestamp=datetime.now()
        )
        
        # Store in history
        self.signal_history.append(fused_signal)
        
        return fused_signal
    
    def _generate_recommendation(self, 
                                direction: SignalDirection,
                                strength: SignalStrength,
                                confidence: float,
                                agreement: float) -> str:
        """Generate trading recommendation based on fused signal"""
        
        if direction == SignalDirection.NEUTRAL:
            return "HOLD - No clear directional bias"
        
        if confidence < 0.7 or agreement < 0.6:
            return "WAIT - Insufficient signal quality"
        
        if strength in [SignalStrength.WEAK, SignalStrength.MODERATE]:
            if direction == SignalDirection.BULLISH:
                return "WEAK_BUY - Consider small position with tight stops"
            else:
                return "WEAK_SELL - Consider small position with tight stops"
        
        if strength == SignalStrength.STRONG:
            if direction == SignalDirection.BULLISH:
                return "BUY - Strong bullish signal across timeframes"
            else:
                return "SELL - Strong bearish signal across timeframes"
        
        if strength == SignalStrength.VERY_STRONG:
            if direction == SignalDirection.BULLISH:
                return "STRONG_BUY - Very strong bullish signal, high conviction"
            else:
                return "STRONG_SELL - Very strong bearish signal, high conviction"
        
        return "HOLD - Insufficient signal strength"
    
    def _calculate_price_levels(self,
                               current_price: float,
                               direction: SignalDirection,
                               strength: SignalStrength,
                               signals: Dict[Timeframe, TimeframeSignal]) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        
        # Get relevant price levels from signals
        price_levels = []
        for signal in signals.values():
            if signal.price_level > 0:
                price_levels.append(signal.price_level)
        
        if not price_levels:
            # Use default levels based on current price
            if direction == SignalDirection.BULLISH:
                entry_price = current_price * 1.001  # Slight premium
                stop_loss = current_price * 0.995   # 0.5% below current
                take_profit = current_price * 1.015  # 1.5% above current
            elif direction == SignalDirection.BEARISH:
                entry_price = current_price * 0.999  # Slight discount
                stop_loss = current_price * 1.005   # 0.5% above current
                take_profit = current_price * 0.985  # 1.5% below current
            else:
                entry_price = current_price
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.005
        else:
            # Use signal price levels
            avg_price_level = np.mean(price_levels)
            
            if direction == SignalDirection.BULLISH:
                entry_price = max(current_price, avg_price_level)
                stop_loss = entry_price * 0.995
                take_profit = entry_price * 1.015
            elif direction == SignalDirection.BEARISH:
                entry_price = min(current_price, avg_price_level)
                stop_loss = entry_price * 1.005
                take_profit = entry_price * 0.985
            else:
                entry_price = current_price
                stop_loss = current_price * 0.995
                take_profit = current_price * 1.005
        
        return entry_price, stop_loss, take_profit
    
    def get_position_size_recommendation(self, 
                                        fused_signal: FusedSignal,
                                        available_capital: float) -> Dict[str, Any]:
        """Get position sizing recommendation for fused signal"""
        
        if not self.position_sizing_optimizer:
            return {
                'position_size': 0.0,
                'reason': 'Position sizing optimizer not available'
            }
        
        # Calculate base confidence for position sizing
        base_confidence = fused_signal.confidence_score * fused_signal.timeframe_agreement
        
        # Adjust confidence based on signal strength
        strength_multiplier = self.strength_values[fused_signal.overall_strength]
        adjusted_confidence = base_confidence * strength_multiplier
        
        # Create sizing parameters
        from .position_sizing import SizingParameters, MarketCondition
        
        # Detect market condition for sizing
        market_condition = self._detect_market_condition_for_sizing(fused_signal)
        
        params = SizingParameters(
            symbol=fused_signal.symbol,
            entry_price=fused_signal.entry_price,
            stop_loss=fused_signal.stop_loss,
            take_profit=fused_signal.take_profit,
            confidence_score=adjusted_confidence,
            win_rate=0.6,  # Default - could be learned from history
            avg_win=abs(fused_signal.take_profit - fused_signal.entry_price),
            avg_loss=abs(fused_signal.stop_loss - fused_signal.entry_price),
            volatility=0.25,  # Default - could be calculated from price data
            market_condition=market_condition,
            current_drawdown=0.0,  # Could be obtained from risk manager
            portfolio_correlation=0.3,  # Could be calculated from portfolio
            available_capital=available_capital
        )
        
        # Get optimal position size
        method, result = self.position_sizing_optimizer.get_optimal_method(params, "balanced")
        
        return {
            'position_size': result.position_size,
            'position_value': result.position_value,
            'risk_percentage': result.risk_percentage,
            'sizing_method': method.value,
            'confidence_level': result.confidence_level,
            'recommendations': result.recommendations,
            'adjusted_confidence': adjusted_confidence
        }
    
    def _detect_market_condition_for_sizing(self, fused_signal: FusedSignal) -> MarketCondition:
        """Detect market condition for position sizing"""
        from .position_sizing import MarketCondition
        
        # Analyze timeframe signals for market condition
        short_term_count = 0
        long_term_count = 0
        volatile_signals = 0
        
        for tf, signal in fused_signal.timeframe_signals.items():
            if tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
                short_term_count += 1
                if signal.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                    volatile_signals += 1
            elif tf in [Timeframe.H1, Timeframe.H4, Timeframe.D1]:
                long_term_count += 1
        
        # Determine market condition
        if volatile_signals >= 2:
            return MarketCondition.VOLATILE
        elif long_term_count >= 2 and fused_signal.timeframe_agreement > 0.7:
            if fused_signal.primary_direction == SignalDirection.BULLISH:
                return MarketCondition.TRENDING_UP
            else:
                return MarketCondition.TRENDING_DOWN
        else:
            return MarketCondition.RANGING
    
    def update_performance_metrics(self, 
                                  signal_id: str, 
                                  pnl: float, 
                                  max_drawdown: float):
        """Update performance metrics for fused signals"""
        # Find the signal in history
        for signal in self.signal_history:
            if signal.symbol == signal_id:  # Using symbol as ID for now
                # Calculate profit factor (simplified)
                profit_factor = 1.0 if pnl > 0 else 0.0
                
                # Update metrics
                self.performance_metrics['accuracy'].append(1.0 if pnl > 0 else 0.0)
                self.performance_metrics['profit_factor'].append(profit_factor)
                self.performance_metrics['max_drawdown'].append(max_drawdown)
                
                # Keep only recent metrics
                max_history = 100
                for metric in self.performance_metrics.values():
                    if len(metric) > max_history:
                        metric[:] = metric[-max_history:]
                
                break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for fused signals"""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                if metric_name == 'accuracy':
                    summary[metric_name] = {
                        'current': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values),
                        'overall': np.mean(values),
                        'total_signals': len(values)
                    }
                elif metric_name == 'profit_factor':
                    summary[metric_name] = {
                        'current': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values),
                        'overall': np.mean(values)
                    }
                elif metric_name == 'max_drawdown':
                    summary[metric_name] = {
                        'current': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values),
                        'worst': np.min(values) if values else 0.0
                    }
            else:
                summary[metric_name] = {'current': 0.0, 'overall': 0.0}
        
        return summary
    
    def get_signal_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed analysis of recent signals for a symbol"""
        recent_signals = [s for s in self.signal_history if s.symbol == symbol][-10:]
        
        if not recent_signals:
            return {'message': 'No recent signals found'}
        
        analysis = {
            'total_signals': len(recent_signals),
            'bullish_signals': sum(1 for s in recent_signals if s.primary_direction == SignalDirection.BULLISH),
            'bearish_signals': sum(1 for s in recent_signals if s.primary_direction == SignalDirection.BEARISH),
            'neutral_signals': sum(1 for s in recent_signals if s.primary_direction == SignalDirection.NEUTRAL),
            'avg_confidence': np.mean([s.confidence_score for s in recent_signals]),
            'avg_agreement': np.mean([s.timeframe_agreement for s in recent_signals]),
            'avg_consistency': np.mean([s.signal_consistency for s in recent_signals]),
            'strength_distribution': {
                'weak': sum(1 for s in recent_signals if s.overall_strength == SignalStrength.WEAK),
                'moderate': sum(1 for s in recent_signals if s.overall_strength == SignalStrength.MODERATE),
                'strong': sum(1 for s in recent_signals if s.overall_strength == SignalStrength.STRONG),
                'very_strong': sum(1 for s in recent_signals if s.overall_strength == SignalStrength.VERY_STRONG)
            }
        }
        
        return analysis
    
    def reset_performance_tracking(self):
        """Reset performance tracking data"""
        self.signal_history.clear()
        for metric in self.performance_metrics.values():
            metric.clear()
        logger.info("Performance tracking reset")

# Global multi-timeframe fusion instance
multi_timeframe_fusion = MultiTimeframeFusion(
    risk_manager=risk_manager,
    position_sizing_optimizer=position_sizing_optimizer
)
