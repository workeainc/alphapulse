"""
Position Sizing Optimizer for AlphaPulse
Dynamic position sizing based on market conditions, volatility, and portfolio constraints
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .risk_management import RiskManager, RiskLevel, PositionType, risk_manager

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing methods"""
    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"
    OPTIMAL_F = "optimal_f"

class MarketCondition(Enum):
    """Market condition classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class SizingParameters:
    """Parameters for position sizing calculation"""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence_score: float
    win_rate: float
    avg_win: float
    avg_loss: float
    volatility: float
    market_condition: MarketCondition
    current_drawdown: float
    portfolio_correlation: float
    available_capital: float

@dataclass
class SizingResult:
    """Result of position sizing calculation"""
    position_size: float
    position_value: float
    risk_amount: float
    risk_percentage: float
    sizing_method: SizingMethod
    confidence_level: str
    recommendations: List[str]
    constraints_applied: List[str]

class PositionSizingOptimizer:
    """
    Advanced position sizing optimizer for AlphaPulse
    """
    
    def __init__(self, 
                 risk_manager: RiskManager = None,
                 max_risk_per_trade: float = 0.02,  # 2% max risk per trade
                 max_portfolio_risk: float = 0.06,  # 6% max portfolio risk
                 kelly_fraction: float = 0.25,  # Conservative Kelly fraction
                 volatility_threshold: float = 0.3,
                 correlation_threshold: float = 0.7):
        
        self.risk_manager = risk_manager or risk_manager
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction
        self.volatility_threshold = volatility_threshold
        self.correlation_threshold = correlation_threshold
        
        # Market condition multipliers
        self.market_multipliers = {
            MarketCondition.TRENDING_UP: 1.2,
            MarketCondition.TRENDING_DOWN: 0.8,
            MarketCondition.RANGING: 1.0,
            MarketCondition.VOLATILE: 0.6,
            MarketCondition.LOW_VOLATILITY: 1.3
        }
        
        # Performance tracking
        self.sizing_history: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'kelly_criterion': [],
            'fixed_fractional': [],
            'volatility_adjusted': [],
            'martingale': [],
            'anti_martingale': [],
            'optimal_f': []
        }
        
        logger.info("PositionSizingOptimizer initialized")
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion position size
        
        Kelly % = (bp - q) / b
        where:
        b = odds received on the bet (avg_win / avg_loss)
        p = probability of winning (win_rate)
        q = probability of losing (1 - win_rate)
        """
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_percentage = (b * p - q) / b
        
        # Apply conservative fraction
        kelly_percentage *= self.kelly_fraction
        
        # Ensure positive and reasonable
        kelly_percentage = max(0.0, min(kelly_percentage, 0.25))
        
        return kelly_percentage
    
    def calculate_fixed_fractional(self, risk_amount: float, available_capital: float) -> float:
        """Calculate fixed fractional position size"""
        if available_capital == 0:
            return 0.0
        
        fractional_size = risk_amount / available_capital
        return min(fractional_size, self.max_risk_per_trade)
    
    def calculate_volatility_adjusted(self, 
                                    base_size: float, 
                                    volatility: float,
                                    market_condition: MarketCondition) -> float:
        """Calculate volatility-adjusted position size"""
        # Volatility adjustment
        if volatility > self.volatility_threshold:
            vol_adjustment = self.volatility_threshold / volatility
        else:
            vol_adjustment = 1.0
        
        # Market condition adjustment
        market_adjustment = self.market_multipliers.get(market_condition, 1.0)
        
        adjusted_size = base_size * vol_adjustment * market_adjustment
        return adjusted_size
    
    def calculate_optimal_f(self, 
                           win_rate: float, 
                           avg_win: float, 
                           avg_loss: float,
                           available_capital: float) -> float:
        """
        Calculate Optimal f position size
        
        Optimal f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        """
        if avg_win == 0:
            return 0.0
        
        optimal_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Convert to percentage of capital
        optimal_f = max(0.0, min(optimal_f, 0.25))  # Cap at 25%
        
        return optimal_f
    
    def detect_market_condition(self, 
                               prices: List[float], 
                               volatility: float) -> MarketCondition:
        """Detect current market condition"""
        if len(prices) < 20:
            return MarketCondition.RANGING
        
        # Calculate trend
        recent_prices = prices[-20:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        # Normalize trend slope by average price
        avg_price = np.mean(recent_prices)
        normalized_slope = trend_slope / avg_price if avg_price > 0 else 0
        
        # Determine market condition
        if volatility > 0.4:
            return MarketCondition.VOLATILE
        elif volatility < 0.15:
            return MarketCondition.LOW_VOLATILITY
        elif normalized_slope > 0.0001:  # More sensitive threshold
            return MarketCondition.TRENDING_UP
        elif normalized_slope < -0.0001:  # More sensitive threshold
            return MarketCondition.TRENDING_DOWN
        else:
            return MarketCondition.RANGING
    
    def calculate_position_size(self, 
                               params: SizingParameters,
                               method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED) -> SizingResult:
        """
        Calculate optimal position size using specified method
        
        Args:
            params: Sizing parameters
            method: Sizing method to use
        
        Returns:
            SizingResult with position size and recommendations
        """
        recommendations = []
        constraints_applied = []
        
        # Calculate risk per share
        risk_per_share = abs(params.entry_price - params.stop_loss)
        if risk_per_share == 0:
            return SizingResult(
                position_size=0.0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percentage=0.0,
                sizing_method=method,
                confidence_level="LOW",
                recommendations=["Invalid stop loss - no risk per share"],
                constraints_applied=["ZERO_RISK"]
            )
        
        # Calculate base position size based on method
        if method == SizingMethod.KELLY_CRITERION:
            base_size_pct = self.calculate_kelly_criterion(
                params.win_rate, params.avg_win, params.avg_loss
            )
            sizing_method = SizingMethod.KELLY_CRITERION
            
        elif method == SizingMethod.FIXED_FRACTIONAL:
            risk_amount = params.available_capital * self.max_risk_per_trade
            base_size_pct = self.calculate_fixed_fractional(risk_amount, params.available_capital)
            sizing_method = SizingMethod.FIXED_FRACTIONAL
            
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            # Start with Kelly criterion
            kelly_pct = self.calculate_kelly_criterion(
                params.win_rate, params.avg_win, params.avg_loss
            )
            base_size_pct = self.calculate_volatility_adjusted(
                kelly_pct, params.volatility, params.market_condition
            )
            sizing_method = SizingMethod.VOLATILITY_ADJUSTED
            
        elif method == SizingMethod.MARTINGALE:
            # Increase position size after losses (not recommended for most traders)
            base_size_pct = self.max_risk_per_trade * (1 + params.current_drawdown)
            sizing_method = SizingMethod.MARTINGALE
            recommendations.append("Martingale sizing increases risk after losses")
            
        elif method == SizingMethod.ANTI_MARTINGALE:
            # Increase position size after wins
            base_size_pct = self.max_risk_per_trade * (1 - params.current_drawdown)
            sizing_method = SizingMethod.ANTI_MARTINGALE
            
        elif method == SizingMethod.OPTIMAL_F:
            base_size_pct = self.calculate_optimal_f(
                params.win_rate, params.avg_win, params.avg_loss, params.available_capital
            )
            sizing_method = SizingMethod.OPTIMAL_F
            
        else:
            base_size_pct = self.max_risk_per_trade
            sizing_method = SizingMethod.FIXED_FRACTIONAL
        
        # Apply confidence score adjustment
        confidence_adjustment = 0.5 + (params.confidence_score * 0.5)
        base_size_pct *= confidence_adjustment
        
        # Apply correlation adjustment
        if params.portfolio_correlation > self.correlation_threshold:
            correlation_adjustment = 1.0 - (params.portfolio_correlation - self.correlation_threshold)
            base_size_pct *= max(0.1, correlation_adjustment)
            constraints_applied.append("HIGH_CORRELATION")
            recommendations.append(f"Reduced position size due to high correlation ({params.portfolio_correlation:.2f})")
        
        # Apply drawdown adjustment
        if params.current_drawdown > 0.1:  # 10% drawdown
            drawdown_adjustment = 1.0 - (params.current_drawdown * 0.5)
            base_size_pct *= max(0.1, drawdown_adjustment)
            constraints_applied.append("HIGH_DRAWDOWN")
            recommendations.append(f"Reduced position size due to high drawdown ({params.current_drawdown:.2%})")
        
        # Calculate final position size
        risk_amount = params.available_capital * base_size_pct
        position_size = risk_amount / risk_per_share
        position_value = position_size * params.entry_price
        
        # Apply maximum position constraints
        max_position_value = params.available_capital * 0.1  # 10% max position value
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / params.entry_price
            constraints_applied.append("MAX_POSITION_SIZE")
            recommendations.append("Position size capped at 10% of available capital")
        
        # Apply minimum position constraints
        min_position_value = params.available_capital * 0.001  # 0.1% minimum
        if position_value < min_position_value:
            position_size = 0.0
            position_value = 0.0
            constraints_applied.append("MIN_POSITION_SIZE")
            recommendations.append("Position size below minimum threshold")
        
        # Calculate final risk percentage
        final_risk_amount = position_size * risk_per_share
        final_risk_percentage = final_risk_amount / params.available_capital
        
        # Determine confidence level
        if params.confidence_score > 0.8:
            confidence_level = "HIGH"
        elif params.confidence_score > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Add method-specific recommendations
        if method == SizingMethod.VOLATILITY_ADJUSTED:
            if params.volatility > self.volatility_threshold:
                recommendations.append(f"High volatility ({params.volatility:.2f}) - position size reduced")
            recommendations.append(f"Market condition: {params.market_condition.value}")
        
        # Store sizing history
        self.sizing_history.append({
            'timestamp': datetime.now(),
            'symbol': params.symbol,
            'method': method.value,
            'position_size': position_size,
            'position_value': position_value,
            'risk_percentage': final_risk_percentage,
            'confidence_score': params.confidence_score,
            'volatility': params.volatility,
            'market_condition': params.market_condition.value
        })
        
        return SizingResult(
            position_size=position_size,
            position_value=position_value,
            risk_amount=final_risk_amount,
            risk_percentage=final_risk_percentage,
            sizing_method=sizing_method,
            confidence_level=confidence_level,
            recommendations=recommendations,
            constraints_applied=constraints_applied
        )
    
    def optimize_position_size(self, 
                              params: SizingParameters) -> Dict[SizingMethod, SizingResult]:
        """
        Calculate position size using all methods and return comparison
        
        Args:
            params: Sizing parameters
        
        Returns:
            Dictionary of results for each sizing method
        """
        results = {}
        
        for method in SizingMethod:
            try:
                result = self.calculate_position_size(params, method)
                results[method] = result
            except Exception as e:
                logger.error(f"Error calculating {method.value}: {e}")
                results[method] = SizingResult(
                    position_size=0.0,
                    position_value=0.0,
                    risk_amount=0.0,
                    risk_percentage=0.0,
                    sizing_method=method,
                    confidence_level="ERROR",
                    recommendations=[f"Error: {str(e)}"],
                    constraints_applied=["ERROR"]
                )
        
        return results
    
    def get_optimal_method(self, 
                          params: SizingParameters,
                          preference: str = "balanced") -> Tuple[SizingMethod, SizingResult]:
        """
        Get optimal sizing method based on preferences
        
        Args:
            params: Sizing parameters
            preference: "conservative", "balanced", "aggressive"
        
        Returns:
            Tuple of (optimal_method, result)
        """
        all_results = self.optimize_position_size(params)
        
        if preference == "conservative":
            # Choose method with lowest risk
            optimal_method = min(all_results.keys(), 
                               key=lambda m: all_results[m].risk_percentage)
        elif preference == "aggressive":
            # Choose method with highest position size
            optimal_method = max(all_results.keys(), 
                               key=lambda m: all_results[m].position_size)
        else:  # balanced
            # Choose volatility-adjusted as default
            optimal_method = SizingMethod.VOLATILITY_ADJUSTED
        
        return optimal_method, all_results[optimal_method]
    
    def update_performance_metrics(self, 
                                  method: SizingMethod, 
                                  pnl: float, 
                                  risk_taken: float):
        """Update performance metrics for a sizing method"""
        if method.value in self.performance_metrics:
            self.performance_metrics[method.value].append(pnl / risk_taken if risk_taken > 0 else 0.0)
            
            # Keep only recent performance data
            if len(self.performance_metrics[method.value]) > 100:
                self.performance_metrics[method.value] = self.performance_metrics[method.value][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all sizing methods"""
        summary = {}
        
        for method_name, performance_data in self.performance_metrics.items():
            if performance_data:
                summary[method_name] = {
                    'avg_return': np.mean(performance_data),
                    'std_return': np.std(performance_data),
                    'sharpe_ratio': np.mean(performance_data) / np.std(performance_data) if np.std(performance_data) > 0 else 0.0,
                    'total_trades': len(performance_data),
                    'win_rate': len([x for x in performance_data if x > 0]) / len(performance_data)
                }
            else:
                summary[method_name] = {
                    'avg_return': 0.0,
                    'std_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0
                }
        
        return summary
    
    def get_sizing_recommendations(self, 
                                  symbol: str,
                                  current_conditions: Dict[str, Any]) -> List[str]:
        """Get general sizing recommendations based on current conditions"""
        recommendations = []
        
        # Volatility-based recommendations
        volatility = current_conditions.get('volatility', 0.0)
        if volatility > 0.4:
            recommendations.append("High volatility detected - consider reducing position sizes")
        elif volatility < 0.15:
            recommendations.append("Low volatility detected - consider increasing position sizes")
        
        # Market condition recommendations
        market_condition = current_conditions.get('market_condition', MarketCondition.RANGING)
        if market_condition == MarketCondition.VOLATILE:
            recommendations.append("Volatile market - use conservative sizing methods")
        elif market_condition == MarketCondition.TRENDING_UP:
            recommendations.append("Uptrend detected - consider trend-following sizing")
        elif market_condition == MarketCondition.TRENDING_DOWN:
            recommendations.append("Downtrend detected - reduce position sizes")
        
        # Portfolio-level recommendations
        current_drawdown = current_conditions.get('current_drawdown', 0.0)
        if current_drawdown > 0.1:
            recommendations.append("High drawdown - implement defensive sizing")
        
        correlation = current_conditions.get('portfolio_correlation', 0.0)
        if correlation > 0.7:
            recommendations.append("High portfolio correlation - consider diversification")
        
        return recommendations
    
    def reset_performance_tracking(self):
        """Reset performance tracking data"""
        self.sizing_history.clear()
        for method in self.performance_metrics:
            self.performance_metrics[method].clear()
        logger.info("Performance tracking reset")

# Global position sizing optimizer instance
position_sizing_optimizer = PositionSizingOptimizer(risk_manager=risk_manager)
