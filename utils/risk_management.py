"""
Risk Management Utilities for AlphaPulse

This module provides comprehensive risk management functionality including
position sizing, risk calculation, portfolio management, and safety checks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels for position sizing."""
    CONSERVATIVE = 0.01  # 1% risk per trade
    MODERATE = 0.02      # 2% risk per trade
    AGGRESSIVE = 0.03    # 3% risk per trade


@dataclass
class RiskMetrics:
    """Risk metrics for a trading position."""
    position_size: float
    risk_amount: float
    max_loss: float
    risk_reward_ratio: float
    portfolio_risk: float
    correlation_risk: float
    volatility_risk: float


@dataclass
class PositionSizingParams:
    """Parameters for position sizing calculations."""
    account_balance: float
    risk_per_trade: float
    stop_loss_pct: float
    max_position_size: float = 0.1  # 10% max position
    correlation_threshold: float = 0.7
    volatility_threshold: float = 0.05


class RiskManager:
    """
    Comprehensive risk management system for AlphaPulse.
    
    Handles position sizing, risk calculation, portfolio management,
    and safety checks for trading operations.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config or {}
        self.risk_level = RiskLevel.MODERATE
        self.max_drawdown = 0.15  # 15% max drawdown
        self.max_correlation = 0.7
        self.position_history = []
        self.risk_metrics_history = []
        
    def calculate_position_size(
        self, 
        account_balance: float, 
        entry_price: float, 
        stop_loss: float,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_per_trade: Risk percentage per trade (optional)
            
        Returns:
            Position size in base currency
        """
        if risk_per_trade is None:
            risk_per_trade = self.risk_level.value
            
        # Calculate risk amount
        risk_amount = account_balance * risk_per_trade
        
        # Calculate price risk
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            logger.warning("Stop loss equals entry price, using minimum risk")
            price_risk = 0.001
            
        # Calculate position size
        position_size = risk_amount / price_risk
        
        # Apply maximum position size limit
        max_position = account_balance * self.config.get('max_position_size', 0.1)
        position_size = min(position_size, max_position)
        
        return position_size
    
    def calculate_risk_metrics(
        self,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        portfolio_positions: List[Dict] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a position.
        
        Args:
            position_size: Position size in base currency
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            account_balance: Current account balance
            portfolio_positions: List of current portfolio positions
            
        Returns:
            RiskMetrics object with calculated risk metrics
        """
        # Calculate basic risk metrics
        risk_amount = abs(entry_price - stop_loss) * position_size
        max_loss = risk_amount
        potential_profit = abs(take_profit - entry_price) * position_size
        risk_reward_ratio = potential_profit / risk_amount if risk_amount > 0 else 0
        
        # Calculate portfolio risk
        portfolio_risk = self._calculate_portfolio_risk(
            position_size, entry_price, account_balance, portfolio_positions
        )
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(
            position_size, portfolio_positions
        )
        
        # Calculate volatility risk
        volatility_risk = self._calculate_volatility_risk(
            position_size, entry_price, stop_loss
        )
        
        return RiskMetrics(
            position_size=position_size,
            risk_amount=risk_amount,
            max_loss=max_loss,
            risk_reward_ratio=risk_reward_ratio,
            portfolio_risk=portfolio_risk,
            correlation_risk=correlation_risk,
            volatility_risk=volatility_risk
        )
    
    def _calculate_portfolio_risk(
        self,
        position_size: float,
        entry_price: float,
        account_balance: float,
        portfolio_positions: List[Dict] = None
    ) -> float:
        """Calculate portfolio risk contribution."""
        if not portfolio_positions:
            return position_size * entry_price / account_balance
            
        total_exposure = sum(pos['size'] * pos['price'] for pos in portfolio_positions)
        new_exposure = position_size * entry_price
        total_risk = (total_exposure + new_exposure) / account_balance
        
        return min(total_risk, 1.0)  # Cap at 100%
    
    def _calculate_correlation_risk(
        self,
        position_size: float,
        portfolio_positions: List[Dict] = None
    ) -> float:
        """Calculate correlation risk with existing positions."""
        if not portfolio_positions:
            return 0.0
            
        # Simplified correlation calculation
        total_correlation = sum(
            pos.get('correlation', 0.5) * pos['size'] 
            for pos in portfolio_positions
        )
        avg_correlation = total_correlation / len(portfolio_positions)
        
        return avg_correlation * position_size
    
    def _calculate_volatility_risk(
        self,
        position_size: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """Calculate volatility-based risk."""
        price_range = abs(entry_price - stop_loss) / entry_price
        volatility_risk = price_range * position_size
        
        return min(volatility_risk, 0.1)  # Cap at 10%
    
    def validate_trade_risk(
        self,
        risk_metrics: RiskMetrics,
        market_conditions: Dict = None
    ) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk management criteria.
        
        Args:
            risk_metrics: Calculated risk metrics
            market_conditions: Current market conditions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check maximum risk per trade
        if risk_metrics.risk_amount > self.config.get('max_risk_per_trade', 0.05):
            return False, "Risk exceeds maximum per trade limit"
        
        # Check portfolio risk
        if risk_metrics.portfolio_risk > self.config.get('max_portfolio_risk', 0.8):
            return False, "Portfolio risk too high"
        
        # Check correlation risk
        if risk_metrics.correlation_risk > self.max_correlation:
            return False, "Correlation risk too high"
        
        # Check risk-reward ratio
        if risk_metrics.risk_reward_ratio < self.config.get('min_risk_reward', 1.5):
            return False, "Risk-reward ratio too low"
        
        # Check volatility risk
        if risk_metrics.volatility_risk > self.config.get('max_volatility_risk', 0.05):
            return False, "Volatility risk too high"
        
        # Market condition checks
        if market_conditions:
            if market_conditions.get('high_volatility', False):
                if risk_metrics.volatility_risk > 0.03:
                    return False, "High volatility market - reducing position size"
            
            if market_conditions.get('low_liquidity', False):
                if risk_metrics.position_size > self.config.get('max_low_liquidity_size', 0.05):
                    return False, "Low liquidity - position size too large"
        
        return True, "Trade meets risk criteria"
    
    def adjust_position_for_risk(
        self,
        original_size: float,
        risk_metrics: RiskMetrics,
        market_conditions: Dict = None
    ) -> float:
        """
        Adjust position size based on risk metrics and market conditions.
        
        Args:
            original_size: Original position size
            risk_metrics: Calculated risk metrics
            market_conditions: Current market conditions
            
        Returns:
            Adjusted position size
        """
        adjustment_factor = 1.0
        
        # Reduce size for high portfolio risk
        if risk_metrics.portfolio_risk > 0.6:
            adjustment_factor *= 0.8
        
        # Reduce size for high correlation
        if risk_metrics.correlation_risk > 0.5:
            adjustment_factor *= 0.7
        
        # Reduce size for high volatility
        if risk_metrics.volatility_risk > 0.03:
            adjustment_factor *= 0.6
        
        # Market condition adjustments
        if market_conditions:
            if market_conditions.get('high_volatility', False):
                adjustment_factor *= 0.5
            
            if market_conditions.get('low_liquidity', False):
                adjustment_factor *= 0.3
        
        adjusted_size = original_size * adjustment_factor
        
        # Ensure minimum position size
        min_size = self.config.get('min_position_size', 0.001)
        return max(adjusted_size, min_size)
    
    def calculate_portfolio_metrics(
        self,
        positions: List[Dict],
        account_balance: float
    ) -> Dict:
        """
        Calculate overall portfolio risk metrics.
        
        Args:
            positions: List of current positions
            account_balance: Current account balance
            
        Returns:
            Dictionary with portfolio metrics
        """
        if not positions:
            return {
                'total_exposure': 0.0,
                'portfolio_risk': 0.0,
                'max_drawdown': 0.0,
                'diversification_score': 1.0,
                'correlation_score': 0.0
            }
        
        total_exposure = sum(pos['size'] * pos['price'] for pos in positions)
        portfolio_risk = total_exposure / account_balance
        
        # Calculate diversification score
        symbols = set(pos['symbol'] for pos in positions)
        diversification_score = len(symbols) / len(positions) if positions else 1.0
        
        # Calculate correlation score
        correlations = [pos.get('correlation', 0.5) for pos in positions]
        avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
        
        # Calculate potential drawdown
        max_losses = [pos.get('max_loss', 0) for pos in positions]
        total_max_loss = sum(max_losses)
        max_drawdown = total_max_loss / account_balance
        
        return {
            'total_exposure': total_exposure,
            'portfolio_risk': portfolio_risk,
            'max_drawdown': max_drawdown,
            'diversification_score': diversification_score,
            'correlation_score': avg_correlation
        }
    
    def should_close_position(
        self,
        position: Dict,
        current_price: float,
        market_conditions: Dict = None
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be closed based on risk criteria.
        
        Args:
            position: Position information
            current_price: Current market price
            market_conditions: Current market conditions
            
        Returns:
            Tuple of (should_close, reason)
        """
        # Check stop loss
        if 'stop_loss' in position and current_price <= position['stop_loss']:
            return True, "Stop loss triggered"
        
        # Check take profit
        if 'take_profit' in position and current_price >= position['take_profit']:
            return True, "Take profit reached"
        
        # Check time-based exit
        if 'entry_time' in position:
            time_in_trade = datetime.now() - position['entry_time']
            max_hold_time = timedelta(hours=self.config.get('max_hold_time_hours', 24))
            if time_in_trade > max_hold_time:
                return True, "Maximum hold time exceeded"
        
        # Check portfolio risk
        if position.get('portfolio_risk', 0) > self.config.get('max_position_risk', 0.1):
            return True, "Position risk too high"
        
        # Market condition checks
        if market_conditions:
            if market_conditions.get('emergency_close', False):
                return True, "Emergency close triggered"
            
            if market_conditions.get('high_volatility', False):
                if position.get('volatility_risk', 0) > 0.05:
                    return True, "High volatility - closing position"
        
        return False, "Position meets criteria"
    
    def log_risk_metrics(self, risk_metrics: RiskMetrics, trade_id: str):
        """Log risk metrics for analysis."""
        self.risk_metrics_history.append({
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'metrics': risk_metrics
        })
        
        logger.info(f"Risk metrics for trade {trade_id}: {risk_metrics}")
    
    def get_risk_summary(self) -> Dict:
        """Get summary of risk management performance."""
        if not self.risk_metrics_history:
            return {'total_trades': 0, 'avg_risk': 0.0}
        
        total_trades = len(self.risk_metrics_history)
        avg_risk = sum(
            m['metrics'].risk_amount for m in self.risk_metrics_history
        ) / total_trades
        
        return {
            'total_trades': total_trades,
            'avg_risk': avg_risk,
            'risk_history': self.risk_metrics_history[-10:]  # Last 10 trades
        }


class PositionSizer:
    """
    Advanced position sizing calculator with multiple strategies.
    """
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            account_balance: Current account balance
            
        Returns:
            Optimal position size as percentage of account
        """
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply fractional Kelly (usually 1/4 or 1/2)
        fractional_kelly = kelly_fraction * 0.25
        
        return max(0.0, min(fractional_kelly, 0.1))  # Cap at 10%
    
    def volatility_adjusted(
        self,
        base_size: float,
        current_volatility: float,
        historical_volatility: float
    ) -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            base_size: Base position size
            current_volatility: Current market volatility
            historical_volatility: Historical average volatility
            
        Returns:
            Volatility-adjusted position size
        """
        if historical_volatility == 0:
            return base_size
        
        volatility_ratio = current_volatility / historical_volatility
        
        # Reduce size when volatility is high
        if volatility_ratio > 1.5:
            adjustment = 0.5
        elif volatility_ratio > 1.2:
            adjustment = 0.7
        elif volatility_ratio < 0.8:
            adjustment = 1.2  # Increase size when volatility is low
        else:
            adjustment = 1.0
        
        return base_size * adjustment


# Example usage
async def example_usage():
    """Example usage of the risk management system."""
    
    # Initialize risk manager
    config = {
        'max_risk_per_trade': 0.02,
        'max_portfolio_risk': 0.8,
        'min_risk_reward': 1.5,
        'max_volatility_risk': 0.05,
        'max_position_size': 0.1,
        'min_position_size': 0.001
    }
    
    risk_manager = RiskManager(config)
    
    # Calculate position size
    account_balance = 10000
    entry_price = 50000
    stop_loss = 48000
    
    position_size = risk_manager.calculate_position_size(
        account_balance, entry_price, stop_loss
    )
    
    print(f"Calculated position size: {position_size}")
    
    # Calculate risk metrics
    take_profit = 52000
    risk_metrics = risk_manager.calculate_risk_metrics(
        position_size, entry_price, stop_loss, take_profit, account_balance
    )
    
    print(f"Risk metrics: {risk_metrics}")
    
    # Validate trade
    is_valid, reason = risk_manager.validate_trade_risk(risk_metrics)
    print(f"Trade valid: {is_valid}, Reason: {reason}")


if __name__ == "__main__":
    asyncio.run(example_usage())
