"""
Risk Management System for AlphaPulse
Dynamic position sizing, portfolio risk controls, and real-time monitoring
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class PositionType(Enum):
    """Position types for risk calculation"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    volatility: float
    var_95: float  # Value at Risk (95% confidence)
    max_drawdown: float
    correlation_risk: float
    concentration_risk: float
    timestamp: datetime

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    portfolio_volatility: float
    portfolio_var_95: float
    max_portfolio_drawdown: float
    sharpe_ratio: float
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    leverage_ratio: float
    margin_utilization: float
    timestamp: datetime

class RiskManager:
    """
    Comprehensive risk management system for AlphaPulse
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size_pct: float = 0.05,  # 5% max per position
                 max_portfolio_risk_pct: float = 0.02,  # 2% max portfolio risk
                 max_drawdown_pct: float = 0.15,  # 15% max drawdown
                 risk_level: RiskLevel = RiskLevel.MODERATE,
                 volatility_lookback: int = 30,
                 correlation_lookback: int = 60,
                 update_interval: int = 60):  # seconds
        
        # Risk parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_level = risk_level
        
        # Lookback periods
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.update_interval = update_interval
        
        # Risk level multipliers
        self.risk_multipliers = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.5
        }
        
        # Data storage
        self.positions: Dict[str, PositionRisk] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.portfolio_history: List[PortfolioRisk] = []
        self.risk_alerts: List[Dict] = []
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        
        # Correlation tracking
        self.correlation_matrix = pd.DataFrame()
        self.last_correlation_update = None
        
        # Risk monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info(f"RiskManager initialized with {risk_level.value} risk level")
    
    def set_risk_level(self, risk_level: RiskLevel):
        """Update risk tolerance level"""
        self.risk_level = risk_level
        logger.info(f"Risk level updated to {risk_level.value}")
    
    def calculate_position_size(self, 
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               confidence_score: float,
                               volatility: float = None) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            confidence_score: Model confidence (0-1)
            volatility: Asset volatility (optional)
        
        Returns:
            Position size in base currency
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
        
        # Calculate maximum risk amount
        max_risk_amount = self.current_capital * self.max_position_size_pct * self.risk_multipliers[self.risk_level]
        
        # Adjust for confidence score
        confidence_adjustment = 0.5 + (confidence_score * 0.5)  # 0.5 to 1.0
        
        # Adjust for volatility if provided
        volatility_adjustment = 1.0
        if volatility is not None:
            # Higher volatility = smaller position
            volatility_adjustment = max(0.1, 1.0 - (volatility * 0.5))
        
        # Calculate position size
        position_size = (max_risk_amount * confidence_adjustment * volatility_adjustment) / risk_per_share
        
        # Apply additional constraints
        max_position_value = self.current_capital * self.max_position_size_pct
        position_size = min(position_size, max_position_value / entry_price)
        
        # Ensure minimum position size
        min_position_value = self.current_capital * 0.001  # 0.1% minimum
        if position_size * entry_price < min_position_value:
            position_size = 0.0
        
        logger.debug(f"Position size calculated for {symbol}: {position_size:.2f} shares")
        return position_size
    
    def calculate_volatility(self, symbol: str, prices: List[float] = None) -> float:
        """Calculate rolling volatility for a symbol"""
        if prices is None:
            prices = self.price_history.get(symbol, [])
        
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate rolling volatility
        if len(returns) >= self.volatility_lookback:
            volatility = np.std(returns[-self.volatility_lookback:]) * np.sqrt(252)  # Annualized
        else:
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        
        return volatility
    
    def calculate_var(self, symbol: str, position_size: float, confidence: float = 0.95) -> float:
        """Calculate Value at Risk for a position"""
        prices = self.price_history.get(symbol, [])
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR using historical simulation
        var_percentile = np.percentile(returns, (1 - confidence) * 100)
        var_amount = abs(var_percentile * position_size * prices[-1])
        
        return var_amount
    
    def update_position(self, 
                       symbol: str,
                       position_type: PositionType,
                       size: float,
                       entry_price: float,
                       current_price: float):
        """Update position information"""
        # Calculate metrics
        if position_type == PositionType.LONG:
            unrealized_pnl = (current_price - entry_price) * size
        else:  # SHORT
            unrealized_pnl = (entry_price - current_price) * size
        
        unrealized_pnl_pct = (unrealized_pnl / (entry_price * size)) * 100
        
        # Calculate volatility
        volatility = self.calculate_volatility(symbol)
        
        # Calculate VaR
        var_95 = self.calculate_var(symbol, size)
        
        # Calculate correlation risk
        correlation_risk = self.calculate_correlation_risk(symbol)
        
        # Calculate concentration risk
        concentration_risk = self.calculate_concentration_risk(symbol, size, current_price)
        
        # Update position
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            position_type=position_type,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            volatility=volatility,
            var_95=var_95,
            max_drawdown=0.0,  # Will be updated in portfolio calculation
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            timestamp=datetime.now()
        )
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(current_price)
        
        # Keep only recent history
        max_history = max(self.volatility_lookback, self.correlation_lookback) * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    def calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        if len(self.positions) <= 1:
            return 0.0
        
        # Get price data for all symbols
        price_data = {}
        for pos_symbol in self.positions.keys():
            if pos_symbol in self.price_history and len(self.price_history[pos_symbol]) > 1:
                price_data[pos_symbol] = self.price_history[pos_symbol]
        
        if symbol not in price_data or len(price_data) < 2:
            return 0.0
        
        # Calculate correlations
        correlations = []
        for other_symbol in price_data:
            if other_symbol != symbol:
                # Align price series
                min_length = min(len(price_data[symbol]), len(price_data[other_symbol]))
                if min_length > 1:
                    corr = np.corrcoef(price_data[symbol][-min_length:], 
                                     price_data[other_symbol][-min_length:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        # Return average correlation risk
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_concentration_risk(self, symbol: str, size: float, price: float) -> float:
        """Calculate concentration risk for a position"""
        position_value = size * price
        total_portfolio_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        
        if total_portfolio_value == 0:
            return 0.0
        
        concentration = position_value / total_portfolio_value
        return concentration
    
    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        if not self.positions:
            return PortfolioRisk(
                total_value=self.current_capital,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                portfolio_volatility=0.0,
                portfolio_var_95=0.0,
                max_portfolio_drawdown=0.0,
                sharpe_ratio=0.0,
                correlation_matrix=pd.DataFrame(),
                concentration_risk=0.0,
                leverage_ratio=1.0,
                margin_utilization=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate total portfolio value and PnL
        total_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Update current capital
        self.current_capital = self.initial_capital + total_pnl
        
        # Calculate drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_historical_drawdown = max(self.max_historical_drawdown, self.current_drawdown)
        
        # Calculate portfolio volatility
        portfolio_volatility = self.calculate_portfolio_volatility()
        
        # Calculate portfolio VaR
        portfolio_var_95 = sum(pos.var_95 for pos in self.positions.values())
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix()
        
        # Calculate concentration risk
        concentration_risk = max(pos.concentration_risk for pos in self.positions.values())
        
        # Calculate leverage and margin utilization
        leverage_ratio = total_value / self.current_capital
        margin_utilization = total_pnl / self.current_capital if self.current_capital > 0 else 0.0
        
        portfolio_risk = PortfolioRisk(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            portfolio_volatility=portfolio_volatility,
            portfolio_var_95=portfolio_var_95,
            max_portfolio_drawdown=self.max_historical_drawdown,
            sharpe_ratio=sharpe_ratio,
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio,
            margin_utilization=margin_utilization,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.portfolio_history.append(portfolio_risk)
        
        return portfolio_risk
    
    def calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio-level volatility"""
        if len(self.positions) == 0:
            return 0.0
        
        # Calculate weighted average volatility
        total_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_volatility = 0.0
        for pos in self.positions.values():
            weight = (pos.size * pos.current_price) / total_value
            weighted_volatility += pos.volatility * weight
        
        return weighted_volatility
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for the portfolio"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
        return sharpe_ratio
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all positions"""
        symbols = list(self.positions.keys())
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Prepare price data
        price_data = {}
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                price_data[symbol] = self.price_history[symbol]
        
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # Calculate correlations
        correlation_data = {}
        for symbol1 in price_data:
            correlation_data[symbol1] = {}
            for symbol2 in price_data:
                if symbol1 == symbol2:
                    correlation_data[symbol1][symbol2] = 1.0
                else:
                    # Align price series
                    min_length = min(len(price_data[symbol1]), len(price_data[symbol2]))
                    if min_length > 1:
                        corr = np.corrcoef(price_data[symbol1][-min_length:], 
                                         price_data[symbol2][-min_length:])[0, 1]
                        correlation_data[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_data[symbol1][symbol2] = 0.0
        
        return pd.DataFrame(correlation_data)
    
    def check_risk_limits(self) -> List[Dict]:
        """Check if any risk limits are exceeded"""
        alerts = []
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_pct:
            alerts.append({
                'type': 'DRAWDOWN_LIMIT',
                'message': f'Drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}',
                'severity': 'HIGH',
                'timestamp': datetime.now()
            })
        
        # Check portfolio risk limit
        if portfolio_risk.portfolio_var_95 > (self.current_capital * self.max_portfolio_risk_pct):
            alerts.append({
                'type': 'PORTFOLIO_RISK_LIMIT',
                'message': f'Portfolio risk limit exceeded: VaR ${portfolio_risk.portfolio_var_95:.2f}',
                'severity': 'HIGH',
                'timestamp': datetime.now()
            })
        
        # Check concentration risk
        if portfolio_risk.concentration_risk > 0.2:  # 20% max concentration
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'message': f'High concentration risk: {portfolio_risk.concentration_risk:.2%}',
                'severity': 'MEDIUM',
                'timestamp': datetime.now()
            })
        
        # Check leverage ratio
        if portfolio_risk.leverage_ratio > 2.0:  # 2x max leverage
            alerts.append({
                'type': 'LEVERAGE_LIMIT',
                'message': f'High leverage ratio: {portfolio_risk.leverage_ratio:.2f}x',
                'severity': 'HIGH',
                'timestamp': datetime.now()
            })
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        return alerts
    
    def get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        portfolio_risk = self.calculate_portfolio_risk()
        
        # High correlation warning
        if not portfolio_risk.correlation_matrix.empty:
            high_corr_pairs = []
            for i in range(len(portfolio_risk.correlation_matrix)):
                for j in range(i+1, len(portfolio_risk.correlation_matrix)):
                    corr = portfolio_risk.correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        symbols = portfolio_risk.correlation_matrix.index
                        high_corr_pairs.append(f"{symbols[i]}-{symbols[j]}: {corr:.2f}")
            
            if high_corr_pairs:
                recommendations.append(f"Consider reducing positions with high correlation: {', '.join(high_corr_pairs)}")
        
        # Volatility-based recommendations
        high_vol_positions = [pos for pos in self.positions.values() if pos.volatility > 0.5]
        if high_vol_positions:
            recommendations.append(f"Consider reducing positions in high volatility assets: {[pos.symbol for pos in high_vol_positions]}")
        
        # Drawdown recommendations
        if self.current_drawdown > 0.1:  # 10% drawdown
            recommendations.append("Consider reducing position sizes or implementing tighter stops")
        
        # Diversification recommendations
        if len(self.positions) < 3:
            recommendations.append("Consider diversifying across more assets to reduce concentration risk")
        
        return recommendations
    
    async def start_monitoring(self):
        """Start continuous risk monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Risk monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous risk monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check risk limits
                alerts = self.check_risk_limits()
                
                # Log alerts
                for alert in alerts:
                    logger.warning(f"Risk Alert: {alert['message']}")
                
                # Update correlation matrix periodically
                if (self.last_correlation_update is None or 
                    (datetime.now() - self.last_correlation_update).seconds > 300):  # 5 minutes
                    self.calculate_correlation_matrix()
                    self.last_correlation_update = datetime.now()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        portfolio_risk = self.calculate_portfolio_risk()
        
        return {
            'portfolio_metrics': {
                'total_value': portfolio_risk.total_value,
                'total_pnl': portfolio_risk.total_pnl,
                'total_pnl_pct': portfolio_risk.total_pnl_pct,
                'portfolio_volatility': portfolio_risk.portfolio_volatility,
                'portfolio_var_95': portfolio_risk.portfolio_var_95,
                'max_drawdown': portfolio_risk.max_portfolio_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': portfolio_risk.sharpe_ratio,
                'leverage_ratio': portfolio_risk.leverage_ratio,
                'margin_utilization': portfolio_risk.margin_utilization
            },
            'position_metrics': {
                symbol: {
                    'size': pos.size,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'volatility': pos.volatility,
                    'var_95': pos.var_95,
                    'correlation_risk': pos.correlation_risk,
                    'concentration_risk': pos.concentration_risk
                }
                for symbol, pos in self.positions.items()
            },
            'risk_alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'recommendations': self.get_risk_recommendations(),
            'risk_level': self.risk_level.value,
            'monitoring_active': self.monitoring_active
        }
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.positions.clear()
        self.price_history.clear()
        self.portfolio_history.clear()
        self.risk_alerts.clear()
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        logger.info("Portfolio reset to initial state")

# Global risk manager instance
risk_manager = RiskManager()
