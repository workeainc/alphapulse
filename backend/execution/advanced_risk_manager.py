"""
Advanced Risk Manager Module

Implements comprehensive risk management with:
- VaR and CVaR calculations
- Stress testing and scenario analysis
- Dynamic risk limits and position sizing
- Correlation and concentration risk monitoring
- Real-time risk alerts
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskMetric(Enum):
    """Risk metrics types"""
    VAR = "var"
    CVAR = "cvar"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    metric: RiskMetric
    threshold: float
    time_horizon: str  # 'daily', 'weekly', 'monthly'
    action: str  # 'warn', 'reduce', 'stop'
    enabled: bool = True

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    max_drawdown: float
    correlation_risk: float
    concentration_risk: float
    beta: float
    sharpe_ratio: float
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk alert information"""
    level: RiskLevel
    metric: RiskMetric
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    action_required: str

class AdvancedRiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Risk configuration
        self.var_confidence = self.config.get('var_confidence', 0.95)
        self.lookback_period = self.config.get('lookback_period', 252)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Risk limits
        self.risk_limits: Dict[str, RiskLimit] = {}
        self._initialize_risk_limits()
        
        # Risk state
        self.current_risk_metrics: Optional[RiskMetrics] = None
        self.risk_alerts: List[RiskAlert] = []
        self.risk_history: List[RiskMetrics] = []
        
        # Stress testing configuration (Week 9 enhancement)
        self.stress_scenarios = {
            'market_crash': {'price_drop': 0.3, 'volatility_spike': 3.0, 'correlation_increase': 0.3},
            'liquidity_crisis': {'volume_drop': 0.7, 'spread_widening': 5.0, 'funding_rate_spike': 0.01},
            'flash_crash': {'price_drop': 0.15, 'duration': 300, 'recovery_time': 1800},
            'regulatory_shock': {'position_limit_reduction': 0.5, 'margin_increase': 2.0},
            'black_swan': {'price_drop': 0.5, 'volatility_spike': 5.0, 'correlation_increase': 0.5}
        }
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_value = 0.0
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'risk_checks': 0,
            'limit_violations': 0,
            'last_risk_check': None
        }
        
    def _initialize_risk_limits(self):
        """Initialize default risk limits"""
        try:
            self.risk_limits = {
                'var_daily': RiskLimit(
                    metric=RiskMetric.VAR,
                    threshold=0.02,  # 2% daily VaR
                    time_horizon='daily',
                    action='warn'
                ),
                'var_weekly': RiskLimit(
                    metric=RiskMetric.VAR,
                    threshold=0.05,  # 5% weekly VaR
                    time_horizon='weekly',
                    action='reduce'
                ),
                'drawdown': RiskLimit(
                    metric=RiskMetric.DRAWDOWN,
                    threshold=0.15,  # 15% max drawdown
                    time_horizon='daily',
                    action='stop'
                ),
                'concentration': RiskLimit(
                    metric=RiskMetric.CONCENTRATION,
                    threshold=0.25,  # 25% max concentration
                    time_horizon='daily',
                    action='warn'
                ),
                'correlation': RiskLimit(
                    metric=RiskMetric.CORRELATION,
                    threshold=0.8,  # 80% max correlation
                    time_horizon='daily',
                    action='warn'
                )
            }
            
            self.logger.info("Risk limits initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk limits: {e}")
    
    async def initialize(self):
        """Initialize the risk manager"""
        try:
            self.logger.info("Initializing Advanced Risk Manager...")
            
            # Load historical risk data if available
            await self._load_risk_history()
            
            self.logger.info("Advanced Risk Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Manager: {e}")
            raise
    
    async def _load_risk_history(self):
        """Load historical risk data"""
        try:
            # For now, we'll create mock history
            # In practice, this would load from database
            self.risk_history = []
            
        except Exception as e:
            self.logger.error(f"Failed to load risk history: {e}")
    
    async def update_positions(self, positions: Dict[str, Dict[str, Any]], 
                              portfolio_value: float):
        """Update current positions and portfolio value"""
        try:
            self.positions = positions
            self.portfolio_value = portfolio_value
            
            self.logger.debug(f"Updated positions: {len(positions)} positions, portfolio value: {portfolio_value}")
            
        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}")
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate VaR and CVaR
            var_95, var_99, cvar_95, cvar_99 = await self._calculate_var_cvar()
            
            # Calculate volatility
            volatility = await self._calculate_volatility()
            
            # Calculate drawdown
            max_drawdown = await self._calculate_max_drawdown()
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk()
            
            # Calculate concentration risk
            concentration_risk = await self._calculate_concentration_risk()
            
            # Calculate beta (simplified)
            beta = await self._calculate_beta()
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Create risk metrics
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                max_drawdown=max_drawdown,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                timestamp=datetime.now()
            )
            
            # Store metrics
            self.current_risk_metrics = metrics
            self.risk_history.append(metrics)
            
            # Maintain history size
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            # Update statistics
            self.stats['risk_checks'] += 1
            self.stats['last_risk_check'] = datetime.now()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk metrics: {e}")
            raise
    
    async def _calculate_var_cvar(self) -> Tuple[float, float, float, float]:
        """Calculate VaR and CVaR"""
        try:
            if not self.positions:
                return 0.0, 0.0, 0.0, 0.0
            
            # Simulate portfolio returns for VaR calculation
            # In practice, this would use historical data
            np.random.seed(42)
            n_simulations = 10000
            
            # Generate correlated returns for all positions
            portfolio_returns = []
            
            for _ in range(n_simulations):
                daily_return = 0.0
                for symbol, position in self.positions.items():
                    # Simulate individual asset returns
                    asset_return = np.random.normal(0, 0.02)  # 2% daily volatility
                    position_value = position.get('value', 0)
                    weight = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
                    daily_return += weight * asset_return
                
                portfolio_returns.append(daily_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate VaR
            var_95 = np.percentile(portfolio_returns, (1 - 0.95) * 100)
            var_99 = np.percentile(portfolio_returns, (1 - 0.99) * 100)
            
            # Calculate CVaR
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            return var_95, var_99, cvar_95, cvar_99
            
        except Exception as e:
            self.logger.error(f"Failed to calculate VaR/CVaR: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    async def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(self.risk_history) < 30:
                return 0.15  # Default volatility
            
            # Calculate volatility from recent risk history
            recent_returns = []
            
            for i in range(1, min(31, len(self.risk_history))):
                prev_value = self.risk_history[i-1].var_95
                curr_value = self.risk_history[i].var_95
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    recent_returns.append(daily_return)
            
            if recent_returns:
                volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
                return volatility
            
            return 0.15  # Default volatility
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volatility: {e}")
            return 0.15
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(self.risk_history) < 2:
                return 0.0
            
            # Calculate drawdown from portfolio values
            # For now, use VaR as proxy for portfolio value
            var_values = [metrics.var_95 for metrics in self.risk_history]
            
            peak = var_values[0]
            max_dd = 0.0
            
            for value in var_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Failed to calculate max drawdown: {e}")
            return 0.0
    
    async def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # Simulate correlation matrix
            # In practice, this would use historical correlation data
            n_assets = len(self.positions)
            
            if n_assets == 1:
                return 0.0
            
            # Generate mock correlation matrix
            np.random.seed(42)
            correlations = []
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    # Simulate correlation between assets
                    correlation = np.random.uniform(-0.3, 0.8)
                    correlations.append(abs(correlation))
            
            # Return average correlation
            if correlations:
                return np.mean(correlations)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation risk: {e}")
            return 0.0
    
    async def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        try:
            if not self.positions or self.portfolio_value <= 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            weights = []
            
            for position in self.positions.values():
                position_value = position.get('value', 0)
                weight = position_value / self.portfolio_value
                weights.append(weight)
            
            if weights:
                hhi = sum(w**2 for w in weights)
                # Normalize HHI to 0-1 scale
                concentration_risk = min(hhi, 1.0)
                return concentration_risk
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate concentration risk: {e}")
            return 0.0
    
    async def _calculate_beta(self) -> float:
        """Calculate portfolio beta"""
        try:
            # Simplified beta calculation
            # In practice, this would compare to market returns
            if len(self.risk_history) < 30:
                return 1.0  # Default beta
            
            # Use volatility as proxy for beta
            volatility = await self._calculate_volatility()
            market_volatility = 0.15  # Assumed market volatility
            
            beta = volatility / market_volatility if market_volatility > 0 else 1.0
            return min(max(beta, 0.5), 2.0)  # Constrain beta
            
        except Exception as e:
            self.logger.error(f"Failed to calculate beta: {e}")
            return 1.0
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.risk_history) < 30:
                return 0.0
            
            # Calculate average return and volatility
            recent_returns = []
            
            for i in range(1, min(31, len(self.risk_history))):
                prev_value = self.risk_history[i-1].var_95
                curr_value = self.risk_history[i].var_95
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    recent_returns.append(daily_return)
            
            if recent_returns:
                avg_return = np.mean(recent_returns) * 252  # Annualized
                volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
                
                if volatility > 0:
                    sharpe_ratio = (avg_return - 0.02) / volatility  # 2% risk-free rate
                    return sharpe_ratio
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    async def check_risk_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts"""
        try:
            if not self.current_risk_metrics:
                await self.calculate_risk_metrics()
            
            alerts = []
            
            # Check VaR limits
            for limit_name, limit in self.risk_limits.items():
                if not limit.enabled:
                    continue
                
                if limit.metric == RiskMetric.VAR:
                    current_value = self.current_risk_metrics.var_95
                    if current_value > limit.threshold:
                        alert = RiskAlert(
                            level=RiskLevel.HIGH if current_value > limit.threshold * 1.5 else RiskLevel.MEDIUM,
                            metric=limit.metric,
                            current_value=current_value,
                            threshold=limit.threshold,
                            message=f"VaR limit exceeded: {current_value:.4f} > {limit.threshold:.4f}",
                            timestamp=datetime.now(),
                            action_required=limit.action
                        )
                        alerts.append(alert)
                
                elif limit.metric == RiskMetric.DRAWDOWN:
                    current_value = self.current_risk_metrics.max_drawdown
                    if current_value > limit.threshold:
                        alert = RiskAlert(
                            level=RiskLevel.CRITICAL if current_value > limit.threshold * 1.2 else RiskLevel.HIGH,
                            metric=limit.metric,
                            current_value=current_value,
                            threshold=limit.threshold,
                            message=f"Drawdown limit exceeded: {current_value:.4f} > {limit.threshold:.4f}",
                            timestamp=datetime.now(),
                            action_required=limit.action
                        )
                        alerts.append(alert)
                
                elif limit.metric == RiskMetric.CONCENTRATION:
                    current_value = self.current_risk_metrics.concentration_risk
                    if current_value > limit.threshold:
                        alert = RiskAlert(
                            level=RiskLevel.MEDIUM,
                            metric=limit.metric,
                            current_value=current_value,
                            threshold=limit.threshold,
                            message=f"Concentration risk high: {current_value:.4f} > {limit.threshold:.4f}",
                            timestamp=datetime.now(),
                            action_required=limit.action
                        )
                        alerts.append(alert)
                
                elif limit.metric == RiskMetric.CORRELATION:
                    current_value = self.current_risk_metrics.correlation_risk
                    if current_value > limit.threshold:
                        alert = RiskAlert(
                            level=RiskLevel.MEDIUM,
                            metric=limit.metric,
                            current_value=current_value,
                            threshold=limit.threshold,
                            message=f"Correlation risk high: {current_value:.4f} > {limit.threshold:.4f}",
                            timestamp=datetime.now(),
                            action_required=limit.action
                        )
                        alerts.append(alert)
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Update statistics
            self.stats['total_alerts'] += len(alerts)
            if alerts:
                self.stats['limit_violations'] += 1
            
            # Maintain alert history
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check risk limits: {e}")
            return []
    
    async def calculate_position_size(self, symbol: str, signal_strength: float, 
                                    current_price: float) -> float:
        """Calculate optimal position size based on risk"""
        try:
            if not self.current_risk_metrics:
                await self.calculate_risk_metrics()
            
            # Base position size
            base_size = self.portfolio_value * self.max_position_size
            
            # Adjust for signal strength
            adjusted_size = base_size * signal_strength
            
            # Adjust for current risk level
            current_var = self.current_risk_metrics.var_95
            risk_adjustment = 1.0 - (current_var / self.max_portfolio_risk)
            risk_adjustment = max(0.1, min(1.0, risk_adjustment))  # Constrain to 0.1-1.0
            
            final_size = adjusted_size * risk_adjustment
            
            # Convert to quantity
            quantity = final_size / current_price if current_price > 0 else 0
            
            self.logger.info(f"Calculated position size for {symbol}: {quantity:.4f} (risk adjusted)")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    async def validate_trade(self, symbol: str, side: str, quantity: float, 
                           price: float) -> Tuple[bool, str]:
        """Validate if a trade meets risk requirements"""
        try:
            # Check position size limits
            trade_value = quantity * price
            position_weight = trade_value / self.portfolio_value if self.portfolio_value > 0 else 0
            
            if position_weight > self.max_position_size:
                return False, f"Position size {position_weight:.2%} exceeds limit {self.max_position_size:.2%}"
            
            # Check portfolio risk limits
            if self.current_risk_metrics:
                if self.current_risk_metrics.var_95 > self.max_portfolio_risk:
                    return False, f"Portfolio VaR {self.current_risk_metrics.var_95:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            
            # Check concentration risk
            if self.current_risk_metrics:
                if self.current_risk_metrics.concentration_risk > 0.3:  # 30% concentration limit
                    return False, f"Concentration risk {self.current_risk_metrics.concentration_risk:.2%} too high"
            
            return True, "Trade validated successfully"
            
        except Exception as e:
            self.logger.error(f"Failed to validate trade: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            if not self.current_risk_metrics:
                await self.calculate_risk_metrics()
            
            # Check current risk limits
            current_alerts = await self.check_risk_limits()
            
            return {
                'current_metrics': {
                    'var_95': self.current_risk_metrics.var_95,
                    'var_99': self.current_risk_metrics.var_99,
                    'cvar_95': self.current_risk_metrics.cvar_95,
                    'cvar_99': self.current_risk_metrics.cvar_99,
                    'volatility': self.current_risk_metrics.volatility,
                    'max_drawdown': self.current_risk_metrics.max_drawdown,
                    'correlation_risk': self.current_risk_metrics.correlation_risk,
                    'concentration_risk': self.current_risk_metrics.concentration_risk,
                    'beta': self.current_risk_metrics.beta,
                    'sharpe_ratio': self.current_risk_metrics.sharpe_ratio
                },
                'risk_limits': {
                    name: {
                        'threshold': limit.threshold,
                        'time_horizon': limit.time_horizon,
                        'action': limit.action,
                        'enabled': limit.enabled
                    }
                    for name, limit in self.risk_limits.items()
                },
                'current_alerts': [
                    {
                        'level': alert.level.value,
                        'metric': alert.metric.value,
                        'message': alert.message,
                        'action_required': alert.action_required,
                        'timestamp': alert.timestamp
                    }
                    for alert in current_alerts
                ],
                'risk_statistics': {
                    'total_alerts': self.stats['total_alerts'],
                    'risk_checks': self.stats['risk_checks'],
                    'limit_violations': self.stats['limit_violations'],
                    'last_risk_check': self.stats['last_risk_check']
                },
                'portfolio_info': {
                    'total_positions': len(self.positions),
                    'portfolio_value': self.portfolio_value,
                    'max_position_size': self.max_position_size,
                    'max_portfolio_risk': self.max_portfolio_risk
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get risk summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check risk manager health"""
        try:
            return {
                'status': 'healthy',
                'risk_state': {
                    'current_metrics_available': self.current_risk_metrics is not None,
                    'total_positions': len(self.positions),
                    'portfolio_value': self.portfolio_value
                },
                'performance': {
                    'total_alerts': self.stats['total_alerts'],
                    'risk_checks': self.stats['risk_checks'],
                    'limit_violations': self.stats['limit_violations']
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def run_stress_test(self, scenario_name: str = 'market_crash', 
                             custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run stress test with specified scenario (Week 9 enhancement)"""
        try:
            scenario = custom_params or self.stress_scenarios.get(scenario_name, {})
            if not scenario:
                raise ValueError(f"Unknown stress scenario: {scenario_name}")
            
            self.logger.info(f"Running stress test: {scenario_name}")
            
            # Simulate stress conditions
            stress_results = await self._simulate_stress_conditions(scenario)
            
            # Calculate stressed risk metrics
            stressed_metrics = await self._calculate_stressed_metrics(scenario)
            
            # Generate stress test report
            report = {
                'scenario_name': scenario_name,
                'scenario_params': scenario,
                'stressed_metrics': stressed_metrics,
                'impact_analysis': stress_results,
                'recommendations': self._generate_stress_recommendations(stressed_metrics),
                'timestamp': datetime.now(),
                'risk_level': self._assess_stress_risk_level(stressed_metrics)
            }
            
            # Store stress test results
            await self._store_stress_test_result(report)
            
            self.logger.info(f"Stress test completed: {scenario_name}")
            return report
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            return {'error': str(e)}
    
    async def _simulate_stress_conditions(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate market stress conditions"""
        try:
            results = {}
            
            # Price impact simulation
            if 'price_drop' in scenario:
                current_prices = {pos: self.positions[pos].get('current_price', 0) for pos in self.positions}
                stressed_prices = {pos: price * (1 - scenario['price_drop']) for pos, price in current_prices.items()}
                
                # Calculate PnL impact
                pnl_impact = {}
                for pos, current_price in current_prices.items():
                    if pos in self.positions:
                        position_size = self.positions[pos].get('size', 0)
                        price_change = stressed_prices[pos] - current_price
                        pnl_impact[pos] = position_size * price_change
                
                results['price_impact'] = {
                    'stressed_prices': stressed_prices,
                    'pnl_impact': pnl_impact,
                    'total_pnl_impact': sum(pnl_impact.values())
                }
            
            # Volatility impact simulation
            if 'volatility_spike' in scenario:
                current_vol = self.current_risk_metrics.volatility if self.current_risk_metrics else 0.2
                stressed_vol = current_vol * scenario['volatility_spike']
                
                # Adjust VaR and CVaR for higher volatility
                vol_multiplier = scenario['volatility_spike']
                results['volatility_impact'] = {
                    'current_volatility': current_vol,
                    'stressed_volatility': stressed_vol,
                    'var_adjustment_factor': vol_multiplier,
                    'cvar_adjustment_factor': vol_multiplier * 1.5
                }
            
            # Liquidity impact simulation
            if 'volume_drop' in scenario:
                results['liquidity_impact'] = {
                    'volume_reduction': scenario['volume_drop'],
                    'estimated_slippage': scenario['volume_drop'] * 0.1,  # 10% of volume drop
                    'execution_risk': 'high' if scenario['volume_drop'] > 0.5 else 'medium'
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error simulating stress conditions: {e}")
            return {}
    
    async def _calculate_stressed_metrics(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics under stress conditions"""
        try:
            if not self.current_risk_metrics:
                await self.calculate_risk_metrics()
            
            stressed_metrics = {}
            base_metrics = self.current_risk_metrics
            
            # Apply stress multipliers
            if 'volatility_spike' in scenario:
                vol_multiplier = scenario['volatility_spike']
                stressed_metrics['var_95'] = base_metrics.var_95 * vol_multiplier
                stressed_metrics['var_99'] = base_metrics.var_99 * vol_multiplier
                stressed_metrics['cvar_95'] = base_metrics.cvar_95 * vol_multiplier
                stressed_metrics['cvar_99'] = base_metrics.cvar_99 * vol_multiplier
                stressed_metrics['volatility'] = base_metrics.volatility * vol_multiplier
            else:
                stressed_metrics.update({
                    'var_95': base_metrics.var_95,
                    'var_99': base_metrics.var_99,
                    'cvar_95': base_metrics.cvar_95,
                    'cvar_99': base_metrics.cvar_99,
                    'volatility': base_metrics.volatility
                })
            
            # Apply correlation stress
            if 'correlation_increase' in scenario:
                base_correlation = base_metrics.correlation_risk
                stressed_metrics['correlation_risk'] = min(1.0, base_correlation + scenario['correlation_increase'])
            else:
                stressed_metrics['correlation_risk'] = base_metrics.correlation_risk
            
            # Copy other metrics
            stressed_metrics.update({
                'max_drawdown': base_metrics.max_drawdown,
                'beta': base_metrics.beta,
                'sharpe_ratio': base_metrics.sharpe_ratio,
                'concentration_risk': base_metrics.concentration_risk
            })
            
            return stressed_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating stressed metrics: {e}")
            return {}
    
    def _generate_stress_recommendations(self, stressed_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        # VaR-based recommendations
        if stressed_metrics.get('var_99', 0) > self.max_portfolio_risk * 2:
            recommendations.append("Immediate position reduction required - VaR exceeds 2x portfolio risk limit")
        elif stressed_metrics.get('var_99', 0) > self.max_portfolio_risk:
            recommendations.append("Consider position reduction - VaR exceeds portfolio risk limit")
        
        # Volatility-based recommendations
        if stressed_metrics.get('volatility', 0) > 0.4:  # 40% volatility
            recommendations.append("High volatility detected - implement dynamic stop-losses and reduce leverage")
        
        # Correlation-based recommendations
        if stressed_metrics.get('correlation_risk', 0) > 0.8:
            recommendations.append("High correlation risk - diversify portfolio across uncorrelated assets")
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append("Portfolio appears resilient to current stress scenario")
        
        return recommendations
    
    def _assess_stress_risk_level(self, stressed_metrics: Dict[str, Any]) -> RiskLevel:
        """Assess overall risk level based on stress test results"""
        try:
            risk_score = 0
            
            # VaR risk scoring
            if stressed_metrics.get('var_99', 0) > self.max_portfolio_risk * 2:
                risk_score += 3
            elif stressed_metrics.get('var_99', 0) > self.max_portfolio_risk:
                risk_score += 2
            
            # Volatility risk scoring
            if stressed_metrics.get('volatility', 0) > 0.4:
                risk_score += 2
            elif stressed_metrics.get('volatility', 0) > 0.2:
                risk_score += 1
            
            # Correlation risk scoring
            if stressed_metrics.get('correlation_risk', 0) > 0.8:
                risk_score += 2
            elif stressed_metrics.get('correlation_risk', 0) > 0.6:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 5:
                return RiskLevel.CRITICAL
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error assessing stress risk level: {e}")
            return RiskLevel.MEDIUM
    
    async def _store_stress_test_result(self, report: Dict[str, Any]):
        """Store stress test result for historical analysis"""
        try:
            # Store in memory for quick access
            if not hasattr(self, 'stress_test_history'):
                self.stress_test_history = []
            
            self.stress_test_history.append(report)
            
            # Keep only last 100 results
            if len(self.stress_test_history) > 100:
                self.stress_test_history = self.stress_test_history[-100:]
            
            self.logger.info(f"Stress test result stored: {report['scenario_name']}")
            
        except Exception as e:
            self.logger.error(f"Error storing stress test result: {e}")
    
    async def get_stress_test_summary(self) -> Dict[str, Any]:
        """Get summary of all stress tests run"""
        try:
            if not hasattr(self, 'stress_test_history'):
                return {'stress_tests': [], 'summary': 'No stress tests run'}
            
            if not self.stress_test_history:
                return {'stress_tests': [], 'summary': 'No stress tests run'}
            
            # Calculate summary statistics
            scenarios_run = [test['scenario_name'] for test in self.stress_test_history]
            risk_levels = [test['risk_level'].value for test in self.stress_test_history]
            
            summary = {
                'total_tests': len(self.stress_test_history),
                'scenarios_run': list(set(scenarios_run)),
                'risk_level_distribution': {
                    level: risk_levels.count(level) for level in set(risk_levels)
                },
                'recent_tests': [
                    {
                        'scenario': test['scenario_name'],
                        'risk_level': test['risk_level'].value,
                        'timestamp': test['timestamp'].isoformat()
                    }
                    for test in self.stress_test_history[-5:]  # Last 5 tests
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting stress test summary: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close the risk manager"""
        try:
            self.logger.info("Advanced Risk Manager closed successfully")
        except Exception as e:
            self.logger.error(f"Failed to close Risk Manager: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
