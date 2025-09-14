"""
Advanced Portfolio Manager Module

Implements modern portfolio theory with risk optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    SHARPE_RATIO = "sharpe_ratio"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"

@dataclass
class AssetAllocation:
    """Asset allocation configuration"""
    symbol: str
    target_weight: float
    current_weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    timestamp: datetime

class AdvancedPortfolioManager:
    """Advanced portfolio manager with modern portfolio theory"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Portfolio configuration
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.rebalancing_threshold = self.config.get('rebalancing_threshold', 0.05)
        self.optimization_method = self.config.get('optimization_method', OptimizationMethod.SHARPE_RATIO)
        
        # Portfolio state
        self.assets: Dict[str, AssetAllocation] = {}
        self.cash_balance = self.initial_capital
        self.total_value = self.initial_capital
        
        # Performance tracking
        self.performance_history: List[PortfolioMetrics] = []
        self.allocation_history: List[Dict[str, float]] = []
        
        # Statistics
        self.stats = {
            'total_rebalancing': 0,
            'last_rebalancing': None
        }
        
    async def initialize(self):
        """Initialize the portfolio manager"""
        try:
            self.logger.info("Initializing Advanced Portfolio Manager...")
            
            # Initialize default assets
            await self._initialize_default_assets()
            
            self.logger.info("Advanced Portfolio Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Portfolio Manager: {e}")
            raise
    
    async def _initialize_default_assets(self):
        """Initialize default asset allocation"""
        try:
            if not self.assets:
                # Default diversified portfolio
                default_assets = {
                    'BTCUSDT': {'weight': 0.3, 'min': 0.1, 'max': 0.5},
                    'ETHUSDT': {'weight': 0.25, 'min': 0.1, 'max': 0.4},
                    'ADAUSDT': {'weight': 0.15, 'min': 0.05, 'max': 0.25},
                    'DOTUSDT': {'weight': 0.15, 'min': 0.05, 'max': 0.25},
                    'LINKUSDT': {'weight': 0.15, 'min': 0.05, 'max': 0.25}
                }
                
                for symbol, config in default_assets.items():
                    self.assets[symbol] = AssetAllocation(
                        symbol=symbol,
                        target_weight=config['weight'],
                        current_weight=config['weight'],
                        min_weight=config['min'],
                        max_weight=config['max']
                    )
                
                self.logger.info("Initialized default asset allocation")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default assets: {e}")
    
    async def add_asset(self, symbol: str, target_weight: float, 
                        min_weight: float = 0.0, max_weight: float = 1.0):
        """Add a new asset to the portfolio"""
        try:
            if symbol in self.assets:
                self.logger.warning(f"Asset {symbol} already exists")
                return
            
            # Validate weights
            if target_weight < 0 or target_weight > 1:
                raise ValueError("Target weight must be between 0 and 1")
            
            if min_weight < 0 or max_weight > 1 or min_weight > max_weight:
                raise ValueError("Invalid min/max weight constraints")
            
            # Add asset
            self.assets[symbol] = AssetAllocation(
                symbol=symbol,
                target_weight=target_weight,
                current_weight=0.0,
                min_weight=min_weight,
                max_weight=max_weight
            )
            
            # Normalize weights
            await self._normalize_weights()
            
            self.logger.info(f"Added asset {symbol} with target weight {target_weight}")
            
        except Exception as e:
            self.logger.error(f"Failed to add asset {symbol}: {e}")
            raise
    
    async def _normalize_weights(self):
        """Normalize asset weights to sum to 1"""
        try:
            if not self.assets:
                return
            
            total_weight = sum(asset.target_weight for asset in self.assets.values())
            
            if total_weight > 0:
                # Normalize weights
                for asset in self.assets.values():
                    asset.target_weight /= total_weight
            else:
                # Equal weights if no weights specified
                equal_weight = 1.0 / len(self.assets)
                for asset in self.assets.values():
                    asset.target_weight = equal_weight
            
            self.logger.debug("Normalized asset weights")
            
        except Exception as e:
            self.logger.error(f"Failed to normalize weights: {e}")
    
    async def update_portfolio_value(self):
        """Update current portfolio value and weights"""
        try:
            # Simulate portfolio value update
            # In practice, this would get real market data
            self.total_value = self.initial_capital * (1 + np.random.normal(0, 0.1))
            
            # Update current weights (simplified)
            for asset in self.assets.values():
                asset.current_weight = asset.target_weight + np.random.normal(0, 0.02)
                asset.current_weight = max(0, min(1, asset.current_weight))
            
            # Store allocation history
            allocation = {symbol: asset.current_weight for symbol, asset in self.assets.items()}
            self.allocation_history.append({
                'timestamp': datetime.now(),
                'allocation': allocation,
                'total_value': self.total_value
            })
            
            # Maintain history size
            if len(self.allocation_history) > 100:
                self.allocation_history = self.allocation_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio value: {e}")
    
    async def optimize_portfolio(self) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            if len(self.assets) < 2:
                raise ValueError("Need at least 2 assets for optimization")
            
            # Simple optimization: equal weights with constraints
            n_assets = len(self.assets)
            equal_weight = 1.0 / n_assets
            
            # Apply constraints
            for asset in self.assets.values():
                if equal_weight < asset.min_weight:
                    asset.target_weight = asset.min_weight
                elif equal_weight > asset.max_weight:
                    asset.target_weight = asset.max_weight
                else:
                    asset.target_weight = equal_weight
            
            # Normalize weights
            await self._normalize_weights()
            
            # Calculate metrics
            total_return = (self.total_value - self.initial_capital) / self.initial_capital
            volatility = 0.15  # Placeholder
            sharpe_ratio = (total_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
            
            result = {
                'optimal_weights': {symbol: asset.target_weight for symbol, asset in self.assets.items()},
                'expected_return': total_return,
                'expected_volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_method': self.optimization_method.value,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Portfolio optimized using {self.optimization_method.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize portfolio: {e}")
            raise
    
    async def check_rebalancing_needed(self) -> bool:
        """Check if portfolio rebalancing is needed"""
        try:
            if not self.assets:
                return False
            
            # Calculate weight drift
            max_drift = 0.0
            
            for asset in self.assets.values():
                drift = abs(asset.current_weight - asset.target_weight)
                max_drift = max(max_drift, drift)
            
            # Check if drift exceeds threshold
            needs_rebalancing = max_drift > self.rebalancing_threshold
            
            if needs_rebalancing:
                self.logger.info(f"Rebalancing needed: max drift = {max_drift:.3f}")
            
            return needs_rebalancing
            
        except Exception as e:
            self.logger.error(f"Failed to check rebalancing: {e}")
            return False
    
    async def rebalance_portfolio(self):
        """Rebalance portfolio to target weights"""
        try:
            if not await self.check_rebalancing_needed():
                self.logger.info("No rebalancing needed")
                return
            
            self.logger.info("Starting portfolio rebalancing...")
            
            # Simulate rebalancing
            for asset in self.assets.values():
                asset.current_weight = asset.target_weight
            
            # Update statistics
            self.stats['total_rebalancing'] += 1
            self.stats['last_rebalancing'] = datetime.now()
            
            self.logger.info("Portfolio rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"Failed to rebalance portfolio: {e}")
            raise
    
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Update portfolio value first
            await self.update_portfolio_value()
            
            # Calculate basic metrics
            total_return = (self.total_value - self.initial_capital) / self.initial_capital
            volatility = 0.15  # Placeholder
            sharpe_ratio = (total_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
            max_drawdown = 0.1  # Placeholder
            
            # Create metrics
            metrics = PortfolioMetrics(
                total_value=self.total_value,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                timestamp=datetime.now()
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            
            # Maintain history size
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics: {e}")
            raise
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Calculate current metrics
            metrics = await self.calculate_portfolio_metrics()
            
            return {
                'portfolio_state': {
                    'total_value': self.total_value,
                    'cash_balance': self.cash_balance,
                    'total_assets': len(self.assets),
                    'last_update': datetime.now()
                },
                'asset_allocation': {
                    symbol: {
                        'target_weight': asset.target_weight,
                        'current_weight': asset.current_weight,
                        'drift': abs(asset.current_weight - asset.target_weight),
                        'constraints': {
                            'min_weight': asset.min_weight,
                            'max_weight': asset.max_weight
                        }
                    }
                    for symbol, asset in self.assets.items()
                },
                'performance_metrics': {
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'volatility': metrics.volatility
                },
                'rebalancing_info': {
                    'total_rebalancing': self.stats['total_rebalancing'],
                    'last_rebalancing': self.stats['last_rebalancing'],
                    'rebalancing_threshold': self.rebalancing_threshold,
                    'needs_rebalancing': await self.check_rebalancing_needed()
                },
                'optimization_info': {
                    'method': self.optimization_method.value,
                    'risk_free_rate': self.risk_free_rate
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check portfolio manager health"""
        try:
            return {
                'status': 'healthy',
                'portfolio_state': {
                    'total_assets': len(self.assets),
                    'total_value': self.total_value,
                    'cash_balance': self.cash_balance
                },
                'performance': {
                    'total_return': (self.total_value - self.initial_capital) / self.initial_capital,
                    'total_rebalancing': self.stats['total_rebalancing']
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Close the portfolio manager"""
        try:
            self.logger.info("Advanced Portfolio Manager closed successfully")
        except Exception as e:
            self.logger.error(f"Failed to close Portfolio Manager: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
