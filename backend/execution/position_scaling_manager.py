#!/usr/bin/env python3
"""
Position Scaling Manager for AlphaPulse
Handles advanced position scaling algorithms and automation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Position scaling strategy types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    CORRELATION_BASED = "correlation_based"

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    PRICE_LEVEL = "price_level"
    TIME_BASED = "time_based"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    CORRELATION = "correlation"
    VOLUME = "volume"

@dataclass
class ScalingLevel:
    """Individual scaling level configuration"""
    level: int
    trigger_price: float
    quantity: float
    trigger_type: ScalingTrigger
    trigger_value: float
    executed: bool = False
    execution_time: Optional[datetime] = None
    execution_price: Optional[float] = None

@dataclass
class ScalingPlan:
    """Complete scaling plan for a position"""
    position_id: str
    symbol: str
    side: str
    base_quantity: float
    total_quantity: float
    entry_price: float
    strategy: ScalingStrategy
    levels: List[ScalingLevel]
    max_levels: int = 5
    max_total_quantity: Optional[float] = None
    created_at: datetime = None
    active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ScalingExecution:
    """Record of scaling execution"""
    scaling_level_id: str
    position_id: str
    level: int
    quantity: float
    execution_price: float
    execution_time: datetime
    trigger_type: ScalingTrigger
    trigger_value: float
    slippage: float = 0.0
    commission: float = 0.0

class PositionScalingManager:
    """Manages advanced position scaling algorithms"""
    
    def __init__(self):
        # Scaling strategies configuration
        self.scaling_strategies = {
            ScalingStrategy.LINEAR: {
                "description": "Linear scaling with equal intervals",
                "default_levels": 3,
                "default_interval": 0.02  # 2% intervals
            },
            ScalingStrategy.EXPONENTIAL: {
                "description": "Exponential scaling with increasing intervals",
                "default_levels": 4,
                "default_base_interval": 0.015  # 1.5% base interval
            },
            ScalingStrategy.FIBONACCI: {
                "description": "Fibonacci-based scaling levels",
                "default_levels": 5,
                "default_base_interval": 0.01  # 1% base interval
            },
            ScalingStrategy.VOLATILITY_BASED: {
                "description": "Scaling based on volatility levels",
                "default_levels": 3,
                "volatility_multiplier": 2.0
            },
            ScalingStrategy.MOMENTUM_BASED: {
                "description": "Scaling based on momentum indicators",
                "default_levels": 3,
                "momentum_threshold": 0.02
            }
        }
        
        # Active scaling plans
        self.active_plans: Dict[str, ScalingPlan] = {}
        
        # Scaling execution history
        self.execution_history: List[ScalingExecution] = []
        
        # Performance tracking
        self.total_scaling_events = 0
        self.successful_scaling_events = 0
        self.failed_scaling_events = 0
        
        logger.info("Position Scaling Manager initialized")
    
    def create_scaling_plan(self, position_id: str, symbol: str, side: str,
                           base_quantity: float, entry_price: float,
                           strategy: ScalingStrategy, **kwargs) -> ScalingPlan:
        """Create a new scaling plan"""
        
        # Get strategy configuration
        strategy_config = self.scaling_strategies.get(strategy, {})
        default_levels = strategy_config.get("default_levels", 3)
        
        # Override with kwargs if provided
        levels = kwargs.get("levels", default_levels)
        max_quantity = kwargs.get("max_total_quantity")
        
        # Generate scaling levels based on strategy
        # Remove levels from kwargs to avoid duplicate argument error
        kwargs_copy = kwargs.copy()
        if 'levels' in kwargs_copy:
            del kwargs_copy['levels']
        
        scaling_levels = self._generate_scaling_levels(
            strategy, entry_price, side, levels, **kwargs_copy
        )
        
        # Calculate total quantity
        total_quantity = base_quantity + sum(level.quantity for level in scaling_levels)
        
        # Create scaling plan
        plan = ScalingPlan(
            position_id=position_id,
            symbol=symbol,
            side=side,
            base_quantity=base_quantity,
            total_quantity=total_quantity,
            entry_price=entry_price,
            strategy=strategy,
            levels=scaling_levels,
            max_levels=levels,
            max_total_quantity=max_quantity
        )
        
        # Store active plan
        self.active_plans[position_id] = plan
        
        logger.info(f"Created scaling plan for {symbol} {side} position: {strategy.value}")
        return plan
    
    def _generate_scaling_levels(self, strategy: ScalingStrategy, entry_price: float,
                                side: str, levels: int, **kwargs) -> List[ScalingLevel]:
        """Generate scaling levels based on strategy"""
        
        if strategy == ScalingStrategy.LINEAR:
            return self._generate_linear_levels(entry_price, side, levels, **kwargs)
        elif strategy == ScalingStrategy.EXPONENTIAL:
            return self._generate_exponential_levels(entry_price, side, levels, **kwargs)
        elif strategy == ScalingStrategy.FIBONACCI:
            return self._generate_fibonacci_levels(entry_price, side, levels, **kwargs)
        elif strategy == ScalingStrategy.VOLATILITY_BASED:
            return self._generate_volatility_levels(entry_price, side, levels, **kwargs)
        elif strategy == ScalingStrategy.MOMENTUM_BASED:
            return self._generate_momentum_levels(entry_price, side, levels, **kwargs)
        else:
            raise ValueError(f"Unsupported scaling strategy: {strategy}")
    
    def _generate_linear_levels(self, entry_price: float, side: str, levels: int,
                               **kwargs) -> List[ScalingLevel]:
        """Generate linear scaling levels"""
        interval = kwargs.get("interval", 0.02)  # 2% intervals
        base_quantity = kwargs.get("base_quantity", 100.0)
        
        scaling_levels = []
        
        for i in range(1, levels + 1):
            if side.lower() == "buy":
                # Scale in on pullbacks (lower prices)
                trigger_price = entry_price * (1 - i * interval)
                trigger_type = ScalingTrigger.PRICE_LEVEL
            else:
                # Scale in on rallies (higher prices)
                trigger_price = entry_price * (1 + i * interval)
                trigger_type = ScalingTrigger.PRICE_LEVEL
            
            # Quantity increases with each level
            quantity = base_quantity * (1 + i * 0.5)
            
            level = ScalingLevel(
                level=i,
                trigger_price=trigger_price,
                quantity=quantity,
                trigger_type=trigger_type,
                trigger_value=trigger_price
            )
            
            scaling_levels.append(level)
        
        return scaling_levels
    
    def _generate_exponential_levels(self, entry_price: float, side: str, levels: int,
                                   **kwargs) -> List[ScalingLevel]:
        """Generate exponential scaling levels"""
        base_interval = kwargs.get("base_interval", 0.015)  # 1.5% base interval
        base_quantity = kwargs.get("base_quantity", 100.0)
        
        scaling_levels = []
        
        for i in range(1, levels + 1):
            if side.lower() == "buy":
                # Exponential intervals for pullbacks
                interval = base_interval * (1.5 ** (i - 1))
                trigger_price = entry_price * (1 - interval)
            else:
                # Exponential intervals for rallies
                interval = base_interval * (1.5 ** (i - 1))
                trigger_price = entry_price * (1 + interval)
            
            # Quantity also increases exponentially
            quantity = base_quantity * (1.2 ** i)
            
            level = ScalingLevel(
                level=i,
                trigger_price=trigger_price,
                quantity=quantity,
                trigger_type=ScalingTrigger.PRICE_LEVEL,
                trigger_value=trigger_price
            )
            
            scaling_levels.append(level)
        
        return scaling_levels
    
    def _generate_fibonacci_levels(self, entry_price: float, side: str, levels: int,
                                 **kwargs) -> List[ScalingLevel]:
        """Generate Fibonacci-based scaling levels"""
        base_interval = kwargs.get("base_interval", 0.01)  # 1% base interval
        base_quantity = kwargs.get("base_quantity", 100.0)
        
        # Fibonacci sequence for intervals
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
        
        scaling_levels = []
        
        for i in range(1, min(levels + 1, len(fibonacci_sequence))):
            if side.lower() == "buy":
                # Fibonacci intervals for pullbacks
                interval = base_interval * fibonacci_sequence[i]
                trigger_price = entry_price * (1 - interval)
            else:
                # Fibonacci intervals for rallies
                interval = base_interval * fibonacci_sequence[i]
                trigger_price = entry_price * (1 + interval)
            
            # Quantity based on Fibonacci ratio
            quantity = base_quantity * (1 + fibonacci_sequence[i] * 0.1)
            
            level = ScalingLevel(
                level=i,
                trigger_price=trigger_price,
                quantity=quantity,
                trigger_type=ScalingTrigger.PRICE_LEVEL,
                trigger_value=trigger_price
            )
            
            scaling_levels.append(level)
        
        return scaling_levels
    
    def _generate_volatility_levels(self, entry_price: float, side: str, levels: int,
                                  **kwargs) -> List[ScalingLevel]:
        """Generate volatility-based scaling levels"""
        volatility = kwargs.get("volatility", 0.02)  # 2% volatility
        volatility_multiplier = kwargs.get("volatility_multiplier", 2.0)
        base_quantity = kwargs.get("base_quantity", 100.0)
        
        scaling_levels = []
        
        for i in range(1, levels + 1):
            if side.lower() == "buy":
                # Volatility-adjusted intervals for pullbacks
                interval = volatility * volatility_multiplier * i
                trigger_price = entry_price * (1 - interval)
            else:
                # Volatility-adjusted intervals for rallies
                interval = volatility * volatility_multiplier * i
                trigger_price = entry_price * (1 + interval)
            
            # Quantity adjusted for volatility
            quantity = base_quantity * (1 + volatility * 10 * i)
            
            level = ScalingLevel(
                level=i,
                trigger_price=trigger_price,
                quantity=quantity,
                trigger_type=ScalingTrigger.VOLATILITY,
                trigger_value=volatility * i
            )
            
            scaling_levels.append(level)
        
        return scaling_levels
    
    def _generate_momentum_levels(self, entry_price: float, side: str, levels: int,
                                **kwargs) -> List[ScalingLevel]:
        """Generate momentum-based scaling levels"""
        momentum_threshold = kwargs.get("momentum_threshold", 0.02)  # 2% momentum
        base_quantity = kwargs.get("base_quantity", 100.0)
        
        scaling_levels = []
        
        for i in range(1, levels + 1):
            if side.lower() == "buy":
                # Momentum-based intervals for pullbacks
                interval = momentum_threshold * i
                trigger_price = entry_price * (1 - interval)
            else:
                # Momentum-based intervals for rallies
                interval = momentum_threshold * i
                trigger_price = entry_price * (1 + interval)
            
            # Quantity based on momentum strength
            quantity = base_quantity * (1 + momentum_threshold * 5 * i)
            
            level = ScalingLevel(
                level=i,
                trigger_price=trigger_price,
                quantity=quantity,
                trigger_type=ScalingTrigger.MOMENTUM,
                trigger_value=momentum_threshold * i
            )
            
            scaling_levels.append(level)
        
        return scaling_levels
    
    def check_scaling_triggers(self, symbol: str, current_price: float,
                             current_volatility: float = None,
                             current_momentum: float = None) -> List[ScalingLevel]:
        """Check for scaling triggers and return triggered levels"""
        triggered_levels = []
        
        for position_id, plan in self.active_plans.items():
            if plan.symbol != symbol or not plan.active:
                continue
            
            for level in plan.levels:
                if level.executed:
                    continue
                
                if self._check_level_trigger(level, current_price, current_volatility, current_momentum):
                    triggered_levels.append(level)
        
        return triggered_levels
    
    def _check_level_trigger(self, level: ScalingLevel, current_price: float,
                           current_volatility: float = None,
                           current_momentum: float = None) -> bool:
        """Check if a specific level should be triggered"""
        
        if level.trigger_type == ScalingTrigger.PRICE_LEVEL:
            # Check price-based trigger
            if level.trigger_price is not None:
                if level.trigger_price > 0:  # Valid price
                    if level.trigger_price <= current_price <= level.trigger_price * 1.001:  # 0.1% tolerance
                        return True
        
        elif level.trigger_type == ScalingTrigger.VOLATILITY:
            # Check volatility-based trigger
            if current_volatility is not None and level.trigger_value is not None:
                if current_volatility >= level.trigger_value:
                    return True
        
        elif level.trigger_type == ScalingTrigger.MOMENTUM:
            # Check momentum-based trigger
            if current_momentum is not None and level.trigger_value is not None:
                if abs(current_momentum) >= level.trigger_value:
                    return True
        
        return False
    
    def execute_scaling_level(self, level: ScalingLevel, execution_price: float,
                            slippage: float = 0.0, commission: float = 0.0) -> ScalingExecution:
        """Execute a scaling level and record the execution"""
        
        execution = ScalingExecution(
            scaling_level_id=f"{level.level}_{int(time.time())}",
            position_id=level.trigger_price,  # This should be position_id from the plan
            level=level.level,
            quantity=level.quantity,
            execution_price=execution_price,
            execution_time=datetime.now(),
            trigger_type=level.trigger_type,
            trigger_value=level.trigger_value,
            slippage=slippage,
            commission=commission
        )
        
        # Mark level as executed
        level.executed = True
        level.execution_time = execution.execution_time
        level.execution_price = execution.execution_price
        
        # Record execution
        self.execution_history.append(execution)
        
        # Update performance tracking
        self.total_scaling_events += 1
        self.successful_scaling_events += 1
        
        logger.info(f"Executed scaling level {level.level} at {execution_price}")
        
        return execution
    
    def get_scaling_plan(self, position_id: str) -> Optional[ScalingPlan]:
        """Get scaling plan for a position"""
        return self.active_plans.get(position_id)
    
    def deactivate_scaling_plan(self, position_id: str) -> bool:
        """Deactivate a scaling plan"""
        if position_id in self.active_plans:
            self.active_plans[position_id].active = False
            logger.info(f"Deactivated scaling plan for position {position_id}")
            return True
        return False
    
    def get_scaling_statistics(self) -> Dict:
        """Get scaling performance statistics"""
        total_plans = len(self.active_plans)
        active_plans = sum(1 for plan in self.active_plans.values() if plan.active)
        
        return {
            "total_scaling_events": self.total_scaling_events,
            "successful_scaling_events": self.successful_scaling_events,
            "failed_scaling_events": self.failed_scaling_events,
            "success_rate": self.successful_scaling_events / max(self.total_scaling_events, 1),
            "total_plans": total_plans,
            "active_plans": active_plans,
            "inactive_plans": total_plans - active_plans
        }
    
    def cleanup_old_executions(self, days: int = 30):
        """Clean up old execution records"""
        cutoff_date = datetime.now() - timedelta(days=days)
        self.execution_history = [
            execution for execution in self.execution_history
            if execution.execution_time > cutoff_date
        ]
        logger.info(f"Cleaned up execution history older than {days} days")

def test_position_scaling_manager():
    """Test the position scaling manager"""
    manager = PositionScalingManager()
    
    # Test linear scaling plan
    plan = manager.create_scaling_plan(
        position_id="test_001",
        symbol="BTCUSDT",
        side="buy",
        base_quantity=100.0,
        entry_price=50000.0,
        strategy=ScalingStrategy.LINEAR,
        levels=3,
        interval=0.02
    )
    
    print(f"Created scaling plan: {plan.strategy.value}")
    print(f"Number of levels: {len(plan.levels)}")
    
    for level in plan.levels:
        print(f"Level {level.level}: {level.trigger_price:.2f} @ {level.quantity:.2f}")
    
    # Test scaling trigger check
    triggered = manager.check_scaling_triggers("BTCUSDT", 49000.0)  # 2% pullback
    print(f"Triggered levels: {len(triggered)}")
    
    # Test execution
    if triggered:
        execution = manager.execute_scaling_level(triggered[0], 49000.0)
        print(f"Executed scaling: {execution}")
    
    # Get statistics
    stats = manager.get_scaling_statistics()
    print(f"Scaling statistics: {stats}")

if __name__ == "__main__":
    test_position_scaling_manager()
