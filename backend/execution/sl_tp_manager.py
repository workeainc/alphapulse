#!/usr/bin/env python3
"""
Dynamic Stop Loss and Take Profit Manager for AlphaPulse
Handles ATR-based stop loss calculations and trailing stops
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Stop loss type enumeration"""
    FIXED = "fixed"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    BREAK_EVEN = "break_even"
    VOLATILITY_ADJUSTED = "volatility_adjusted"

class TakeProfitType(Enum):
    """Take profit type enumeration"""
    FIXED = "fixed"
    RISK_REWARD = "risk_reward"
    ATR_BASED = "atr_based"
    SCALING = "scaling"

@dataclass
class Position:
    """Position data structure for SL/TP management"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    atr: float
    current_price: float
    unrealized_pnl: float = 0.0
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    take_profit_type: TakeProfitType = StopLossType.FIXED

@dataclass
class SLTPUpdate:
    """Stop loss or take profit update"""
    position_id: str
    field: str  # 'stop_loss' or 'take_profit'
    old_value: float
    new_value: float
    reason: str
    timestamp: datetime

class SLTPManager:
    """Manages dynamic stop loss and take profit levels"""
    
    def __init__(self):
        # ATR settings
        self.atr_period = 14
        self.atr_multiplier = 2.0
        
        # Trailing stop settings
        self.trailing_stop_enabled = True
        self.trailing_multiplier = 1.5
        self.trailing_activation_pct = 0.02  # 2% profit to activate trailing
        
        # Break-even settings
        self.break_even_enabled = True
        self.break_even_pct = 0.015  # 1.5% profit to move to break-even
        
        # Risk-reward settings
        self.default_risk_reward_ratio = 2.0
        self.min_risk_reward_ratio = 1.5
        
        # Volatility adjustment
        self.volatility_adjustment_enabled = True
        self.volatility_threshold = 0.03  # 3% volatility threshold
        
        # Update tracking
        self.update_history: List[SLTPUpdate] = []
        self.max_update_history = 1000
        
        logger.info("SL/TP Manager initialized")
    
    def calculate_atr_stop_loss(self, entry_price: float, side: str, 
                               atr: float, risk_multiplier: float = None) -> float:
        """
        Calculate ATR-based stop loss
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('buy' or 'sell')
            atr: Average True Range value
            risk_multiplier: ATR multiplier for risk (defaults to self.atr_multiplier)
            
        Returns:
            Stop loss price
        """
        if risk_multiplier is None:
            risk_multiplier = self.atr_multiplier
        
        atr_distance = atr * risk_multiplier
        
        if side.lower() == 'buy':
            # Long position: stop loss below entry
            stop_loss = entry_price - atr_distance
        else:
            # Short position: stop loss above entry
            stop_loss = entry_price + atr_distance
        
        logger.debug(f"ATR stop loss calculated: {side} {entry_price} ± {atr_distance} = {stop_loss}")
        return stop_loss
    
    def calculate_risk_reward_take_profit(self, entry_price: float, side: str,
                                        stop_loss: float, risk_reward_ratio: float = None) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('buy' or 'sell')
            stop_loss: Stop loss price
            risk_reward_ratio: Risk-reward ratio (defaults to self.default_risk_reward_ratio)
            
        Returns:
            Take profit price
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.default_risk_reward_ratio
        
        # Calculate risk (distance from entry to stop loss)
        risk_distance = abs(entry_price - stop_loss)
        
        # Calculate reward (risk * risk-reward ratio)
        reward_distance = risk_distance * risk_reward_ratio
        
        if side.lower() == 'buy':
            # Long position: take profit above entry
            take_profit = entry_price + reward_distance
        else:
            # Short position: take profit below entry
            take_profit = entry_price - reward_distance
        
        logger.debug(f"Risk-reward take profit calculated: {side} {entry_price} ± {reward_distance} = {take_profit}")
        return take_profit
    
    def calculate_atr_take_profit(self, entry_price: float, side: str,
                                 atr: float, atr_multiplier: float = 3.0) -> float:
        """
        Calculate ATR-based take profit
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('buy' or 'sell')
            atr: Average True Range value
            atr_multiplier: ATR multiplier for take profit
            
        Returns:
            Take profit price
        """
        atr_distance = atr * atr_multiplier
        
        if side.lower() == 'buy':
            # Long position: take profit above entry
            take_profit = entry_price + atr_distance
        else:
            # Short position: take profit below entry
            take_profit = entry_price - atr_distance
        
        logger.debug(f"ATR take profit calculated: {side} {entry_price} ± {atr_distance} = {take_profit}")
        return take_profit
    
    def update_trailing_stop(self, position: Position, current_price: float) -> Optional[float]:
        """
        Update trailing stop loss for a position
        
        Args:
            position: Position to update
            current_price: Current market price
            
        Returns:
            New stop loss price if updated, None otherwise
        """
        if not self.trailing_stop_enabled:
            return None
        
        # Calculate current profit percentage
        if position.side.lower() == 'buy':
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
        
        # Check if trailing should be activated
        if profit_pct < self.trailing_activation_pct:
            return None
        
        # Calculate new trailing stop
        if position.side.lower() == 'buy':
            # Long position: trail below current price
            new_stop = current_price - (position.atr * self.trailing_multiplier)
            
            # Only update if new stop is higher than current stop
            if new_stop > position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                
                # Record update
                self._record_update(position.id, 'stop_loss', old_stop, new_stop, 'trailing_stop')
                
                logger.info(f"Trailing stop updated for {position.symbol}: {old_stop:.4f} → {new_stop:.4f}")
                return new_stop
        else:
            # Short position: trail above current price
            new_stop = current_price + (position.atr * self.trailing_multiplier)
            
            # Only update if new stop is lower than current stop
            if new_stop < position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                
                # Record update
                self._record_update(position.id, 'stop_loss', old_stop, new_stop, 'trailing_stop')
                
                logger.info(f"Trailing stop updated for {position.symbol}: {old_stop:.4f} → {new_stop:.4f}")
                return new_stop
        
        return None
    
    def check_break_even(self, position: Position, current_price: float) -> Optional[float]:
        """
        Check if position should move to break-even
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            New stop loss price if moving to break-even, None otherwise
        """
        if not self.break_even_enabled:
            return None
        
        # Calculate current profit percentage
        if position.side.lower() == 'buy':
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
        
        # Check if break-even should be activated
        if profit_pct >= self.break_even_pct:
            # Move stop loss to entry price (break-even)
            if position.stop_loss != position.entry_price:
                old_stop = position.stop_loss
                position.stop_loss = position.entry_price
                
                # Record update
                self._record_update(position.id, 'stop_loss', old_stop, position.entry_price, 'break_even')
                
                logger.info(f"Stop loss moved to break-even for {position.symbol}: {old_stop:.4f} → {position.entry_price:.4f}")
                return position.entry_price
        
        return None
    
    def adjust_for_volatility(self, position: Position, current_volatility: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Adjust stop loss and take profit for current volatility
        
        Args:
            position: Position to adjust
            current_volatility: Current market volatility (as percentage)
            
        Returns:
            Tuple of (new_stop_loss, new_take_profit) if adjusted, (None, None) otherwise
        """
        if not self.volatility_adjustment_enabled:
            return None, None
        
        # Check if volatility is above threshold
        if current_volatility <= self.volatility_threshold:
            return None, None
        
        # Calculate volatility adjustment factor
        volatility_factor = current_volatility / self.volatility_threshold
        
        # Adjust ATR multiplier based on volatility
        adjusted_atr_multiplier = self.atr_multiplier * volatility_factor
        
        # Recalculate stop loss and take profit
        new_stop_loss = self.calculate_atr_stop_loss(
            position.entry_price, 
            position.side, 
            position.atr, 
            adjusted_atr_multiplier
        )
        
        new_take_profit = self.calculate_atr_take_profit(
            position.entry_price,
            position.side,
            position.atr,
            adjusted_atr_multiplier
        )
        
        # Check if adjustment is significant enough
        stop_loss_change = abs(new_stop_loss - position.stop_loss) / position.entry_price
        take_profit_change = abs(new_take_profit - position.take_profit) / position.entry_price
        
        changes_made = False
        
        # Update stop loss if change is significant
        if stop_loss_change > 0.005:  # 0.5% change threshold
            old_stop = position.stop_loss
            position.stop_loss = new_stop_loss
            self._record_update(position.id, 'stop_loss', old_stop, new_stop_loss, 'volatility_adjustment')
            changes_made = True
        
        # Update take profit if change is significant
        if take_profit_change > 0.005:  # 0.5% change threshold
            old_tp = position.take_profit
            position.take_profit = new_take_profit
            self._record_update(position.id, 'take_profit', old_tp, new_take_profit, 'volatility_adjustment')
            changes_made = True
        
        if changes_made:
            logger.info(f"Volatility adjustment applied to {position.symbol}: volatility={current_volatility:.2%}")
            return new_stop_loss, new_take_profit
        
        return None, None
    
    def calculate_scaling_take_profits(self, entry_price: float, side: str,
                                     stop_loss: float, risk_reward_ratios: List[float] = None) -> List[Tuple[float, float]]:
        """
        Calculate multiple take profit levels for position scaling
        
        Args:
            entry_price: Entry price of the position
            side: Position side ('buy' or 'sell')
            stop_loss: Stop loss price
            risk_reward_ratios: List of risk-reward ratios for each level
            
        Returns:
            List of (price, percentage) tuples for each take profit level
        """
        if risk_reward_ratios is None:
            risk_reward_ratios = [1.5, 2.0, 3.0, 5.0]
        
        scaling_levels = []
        
        for ratio in risk_reward_ratios:
            take_profit = self.calculate_risk_reward_take_profit(
                entry_price, side, stop_loss, ratio
            )
            
            # Calculate percentage from entry
            if side.lower() == 'buy':
                percentage = (take_profit - entry_price) / entry_price
            else:
                percentage = (entry_price - take_profit) / entry_price
            
            scaling_levels.append((take_profit, percentage))
        
        logger.debug(f"Scaling take profits calculated for {side}: {scaling_levels}")
        return scaling_levels
    
    def validate_stop_loss(self, entry_price: float, stop_loss: float, 
                          side: str, max_risk_pct: float = 0.10) -> bool:
        """
        Validate stop loss is within acceptable risk parameters
        
        Args:
            entry_price: Entry price of the position
            stop_loss: Stop loss price
            side: Position side ('buy' or 'sell')
            max_risk_pct: Maximum allowed risk percentage
            
        Returns:
            True if stop loss is valid, False otherwise
        """
        # Calculate risk percentage
        if side.lower() == 'buy':
            risk_pct = (entry_price - stop_loss) / entry_price
        else:
            risk_pct = (stop_loss - entry_price) / entry_price
        
        # Check if risk is within acceptable range
        if risk_pct > max_risk_pct:
            logger.warning(f"Stop loss risk {risk_pct:.2%} exceeds maximum {max_risk_pct:.2%}")
            return False
        
        if risk_pct <= 0:
            logger.warning(f"Invalid stop loss: risk percentage {risk_pct:.2%}")
            return False
        
        return True
    
    def validate_take_profit(self, entry_price: float, take_profit: float,
                            stop_loss: float, side: str, min_risk_reward: float = 1.0) -> bool:
        """
        Validate take profit is within acceptable parameters
        
        Args:
            entry_price: Entry price of the position
            take_profit: Take profit price
            stop_loss: Stop loss price
            side: Position side ('buy' or 'sell')
            min_risk_reward: Minimum required risk-reward ratio
            
        Returns:
            True if take profit is valid, False otherwise
        """
        # Calculate risk and reward
        if side.lower() == 'buy':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        # Check if take profit is in correct direction
        if reward <= 0:
            logger.warning(f"Take profit {take_profit} is not profitable for {side} position")
            return False
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Check minimum risk-reward ratio
        if risk_reward_ratio < min_risk_reward:
            logger.warning(f"Risk-reward ratio {risk_reward_ratio:.2f} below minimum {min_risk_reward}")
            return False
        
        return True
    
    def _record_update(self, position_id: str, field: str, old_value: float, 
                      new_value: float, reason: str):
        """Record an SL/TP update for tracking"""
        update = SLTPUpdate(
            position_id=position_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            timestamp=datetime.now()
        )
        
        self.update_history.append(update)
        
        # Maintain history size limit
        if len(self.update_history) > self.max_update_history:
            self.update_history = self.update_history[-self.max_update_history:]
    
    def get_update_history(self, position_id: Optional[str] = None, 
                          field: Optional[str] = None) -> List[SLTPUpdate]:
        """Get update history, optionally filtered"""
        updates = self.update_history
        
        if position_id:
            updates = [u for u in updates if u.position_id == position_id]
        
        if field:
            updates = [u for u in updates if u.field == field]
        
        return sorted(updates, key=lambda x: x.timestamp, reverse=True)
    
    def get_performance_stats(self) -> Dict:
        """Get SL/TP manager performance statistics"""
        if not self.update_history:
            return {}
        
        # Calculate update frequency
        total_updates = len(self.update_history)
        if total_updates > 1:
            time_span = (self.update_history[-1].timestamp - self.update_history[0].timestamp).total_seconds() / 3600
            updates_per_hour = total_updates / time_span if time_span > 0 else 0
        else:
            updates_per_hour = 0
        
        # Count updates by reason
        reason_counts = {}
        for update in self.update_history:
            reason = update.reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            'total_updates': total_updates,
            'updates_per_hour': updates_per_hour,
            'reason_counts': reason_counts,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'break_even_enabled': self.break_even_enabled,
            'volatility_adjustment_enabled': self.volatility_adjustment_enabled
        }

# Example usage
def test_sltp_manager():
    """Test the SL/TP manager functionality"""
    manager = SLTPManager()
    
    # Create test position
    position = Position(
        id="POS_001",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        quantity=0.1,
        stop_loss=48000.0,
        take_profit=55000.0,
        entry_time=datetime.now(),
        atr=2000.0,
        current_price=52000.0
    )
    
    # Test ATR stop loss calculation
    atr_stop = manager.calculate_atr_stop_loss(position.entry_price, position.side, position.atr)
    print(f"ATR Stop Loss: {atr_stop:.2f}")
    
    # Test risk-reward take profit
    rr_tp = manager.calculate_risk_reward_take_profit(position.entry_price, position.side, position.stop_loss)
    print(f"Risk-Reward Take Profit: {rr_tp:.2f}")
    
    # Test trailing stop update
    new_stop = manager.update_trailing_stop(position, 53000.0)
    if new_stop:
        print(f"Trailing Stop Updated: {new_stop:.2f}")
    
    # Test break-even check
    be_stop = manager.check_break_even(position, 53000.0)
    if be_stop:
        print(f"Break-Even Stop: {be_stop:.2f}")
    
    # Test volatility adjustment
    new_stop, new_tp = manager.adjust_for_volatility(position, 0.05)  # 5% volatility
    if new_stop or new_tp:
        print(f"Volatility Adjustment: Stop={new_stop}, TP={new_tp}")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    print(f"Performance Stats: {stats}")
    
    return manager

if __name__ == "__main__":
    # Run test if script is executed directly
    test_sltp_manager()
