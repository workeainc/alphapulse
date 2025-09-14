"""
Funding Rate Signal Execution Service for AlphaPulse
Week 7.3 Phase 3: Funding Rate Strategy Integration

Features:
- Signal execution and position management
- Risk management and position sizing
- Performance tracking and optimization
- Integration with trading engine

Author: AlphaPulse Team
Date: 2025
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"

@dataclass
class ExecutionOrder:
    """Execution order details"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: float
    status: ExecutionStatus
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Trading position"""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    status: PositionStatus
    entry_time: datetime
    last_update: datetime
    pnl: float
    metadata: Dict[str, Any]

class FundingRateExecutor:
    """Funding rate signal execution service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Execution configuration
        self.max_slippage = self.config.get('max_slippage', 0.001)  # 0.1%
        self.execution_timeout = self.config.get('execution_timeout', 30)  # seconds
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.min_order_size = self.config.get('min_order_size', 0.001)
        
        # Risk management
        self.max_position_size = self.config.get('max_position_size', 0.2)  # 20% of balance
        self.max_daily_loss = self.config.get('max_daily_loss', 0.05)  # 5% daily loss limit
        self.max_open_positions = self.config.get('max_open_positions', 10)
        
        # Data storage
        self.orders = defaultdict(list)  # symbol -> orders
        self.positions = defaultdict(dict)  # symbol -> position
        self.execution_history = defaultdict(list)  # symbol -> history
        
        # Performance tracking
        self.stats = {
            'orders_executed': 0,
            'orders_failed': 0,
            'total_volume': 0.0,
            'total_pnl': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'last_update': None
        }
        
        # Callbacks
        self.execution_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_start = datetime.now(timezone.utc).date()
        
        # Initialize risk management
        self._initialize_risk_management()
    
    def _initialize_risk_management(self):
        """Initialize risk management parameters"""
        try:
            self.logger.info("Risk management initialized for funding rate executor")
        except Exception as e:
            self.logger.error(f"Failed to initialize risk management: {e}")
    
    async def execute_signal(self, signal: Any, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a funding rate trading signal"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return {'success': False, 'error': 'Invalid signal'}
            
            # Check risk limits
            if not self._check_risk_limits(signal, market_data):
                return {'success': False, 'error': 'Risk limits exceeded'}
            
            # Calculate execution parameters
            execution_params = await self._calculate_execution_params(signal, market_data)
            if not execution_params:
                return {'success': False, 'error': 'Failed to calculate execution parameters'}
            
            # Execute order
            execution_result = await self._execute_order(execution_params)
            if not execution_result['success']:
                return execution_result
            
            # Create position
            position = await self._create_position(signal, execution_result, market_data)
            if not position:
                return {'success': False, 'error': 'Failed to create position'}
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            # Trigger callbacks
            await self._trigger_callbacks('signal_executed', {
                'signal': signal,
                'execution_result': execution_result,
                'position': position
            })
            
            self.logger.info(f"Signal executed successfully for {signal.symbol}: {signal.signal_type}")
            
            return {
                'success': True,
                'order_id': execution_result['order_id'],
                'position_id': position['position_id'],
                'execution_price': execution_result['execution_price'],
                'quantity': execution_result['quantity']
            }
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_signal(self, signal: Any) -> bool:
        """Validate trading signal"""
        try:
            # Check required attributes
            required_attrs = ['symbol', 'direction', 'confidence', 'signal_type']
            for attr in required_attrs:
                if not hasattr(signal, attr):
                    return False
            
            # Check confidence threshold
            if signal.confidence < 0.7:
                return False
            
            # Check symbol format
            if not isinstance(signal.symbol, str) or '/' not in signal.symbol:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def _check_risk_limits(self, signal: Any, market_data: Dict[str, Any]) -> bool:
        """Check risk management limits"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning("Daily loss limit exceeded")
                return False
            
            # Check maximum open positions
            if len(self.positions) >= self.max_open_positions:
                self.logger.warning("Maximum open positions reached")
                return False
            
            # Check if symbol already has an open position
            if signal.symbol in self.positions:
                self.logger.warning(f"Position already open for {signal.symbol}")
                return False
            
            # Check position size limits
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return False
            
            # Additional risk checks can be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def _calculate_execution_params(self, signal: Any, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate execution parameters for the signal"""
        try:
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None
            
            # Calculate position size based on signal confidence and risk
            base_size = signal.confidence * 0.1  # Base 10% position size
            
            # Adjust for signal strength
            if hasattr(signal, 'strength'):
                if signal.strength.value == 'extreme':
                    base_size *= 1.5
                elif signal.strength.value == 'strong':
                    base_size *= 1.2
                elif signal.strength.value == 'weak':
                    base_size *= 0.7
            
            # Ensure position size doesn't exceed maximum
            position_size = min(base_size, self.max_position_size)
            
            # Calculate quantity
            quantity = (position_size * 10000) / current_price  # Assuming $10k account
            
            # Ensure minimum order size
            if quantity < self.min_order_size:
                quantity = self.min_order_size
            
            # Calculate execution price with slippage consideration
            if signal.direction.value == 'long':
                execution_price = current_price * (1 + self.max_slippage)
                side = 'buy'
            elif signal.direction.value == 'short':
                execution_price = current_price * (1 - self.max_slippage)
                side = 'sell'
            else:
                # Neutral signal - determine direction based on market conditions
                execution_price = current_price
                side = 'buy'  # Default to buy for neutral signals
            
            # Debug logging
            self.logger.debug(f"Signal direction: {signal.direction.value}, Execution side: {side}")
            
            return {
                'symbol': signal.symbol,
                'side': side,
                'quantity': quantity,
                'price': execution_price,
                'order_type': 'market',
                'signal': signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating execution parameters: {e}")
            return None
    
    async def _execute_order(self, execution_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trading order"""
        try:
            # Simulate order execution (replace with actual exchange integration)
            order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Simulate execution price (with some slippage)
            execution_price = execution_params['price']
            if execution_params['side'] == 'buy':
                execution_price *= (1 + np.random.uniform(0, self.max_slippage))
            else:
                execution_price *= (1 - np.random.uniform(0, self.max_slippage))
            
            execution_result = {
                'success': True,
                'order_id': order_id,
                'execution_price': execution_price,
                'quantity': execution_params['quantity'],
                'timestamp': datetime.now(timezone.utc),
                'side': execution_params['side'],
                'symbol': execution_params['symbol']
            }
            
            # Store order
            self.orders[execution_params['symbol']].append(ExecutionOrder(
                order_id=order_id,
                symbol=execution_params['symbol'],
                side=execution_params['side'],
                order_type=execution_params['order_type'],
                quantity=execution_params['quantity'],
                price=execution_price,
                status=ExecutionStatus.EXECUTED,
                timestamp=datetime.now(timezone.utc),
                metadata={'signal_type': execution_params['signal'].signal_type}
            ))
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _create_position(self, signal: Any, execution_result: Dict[str, Any], 
                              market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a trading position from executed signal"""
        try:
            position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Calculate stop loss and take profit
            entry_price = execution_result['execution_price']
            if signal.direction.value == 'long':
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.04  # 4% take profit
            elif signal.direction.value == 'short':
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profit = entry_price * 0.96  # 4% take profit
            else:
                # Neutral signal - use tighter stops
                stop_loss = entry_price * 0.99
                take_profit = entry_price * 1.02
            
            # Debug logging
            self.logger.debug(f"Position creation: direction={signal.direction.value}, entry_price={entry_price}, stop_loss={stop_loss}, take_profit={take_profit}")
            
            position = Position(
                position_id=position_id,
                symbol=signal.symbol,
                side=execution_result['side'],
                quantity=execution_result['quantity'],
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=PositionStatus.OPEN,
                entry_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc),
                pnl=0.0,
                metadata={
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'order_id': execution_result['order_id']
                }
            )
            
            # Store position
            self.positions[signal.symbol] = position
            
            return position.__dict__
            
        except Exception as e:
            self.logger.error(f"Error creating position: {e}")
            return None
    
    async def update_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Update position with current market data"""
        try:
            if symbol not in self.positions:
                return {}
            
            position = self.positions[symbol]
            
            # Update current price and PnL
            position.current_price = current_price
            position.last_update = datetime.now(timezone.utc)
            
            # Calculate PnL
            if position.side == 'buy':
                position.pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.pnl = (position.entry_price - current_price) * position.quantity
            
            # Check if position should be closed
            should_close = False
            close_reason = ""
            
            # Debug logging
            self.logger.debug(f"Position check for {symbol}: side={position.side}, current_price={current_price}, stop_loss={position.stop_loss}, take_profit={position.take_profit}")
            
            if position.side == 'buy':
                if current_price <= position.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= position.take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:
                if current_price >= position.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= position.take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                await self._close_position(symbol, current_price, close_reason)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'entry_price': position.entry_price,
                'pnl': position.pnl,
                'should_close': should_close,
                'close_reason': close_reason
            }
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
            return {}
    
    async def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a trading position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Calculate final PnL
            if position.side == 'buy':
                final_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                final_pnl = (position.entry_price - exit_price) * position.quantity
            
            # Update daily PnL
            self.daily_pnl += final_pnl
            
            # Update global statistics
            self.stats['total_pnl'] += final_pnl
            if final_pnl > 0:
                self.stats['successful_trades'] += 1
            else:
                self.stats['failed_trades'] += 1
            
            # Store execution history
            self.execution_history[symbol].append({
                'position_id': position.position_id,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'pnl': final_pnl,
                'reason': reason,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(timezone.utc)
            })
            
            # Remove position
            del self.positions[symbol]
            
            # Trigger callback
            await self._trigger_callbacks('position_closed', {
                'symbol': symbol,
                'exit_price': exit_price,
                'pnl': final_pnl,
                'reason': reason,
                'position': position.__dict__
            })
            
            self.logger.info(f"Position closed for {symbol}: {reason}, PnL: {final_pnl:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _update_execution_stats(self, execution_result: Dict[str, Any]):
        """Update execution statistics"""
        try:
            self.stats['orders_executed'] += 1
            self.stats['total_volume'] += execution_result['quantity'] * execution_result['execution_price']
            self.stats['last_update'] = datetime.now(timezone.utc)
        except Exception as e:
            self.logger.error(f"Error updating execution stats: {e}")
    
    def add_callback(self, event_type: str, callback):
        """Add callback for execution events"""
        self.execution_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for execution events"""
        callbacks = self.execution_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_executor_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get executor summary"""
        try:
            summary = {
                'stats': self.stats,
                'active_positions': len(self.positions),
                'total_orders': sum(len(orders) for orders in self.orders.values()),
                'daily_pnl': self.daily_pnl,
                'risk_limits': {
                    'max_position_size': self.max_position_size,
                    'max_daily_loss': self.max_daily_loss,
                    'max_open_positions': self.max_open_positions
                }
            }
            
            if symbol:
                symbol_summary = {
                    'orders': len(self.orders.get(symbol, [])),
                    'active_position': symbol in self.positions,
                    'execution_history': len(self.execution_history.get(symbol, []))
                }
                summary['symbol_details'] = {symbol: symbol_summary}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting executor summary: {e}")
            return {}
    
    async def close(self):
        """Close the executor service"""
        try:
            # Close all open positions
            for symbol in list(self.positions.keys()):
                await self._close_position(symbol, 0, "service_shutdown")
            
            self.logger.info("Funding Rate Executor service closed")
        except Exception as e:
            self.logger.error(f"Error closing executor service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
