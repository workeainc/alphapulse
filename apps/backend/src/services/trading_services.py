"""
Trading Services for AlphaPulse

This module provides comprehensive trading services including order management,
portfolio management, position tracking, and execution services.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import redis
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides for trading."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Trading position representation."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    """Portfolio representation."""
    total_balance: float
    available_balance: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    positions: List[Position] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Order management system for AlphaPulse.
    
    Handles order creation, tracking, and execution.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize order manager.
        
        Args:
            redis_client: Redis client for persistence
        """
        self.redis_client = redis_client
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.lock = asyncio.Lock()
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new trading order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Type of order
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            metadata: Additional order metadata
            
        Returns:
            Created order
        """
        order_id = f"order_{int(time.time() * 1000)}_{symbol}_{side.value}"
        
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        async with self.lock:
            self.orders[order_id] = order
            await self._save_order_to_redis(order)
        
        logger.info(f"Created order {order_id}: {side.value} {quantity} {symbol} @ {price}")
        return order
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order object or None if not found
        """
        async with self.lock:
            return self.orders.get(order_id)
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None
    ) -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            side: Filter by side
            
        Returns:
            List of matching orders
        """
        async with self.lock:
            orders = list(self.orders.values())
        
        # Apply filters
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        if side:
            orders = [o for o in orders if o.side == side]
        
        return orders
    
    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[float] = None,
        average_price: Optional[float] = None
    ):
        """
        Update order status and execution details.
        
        Args:
            order_id: Order identifier
            status: New order status
            filled_quantity: Filled quantity
            average_price: Average fill price
        """
        async with self.lock:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return
            
            order = self.orders[order_id]
            order.status = status
            order.updated_at = datetime.now()
            
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
            
            if average_price is not None:
                order.average_price = average_price
            
            # Move to history if completed
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.order_history.append(order)
                del self.orders[order_id]
            
            await self._save_order_to_redis(order)
        
        logger.info(f"Updated order {order_id} status to {status.value}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order was cancelled successfully
        """
        async with self.lock:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            order = self.orders[order_id]
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Order {order_id} cannot be cancelled in status {order.status.value}")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Move to history
            self.order_history.append(order)
            del self.orders[order_id]
            
            await self._save_order_to_redis(order)
        
        logger.info(f"Cancelled order {order_id}")
        return True
    
    async def _save_order_to_redis(self, order: Order):
        """Save order to Redis for persistence."""
        if not self.redis_client:
            return
        
        try:
            key = f"order:{order.id}"
            order_data = {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'average_price': order.average_price,
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat(),
                'metadata': order.metadata
            }
            
            self.redis_client.hset(key, mapping=order_data)
            self.redis_client.expire(key, 86400)  # 24 hours TTL
        except Exception as e:
            logger.error(f"Error saving order to Redis: {e}")


class PortfolioManager:
    """
    Portfolio management system for AlphaPulse.
    
    Handles position tracking, P&L calculation, and portfolio updates.
    """
    
    def __init__(self, initial_balance: float, redis_client: Optional[redis.Redis] = None):
        """
        Initialize portfolio manager.
        
        Args:
            initial_balance: Initial portfolio balance
            redis_client: Redis client for persistence
        """
        self.initial_balance = initial_balance
        self.redis_client = redis_client
        self.positions: Dict[str, Position] = {}
        self.balance = initial_balance
        self.realized_pnl = 0.0
        self.lock = asyncio.Lock()
    
    async def open_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Open a new trading position.
        
        Args:
            symbol: Trading symbol
            side: Position side (buy/sell)
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional position metadata
            
        Returns:
            Created position
        """
        position_id = f"pos_{symbol}_{side.value}_{int(time.time() * 1000)}"
        
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        
        async with self.lock:
            self.positions[position_id] = position
            await self._save_position_to_redis(position_id, position)
        
        logger.info(f"Opened position {position_id}: {side.value} {quantity} {symbol} @ {entry_price}")
        return position
    
    async def update_position_price(
        self,
        symbol: str,
        current_price: float
    ):
        """
        Update position price and calculate unrealized P&L.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        async with self.lock:
            for position_id, position in self.positions.items():
                if position.symbol == symbol:
                    position.current_price = current_price
                    position.updated_at = datetime.now()
                    
                    # Calculate unrealized P&L
                    if position.side == OrderSide.BUY:
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    await self._save_position_to_redis(position_id, position)
    
    async def close_position(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None
    ) -> float:
        """
        Close a position and calculate realized P&L.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            quantity: Quantity to close (None for full position)
            
        Returns:
            Realized P&L
        """
        async with self.lock:
            positions_to_close = [
                (pos_id, pos) for pos_id, pos in self.positions.items()
                if pos.symbol == symbol
            ]
            
            if not positions_to_close:
                logger.warning(f"No positions found for symbol {symbol}")
                return 0.0
            
            total_realized_pnl = 0.0
            
            for position_id, position in positions_to_close:
                close_qty = quantity or position.quantity
                
                if close_qty > position.quantity:
                    logger.warning(f"Close quantity {close_qty} exceeds position quantity {position.quantity}")
                    close_qty = position.quantity
                
                # Calculate realized P&L
                if position.side == OrderSide.BUY:
                    realized_pnl = (exit_price - position.entry_price) * close_qty
                else:
                    realized_pnl = (position.entry_price - exit_price) * close_qty
                
                total_realized_pnl += realized_pnl
                
                # Update position
                if close_qty == position.quantity:
                    # Close entire position
                    del self.positions[position_id]
                    await self._delete_position_from_redis(position_id)
                else:
                    # Partial close
                    position.quantity -= close_qty
                    position.realized_pnl += realized_pnl
                    position.updated_at = datetime.now()
                    await self._save_position_to_redis(position_id, position)
            
            # Update portfolio
            self.realized_pnl += total_realized_pnl
            self.balance += total_realized_pnl
            
            await self._save_portfolio_to_redis()
        
        logger.info(f"Closed position for {symbol} at {exit_price}, realized P&L: {total_realized_pnl}")
        return total_realized_pnl
    
    async def get_portfolio_summary(self) -> Portfolio:
        """
        Get current portfolio summary.
        
        Returns:
            Portfolio summary
        """
        async with self.lock:
            # Calculate total unrealized P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Get open orders (would need to integrate with OrderManager)
            open_orders = []  # Placeholder
            
            portfolio = Portfolio(
                total_balance=self.balance + total_unrealized_pnl,
                available_balance=self.balance,
                total_pnl=self.realized_pnl + total_unrealized_pnl,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=self.realized_pnl,
                positions=list(self.positions.values()),
                orders=open_orders,
                updated_at=datetime.now()
            )
        
        return portfolio
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position or None if not found
        """
        async with self.lock:
            for position in self.positions.values():
                if position.symbol == symbol:
                    return position
        return None
    
    async def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List of all positions
        """
        async with self.lock:
            return list(self.positions.values())
    
    async def _save_position_to_redis(self, position_id: str, position: Position):
        """Save position to Redis for persistence."""
        if not self.redis_client:
            return
        
        try:
            key = f"position:{position_id}"
            position_data = {
                'symbol': position.symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'created_at': position.created_at.isoformat(),
                'updated_at': position.updated_at.isoformat(),
                'metadata': position.metadata
            }
            
            self.redis_client.hset(key, mapping=position_data)
            self.redis_client.expire(key, 86400)  # 24 hours TTL
        except Exception as e:
            logger.error(f"Error saving position to Redis: {e}")
    
    async def _delete_position_from_redis(self, position_id: str):
        """Delete position from Redis."""
        if not self.redis_client:
            return
        
        try:
            key = f"position:{position_id}"
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting position from Redis: {e}")
    
    async def _save_portfolio_to_redis(self):
        """Save portfolio state to Redis."""
        if not self.redis_client:
            return
        
        try:
            key = "portfolio:state"
            portfolio_data = {
                'initial_balance': self.initial_balance,
                'balance': self.balance,
                'realized_pnl': self.realized_pnl,
                'updated_at': datetime.now().isoformat()
            }
            
            self.redis_client.hset(key, mapping=portfolio_data)
        except Exception as e:
            logger.error(f"Error saving portfolio to Redis: {e}")


class ExecutionService:
    """
    Order execution service for AlphaPulse.
    
    Handles order routing, execution, and market interaction.
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        portfolio_manager: PortfolioManager,
        exchange_config: Dict[str, Any]
    ):
        """
        Initialize execution service.
        
        Args:
            order_manager: Order manager instance
            portfolio_manager: Portfolio manager instance
            exchange_config: Exchange configuration
        """
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.exchange_config = exchange_config
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_signal(
        self,
        signal: Dict[str, Any],
        risk_params: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Order]]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal dictionary
            risk_params: Risk management parameters
            
        Returns:
            Tuple of (success, message, order)
        """
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            confidence = signal.get('confidence', 0.5)
            entry_price = signal.get('entry_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            # Validate signal
            if not all([symbol, direction, entry_price]):
                return False, "Invalid signal parameters", None
            
            # Check if we already have a position
            existing_position = await self.portfolio_manager.get_position(symbol)
            if existing_position:
                return False, f"Position already exists for {symbol}", None
            
            # Calculate position size based on risk
            position_size = self._calculate_position_size(
                risk_params, entry_price, stop_loss
            )
            
            if position_size <= 0:
                return False, "Position size too small", None
            
            # Create order
            side = OrderSide.BUY if direction == 'long' else OrderSide.SELL
            order_type = OrderType.MARKET if confidence > 0.8 else OrderType.LIMIT
            
            order = await self.order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=position_size,
                price=entry_price if order_type == OrderType.LIMIT else None,
                metadata={
                    'signal_confidence': confidence,
                    'signal_id': signal.get('signal_id'),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            )
            
            # Simulate order execution (in real implementation, this would interact with exchange)
            await self._simulate_order_execution(order)
            
            # Open position
            position = await self.portfolio_manager.open_position(
                symbol=symbol,
                side=side,
                quantity=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={'order_id': order.id}
            )
            
            # Log execution
            self.execution_history.append({
                'timestamp': datetime.now(),
                'signal': signal,
                'order': order,
                'position': position,
                'risk_params': risk_params
            })
            
            return True, f"Signal executed successfully: {order.id}", order
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False, f"Execution error: {str(e)}", None
    
    def _calculate_position_size(
        self,
        risk_params: Dict[str, Any],
        entry_price: float,
        stop_loss: Optional[float]
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            risk_params: Risk management parameters
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size
        """
        account_balance = risk_params.get('account_balance', 10000)
        risk_per_trade = risk_params.get('risk_per_trade', 0.02)
        
        if not stop_loss:
            # Default to 2% risk if no stop loss
            risk_amount = account_balance * risk_per_trade
            return risk_amount / entry_price
        
        # Calculate based on stop loss
        risk_amount = account_balance * risk_per_trade
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        
        # Apply maximum position size limit
        max_position_size = risk_params.get('max_position_size', 0.1)
        max_size = account_balance * max_position_size / entry_price
        
        return min(position_size, max_size)
    
    async def _simulate_order_execution(self, order: Order):
        """
        Simulate order execution (for testing/demo).
        
        Args:
            order: Order to execute
        """
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # Update order status
        if order.order_type == OrderType.MARKET:
            await self.order_manager.update_order_status(
                order.id,
                OrderStatus.FILLED,
                filled_quantity=order.quantity,
                average_price=order.price or 0.0
            )
        else:
            # For limit orders, simulate partial fill
            fill_probability = 0.7  # 70% chance of fill
            if np.random.random() < fill_probability:
                filled_qty = order.quantity * 0.8  # 80% fill
                await self.order_manager.update_order_status(
                    order.id,
                    OrderStatus.PARTIALLY_FILLED,
                    filled_quantity=filled_qty,
                    average_price=order.price
                )
            else:
                await self.order_manager.update_order_status(
                    order.id,
                    OrderStatus.OPEN
                )
    
    async def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution summary statistics.
        
        Returns:
            Execution summary
        """
        if not self.execution_history:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_pnl': 0.0,
                'avg_execution_time': 0.0
            }
        
        successful = sum(1 for exec_record in self.execution_history 
                        if exec_record.get('order') and 
                        exec_record['order'].status == OrderStatus.FILLED)
        
        total_pnl = sum(
            exec_record.get('position', {}).unrealized_pnl 
            for exec_record in self.execution_history
            if exec_record.get('position')
        )
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': successful,
            'failed_executions': len(self.execution_history) - successful,
            'total_pnl': total_pnl,
            'avg_execution_time': 0.1  # Placeholder
        }


# Example usage
async def example_usage():
    """Example usage of trading services."""
    
    # Initialize services
    order_manager = OrderManager()
    portfolio_manager = PortfolioManager(initial_balance=10000)
    execution_service = ExecutionService(
        order_manager, portfolio_manager, {}
    )
    
    # Create a sample signal
    signal = {
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'confidence': 0.85,
        'entry_price': 50000,
        'stop_loss': 48000,
        'take_profit': 52000,
        'signal_id': 'signal_001'
    }
    
    risk_params = {
        'account_balance': 10000,
        'risk_per_trade': 0.02,
        'max_position_size': 0.1
    }
    
    # Execute signal
    success, message, order = await execution_service.execute_signal(signal, risk_params)
    print(f"Signal execution: {success}, {message}")
    
    if success and order:
        print(f"Order created: {order.id}")
        
        # Get portfolio summary
        portfolio = await portfolio_manager.get_portfolio_summary()
        print(f"Portfolio balance: {portfolio.total_balance}")
        print(f"Total P&L: {portfolio.total_pnl}")
        
        # Get execution summary
        summary = await execution_service.get_execution_summary()
        print(f"Execution summary: {summary}")


if __name__ == "__main__":
    asyncio.run(example_usage())
