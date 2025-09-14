"""
Enhanced Order Manager for AlphaPlus
Manages order creation, execution, and position management
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from .exchange_trading_connector import ExchangeTradingConnector
from .portfolio_manager import PortfolioManager
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order structure"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Position:
    """Position structure"""
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    last_update: datetime = None
    metadata: Optional[Dict[str, Any]] = None

class OrderManager:
    """Enhanced order manager"""
    
    def __init__(self):
        self.logger = logger
        
        # Core components
        self.exchange_connector = ExchangeTradingConnector()
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.order_timeout = 300  # 5 minutes
        
        # Performance tracking
        self.orders_created = 0
        self.orders_filled = 0
        self.orders_cancelled = 0
        self.orders_rejected = 0
    
    async def initialize(self):
        """Initialize the order manager"""
        try:
            self.logger.info("Initializing Order Manager...")
            
            # Initialize components
            await self.exchange_connector.initialize()
            await self.portfolio_manager.initialize()
            await self.risk_manager.initialize()
            
            # Load existing positions
            await self._load_positions()
            
            self.logger.info("Order Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Order Manager: {e}")
            raise
    
    async def create_order(self, signal: Dict[str, Any]) -> Order:
        """Create a new order from a trading signal"""
        try:
            self.logger.info(f"Creating order for signal: {signal.get('id')}")
            
            # Validate signal
            if not await self._validate_signal(signal):
                raise ValueError("Invalid trading signal")
            
            # Calculate order parameters
            order_params = await self._calculate_order_parameters(signal)
            
            # Create order object
            order = Order(
                id=str(uuid.uuid4()),
                symbol=signal['symbol'],
                side=OrderSide(signal['side']),
                order_type=OrderType.MARKET,  # Default to market order
                quantity=order_params['quantity'],
                price=order_params.get('price'),
                stop_price=order_params.get('stop_price'),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'signal_id': signal.get('id'),
                    'strategy': signal.get('strategy'),
                    'confidence': signal.get('confidence'),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit')
                }
            )
            
            # Store order
            self.active_orders[order.id] = order
            self.order_history.append(order)
            self.orders_created += 1
            
            self.logger.info(f"Created order: {order.id} for {order.symbol} {order.side.value}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            raise
    
    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute an order through the exchange"""
        try:
            self.logger.info(f"Executing order: {order.id}")
            
            # Check if order is still valid
            if order.status != OrderStatus.PENDING:
                raise ValueError(f"Order {order.id} is not in pending status")
            
            # Execute through exchange
            execution_result = await self.exchange_connector.execute_order(order)
            
            if execution_result['status'] == 'success':
                # Update order status
                order.status = OrderStatus.FILLED
                order.filled_quantity = execution_result['filled_quantity']
                order.average_fill_price = execution_result['fill_price']
                order.commission = execution_result.get('commission', 0.0)
                
                # Remove from active orders
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                
                # Update position
                await self._update_position_from_order(order, execution_result)
                
                self.orders_filled += 1
                
                result = {
                    'status': 'filled',
                    'order_id': order.id,
                    'fill_price': execution_result['fill_price'],
                    'fill_quantity': execution_result['filled_quantity'],
                    'commission': execution_result.get('commission', 0.0),
                    'position_id': execution_result.get('position_id')
                }
                
            else:
                # Order failed
                order.status = OrderStatus.REJECTED
                self.orders_rejected += 1
                
                result = {
                    'status': 'rejected',
                    'order_id': order.id,
                    'error': execution_result.get('error', 'Unknown error')
                }
            
            self.logger.info(f"Order execution result: {result['status']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.id}: {e}")
            
            # Mark order as rejected
            order.status = OrderStatus.REJECTED
            self.orders_rejected += 1
            
            return {
                'status': 'error',
                'order_id': order.id,
                'error': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            order = self.active_orders[order_id]
            
            # Cancel through exchange
            cancel_result = await self.exchange_connector.cancel_order(order_id)
            
            if cancel_result['status'] == 'success':
                order.status = OrderStatus.CANCELLED
                del self.active_orders[order_id]
                self.orders_cancelled += 1
                
                self.logger.info(f"Cancelled order: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}: {cancel_result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def close_position(self, position_id: str) -> bool:
        """Close a position"""
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            
            # Create closing order
            close_order = Order(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                timestamp=datetime.now(timezone.utc),
                metadata={'action': 'close_position', 'position_id': position_id}
            )
            
            # Execute closing order
            result = await self.execute_order(close_order)
            
            if result['status'] == 'filled':
                # Remove position
                del self.positions[position_id]
                
                self.logger.info(f"Closed position: {position_id}")
                return True
            else:
                self.logger.error(f"Failed to close position {position_id}: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    async def update_positions(self):
        """Update all position prices and PnL"""
        try:
            for position_id, position in self.positions.items():
                # Get current price
                current_price = await self.exchange_connector.get_current_price(position.symbol)
                
                if current_price is not None:
                    position.current_price = current_price
                    position.last_update = datetime.now(timezone.utc)
                    
                    # Calculate unrealized PnL
                    if position.side == OrderSide.BUY:
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    
                    # Check stop loss and take profit
                    await self._check_position_exits(position)
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trading signal"""
        try:
            required_fields = ['symbol', 'side', 'strategy', 'confidence']
            
            for field in required_fields:
                if field not in signal:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Check confidence level
            if signal['confidence'] < 0.5:
                self.logger.warning(f"Low confidence signal: {signal['confidence']}")
                return False
            
            # Check risk limits
            if not await self.risk_manager.check_signal_risk(signal):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    async def _calculate_order_parameters(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate order parameters from signal"""
        try:
            # Get current price
            current_price = await self.exchange_connector.get_current_price(signal['symbol'])
            
            if current_price is None:
                raise ValueError(f"Could not get current price for {signal['symbol']}")
            
            # Calculate position size based on risk management
            position_size = await self.risk_manager.calculate_position_size(
                signal['symbol'], 
                signal.get('stop_loss'), 
                signal.get('confidence', 0.7)
            )
            
            # Calculate stop loss and take profit
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            if not stop_loss:
                # Calculate default stop loss based on ATR or percentage
                atr = await self._get_atr(signal['symbol'])
                if signal['side'] == 'buy':
                    stop_loss = current_price - (atr * 2)
                else:
                    stop_loss = current_price + (atr * 2)
            
            if not take_profit:
                # Calculate default take profit (2:1 risk:reward)
                if signal['side'] == 'buy':
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * 2)
                else:
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * 2)
            
            return {
                'quantity': position_size,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order parameters: {e}")
            raise
    
    async def _update_position_from_order(self, order: Order, execution_result: Dict[str, Any]):
        """Update position tracking after order execution"""
        try:
            position_id = execution_result.get('position_id') or str(uuid.uuid4())
            
            position = Position(
                id=position_id,
                symbol=order.symbol,
                side=order.side,
                entry_price=execution_result['fill_price'],
                quantity=execution_result['filled_quantity'],
                current_price=execution_result['fill_price'],
                entry_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc),
                stop_loss=order.metadata.get('stop_loss'),
                take_profit=order.metadata.get('take_profit'),
                metadata=order.metadata
            )
            
            self.positions[position_id] = position
            
            self.logger.info(f"Created position: {position_id} for {order.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating position from order: {e}")
    
    async def _check_position_exits(self, position: Position):
        """Check if position should be closed based on stop loss or take profit"""
        try:
            if position.stop_loss is not None:
                if (position.side == OrderSide.BUY and position.current_price <= position.stop_loss) or \
                   (position.side == OrderSide.SELL and position.current_price >= position.stop_loss):
                    
                    self.logger.info(f"Stop loss triggered for position {position.id}")
                    await self.close_position(position.id)
                    return
            
            if position.take_profit is not None:
                if (position.side == OrderSide.BUY and position.current_price >= position.take_profit) or \
                   (position.side == OrderSide.SELL and position.current_price <= position.take_profit):
                    
                    self.logger.info(f"Take profit triggered for position {position.id}")
                    await self.close_position(position.id)
                    return
                    
        except Exception as e:
            self.logger.error(f"Error checking position exits: {e}")
    
    async def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Get Average True Range for a symbol"""
        try:
            # This would typically get ATR from technical indicators
            # For now, return a default value
            return 0.02  # 2% default volatility
        except Exception as e:
            self.logger.error(f"Error getting ATR for {symbol}: {e}")
            return 0.02
    
    async def _load_positions(self):
        """Load existing positions from database or exchange"""
        try:
            # This would typically load from database
            # For now, start with empty positions
            self.positions = {}
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    async def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get a specific order by ID"""
        return self.active_orders.get(order_id)
    
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    async def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get a specific position by ID"""
        return self.positions.get(position_id)
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get order manager performance summary"""
        try:
            total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self.positions.values())
            
            return {
                'orders_created': self.orders_created,
                'orders_filled': self.orders_filled,
                'orders_cancelled': self.orders_cancelled,
                'orders_rejected': self.orders_rejected,
                'active_orders': len(self.active_orders),
                'active_positions': len(self.positions),
                'total_pnl': total_pnl,
                'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
                'realized_pnl': sum(p.realized_pnl for p in self.positions.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for order manager"""
        try:
            return {
                'status': 'healthy',
                'active_orders': len(self.active_orders),
                'active_positions': len(self.positions),
                'exchange_connected': await self.exchange_connector.health_check(),
                'portfolio_healthy': await self.portfolio_manager.health_check(),
                'risk_manager_healthy': await self.risk_manager.health_check()
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
