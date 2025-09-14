"""
Core Trading Engine for AlphaPlus
Central orchestrator for all trading operations
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from strategies.strategy_manager import StrategyManager
from data.candlestick_collector import CandlestickCollector
from execution.order_manager import OrderManager
from database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Trading engine states"""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class TradingConfig:
    """Trading configuration"""
    max_positions: int = 10
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    enable_short_trading: bool = True
    enable_leverage: bool = False
    max_leverage: float = 1.0

class TradingEngine:
    """Core trading engine orchestrator"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.logger = logger
        
        # Core components
        self.strategy_manager = StrategyManager()
        self.data_collector = CandlestickCollector()
        self.order_manager = OrderManager()
        self.db_connection = TimescaleDBConnection()
        
        # State management
        self.state = TradingState.IDLE
        self.active_positions: Dict[str, Any] = {}
        self.daily_pnl = 0.0
        self.last_update = datetime.now(timezone.utc)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    async def initialize(self):
        """Initialize the trading engine"""
        try:
            self.logger.info("Initializing Trading Engine...")
            
            # Initialize components
            await self.strategy_manager.initialize()
            await self.data_collector.initialize()
            await self.order_manager.initialize()
            await self.db_connection.initialize()
            
            self.state = TradingState.IDLE
            self.logger.info("Trading Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Trading Engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def start_trading(self):
        """Start the trading engine"""
        if self.state != TradingState.IDLE:
            raise ValueError(f"Cannot start trading from state: {self.state}")
        
        try:
            self.logger.info("Starting Trading Engine...")
            self.state = TradingState.ACTIVE
            
            # Start data collection
            await self.data_collector.start_streaming()
            
            # Start strategy monitoring
            await self.strategy_manager.start_monitoring()
            
            self.logger.info("Trading Engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Trading Engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def stop_trading(self):
        """Stop the trading engine"""
        try:
            self.logger.info("Stopping Trading Engine...")
            
            # Stop components
            await self.data_collector.stop_streaming()
            await self.strategy_manager.stop_monitoring()
            
            # Close all positions
            await self.close_all_positions()
            
            self.state = TradingState.STOPPED
            self.logger.info("Trading Engine stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop Trading Engine: {e}")
            self.state = TradingState.ERROR
            raise
    
    async def process_signals(self):
        """Process trading signals from strategies"""
        try:
            signals = await self.strategy_manager.get_active_signals()
            
            for signal in signals:
                if await self.validate_signal(signal):
                    await self.execute_signal(signal)
                    
        except Exception as e:
            self.logger.error(f"Error processing signals: {e}")
    
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trading signal"""
        try:
            # Check risk limits
            if not await self.check_risk_limits(signal):
                return False
            
            # Check position limits
            if not await self.check_position_limits(signal):
                return False
            
            # Check signal quality
            if not await self.check_signal_quality(signal):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    async def execute_signal(self, signal: Dict[str, Any]):
        """Execute a trading signal"""
        try:
            self.logger.info(f"Executing signal: {signal['id']}")
            
            # Create order
            order = await self.order_manager.create_order(signal)
            
            # Execute order
            result = await self.order_manager.execute_order(order)
            
            if result['status'] == 'filled':
                await self.record_trade(signal, result)
                self.total_trades += 1
                
                # Update position tracking
                await self.update_position_tracking(signal, result)
                
            self.logger.info(f"Signal executed: {result['status']}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    async def check_risk_limits(self, signal: Dict[str, Any]) -> bool:
        """Check if signal meets risk limits"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.config.max_daily_loss:
                self.logger.warning("Daily loss limit reached")
                return False
            
            # Check per-trade risk
            signal_risk = signal.get('risk_amount', 0)
            if signal_risk > self.config.risk_per_trade:
                self.logger.warning(f"Signal risk {signal_risk} exceeds limit {self.config.risk_per_trade}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def check_position_limits(self, signal: Dict[str, Any]) -> bool:
        """Check if signal meets position limits"""
        try:
            # Check max positions
            if len(self.active_positions) >= self.config.max_positions:
                self.logger.warning("Maximum positions reached")
                return False
            
            # Check symbol-specific limits
            symbol = signal.get('symbol')
            if symbol in self.active_positions:
                self.logger.warning(f"Position already exists for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False
    
    async def check_signal_quality(self, signal: Dict[str, Any]) -> bool:
        """Check signal quality metrics"""
        try:
            # Check confidence score
            confidence = signal.get('confidence', 0)
            if confidence < 0.7:  # Minimum 70% confidence
                self.logger.warning(f"Low confidence signal: {confidence}")
                return False
            
            # Check signal age
            signal_time = signal.get('timestamp')
            if signal_time:
                age = datetime.now(timezone.utc) - signal_time
                if age.total_seconds() > 300:  # 5 minutes max age
                    self.logger.warning(f"Signal too old: {age}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking signal quality: {e}")
            return False
    
    async def close_all_positions(self):
        """Close all active positions"""
        try:
            self.logger.info("Closing all positions...")
            
            for position_id, position in self.active_positions.items():
                await self.order_manager.close_position(position_id)
            
            self.active_positions.clear()
            self.logger.info("All positions closed")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    async def update_position_tracking(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """Update position tracking after trade execution"""
        try:
            position_id = result.get('position_id')
            if position_id:
                self.active_positions[position_id] = {
                    'symbol': signal.get('symbol'),
                    'side': signal.get('side'),
                    'entry_price': result.get('fill_price'),
                    'quantity': result.get('fill_quantity'),
                    'timestamp': datetime.now(timezone.utc)
                }
                
        except Exception as e:
            self.logger.error(f"Error updating position tracking: {e}")
    
    async def record_trade(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """Record trade in database"""
        try:
            trade_data = {
                'signal_id': signal.get('id'),
                'symbol': signal.get('symbol'),
                'side': signal.get('side'),
                'entry_price': result.get('fill_price'),
                'quantity': result.get('fill_quantity'),
                'timestamp': datetime.now(timezone.utc),
                'strategy': signal.get('strategy'),
                'confidence': signal.get('confidence')
            }
            
            # Save to database
            await self.db_connection.save_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        try:
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            return {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'daily_pnl': self.daily_pnl,
                'active_positions': len(self.active_positions),
                'state': self.state.value,
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for trading engine"""
        try:
            return {
                'status': 'healthy' if self.state != TradingState.ERROR else 'unhealthy',
                'state': self.state.value,
                'components': {
                    'strategy_manager': await self.strategy_manager.health_check(),
                    'data_collector': await self.data_collector.health_check(),
                    'order_manager': await self.order_manager.health_check(),
                    'database': await self.db_connection.health_check()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
