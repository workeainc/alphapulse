import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.core.config import settings
from app.services.market_data_service import MarketDataService
from app.services.sentiment_service import SentimentService
from app.services.risk_manager import RiskManager
from app.strategies.strategy_manager import StrategyManager
from app.database.models import Trade, Strategy, MarketData
from app.database.connection import get_db

# Import real-time components
from app.strategies.real_time_signal_generator import RealTimeSignalGenerator
from app.data.real_time_processor import RealTimeCandlestickProcessor
from app.core.unified_websocket_client import UnifiedWebSocketClient, UnifiedWebSocketManager

# Import execution layer components
from execution.order_manager import OrderManager
from execution.portfolio_manager import PortfolioManager
from execution.sl_tp_manager import SLTPManager
from execution.position_scaling_manager import PositionScalingManager
from execution.exchange_trading_connector import ExchangeCredentials, ExchangeType

# Import latency tracking
from app.core.latency_tracker import track_trading_pipeline, latency_tracker

logger = logging.getLogger(__name__)

class TradingEngine:
    """Main trading engine that orchestrates all trading operations"""
    
    def __init__(self, exchange_credentials: Optional[ExchangeCredentials] = None):
        self.is_running = False
        self.market_data_service = MarketDataService()
        self.sentiment_service = SentimentService()
        self.risk_manager = RiskManager()
        self.strategy_manager = StrategyManager()
        
        # Execution layer components
        self.order_manager = OrderManager(exchange_credentials)
        self.portfolio_manager = PortfolioManager()
        self.sl_tp_manager = SLTPManager()
        self.position_scaling_manager = PositionScalingManager()
        
        # Real-time components
        self.signal_generator = RealTimeSignalGenerator()
        self.candlestick_processor = RealTimeCandlestickProcessor()
        self.websocket_client = UnifiedWebSocketClient()
        
        # Trading state
        self.open_positions = {}
        self.pending_orders = {}
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.consecutive_losses = 0
        
        # Performance tracking
        self.current_model_id = "trading_engine_v1"  # Current model identifier

    async def start(self):
        """Start the trading engine"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
            
        logger.info("🚀 Starting AlphaPulse Trading Engine...")
        self.is_running = True
        
        try:
            # Start all services
            await self.market_data_service.start()
            await self.sentiment_service.start()
            await self.risk_manager.start()
            await self.strategy_manager.start()
            
            # Start real-time components
            await self.signal_generator.start()
            await self.candlestick_processor.start()
            await self.websocket_client.start()
            
            # Start background tasks
            asyncio.create_task(self._trading_loop())
            asyncio.create_task(self._position_monitor())
            asyncio.create_task(self._performance_tracker())
            
            logger.info("✅ Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting trading engine: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            logger.warning("Trading engine is not running")
            return
            
        logger.info("🛑 Stopping Trading Engine...")
        self.is_running = False
        
        try:
            # Stop all services
            await self.market_data_service.stop()
            await self.sentiment_service.stop()
            await self.risk_manager.stop()
            await self.strategy_manager.stop()
            
            # Stop real-time components
            await self.signal_generator.stop()
            await self.candlestick_processor.stop()
            await self.websocket_client.stop()
            
            logger.info("✅ Trading Engine stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading engine: {e}")
    
    async def _trading_loop(self):
        """Main trading loop that processes signals and executes trades"""
        while self.is_running:
            try:
                # Wait for next update cycle
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
                # Get latest market data
                market_data = await self._get_market_data()
                if market_data is None or market_data.empty:
                    continue
                
                # Generate signals with latency tracking
                signals = await self._generate_signals(market_data)
                
                # Execute trades for each signal
                for signal in signals:
                    if await self.risk_manager.validate_signal(signal):
                        await self._execute_trade(signal, market_data)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get latest market data for all trading pairs"""
        try:
            market_data = {}
            
            for pair in self.trading_pairs:
                # Get data from candlestick processor
                pair_data = await self.candlestick_processor.get_latest_data(pair)
                if pair_data is not None:
                    market_data[pair] = pair_data
            
            if market_data:
                # Combine all pair data
                combined_data = pd.concat(market_data.values(), keys=market_data.keys())
                return combined_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return None
    
    @track_trading_pipeline(model_id="signal_generation", strategy_name="multi_strategy")
    async def _generate_signals(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals with latency tracking"""
        try:
            # Get market data for multiple timeframes
            signals = []
            
            # Generate signals from all strategies
            for strategy_name, strategy in self.strategy_manager.strategies.items():
                if not strategy.is_active:
                    continue
                
                # Calculate indicators
                data_with_indicators = strategy.calculate_indicators(market_data)
                
                # Generate signals
                strategy_signals = strategy.generate_signals(data_with_indicators, market_data['symbol'].iloc[0])
                
                # Add strategy information
                for signal in strategy_signals:
                    signal['strategy_name'] = strategy_name
                
                signals.extend(strategy_signals)
            
            # Filter and rank signals
            filtered_signals = await self._filter_signals(signals)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            return []
    
    async def _filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and rank trading signals"""
        try:
            filtered_signals = []
            
            for signal in signals:
                # Validate signal with risk manager
                if await self.risk_manager.validate_signal(signal):
                    # Calculate signal score
                    score = await self._calculate_signal_score(signal)
                    signal['score'] = score
                    
                    # Only include high-confidence signals
                    if score >= 0.7:
                        filtered_signals.append(signal)
            
            # Sort by score (highest first)
            filtered_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"❌ Error filtering signals: {e}")
            return []
    
    async def _calculate_signal_score(self, signal: Dict[str, Any]) -> float:
        """Calculate confidence score for a trading signal"""
        try:
            score = 0.0
            
            # Base confidence from signal
            base_confidence = signal.get('confidence', 0.5)
            score += base_confidence * 0.4
            
            # Strategy performance bonus
            strategy_name = signal.get('strategy_name', 'unknown')
            if strategy_name in self.strategy_manager.strategies:
                performance = self.strategy_manager.get_strategy_performance(strategy_name)
                win_rate = performance.get('win_rate', 0.5)
                score += win_rate * 0.3
            
            # Market condition bonus
            market_condition = await self._assess_market_condition()
            score += market_condition * 0.2
            
            # Sentiment bonus
            sentiment_score = await self.sentiment_service.get_market_sentiment()
            score += sentiment_score * 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating signal score: {e}")
            return 0.5
    
    async def _assess_market_condition(self) -> float:
        """Assess current market conditions (0.0 = poor, 1.0 = excellent)"""
        try:
            # This would typically analyze volatility, trend strength, etc.
            # For now, return a neutral score
            return 0.6
            
        except Exception as e:
            logger.error(f"❌ Error assessing market condition: {e}")
            return 0.5
    
    async def _process_signal(self, signal: Dict[str, Any], market_data: pd.DataFrame):
        """Process a trading signal and execute if appropriate"""
        try:
            signal_type = signal.get('signal_type', 'unknown')
            symbol = signal.get('symbol', 'unknown')
            confidence = signal.get('confidence', 0.0)
            
            logger.info(f"📊 Processing {signal_type} signal for {symbol} (confidence: {confidence:.2f})")
            
            # Check if we already have a position in this symbol
            if symbol in self.open_positions:
                logger.info(f"⚠️ Already have position in {symbol}, skipping signal")
                return
            
            # Check if signal conflicts with existing positions
            if await self._signal_conflicts_with_positions(signal):
                logger.info(f"⚠️ Signal conflicts with existing positions, skipping")
                return
            
            # Execute the trade
            await self._execute_trade(signal, market_data)
            
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
    
    async def _signal_conflicts_with_positions(self, signal: Dict[str, Any]) -> bool:
        """Check if signal conflicts with existing positions"""
        try:
            signal_type = signal.get('signal_type', 'unknown')
            
            # Check for conflicting signals in open positions
            for position in self.open_positions.values():
                if position['side'] == 'long' and signal_type == 'sell':
                    return True
                elif position['side'] == 'short' and signal_type == 'buy':
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking signal conflicts: {e}")
            return True  # Assume conflict on error for safety
    
    @track_trading_pipeline(model_id="trade_execution", strategy_name="execution_engine")
    async def _execute_trade(self, signal: Dict[str, Any], market_data: pd.DataFrame):
        """Execute trade with latency tracking"""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            
            # 1. Get current market price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ Could not get current price for {symbol}")
                return False
            
            # 2. Calculate position size
            position_size = await self._calculate_position_size(signal, current_price)
            
            # 3. Calculate stop loss and take profit
            stop_loss, take_profit = await self._calculate_sl_tp(signal, market_data, current_price)
            
            # 4. Place trading order
            order_result = await self._place_trading_order(
                symbol, signal_type, position_size, current_price, stop_loss, take_profit
            )
            
            if order_result and order_result.success:
                # 5. Update trading state
                position = {
                    'symbol': symbol,
                    'side': signal_type,
                    'entry_price': current_price,
                    'quantity': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(),
                    'order_id': order_result.order_id,
                    'exchange_order_id': order_result.exchange_order_id
                }
                
                self.open_positions[symbol] = position
                self.total_trades += 1
                
                # 6. Create position scaling plan
                await self._create_scaling_plan(symbol, position, market_data)
                
                logger.info(f"✅ Trade executed successfully: {symbol} {signal_type} {position_size} @ {current_price}")
                logger.info(f"   Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                
                # 7. Store trade in database
                await self._store_trade_record(signal, position, order_result)
                
                return True
            else:
                logger.error(f"❌ Failed to execute trade for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return False
    
    async def _calculate_position_size(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate position size using portfolio manager"""
        try:
            symbol = signal['symbol']
            confidence = signal['confidence']
            current_price = market_data.iloc[-1]['close']
            
            # Get account balance and risk parameters
            account_balance = await self._get_account_balance()
            risk_per_trade = self.risk_manager.get_risk_per_trade()
            
            # Calculate position size using portfolio manager
            position_size = self.portfolio_manager.calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=risk_per_trade,
                entry_price=current_price,
                stop_loss_distance=0.02,  # 2% default stop loss
                confidence=confidence
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _calculate_sl_tp(self, signal: Dict[str, Any], market_data: pd.DataFrame, entry_price: float) -> tuple:
        """Calculate stop loss and take profit using SL/TP manager"""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            
            # Calculate ATR for volatility-based stops
            atr = self._calculate_atr(market_data)
            
            # Use SL/TP manager to calculate levels
            if signal_type.lower() == 'buy':
                stop_loss = self.sl_tp_manager.calculate_atr_stop(
                    entry_price=entry_price,
                    side='buy',
                    atr=atr,
                    risk_multiplier=2.0
                )
                take_profit = self.sl_tp_manager.calculate_risk_reward_tp(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=2.0
                )
            else:  # sell/short
                stop_loss = self.sl_tp_manager.calculate_atr_stop(
                    entry_price=entry_price,
                    side='sell',
                    atr=atr,
                    risk_multiplier=2.0
                )
                take_profit = self.sl_tp_manager.calculate_risk_reward_tp(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=2.0
                )
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}")
            # Return default values
            if signal['signal_type'].lower() == 'buy':
                return entry_price * 0.98, entry_price * 1.04  # 2% SL, 4% TP
            else:
                return entry_price * 1.02, entry_price * 0.96  # 2% SL, 4% TP
    
    async def _place_trading_order(self, symbol: str, side: str, quantity: float, 
                                  price: float, stop_loss: float, take_profit: float):
        """Place trading order using order manager"""
        try:
            from execution.order_manager import Order, OrderType, OrderSide
            
            # Create order object
            order = Order(
                id="",
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,  # Start with market orders for simplicity
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Place order on exchange (default to binance)
            result = await self.order_manager.place_order(order, exchange="binance")
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing trading order: {e}")
            return None
    
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # This should integrate with the exchange connector
            # For now, return a default balance
            return 10000.0  # $10,000 default
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 10000.0
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean()
            
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def _store_trade_record(self, signal: Dict[str, Any], position: Dict[str, Any], order_result):
        """Store trade record in database"""
        try:
            # This should integrate with your database models
            # For now, just log the trade
            logger.info(f"📊 Trade recorded: {position['symbol']} {position['side']} "
                       f"{position['quantity']} @ {position['entry_price']}")
            
        except Exception as e:
            logger.error(f"Error storing trade record: {e}")
    
    async def _create_scaling_plan(self, symbol: str, position: Dict[str, Any], market_data: pd.DataFrame):
        """Create a position scaling plan for the new position"""
        try:
            from execution.position_scaling_manager import ScalingStrategy
            
            # Determine scaling strategy based on position confidence and market conditions
            confidence = position.get('confidence', 0.5)
            volatility = self._calculate_atr(market_data)
            
            if confidence > 0.8:
                strategy = ScalingStrategy.EXPONENTIAL
                levels = 4
            elif confidence > 0.6:
                strategy = ScalingStrategy.LINEAR
                levels = 3
            else:
                strategy = ScalingStrategy.LINEAR
                levels = 2
            
            # Create scaling plan
            scaling_plan = self.position_scaling_manager.create_scaling_plan(
                position_id=position.get('order_id', f"{symbol}_{datetime.now().timestamp()}"),
                symbol=symbol,
                side=position['side'],
                base_quantity=position['quantity'],
                entry_price=position['entry_price'],
                strategy=strategy,
                levels=levels
            )
            
            logger.info(f"📈 Created scaling plan for {symbol}: {strategy.value} strategy with {levels} levels")
            
        except Exception as e:
            logger.error(f"❌ Error creating scaling plan for {symbol}: {e}")
    
    async def _position_monitor(self):
        """Monitor open positions and manage exits"""
        while self.is_running:
            try:
                # Wait for next check
                await asyncio.sleep(10)
                
                # Check each open position
                for symbol, position in list(self.open_positions.items()):
                    await self._check_position_exit(symbol, position)
                
            except Exception as e:
                logger.error(f"❌ Error in position monitor: {e}")
                await asyncio.sleep(30)
    
    async def _check_position_exit(self, symbol: str, position: Dict[str, Any]):
        """Check if a position should be exited"""
        try:
            # Get current price
            current_data = await self.candlestick_processor.get_latest_data(symbol)
            if current_data is None:
                return
            
            current_price = current_data['close'].iloc[-1]
            
            # Check stop loss
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    await self._close_position(symbol, position, current_price, 'stop_loss')
                    return
                
                if current_price >= position['take_profit']:
                    await self._close_position(symbol, position, current_price, 'take_profit')
                    return
            else:  # short position
                if current_price >= position['stop_loss']:
                    await self._close_position(symbol, position, current_price, 'stop_loss')
                    return
                
                if current_price <= position['take_profit']:
                    await self._close_position(symbol, position, current_price, 'take_profit')
                    return
            
            # Check for manual exit signals
            exit_signal = await self._check_exit_signal(symbol, position, current_price)
            if exit_signal:
                await self._close_position(symbol, position, current_price, 'signal')
            
            # Check for scaling opportunities
            await self._check_scaling_opportunities(symbol, position, current_price)
                
        except Exception as e:
            logger.error(f"❌ Error checking position exit for {symbol}: {e}")
    
    async def _check_exit_signal(self, symbol: str, position: Dict[str, Any], current_price: float) -> bool:
        """Check if we should exit based on exit signals"""
        try:
            # This would check for reversal signals, trend changes, etc.
            # For now, return False (no exit signal)
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking exit signal for {symbol}: {e}")
            return False
    
    async def _check_scaling_opportunities(self, symbol: str, position: Dict[str, Any], current_price: float):
        """Check for position scaling opportunities"""
        try:
            # Get current market data for volatility calculation
            market_data = await self._get_market_data()
            if market_data is None:
                return
            
            current_volatility = self._calculate_atr(market_data)
            
            # Check scaling triggers
            scaling_levels = self.position_scaling_manager.check_scaling_triggers(
                symbol=symbol,
                current_price=current_price,
                current_volatility=current_volatility
            )
            
            # Execute any triggered scaling levels
            for level in scaling_levels:
                if not level.executed:
                    await self._execute_scaling_level(symbol, position, level, current_price)
                    
        except Exception as e:
            logger.error(f"❌ Error checking scaling opportunities for {symbol}: {e}")
    
    async def _execute_scaling_level(self, symbol: str, position: Dict[str, Any], level, current_price: float):
        """Execute a scaling level for position management"""
        try:
            # Calculate scaling quantity
            scaling_quantity = level.quantity
            
            # Determine scaling side (opposite of original position for scale-in)
            scaling_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            # Place scaling order
            order_result = await self._place_trading_order(
                symbol=symbol,
                side=scaling_side,
                quantity=scaling_quantity,
                price=current_price,
                stop_loss=0,  # No stop loss for scaling orders
                take_profit=0   # No take profit for scaling orders
            )
            
            if order_result.success:
                # Record scaling execution
                scaling_execution = self.position_scaling_manager.execute_scaling_level(
                    level=level,
                    execution_price=current_price
                )
                
                # Update position quantity
                if scaling_side == 'buy':  # Scale in
                    position['quantity'] += scaling_quantity
                    logger.info(f"📈 Scaled IN {symbol}: +{scaling_quantity} @ {current_price}")
                else:  # Scale out
                    position['quantity'] -= scaling_quantity
                    logger.info(f"📉 Scaled OUT {symbol}: -{scaling_quantity} @ {current_price}")
                
                # Remove position if fully scaled out
                if position['quantity'] <= 0:
                    await self._close_position(symbol, position, current_price, 'scaled_out')
                    
            else:
                logger.error(f"❌ Failed to execute scaling level for {symbol}: {order_result.error}")
                
        except Exception as e:
            logger.error(f"❌ Error executing scaling level for {symbol}: {e}")
    
    async def _close_position(self, symbol: str, position: Dict[str, Any], exit_price: float, reason: str):
        """Close an open position"""
        try:
            # Calculate P&L
            if position['side'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Update position
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['status'] = 'closed'
            position['exit_reason'] = reason
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            # Update performance metrics
            self.daily_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            
            # Log the exit
            logger.info(f"🔒 Closed {position['side']} position in {symbol}")
            logger.info(f"   Entry: {position['entry_price']:.4f}, Exit: {exit_price:.4f}")
            logger.info(f"   P&L: {pnl:.2f} USDT, Reason: {reason}")
            
            # Update risk manager
            self.risk_manager.update_portfolio_metrics(
                self.daily_pnl, 
                len(self.open_positions)
            )
            
        except Exception as e:
            logger.error(f"❌ Error closing position for {symbol}: {e}")
    
    async def _performance_tracker(self):
        """Track and log trading performance"""
        while self.is_running:
            try:
                # Wait for next update
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Calculate performance metrics
                win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
                
                # Log performance
                logger.info(f"📊 Performance Update:")
                logger.info(f"   Daily P&L: {self.daily_pnl:.2f} USDT")
                logger.info(f"   Total Trades: {self.total_trades}")
                logger.info(f"   Win Rate: {win_rate:.2%}")
                logger.info(f"   Open Positions: {len(self.open_positions)}")
                
                # Get risk summary
                risk_summary = self.risk_manager.get_risk_summary()
                logger.info(f"   Risk Status: {risk_summary.get('risk_status', 'unknown')}")
                
            except Exception as e:
                logger.error(f"❌ Error in performance tracker: {e}")
                await asyncio.sleep(60)
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        try:
            return {
                "is_running": self.is_running,
                "open_positions": len(self.open_positions),
                "daily_pnl": self.daily_pnl,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                "trading_pairs": self.trading_pairs,
                "risk_summary": self.risk_manager.get_risk_summary(),
                "strategy_status": self.strategy_manager.get_strategy_status()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting trading status: {e}")
            return {}
    
    async def get_open_positions(self) -> Dict[str, Any]:
        """Get details of all open positions"""
        return self.open_positions.copy()
    
    async def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.info("🔄 Closing all open positions...")
            
            for symbol, position in list(self.open_positions.items()):
                # Get current price
                current_data = await self.candlestick_processor.get_latest_data(symbol)
                if current_data is not None:
                    current_price = current_data['close'].iloc[-1]
                    await self._close_position(symbol, position, current_price, 'manual_close')
            
            logger.info("✅ All positions closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing all positions: {e}")
    
    async def update_trading_parameters(self, parameters: Dict[str, Any]):
        """Update trading parameters"""
        try:
            # Update risk parameters
            if 'risk_limits' in parameters:
                await self.risk_manager.update_risk_limits(parameters['risk_limits'])
            
            # Update strategy parameters
            if 'strategy_parameters' in parameters:
                for strategy_name, params in parameters['strategy_parameters'].items():
                    await self.strategy_manager.update_strategy_parameters(strategy_name, params)
            
            logger.info("✅ Trading parameters updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error updating trading parameters: {e}")

    async def get_latency_summary(self) -> Dict[str, Any]:
        """Get latency summary for the trading engine"""
        try:
            # Get overall latency summary
            overall_summary = latency_tracker.get_latency_summary()
            
            # Get pipeline-specific summary
            pipeline_summary = latency_tracker.get_pipeline_summary()
            
            # Get recent metrics by strategy
            strategy_metrics = {}
            for strategy_name in ['signal_generation', 'trade_execution', 'multi_strategy']:
                metrics = latency_tracker.get_pipeline_metrics_by_strategy(strategy_name, minutes=60)
                if metrics:
                    strategy_metrics[strategy_name] = {
                        'count': len(metrics),
                        'avg_latency_ms': sum(m.total_latency_ms for m in metrics) / len(metrics),
                        'success_rate': sum(1 for m in metrics if m.success) / len(metrics)
                    }
            
            return {
                'overall_summary': overall_summary,
                'pipeline_summary': pipeline_summary,
                'strategy_metrics': strategy_metrics,
                'current_model_id': self.current_model_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting latency summary: {e}")
            return {"error": str(e)}
