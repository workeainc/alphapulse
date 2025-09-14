"""
Advanced Backtesting Framework for AlphaPulse
Provides comprehensive strategy testing, optimization, and performance analysis
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np

# Import our components
try:
    from ..database.connection import TimescaleDBConnection
    from ..strategies.strategy_manager import StrategyManager
except ImportError as e:
    logging.warning(f"Some imports not available: {e}")
    TimescaleDBConnection = None
    StrategyManager = None

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Backtest configuration parameters"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    timeframe: str
    commission_rate: float
    slippage_rate: float
    risk_free_rate: float
    max_position_size: float
    strategy_params: Dict[str, Any]

@dataclass
class BacktestResult:
    """Comprehensive backtest result"""
    config: BacktestConfig
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    execution_time: float

class AdvancedBacktestingFramework:
    """Advanced backtesting framework with optimization capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Framework configuration
        self.enable_parallel_backtests = self.config.get('enable_parallel_backtests', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Component references
        self.db_connection = None
        self.strategy_manager = None
        
        # Backtest state
        self.active_backtests = {}  # backtest_id -> backtest_task
        self.backtest_results = {}  # backtest_id -> result
        
        # Performance tracking
        self.stats = {
            'total_backtests': 0,
            'completed_backtests': 0,
            'failed_backtests': 0,
            'last_backtest': None,
            'execution_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.backtest_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize backtesting components"""
        try:
            # Initialize database connection if available
            if TimescaleDBConnection:
                db_config = self.config.get('database', {})
                self.db_connection = TimescaleDBConnection(db_config)
                self.logger.info("Database connection initialized for backtesting")
            
            # Initialize strategy manager if available
            if StrategyManager:
                self.strategy_manager = StrategyManager()
                self.logger.info("Strategy manager initialized for backtesting")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize backtesting components: {e}")
    
    async def initialize(self):
        """Initialize the backtesting framework"""
        try:
            self.logger.info("Initializing Advanced Backtesting Framework...")
            
            # Initialize database connection
            if self.db_connection:
                await self.db_connection.initialize()
            
            # Initialize strategy manager
            if self.strategy_manager:
                await self.strategy_manager.initialize()
            
            self.logger.info("Advanced Backtesting Framework initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backtesting framework: {e}")
            raise
    
    async def run_backtest(self, config: BacktestConfig) -> str:
        """Run a single backtest"""
        try:
            backtest_id = f"backtest_{int(time.time())}_{len(self.active_backtests)}"
            
            # Create backtest task
            backtest_task = asyncio.create_task(self._execute_backtest(backtest_id, config))
            self.active_backtests[backtest_id] = backtest_task
            
            # Update statistics
            self.stats['total_backtests'] += 1
            self.stats['last_backtest'] = datetime.now()
            
            self.logger.info(f"Started backtest {backtest_id} for {config.symbols}")
            return backtest_id
            
        except Exception as e:
            self.logger.error(f"Failed to start backtest: {e}")
            raise
    
    async def _execute_backtest(self, backtest_id: str, config: BacktestConfig) -> BacktestResult:
        """Execute a single backtest"""
        try:
            start_time = time.time()
            
            # Initialize backtest environment
            portfolio = await self._initialize_backtest_portfolio(config)
            trades = []
            equity_curve = []
            
            # Get historical data
            historical_data = await self._get_historical_data(config)
            if not historical_data:
                raise ValueError("No historical data available for backtest")
            
            # Run backtest simulation
            for timestamp, data_point in historical_data:
                try:
                    # Update portfolio with current data
                    await self._update_portfolio(portfolio, data_point, timestamp)
                    
                    # Generate strategy signals
                    signals = await self._generate_strategy_signals(data_point, config)
                    
                    # Execute trades based on signals
                    new_trades = await self._execute_backtest_trades(portfolio, signals, data_point, timestamp, config)
                    trades.extend(new_trades)
                    
                    # Record equity curve
                    equity = await self._calculate_portfolio_equity(portfolio, data_point, timestamp)
                    equity_curve.append({
                        'timestamp': timestamp,
                        'equity': equity,
                        'cash': portfolio.get('cash', 0),
                        'positions': portfolio.get('positions', {})
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing data point at {timestamp}: {e}")
                    continue
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(trades, equity_curve, config)
            risk_metrics = await self._calculate_risk_metrics(trades, equity_curve, config)
            
            # Create backtest result
            execution_time = time.time() - start_time
            result = BacktestResult(
                config=config,
                trades=trades,
                equity_curve=equity_curve,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                execution_time=execution_time
            )
            
            # Store result
            self.backtest_results[backtest_id] = result
            
            # Update statistics
            self.stats['completed_backtests'] += 1
            self.stats['execution_times'].append(execution_time)
            
            # Trigger callbacks
            await self._trigger_callbacks('backtest_completed', result)
            
            self.logger.info(f"Completed backtest {backtest_id} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest {backtest_id} failed: {e}")
            self.stats['failed_backtests'] += 1
            raise
        finally:
            # Clean up active backtest
            if backtest_id in self.active_backtests:
                del self.active_backtests[backtest_id]
    
    async def _initialize_backtest_portfolio(self, config: BacktestConfig) -> Dict[str, Any]:
        """Initialize portfolio for backtest"""
        try:
            portfolio = {
                'cash': config.initial_capital,
                'positions': {},
                'total_value': config.initial_capital,
                'initial_capital': config.initial_capital,
                'trades': []
            }
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backtest portfolio: {e}")
            raise
    
    async def _get_historical_data(self, config: BacktestConfig) -> List[Tuple[datetime, Dict[str, Any]]]:
        """Get historical data for backtest period"""
        try:
            if not self.db_connection:
                # Generate mock data for testing
                return self._generate_mock_historical_data(config)
            
            # Get candlestick data from database
            data = []
            for symbol in config.symbols:
                candlesticks = await self.db_connection.get_candlestick_data(
                    symbol=symbol,
                    timeframe=config.timeframe,
                    start_time=config.start_date,
                    end_time=config.end_date,
                    limit=10000
                )
                
                for candle in candlesticks:
                    timestamp = candle['timestamp']
                    data_point = {
                        'symbol': symbol,
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': candle['volume'],
                        'timestamp': timestamp
                    }
                    data.append((timestamp, data_point))
            
            # Sort by timestamp
            data.sort(key=lambda x: x[0])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return []
    
    def _generate_mock_historical_data(self, config: BacktestConfig) -> List[Tuple[datetime, Dict[str, Any]]]:
        """Generate mock historical data for testing"""
        try:
            data = []
            current_time = config.start_date
            base_prices = {'BTC/USDT': 50000, 'ETH/USDT': 3000, 'ADA/USDT': 1.0}
            
            while current_time <= config.end_date:
                for symbol in config.symbols:
                    base_price = base_prices.get(symbol, 100)
                    
                    # Generate realistic price movement
                    price_change = np.random.normal(0, 0.02)  # 2% daily volatility
                    current_price = base_price * (1 + price_change)
                    
                    # Generate OHLCV data
                    high = current_price * (1 + abs(np.random.normal(0, 0.01)))
                    low = current_price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = base_price
                    close_price = current_price
                    volume = np.random.uniform(1000, 10000)
                    
                    data_point = {
                        'symbol': symbol,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close_price,
                        'volume': volume,
                        'timestamp': current_time
                    }
                    
                    data.append((current_time, data_point))
                    base_prices[symbol] = current_price
                
                # Move to next timeframe
                if config.timeframe == '1h':
                    current_time += timedelta(hours=1)
                elif config.timeframe == '1d':
                    current_time += timedelta(days=1)
                else:
                    current_time += timedelta(minutes=1)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate mock data: {e}")
            return []
    
    async def _update_portfolio(self, portfolio: Dict[str, Any], data_point: Dict[str, Any], timestamp: datetime):
        """Update portfolio with current market data"""
        try:
            symbol = data_point['symbol']
            current_price = data_point['close']
            
            # Update position values
            if symbol in portfolio['positions']:
                position = portfolio['positions'][symbol]
                position['current_value'] = position['quantity'] * current_price
                position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
            
            # Recalculate total portfolio value
            total_value = portfolio['cash']
            for pos in portfolio['positions'].values():
                total_value += pos['current_value']
            
            portfolio['total_value'] = total_value
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio: {e}")
    
    async def _generate_strategy_signals(self, data_point: Dict[str, Any], config: BacktestConfig) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy parameters"""
        try:
            signals = []
            
            # Simple moving average crossover strategy for demonstration
            if 'ma_short' in config.strategy_params and 'ma_long' in config.strategy_params:
                # Generate random signals for testing
                if np.random.random() < 0.1:  # 10% chance of signal
                    signal_type = 'buy' if np.random.random() < 0.5 else 'sell'
                    confidence = np.random.uniform(0.6, 0.9)
                    
                    signals.append({
                        'symbol': data_point['symbol'],
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'price': data_point['close'],
                        'timestamp': data_point['timestamp'],
                        'strategy': 'moving_average_crossover'
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to generate strategy signals: {e}")
            return []
    
    async def _execute_backtest_trades(self, portfolio: Dict[str, Any], signals: List[Dict[str, Any]], 
                                     data_point: Dict[str, Any], timestamp: datetime, config: BacktestConfig) -> List[Dict[str, Any]]:
        """Execute trades based on signals in backtest environment"""
        try:
            trades = []
            
            for signal in signals:
                try:
                    symbol = signal['symbol']
                    signal_type = signal['signal_type']
                    price = signal['price']
                    
                    # Calculate position size
                    position_size = await self._calculate_position_size(portfolio, signal, config)
                    
                    if position_size <= 0:
                        continue
                    
                    # Execute trade
                    trade = await self._execute_single_trade(portfolio, signal, position_size, price, timestamp, config)
                    if trade:
                        trades.append(trade)
                        portfolio['trades'].append(trade)
                
                except Exception as e:
                    self.logger.warning(f"Failed to execute signal {signal}: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to execute backtest trades: {e}")
            return []
    
    async def _calculate_position_size(self, portfolio: Dict[str, Any], signal: Dict[str, Any], 
                                     config: BacktestConfig) -> float:
        """Calculate position size based on risk management rules"""
        try:
            available_capital = portfolio['cash']
            max_position_value = available_capital * config.max_position_size
            
            # Simple position sizing - 10% of available capital per trade
            position_value = min(max_position_value, available_capital * 0.1)
            
            return position_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return 0.0
    
    async def _execute_single_trade(self, portfolio: Dict[str, Any], signal: Dict[str, Any], 
                                   position_size: float, price: float, timestamp: datetime, 
                                   config: BacktestConfig) -> Optional[Dict[str, Any]]:
        """Execute a single trade in backtest environment"""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            
            # Apply slippage
            execution_price = price * (1 + config.slippage_rate) if signal_type == 'buy' else price * (1 - config.slippage_rate)
            
            # Calculate quantity
            quantity = position_size / execution_price
            
            # Calculate commission
            commission = position_size * config.commission_rate
            
            # Update portfolio
            if signal_type == 'buy':
                if symbol not in portfolio['positions']:
                    portfolio['positions'][symbol] = {
                        'quantity': 0,
                        'cost_basis': 0,
                        'current_value': 0,
                        'unrealized_pnl': 0
                    }
                
                position = portfolio['positions'][symbol]
                total_cost = position['cost_basis'] + position_size + commission
                total_quantity = position['quantity'] + quantity
                
                position['quantity'] = total_quantity
                position['cost_basis'] = total_cost
                position['current_value'] = total_quantity * execution_price
                
                portfolio['cash'] -= (position_size + commission)
                
            elif signal_type == 'sell':
                if symbol in portfolio['positions']:
                    position = portfolio['positions'][symbol]
                    
                    if position['quantity'] >= quantity:
                        # Partial or full sale
                        sale_ratio = quantity / position['quantity']
                        cost_basis_sold = position['cost_basis'] * sale_ratio
                        
                        position['quantity'] -= quantity
                        position['cost_basis'] -= cost_basis_sold
                        position['current_value'] = position['quantity'] * execution_price
                        
                        # Remove position if fully sold
                        if position['quantity'] <= 0:
                            del portfolio['positions'][symbol]
                        
                        portfolio['cash'] += (position_size - commission)
                    else:
                        # Not enough shares to sell
                        continue
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'side': signal_type,
                'quantity': quantity,
                'price': execution_price,
                'value': position_size,
                'commission': commission,
                'timestamp': timestamp,
                'strategy': signal.get('strategy', 'unknown'),
                'confidence': signal.get('confidence', 0.0)
            }
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Failed to execute single trade: {e}")
            return None
    
    async def _calculate_portfolio_equity(self, portfolio: Dict[str, Any], data_point: Dict[str, Any], 
                                        timestamp: datetime) -> float:
        """Calculate current portfolio equity"""
        try:
            total_equity = portfolio['cash']
            
            for symbol, position in portfolio['positions'].items():
                if symbol == data_point['symbol']:
                    # Use current price from data point
                    current_price = data_point['close']
                    position['current_value'] = position['quantity'] * current_price
                
                total_equity += position['current_value']
            
            return total_equity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio equity: {e}")
            return portfolio.get('cash', 0)
    
    async def _calculate_performance_metrics(self, trades: List[Dict[str, Any]], 
                                           equity_curve: List[Dict[str, Any]], 
                                           config: BacktestConfig) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not equity_curve:
                return {}
            
            # Basic metrics
            initial_equity = config.initial_capital
            final_equity = equity_curve[-1]['equity']
            total_return = (final_equity - initial_equity) / initial_equity
            
            # Calculate returns
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1]['equity']
                curr_equity = equity_curve[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if not returns:
                return {}
            
            # Risk-adjusted metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            sharpe_ratio = (mean_return * 252 - config.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = initial_equity
            max_drawdown = 0
            
            for equity_point in equity_curve:
                equity = equity_point['equity']
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Trade metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': mean_return * 252,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return {}
    
    async def _calculate_risk_metrics(self, trades: List[Dict[str, Any]], 
                                    equity_curve: List[Dict[str, Any]], 
                                    config: BacktestConfig) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            if not equity_curve:
                return {}
            
            # Calculate returns
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1]['equity']
                curr_equity = equity_curve[i]['equity']
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if not returns:
                return {}
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'volatility': volatility
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk metrics: {e}")
            return {}
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for backtest events"""
        self.backtest_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for backtest events"""
        callbacks = self.backtest_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_backtesting_statistics(self) -> Dict[str, Any]:
        """Get backtesting framework statistics"""
        return {
            'stats': self.stats,
            'active_backtests': len(self.active_backtests),
            'completed_backtests': len(self.backtest_results),
            'last_backtest_time': self.stats['last_backtest'].isoformat() if self.stats['last_backtest'] else None
        }
    
    async def close(self):
        """Close the backtesting framework"""
        try:
            # Cancel active backtests
            for backtest_id, task in self.active_backtests.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Close components
            if self.db_connection:
                await self.db_connection.close()
            
            if self.strategy_manager:
                await self.strategy_manager.close()
            
            self.logger.info("Advanced Backtesting Framework closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close backtesting framework: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
