"""
Backtesting Framework for AlphaPlus
Tests trading strategies with historical data
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class BacktestResult(Enum):
    """Backtest result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class BacktestMetrics:
    """Backtest performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float

class BacktestingFramework:
    """Comprehensive backtesting framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Backtest configuration
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.commission_rate = self.config.get('commission_rate', 0.001)
        self.slippage = self.config.get('slippage', 0.0005)
        
        # Performance tracking
        self.results = []
        self.current_position = None
        self.cash = self.initial_capital
        self.portfolio_value = []
        self.trades = []
        
    async def run_backtest(self, strategy, data: pd.DataFrame, 
                          start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run a complete backtest"""
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Filter data for backtest period
            mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
            backtest_data = data[mask].copy().reset_index(drop=True)
            
            # Initialize backtest
            self._initialize_backtest()
            
            # Run strategy simulation
            await self._simulate_strategy(strategy, backtest_data)
            
            # Calculate results
            results = self._calculate_results()
            
            self.logger.info("Backtest completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _initialize_backtest(self):
        """Initialize backtest state"""
        self.cash = self.initial_capital
        self.current_position = None
        self.portfolio_value = []
        self.trades = []
        self.results = []
    
    async def _simulate_strategy(self, strategy, data: pd.DataFrame):
        """Simulate strategy execution"""
        try:
            for i, row in data.iterrows():
                # Update portfolio value
                self._update_portfolio_value(row)
                
                # Get strategy signals
                signal = await strategy.generate_signal(data.iloc[:i+1])
                
                if signal:
                    # Execute signal
                    await self._execute_signal(signal, row)
                
                # Update position
                if self.current_position:
                    await self._update_position(row)
            
            # Close any remaining position
            if self.current_position:
                await self._close_position(data.iloc[-1])
                
        except Exception as e:
            self.logger.error(f"Error simulating strategy: {e}")
    
    def _update_portfolio_value(self, row):
        """Update current portfolio value"""
        current_value = self.cash
        
        if self.current_position:
            if self.current_position['side'] == 'long':
                current_value += self.current_position['quantity'] * row['close']
            else:
                current_value += self.current_position['quantity'] * (2 * self.current_position['entry_price'] - row['close'])
        
        self.portfolio_value.append({
            'timestamp': row['timestamp'],
            'value': current_value,
            'cash': self.cash,
            'position_value': current_value - self.cash
        })
    
    async def _execute_signal(self, signal: Dict[str, Any], row):
        """Execute a trading signal"""
        try:
            if self.current_position:
                # Close existing position
                await self._close_position(row)
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, row)
            
            if position_size > 0:
                # Open new position
                self.current_position = {
                    'side': signal['side'],
                    'quantity': position_size,
                    'entry_price': row['close'],
                    'entry_time': row['timestamp'],
                    'signal': signal
                }
                
                # Record trade
                self.trades.append({
                    'timestamp': row['timestamp'],
                    'action': 'open',
                    'side': signal['side'],
                    'quantity': position_size,
                    'price': row['close'],
                    'commission': position_size * row['close'] * self.commission_rate
                })
                
                self.logger.info(f"Opened {signal['side']} position: {position_size} @ {row['close']}")
                
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def _calculate_position_size(self, signal: Dict[str, Any], row) -> float:
        """Calculate position size based on risk management"""
        try:
            # Risk per trade (1% of portfolio)
            risk_amount = self.cash * 0.01
            
            # Calculate stop loss distance
            stop_loss = signal.get('stop_loss')
            if stop_loss:
                risk_per_share = abs(row['close'] - stop_loss)
            else:
                # Default risk per share (2% of current price)
                risk_per_share = row['close'] * 0.02
            
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # Apply position limits
            max_position_value = self.cash * 0.2  # Max 20% of portfolio
            max_position_size = max_position_value / row['close']
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _update_position(self, row):
        """Update current position"""
        try:
            if not self.current_position:
                return
            
            # Check stop loss and take profit
            if await self._should_close_position(row):
                await self._close_position(row)
                
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    async def _should_close_position(self, row) -> bool:
        """Check if position should be closed"""
        try:
            if not self.current_position:
                return False
            
            signal = self.current_position['signal']
            current_price = row['close']
            
            # Check stop loss
            stop_loss = signal.get('stop_loss')
            if stop_loss:
                if (self.current_position['side'] == 'long' and current_price <= stop_loss) or \
                   (self.current_position['side'] == 'short' and current_price >= stop_loss):
                    return True
            
            # Check take profit
            take_profit = signal.get('take_profit')
            if take_profit:
                if (self.current_position['side'] == 'long' and current_price >= take_profit) or \
                   (self.current_position['side'] == 'short' and current_price <= take_profit):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking position close: {e}")
            return False
    
    async def _close_position(self, row):
        """Close current position"""
        try:
            if not self.current_position:
                return
            
            # Calculate P&L
            entry_price = self.current_position['entry_price']
            current_price = row['close']
            quantity = self.current_position['quantity']
            
            if self.current_position['side'] == 'long':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            # Apply commission
            commission = quantity * current_price * self.commission_rate
            pnl -= commission
            
            # Update cash
            self.cash += pnl
            
            # Record trade
            self.trades.append({
                'timestamp': row['timestamp'],
                'action': 'close',
                'side': self.current_position['side'],
                'quantity': quantity,
                'price': current_price,
                'pnl': pnl,
                'commission': commission,
                'duration': (row['timestamp'] - self.current_position['entry_time']).total_seconds() / 3600  # hours
            })
            
            self.logger.info(f"Closed {self.current_position['side']} position: P&L = {pnl:.2f}")
            
            # Clear position
            self.current_position = None
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results and metrics"""
        try:
            if not self.portfolio_value:
                return {'error': 'No portfolio data available'}
            
            # Calculate basic metrics
            initial_value = self.portfolio_value[0]['value']
            final_value = self.portfolio_value[-1]['value']
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate returns
            returns = []
            for i in range(1, len(self.portfolio_value)):
                prev_value = self.portfolio_value[i-1]['value']
                curr_value = self.portfolio_value[i]['value']
                returns.append((curr_value - prev_value) / prev_value)
            
            returns = np.array(returns)
            
            # Calculate metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown()
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            
            # Create metrics object
            metrics = BacktestMetrics(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len([t for t in self.trades if t['action'] == 'close']),
                avg_trade_duration=np.mean([t.get('duration', 0) for t in self.trades if t['action'] == 'close'])
            )
            
            return {
                'metrics': metrics,
                'portfolio_value': self.portfolio_value,
                'trades': self.trades,
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_value': final_value,
                    'total_return_pct': total_return * 100,
                    'total_trades': metrics.total_trades,
                    'win_rate_pct': win_rate * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating results: {e}")
            return {'error': str(e)}
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            if np.std(excess_returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.portfolio_value:
                return 0.0
            
            values = [pv['value'] for pv in self.portfolio_value]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        try:
            closed_trades = [t for t in self.trades if t['action'] == 'close']
            if not closed_trades:
                return 0.0
            
            winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
            return len(winning_trades) / len(closed_trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            closed_trades = [t for t in self.trades if t['action'] == 'close']
            if not closed_trades:
                return 0.0
            
            gross_profit = sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0])
            gross_loss = abs(sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) < 0]))
            
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    async def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive backtest report"""
        try:
            if 'error' in results:
                return f"Backtest Error: {results['error']}"
            
            metrics = results['metrics']
            summary = results['summary']
            
            report = f"""
=== ALPHAPLUS BACKTEST REPORT ===

PERFORMANCE SUMMARY:
- Initial Capital: ${summary['initial_capital']:,.2f}
- Final Value: ${summary['final_value']:,.2f}
- Total Return: {summary['total_return_pct']:.2f}%
- Total Trades: {summary['total_trades']}
- Win Rate: {summary['win_rate_pct']:.2f}%

RISK METRICS:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.2f}%
- Profit Factor: {metrics.profit_factor:.2f}

TRADE ANALYSIS:
- Average Trade Duration: {metrics.avg_trade_duration:.2f} hours

PORTFOLIO PERFORMANCE:
- Peak Value: ${max([pv['value'] for pv in results['portfolio_value']]):,.2f}
- Final Cash: ${results['portfolio_value'][-1]['cash']:,.2f}
- Final Position Value: ${results['portfolio_value'][-1]['position_value']:,.2f}

=== END REPORT ===
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    async def export_results(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Export backtest results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                filename = f"backtest_results_{timestamp}.json"
                import json
                
                # Convert datetime objects to strings
                export_data = results.copy()
                for pv in export_data.get('portfolio_value', []):
                    pv['timestamp'] = pv['timestamp'].isoformat()
                for trade in export_data.get('trades', []):
                    trade['timestamp'] = trade['timestamp'].isoformat()
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filename
            
            elif format == 'csv':
                filename = f"backtest_results_{timestamp}.csv"
                
                # Export portfolio values
                portfolio_df = pd.DataFrame(results.get('portfolio_value', []))
                portfolio_df.to_csv(f"portfolio_{filename}", index=False)
                
                # Export trades
                trades_df = pd.DataFrame(results.get('trades', []))
                trades_df.to_csv(f"trades_{filename}", index=False)
                
                return filename
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return f"Error exporting results: {e}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for backtesting framework"""
        try:
            return {
                'status': 'healthy',
                'initial_capital': self.initial_capital,
                'current_cash': self.cash,
                'total_trades': len(self.trades),
                'portfolio_values': len(self.portfolio_value)
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
