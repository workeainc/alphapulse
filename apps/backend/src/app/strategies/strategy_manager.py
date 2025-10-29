"""
Strategy Manager Service for AlphaPulse
Manages multiple trading strategies and their execution with integrated candlestick analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psutil
import time

# Import candlestick analysis components
from src.app.data.real_time_processor import RealTimeCandlestickProcessor
from .ml_pattern_detector import MLPatternDetector
from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
from src.app.strategies.market_regime_detector import MarketRegimeDetector

# Get standardized logger
try:
    from src.app.core.unified_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class StrategyManager:
    """Service for managing multiple trading strategies with candlestick analysis"""
    
    def __init__(self):
        self.is_running = False
        self.strategies = {}
        self.active_strategies = set()
        self.strategy_performance = {}
        self.last_update = {}
        
        # Performance optimization settings
        self.base_monitor_interval = 30  # Base interval in seconds
        self.min_interval = 10  # Minimum interval
        self.max_interval = 120  # Maximum interval
        self.cpu_threshold_high = 80.0  # CPU threshold for high load
        self.cpu_threshold_low = 30.0   # CPU threshold for low load
        
        # Initialize candlestick analysis components
        self.candlestick_processor = RealTimeCandlestickProcessor({
            'min_confidence': 0.7,
            'min_strength': 0.6,
            'confirmation_required': True,
            'volume_confirmation': True,
            'trend_confirmation': True,
            'min_data_points': 50,
            'max_data_points': 1000,
            'signal_cooldown': 300
        })
        
        self.ml_pattern_detector = MLPatternDetector()
        self.real_time_signal_generator = RealTimeSignalGenerator()
        
        # Candlestick analysis state
        self.active_symbols = set()
        self.active_timeframes = ['15m', '1h', '4h']
        self.analysis_enabled = False
        
    async def start(self):
        """Start the strategy manager with candlestick analysis"""
        if self.is_running:
            logger.warning("Strategy manager is already running")
            return
            
        logger.info("Starting Strategy Manager with Candlestick Analysis...")
        self.is_running = True
        self._start_time = datetime.now()
        
        # Start background tasks
        asyncio.create_task(self._monitor_strategies())
        asyncio.create_task(self._candlestick_analysis_loop())
        
        logger.info("Strategy Manager started successfully")
    
    async def stop(self):
        """Stop the strategy manager"""
        if not self.is_running:
            logger.warning("Strategy manager is not running")
            return
            
        logger.info("Stopping Strategy Manager...")
        self.is_running = False
        
        # Stop candlestick analysis
        if self.analysis_enabled:
            await self.stop_candlestick_analysis()
            
        logger.info("Strategy Manager stopped successfully")
    
    def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive sleep interval based on CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if cpu_percent > self.cpu_threshold_high:
                # High CPU usage - increase interval to reduce load
                adaptive_interval = min(self.base_monitor_interval * 2, self.max_interval)
                logger.debug(f"High CPU usage ({cpu_percent:.1f}%), using interval: {adaptive_interval}s")
            elif cpu_percent < self.cpu_threshold_low:
                # Low CPU usage - decrease interval for more frequent updates
                adaptive_interval = max(self.base_monitor_interval * 0.5, self.min_interval)
                logger.debug(f"Low CPU usage ({cpu_percent:.1f}%), using interval: {adaptive_interval}s")
            else:
                # Normal CPU usage - use base interval
                adaptive_interval = self.base_monitor_interval
                logger.debug(f"Normal CPU usage ({cpu_percent:.1f}%), using interval: {adaptive_interval}s")
            
            return adaptive_interval
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive interval: {e}, using base interval")
            return self.base_monitor_interval
    
    async def _monitor_strategies(self):
        """Background task to monitor strategy performance with adaptive intervals"""
        while self.is_running:
            try:
                # Calculate adaptive interval based on CPU usage
                interval = self._calculate_adaptive_interval()
                
                # Update strategy performance
                await self._update_strategy_performance()
                
                # Check for strategy activation/deactivation
                await self._check_strategy_status()
                
                # Sleep for adaptive interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring strategies: {e}", exc_info=True)
                # Use longer interval on error to prevent rapid retries
                await asyncio.sleep(min(60, self.max_interval))
    
    async def _update_strategy_performance(self):
        """Update performance metrics for all strategies"""
        try:
            for strategy_name in self.strategies:
                if strategy_name in self.active_strategies:
                    # Update performance metrics
                    performance = await self._calculate_strategy_performance(strategy_name)
                    self.strategy_performance[strategy_name] = performance
                    
        except Exception as e:
            logger.error(f"❌ Error updating strategy performance: {e}")
    
    async def _calculate_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy"""
        try:
            # This would typically query the database for actual trade data
            # For now, return mock performance data
            
            # Simulate performance metrics
            win_rate = np.random.uniform(0.4, 0.8)  # 40-80% win rate
            total_trades = np.random.randint(10, 100)
            avg_profit = np.random.uniform(0.01, 0.05)  # 1-5% avg profit
            max_drawdown = np.random.uniform(0.05, 0.15)  # 5-15% max drawdown
            
            return {
                "win_rate": win_rate,
                "total_trades": total_trades,
                "avg_profit": avg_profit,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": avg_profit / max_drawdown if max_drawdown > 0 else 0,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance for {strategy_name}: {e}")
            return {}
    
    async def _check_strategy_status(self):
        """Check if strategies should be activated or deactivated"""
        try:
            for strategy_name, performance in self.strategy_performance.items():
                if strategy_name not in self.active_strategies:
                    # Check if strategy should be activated
                    if await self._should_activate_strategy(strategy_name, performance):
                        await self.activate_strategy(strategy_name)
                else:
                    # Check if strategy should be deactivated
                    if await self._should_deactivate_strategy(strategy_name, performance):
                        await self.deactivate_strategy(strategy_name)
                        
        except Exception as e:
            logger.error(f"❌ Error checking strategy status: {e}")
    
    async def _should_activate_strategy(self, strategy_name: str, performance: Dict[str, Any]) -> bool:
        """Check if a strategy should be activated"""
        try:
            # Check if strategy has enough historical data
            if performance.get("total_trades", 0) < 20:
                return False
            
            # Check if win rate is acceptable
            if performance.get("win_rate", 0) < 0.45:
                return False
            
            # Check if drawdown is manageable
            if performance.get("max_drawdown", 1.0) > 0.2:
                return False
            
            # Check if Sharpe ratio is positive
            if performance.get("sharpe_ratio", 0) < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking activation for {strategy_name}: {e}")
            return False
    
    async def _should_deactivate_strategy(self, strategy_name: str, performance: Dict[str, Any]) -> bool:
        """Check if a strategy should be deactivated"""
        try:
            # Deactivate if win rate drops too low
            if performance.get("win_rate", 0) < 0.35:
                return True
            
            # Deactivate if drawdown becomes too high
            if performance.get("max_drawdown", 0) > 0.25:
                return True
            
            # Deactivate if Sharpe ratio becomes negative
            if performance.get("sharpe_ratio", 0) < 0:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking deactivation for {strategy_name}: {e}")
            return False
    
    async def register_strategy(self, strategy_name: str, strategy_instance: Any):
        """Register a new trading strategy"""
        try:
            if strategy_name in self.strategies:
                logger.warning(f"Strategy {strategy_name} is already registered")
                return False
            
            self.strategies[strategy_name] = strategy_instance
            logger.info(f"✅ Strategy {strategy_name} registered successfully")
            
            # Initialize performance tracking
            self.strategy_performance[strategy_name] = {}
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registering strategy {strategy_name}: {e}")
            return False
    
    async def unregister_strategy(self, strategy_name: str):
        """Unregister a trading strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.warning(f"Strategy {strategy_name} is not registered")
                return False
            
            # Deactivate if currently active
            if strategy_name in self.active_strategies:
                await self.deactivate_strategy(strategy_name)
            
            # Remove from strategies
            del self.strategies[strategy_name]
            
            # Remove performance data
            if strategy_name in self.strategy_performance:
                del self.strategy_performance[strategy_name]
            
            logger.info(f"✅ Strategy {strategy_name} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unregistering strategy {strategy_name}: {e}")
            return False
    
    async def activate_strategy(self, strategy_name: str):
        """Activate a trading strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Cannot activate {strategy_name}: strategy not registered")
                return False
            
            if strategy_name in self.active_strategies:
                logger.warning(f"Strategy {strategy_name} is already active")
                return False
            
            # Activate the strategy
            self.active_strategies.add(strategy_name)
            logger.info(f"✅ Strategy {strategy_name} activated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error activating strategy {strategy_name}: {e}")
            return False
    
    async def deactivate_strategy(self, strategy_name: str):
        """Deactivate a trading strategy"""
        try:
            if strategy_name not in self.active_strategies:
                logger.warning(f"Strategy {strategy_name} is not active")
                return False
            
            # Deactivate the strategy
            self.active_strategies.remove(strategy_name)
            logger.info(f"✅ Strategy {strategy_name} deactivated")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deactivating strategy {strategy_name}: {e}")
            return False
    
    async def get_strategy_signals(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get trading signals from all active strategies"""
        try:
            signals = []
            
            for strategy_name in self.active_strategies:
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    
                    try:
                        # Generate signals from the strategy
                        if hasattr(strategy, 'generate_signals'):
                            strategy_signals = await strategy.generate_signals(market_data)
                            
                            # Add strategy metadata to signals
                            for signal in strategy_signals:
                                signal['strategy_name'] = strategy_name
                                signal['timestamp'] = datetime.now().isoformat()
                                signals.append(signal)
                        
                        elif hasattr(strategy, 'detect_patterns'):
                            # For pattern-based strategies
                            patterns = await strategy.detect_patterns(market_data)
                            
                            for pattern in patterns:
                                signal = {
                                    'strategy_name': strategy_name,
                                    'signal_type': pattern.get('signal_type', 'unknown'),
                                    'confidence': pattern.get('confidence', 0.5),
                                    'entry_price': pattern.get('entry_price', 0.0),
                                    'stop_loss': pattern.get('stop_loss', 0.0),
                                    'take_profit': pattern.get('take_profit', 0.0),
                                    'timestamp': datetime.now().isoformat()
                                }
                                signals.append(signal)
                    
                    except Exception as e:
                        logger.error(f"❌ Error getting signals from {strategy_name}: {e}")
                        continue
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy signals: {e}")
            return []
    
    async def _candlestick_analysis_loop(self):
        """Background task for candlestick analysis"""
        while self.is_running:
            try:
                if self.analysis_enabled and self.active_symbols:
                    # Process candlestick analysis for active symbols
                    await self._process_candlestick_analysis()
                
                # Update every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error in candlestick analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_candlestick_analysis(self):
        """Process candlestick analysis for all active symbols"""
        try:
            for symbol in self.active_symbols:
                for timeframe in self.active_timeframes:
                    # Get symbol data
                    symbol_data = self.candlestick_processor.get_symbol_data(symbol, timeframe)
                    
                    if not symbol_data['candlesticks']:
                        continue
                    
                    # Convert to DataFrame for analysis
                    df = self.candlestick_processor._to_dataframe(symbol, timeframe)
                    
                    if len(df) < 50:  # Need sufficient data
                        continue
                    
                    # Detect patterns using ML
                    patterns = self.ml_pattern_detector.detect_patterns_ml(df)
                    
                    # Generate real-time signals
                    signals = self.real_time_signal_generator.generate_signals(df, patterns)
                    
                    # Store signals in processor
                    for signal in signals:
                        self.candlestick_processor._store_signal(symbol, signal)
                    
                    # Update last update time
                    self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
                    
        except Exception as e:
            logger.error(f"❌ Error processing candlestick analysis: {e}")
    
    async def start_candlestick_analysis(self, symbols: List[str], timeframes: List[str] = None):
        """Start candlestick analysis for specified symbols"""
        try:
            if timeframes is None:
                timeframes = self.active_timeframes
            
            self.active_symbols.update(symbols)
            self.active_timeframes = list(set(self.active_timeframes + timeframes))
            self.analysis_enabled = True
            
            logger.info(f"Started candlestick analysis for {len(symbols)} symbols: {symbols}")
            logger.info(f"📊 Active timeframes: {self.active_timeframes}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting candlestick analysis: {e}")
            return False
    
    async def stop_candlestick_analysis(self):
        """Stop candlestick analysis"""
        try:
            self.analysis_enabled = False
            self.active_symbols.clear()
            logger.info("Candlestick analysis stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping candlestick analysis: {e}")
            return False
    
    async def get_candlestick_patterns(self, symbol: str, timeframe: str = "15m", limit: int = 100) -> List[Dict]:
        """Get detected candlestick patterns for a symbol"""
        try:
            if not self.analysis_enabled:
                return []
            
            # Get symbol data
            symbol_data = self.candlestick_processor.get_symbol_data(symbol, timeframe)
            
            if not symbol_data['candlesticks']:
                return []
            
            # Convert to DataFrame for pattern detection
            df = self.candlestick_processor._to_dataframe(symbol, timeframe)
            
            # Detect patterns
            patterns = self.ml_pattern_detector.detect_patterns_ml(df)
            
            # Format patterns for response
            formatted_patterns = []
            for pattern in patterns[-limit:]:  # Get latest patterns
                formatted_patterns.append({
                    "pattern": pattern.pattern,
                    "type": pattern.type,
                    "strength": pattern.strength,
                    "confidence": pattern.ml_confidence,
                    "market_regime": pattern.market_regime,
                    "timestamp": pattern.timestamp,
                    "features": pattern.features
                })
            
            return formatted_patterns
            
        except Exception as e:
            logger.error(f"❌ Error getting candlestick patterns for {symbol}: {e}")
            return []
    
    async def get_candlestick_signals(self, symbol: str, timeframe: str = "15m", limit: int = 50) -> List[Dict]:
        """Get trading signals from candlestick analysis"""
        try:
            if not self.analysis_enabled:
                return []
            
            # Get signals from processor
            signals = self.candlestick_processor.get_signal_summary(symbol)
            
            if not signals or 'signals' not in signals:
                return []
            
            # Format and limit signals
            formatted_signals = []
            for signal in signals['signals'][-limit:]:
                formatted_signals.append({
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timestamp": signal.timestamp,
                    "pattern": signal.pattern,
                    "strength": signal.strength
                })
            
            return formatted_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting candlestick signals for {symbol}: {e}")
            return []
    
    async def get_comprehensive_analysis(self, symbol: str, timeframe: str = "15m") -> Dict:
        """Get comprehensive candlestick analysis for a symbol"""
        try:
            if not self.analysis_enabled:
                return {"error": "Candlestick analysis not enabled"}
            
            # Get patterns and signals
            patterns = await self.get_candlestick_patterns(symbol, timeframe)
            signals = await self.get_candlestick_signals(symbol, timeframe)
            
            # Get symbol data
            symbol_data = self.candlestick_processor.get_symbol_data(symbol, timeframe)
            
            # Get processing stats
            processing_stats = self.candlestick_processor.get_processing_stats()
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis_timestamp": datetime.now().isoformat(),
                "patterns": {
                    "total": len(patterns),
                    "recent": patterns[-10:] if patterns else []  # Last 10 patterns
                },
                "signals": {
                    "total": len(signals),
                    "recent": signals[-10:] if signals else []  # Last 10 signals
                },
                "data_summary": {
                    "total_candlesticks": len(symbol_data.get('candlesticks', [])),
                    "last_update": self.last_update.get(f"{symbol}_{timeframe}", "Never")
                },
                "processing_stats": processing_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_candlestick_status(self) -> Dict[str, Any]:
        """Get status of candlestick analysis system"""
        try:
            return {
                "analysis_enabled": self.analysis_enabled,
                "active_symbols": list(self.active_symbols),
                "active_timeframes": self.active_timeframes,
                "total_symbols": len(self.active_symbols),
                "processing_stats": self.candlestick_processor.get_processing_stats(),
                "last_update": {
                    k: v.isoformat() if isinstance(v, datetime) else str(v)
                    for k, v in self.last_update.items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting candlestick status: {e}")
            return {}
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies"""
        try:
            status = {
                "total_strategies": len(self.strategies),
                "active_strategies": len(self.active_strategies),
                "strategies": {}
            }
            
            for strategy_name in self.strategies:
                status["strategies"][strategy_name] = {
                    "active": strategy_name in self.active_strategies,
                    "performance": self.strategy_performance.get(strategy_name, {}),
                    "last_update": self.last_update.get(strategy_name, "Never")
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy status: {e}")
            return {}
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names"""
        return list(self.active_strategies)
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy"""
        return self.strategy_performance.get(strategy_name, {})
    
    async def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]):
        """Update parameters for a specific strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Cannot update {strategy_name}: strategy not registered")
                return False
            
            strategy = self.strategies[strategy_name]
            
            # Update strategy parameters if the strategy supports it
            if hasattr(strategy, 'update_parameters'):
                await strategy.update_parameters(parameters)
                logger.info(f"✅ Parameters updated for strategy {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} does not support parameter updates")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating parameters for {strategy_name}: {e}")
            return False
    
    async def update_candlestick_parameters(self, parameters: Dict[str, Any]):
        """Update candlestick analysis parameters"""
        try:
            # Update processor configuration
            if 'min_confidence' in parameters:
                self.candlestick_processor.config['min_confidence'] = parameters['min_confidence']
            
            if 'min_strength' in parameters:
                self.candlestick_processor.config['min_strength'] = parameters['min_strength']
            
            if 'confirmation_required' in parameters:
                self.candlestick_processor.config['confirmation_required'] = parameters['confirmation_required']
            
            if 'volume_confirmation' in parameters:
                self.candlestick_processor.config['volume_confirmation'] = parameters['volume_confirmation']
            
            if 'trend_confirmation' in parameters:
                self.candlestick_processor.config['trend_confirmation'] = parameters['trend_confirmation']
            
            if 'min_data_points' in parameters:
                self.candlestick_processor.config['min_data_points'] = parameters['min_data_points']
            
            if 'max_data_points' in parameters:
                self.candlestick_processor.config['max_data_points'] = parameters['max_data_points']
            
            if 'signal_cooldown' in parameters:
                self.candlestick_processor.config['signal_cooldown'] = parameters['signal_cooldown']
            
            logger.info(f"✅ Candlestick analysis parameters updated: {parameters}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating candlestick parameters: {e}")
            return False
    
    async def add_symbol_to_analysis(self, symbol: str, timeframes: List[str] = None):
        """Add a symbol to candlestick analysis"""
        try:
            if timeframes is None:
                timeframes = self.active_timeframes
            
            self.active_symbols.add(symbol)
            self.active_timeframes = list(set(self.active_timeframes + timeframes))
            
            logger.info(f"Added {symbol} to candlestick analysis with timeframes: {timeframes}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding symbol {symbol} to analysis: {e}")
            return False
    
    async def remove_symbol_from_analysis(self, symbol: str):
        """Remove a symbol from candlestick analysis"""
        try:
            if symbol in self.active_symbols:
                self.active_symbols.remove(symbol)
                logger.info(f"Removed {symbol} from candlestick analysis")
                return True
            else:
                logger.warning(f"Symbol {symbol} not found in active analysis")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error removing symbol {symbol} from analysis: {e}")
            return False
    
    async def get_enhanced_strategy_signals(self, market_data: pd.DataFrame, symbol: str = None) -> List[Dict[str, Any]]:
        """Get enhanced trading signals combining strategy and candlestick analysis"""
        try:
            signals = []
            
            # Get signals from traditional strategies
            strategy_signals = await self.get_strategy_signals(market_data)
            signals.extend(strategy_signals)
            
            # Get signals from candlestick analysis if enabled
            if self.analysis_enabled and symbol:
                candlestick_signals = await self.get_candlestick_signals(symbol)
                
                # Merge candlestick signals with strategy signals
                for signal in candlestick_signals:
                    # Check if we already have a similar signal from strategies
                    duplicate_found = False
                    for existing_signal in signals:
                        if (existing_signal.get('symbol') == signal['symbol'] and
                            existing_signal.get('signal_type') == signal['signal_type'] and
                            abs(existing_signal.get('entry_price', 0) - signal['entry_price']) < 0.001):
                            duplicate_found = True
                            break
                    
                    if not duplicate_found:
                        signals.append(signal)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced strategy signals: {e}")
            return []
    
    async def reset_strategy_performance(self, strategy_name: str):
        """Reset performance metrics for a strategy"""
        try:
            if strategy_name in self.strategy_performance:
                self.strategy_performance[strategy_name] = {}
                logger.info(f"✅ Performance reset for strategy {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} not found in performance tracking")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error resetting performance for {strategy_name}: {e}")
            return False
    
    async def clear_candlestick_data(self, symbol: str = None, timeframe: str = None):
        """Clear candlestick analysis data"""
        try:
            self.candlestick_processor.clear_data(symbol, timeframe)
            logger.info(f"Cleared candlestick data for {symbol or 'all symbols'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing candlestick data: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including candlestick analysis"""
        try:
            strategy_status = self.get_strategy_status()
            candlestick_status = self.get_candlestick_status()
            
            return {
                "system_status": {
                    "running": self.is_running,
                    "start_time": getattr(self, '_start_time', None),
                    "uptime": self._calculate_uptime() if hasattr(self, '_start_time') else None
                },
                "strategy_management": strategy_status,
                "candlestick_analysis": candlestick_status,
                "total_active_components": (
                    strategy_status.get("active_strategies", 0) + 
                    candlestick_status.get("total_symbols", 0)
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {}
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        try:
            if hasattr(self, '_start_time') and self._start_time:
                uptime = datetime.now() - self._start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                if days > 0:
                    return f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    return f"{hours}h {minutes}m"
                else:
                    return f"{minutes}m {seconds}s"
            return "Unknown"
        except Exception:
            return "Unknown"
