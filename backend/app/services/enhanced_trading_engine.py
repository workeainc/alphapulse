#!/usr/bin/env python3
"""
AlphaPulse Enhanced Trading Engine
Integrates optimized components with existing trading system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Import existing components
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

# Import optimized integration
from app.services.optimized_trading_integration import OptimizedTradingIntegration

logger = logging.getLogger(__name__)

class EnhancedTradingEngine:
    """
    Enhanced trading engine that integrates optimized components with existing AlphaPulse system
    """
    
    def __init__(self, exchange_credentials: Optional[ExchangeCredentials] = None, 
                 use_optimization: bool = True, max_workers: int = 4):
        self.is_running = False
        self.use_optimization = use_optimization
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Existing components
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
        
        # Optimized integration (if enabled)
        self.optimized_integration = None
        if self.use_optimization:
            self.optimized_integration = OptimizedTradingIntegration(max_workers=max_workers)
        
        # Trading state
        self.open_positions = {}
        self.pending_orders = {}
        self.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.performance_stats = {
            'total_cycles': 0,
            'optimized_cycles': 0,
            'legacy_cycles': 0,
            'avg_cycle_time': 0.0,
            'total_signals': 0,
            'total_patterns': 0
        }
        
    async def start(self):
        """Start the enhanced trading engine"""
        if self.is_running:
            logger.warning("Enhanced trading engine is already running")
            return
            
        logger.info("🚀 Starting AlphaPulse Enhanced Trading Engine...")
        logger.info(f"Optimization enabled: {self.use_optimization}")
        self.is_running = True
        
        try:
            # Start existing services
            await self.market_data_service.start()
            await self.sentiment_service.start()
            await self.risk_manager.start()
            await self.strategy_manager.start()
            
            # Start real-time components
            await self.signal_generator.start()
            await self.candlestick_processor.start()
            await self.websocket_client.start()
            
            # Start optimized integration if enabled
            if self.optimized_integration:
                await self.optimized_integration.start()
                logger.info("✅ Optimized integration started")
            
            # Start background tasks
            asyncio.create_task(self._enhanced_trading_loop())
            asyncio.create_task(self._position_monitor())
            asyncio.create_task(self._performance_tracker())
            asyncio.create_task(self._optimization_monitor())
            
            logger.info("✅ Enhanced Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting enhanced trading engine: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the enhanced trading engine"""
        if not self.is_running:
            logger.warning("Enhanced trading engine is not running")
            return
            
        logger.info("🛑 Stopping Enhanced Trading Engine...")
        self.is_running = False
        
        try:
            # Stop existing services
            await self.market_data_service.stop()
            await self.sentiment_service.stop()
            await self.risk_manager.stop()
            await self.strategy_manager.stop()
            
            # Stop real-time components
            await self.signal_generator.stop()
            await self.candlestick_processor.stop()
            await self.websocket_client.stop()
            
            # Stop optimized integration if enabled
            if self.optimized_integration:
                await self.optimized_integration.stop()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Enhanced Trading Engine stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping enhanced trading engine: {e}")
    
    async def _enhanced_trading_loop(self):
        """Enhanced trading loop that uses optimized components when available"""
        while self.is_running:
            try:
                cycle_start = datetime.now()
                
                # Wait for next update cycle
                await asyncio.sleep(settings.UPDATE_INTERVAL)
                
                # Get latest market data
                market_data = await self._get_market_data()
                
                if market_data is not None and not market_data.empty:
                    # Process with enhanced pipeline
                    results = await self._process_market_data_enhanced(market_data)
                    
                    # Process signals
                    if results['signals']:
                        for signal in results['signals']:
                            await self._process_signal(signal, market_data)
                    
                    # Update performance stats
                    cycle_time = (datetime.now() - cycle_start).total_seconds()
                    self.performance_stats['total_cycles'] += 1
                    self.performance_stats['avg_cycle_time'] = (
                        (self.performance_stats['avg_cycle_time'] * (self.performance_stats['total_cycles'] - 1) + cycle_time) 
                        / self.performance_stats['total_cycles']
                    )
                    self.performance_stats['total_signals'] += len(results['signals'])
                    self.performance_stats['total_patterns'] += len(results['patterns'])
                    
                    if results.get('optimization_used', False):
                        self.performance_stats['optimized_cycles'] += 1
                    else:
                        self.performance_stats['legacy_cycles'] += 1
                
            except Exception as e:
                logger.error(f"❌ Error in enhanced trading loop: {e}")
                await asyncio.sleep(30)
    
    async def _process_market_data_enhanced(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Process market data using enhanced pipeline with optimization"""
        try:
            if self.optimized_integration and self.optimized_integration.is_running:
                # Use optimized integration
                logger.debug("🔄 Using optimized processing pipeline...")
                results = await self.optimized_integration.process_market_data(market_data, self.trading_pairs)
                return results
            else:
                # Fallback to legacy processing
                logger.debug("🔄 Using legacy processing pipeline...")
                return await self._process_market_data_legacy(market_data)
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced market data processing: {e}")
            return await self._process_market_data_legacy(market_data)
    
    async def _process_market_data_legacy(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Legacy market data processing"""
        try:
            signals = []
            patterns = []
            
            # Get signals from strategy manager
            strategy_signals = await self.strategy_manager.get_strategy_signals(market_data)
            signals.extend(strategy_signals)
            
            # Get signals from real-time signal generator
            real_time_signals = await self.signal_generator.generate_signals(market_data)
            signals.extend(real_time_signals)
            
            # Get sentiment-based signals
            sentiment_signals = await self.sentiment_service.get_trading_signals(market_data)
            signals.extend(sentiment_signals)
            
            # Get patterns from legacy pattern detector
            patterns = await self._get_legacy_patterns(market_data)
            
            return {
                'signals': signals,
                'patterns': patterns,
                'optimization_used': False
            }
            
        except Exception as e:
            logger.error(f"❌ Error in legacy market data processing: {e}")
            return {'signals': [], 'patterns': [], 'optimization_used': False}
    
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
    
    async def _get_legacy_patterns(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get patterns using legacy pattern detection"""
        try:
            # Import legacy pattern detector
            from ..strategies.pattern_detector import PatternDetector
            
            pattern_detector = PatternDetector()
            patterns = []
            
            for symbol in market_data.index.get_level_values(0).unique():
                symbol_data = market_data.loc[symbol]
                if not symbol_data.empty:
                    symbol_patterns = pattern_detector.detect_patterns(symbol_data)
                    for pattern in symbol_patterns:
                        pattern['symbol'] = symbol
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error getting legacy patterns: {e}")
            return []
    
    async def _process_signal(self, signal: Dict[str, Any], market_data: pd.DataFrame):
        """Process a trading signal"""
        try:
            # Validate signal with risk manager
            if await self.risk_manager.validate_signal(signal):
                # Calculate signal score
                score = await self._calculate_signal_score(signal)
                signal['score'] = score
                
                # Execute trade if score is high enough
                if score >= settings.MIN_SIGNAL_SCORE:
                    await self._execute_trade(signal, market_data)
                else:
                    logger.debug(f"Signal score {score} below threshold {settings.MIN_SIGNAL_SCORE}")
            else:
                logger.debug("Signal failed risk validation")
                
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
    
    async def _calculate_signal_score(self, signal: Dict[str, Any]) -> float:
        """Calculate signal score based on multiple factors"""
        try:
            score = 0.0
            
            # Base score from signal strength
            score += signal.get('strength', 0.0) * 0.3
            
            # Pattern confirmation score
            if signal.get('pattern_confirmed', False):
                score += 0.2
            
            # Volume confirmation score
            if signal.get('volume_confirmed', False):
                score += 0.15
            
            # Trend alignment score
            if signal.get('trend_aligned', False):
                score += 0.15
            
            # Risk-adjusted score
            risk_score = signal.get('risk_score', 0.5)
            score += (1.0 - risk_score) * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating signal score: {e}")
            return 0.0
    
    async def _execute_trade(self, signal: Dict[str, Any], market_data: pd.DataFrame):
        """Execute a trade based on signal"""
        try:
            symbol = signal.get('symbol')
            side = signal.get('side', 'BUY')
            quantity = signal.get('quantity', 0.0)
            
            if not symbol or quantity <= 0:
                logger.warning(f"Invalid trade parameters: {signal}")
                return
            
            # Create order
            order = await self.order_manager.create_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            if order:
                logger.info(f"✅ Executed {side} order for {quantity} {symbol}")
                self.total_trades += 1
                
                # Update portfolio
                await self.portfolio_manager.update_position(order)
                
                # Set stop loss and take profit
                await self.sl_tp_manager.set_sl_tp(order, signal)
                
            else:
                logger.error(f"❌ Failed to execute trade for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
    
    async def _position_monitor(self):
        """Monitor open positions"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Update position P&L
                positions = await self.portfolio_manager.get_positions()
                for position in positions:
                    pnl = await self.portfolio_manager.calculate_pnl(position)
                    if pnl > 0:
                        self.winning_trades += 1
                    
                    self.daily_pnl += pnl
                
            except Exception as e:
                logger.error(f"❌ Error in position monitor: {e}")
    
    async def _performance_tracker(self):
        """Track and log performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Log every 5 minutes
                
                # Log performance stats
                logger.info(f"📊 Performance Stats:")
                logger.info(f"   Total Cycles: {self.performance_stats['total_cycles']}")
                logger.info(f"   Optimized Cycles: {self.performance_stats['optimized_cycles']}")
                logger.info(f"   Legacy Cycles: {self.performance_stats['legacy_cycles']}")
                logger.info(f"   Avg Cycle Time: {self.performance_stats['avg_cycle_time']:.3f}s")
                logger.info(f"   Total Signals: {self.performance_stats['total_signals']}")
                logger.info(f"   Total Patterns: {self.performance_stats['total_patterns']}")
                logger.info(f"   Daily P&L: ${self.daily_pnl:.2f}")
                logger.info(f"   Win Rate: {(self.winning_trades/self.total_trades*100):.1f}%" if self.total_trades > 0 else "N/A")
                
            except Exception as e:
                logger.error(f"❌ Error in performance tracker: {e}")
    
    async def _optimization_monitor(self):
        """Monitor optimization performance and status"""
        while self.is_running:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                if self.optimized_integration:
                    # Get optimization stats
                    opt_stats = await self.optimized_integration.get_performance_stats()
                    opt_status = await self.optimized_integration.get_optimization_status()
                    
                    logger.info(f"🚀 Optimization Status:")
                    logger.info(f"   Available: {opt_status['optimized_components_available']}")
                    logger.info(f"   Active: {opt_status['optimized_system_active']}")
                    logger.info(f"   Cache Efficiency: {opt_stats.get('cache_efficiency', 0):.1%}")
                    logger.info(f"   Avg Processing Time: {opt_stats.get('avg_processing_time', 0):.3f}s")
                    
            except Exception as e:
                logger.error(f"❌ Error in optimization monitor: {e}")
    
    async def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced trading engine statistics"""
        stats = {
            'engine_status': {
                'is_running': self.is_running,
                'use_optimization': self.use_optimization,
                'max_workers': self.max_workers
            },
            'performance': self.performance_stats.copy(),
            'trading': {
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': (self.winning_trades/self.total_trades*100) if self.total_trades > 0 else 0.0,
                'open_positions': len(self.open_positions),
                'pending_orders': len(self.pending_orders)
            }
        }
        
        # Add optimization stats if available
        if self.optimized_integration:
            try:
                opt_stats = await self.optimized_integration.get_performance_stats()
                opt_status = await self.optimized_integration.get_optimization_status()
                stats['optimization'] = {
                    'stats': opt_stats,
                    'status': opt_status
                }
            except Exception as e:
                logger.error(f"❌ Error getting optimization stats: {e}")
                stats['optimization'] = {'error': str(e)}
        
        return stats
    
    async def toggle_optimization(self, enable: bool):
        """Toggle optimization on/off"""
        if self.is_running:
            logger.warning("Cannot toggle optimization while engine is running")
            return
        
        self.use_optimization = enable
        if enable and not self.optimized_integration:
            self.optimized_integration = OptimizedTradingIntegration(max_workers=self.max_workers)
        elif not enable:
            self.optimized_integration = None
        
        logger.info(f"Optimization {'enabled' if enable else 'disabled'}")
    
    async def run_optimization_benchmark(self) -> Dict[str, Any]:
        """Run optimization benchmark"""
        if not self.optimized_integration:
            return {'error': 'Optimization not available'}
        
        try:
            return await self.optimized_integration.run_optimization_benchmark()
        except Exception as e:
            logger.error(f"❌ Error running optimization benchmark: {e}")
            return {'error': str(e)}
    
    async def clear_optimization_cache(self):
        """Clear optimization caches"""
        if self.optimized_integration:
            await self.optimized_integration.clear_optimization_cache()
        else:
            logger.warning("Optimization not available")
