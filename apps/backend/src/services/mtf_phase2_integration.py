import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..src.services.mtf_pattern_integrator import MTFPatternIntegrator, MTFPatternResult
from ..src.services.mtf_scheduler import MTFScheduler
from ..src.services.mtf_signal_generator import MTFSignalGenerator, RealTimeSignal, SignalPriority
from ..src.services.mtf_orchestrator import MTFOrchestrator
from ..src.services.mtf_cache_manager import MTFCacheManager

logger = logging.getLogger(__name__)

@dataclass
class MTFPhase2Status:
    is_running: bool
    scheduler_active: bool
    cache_healthy: bool
    total_signals_generated: int
    active_symbols: List[str]
    last_update: datetime
    performance_metrics: Dict[str, Any]

class MTFPhase2Integration:
    """
    Phase 2: Advanced Integration Service
    Coordinates all MTF components and provides unified interface for AlphaPulse
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        # Core MTF components
        self.mtf_pattern_integrator = MTFPatternIntegrator(redis_url)
        self.mtf_scheduler = MTFScheduler(redis_url)
        self.mtf_signal_generator = MTFSignalGenerator(redis_url)
        self.mtf_orchestrator = MTFOrchestrator(redis_url)
        self.cache_manager = MTFCacheManager(redis_url)
        
        # Integration state
        self.is_running = False
        self.active_symbols = set()
        self.processing_queue = []
        
        # Performance tracking
        self.stats = {
            'total_integration_runs': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'total_signals_processed': 0,
            'mtf_enhanced_signals': 0,
            'processing_times': [],
            'last_run_time': None
        }
        
        logger.info("🚀 MTF Phase 2 Integration initialized")
    
    async def start_integration(self, symbols: List[str] = None):
        """
        Start the Phase 2 MTF integration
        """
        if self.is_running:
            logger.warning("⚠️ MTF Phase 2 Integration is already running")
            return
        
        try:
            # Set active symbols
            if symbols:
                self.active_symbols = set(symbols)
            else:
                # Default symbols
                self.active_symbols = {"BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"}
            
            # Start MTF scheduler
            await self.mtf_scheduler.start(list(self.active_symbols))
            
            # Start MTF orchestrator
            await self.mtf_orchestrator.start()
            
            self.is_running = True
            
            logger.info(f"🚀 MTF Phase 2 Integration started with {len(self.active_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Error starting MTF Phase 2 Integration: {e}")
            raise
    
    async def stop_integration(self):
        """
        Stop the Phase 2 MTF integration
        """
        if not self.is_running:
            return
        
        try:
            # Stop MTF scheduler
            await self.mtf_scheduler.stop()
            
            # Stop MTF orchestrator
            await self.mtf_orchestrator.stop()
            
            self.is_running = False
            
            logger.info("🛑 MTF Phase 2 Integration stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping MTF Phase 2 Integration: {e}")
            raise
    
    async def process_symbol_timeframe(
        self, 
        symbol: str, 
        timeframe: str, 
        data: pd.DataFrame
    ) -> List[RealTimeSignal]:
        """
        Process a symbol and timeframe with full MTF integration
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Ensure MTF context is available
            await self._ensure_mtf_context(symbol, timeframe)
            
            # Step 2: Generate real-time signals with MTF enhancement
            signals = await self.mtf_signal_generator.generate_real_time_signals(
                symbol, timeframe, data
            )
            
            # Step 3: Filter and prioritize signals
            filtered_signals = await self._filter_and_prioritize_signals(signals)
            
            # Step 4: Update processing queue
            self._update_processing_queue(symbol, timeframe, filtered_signals)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(len(filtered_signals), processing_time, True)
            
            logger.info(f"✅ MTF Phase 2 processed {symbol} {timeframe}: {len(filtered_signals)} signals in {processing_time:.3f}s")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"❌ Error in MTF Phase 2 processing: {e}")
            self._update_stats(0, 0, False)
            return []
    
    async def _ensure_mtf_context(self, symbol: str, timeframe: str):
        """
        Ensure MTF context is available for the symbol and timeframe
        """
        try:
            # Check if context exists in cache
            cached_context = self.cache_manager.get_mtf_context(symbol, timeframe)
            
            if cached_context is None:
                # Force run the scheduler for this symbol/timeframe
                await self.mtf_scheduler.force_run(symbol, timeframe)
                
                # Wait a moment for processing
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.warning(f"⚠️ Could not ensure MTF context for {symbol} {timeframe}: {e}")
    
    async def _filter_and_prioritize_signals(
        self, 
        signals: List[RealTimeSignal]
    ) -> List[RealTimeSignal]:
        """
        Filter and prioritize signals based on MTF criteria
        """
        try:
            if not signals:
                return []
            
            # Filter by minimum confidence
            filtered_signals = [
                signal for signal in signals 
                if signal.final_confidence >= self.mtf_signal_generator.min_confidence_threshold
            ]
            
            # Filter by priority (at least medium priority)
            filtered_signals = await self.mtf_signal_generator.filter_signals_by_priority(
                filtered_signals, SignalPriority.MEDIUM
            )
            
            # Sort by confidence (highest first)
            filtered_signals.sort(key=lambda x: x.final_confidence, reverse=True)
            
            # Limit to top 5 signals per symbol/timeframe
            return filtered_signals[:5]
            
        except Exception as e:
            logger.error(f"❌ Error filtering signals: {e}")
            return signals
    
    def _update_processing_queue(self, symbol: str, timeframe: str, signals: List[RealTimeSignal]):
        """
        Update the processing queue with new signals
        """
        try:
            for signal in signals:
                queue_entry = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'processed': False
                }
                
                self.processing_queue.append(queue_entry)
                
                # Keep only last 100 entries
                if len(self.processing_queue) > 100:
                    self.processing_queue = self.processing_queue[-100:]
                    
        except Exception as e:
            logger.error(f"❌ Error updating processing queue: {e}")
    
    def _update_stats(self, signals_count: int, processing_time: float, success: bool):
        """
        Update performance statistics
        """
        self.stats['total_integration_runs'] += 1
        
        if success:
            self.stats['successful_integrations'] += 1
            self.stats['total_signals_processed'] += signals_count
            self.stats['processing_times'].append(processing_time)
        else:
            self.stats['failed_integrations'] += 1
        
        # Keep only last 100 processing times
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'] = self.stats['processing_times'][-100:]
        
        self.stats['last_run_time'] = datetime.now()
    
    async def get_integration_status(self) -> MTFPhase2Status:
        """
        Get current integration status
        """
        try:
            # Get scheduler status
            scheduler_status = await self.mtf_scheduler.get_scheduler_status()
            
            # Get cache stats
            cache_stats = self.cache_manager.get_stats()
            
            # Calculate performance metrics
            avg_processing_time = (
                sum(self.stats['processing_times']) / len(self.stats['processing_times'])
                if self.stats['processing_times'] else 0.0
            )
            
            performance_metrics = {
                'average_processing_time': avg_processing_time,
                'success_rate': (
                    self.stats['successful_integrations'] / max(1, self.stats['total_integration_runs'])
                ),
                'signals_per_run': (
                    self.stats['total_signals_processed'] / max(1, self.stats['successful_integrations'])
                ),
                'cache_hit_rate': cache_stats.get('hit_rate', 0.0),
                'scheduler_uptime': scheduler_status.get('is_running', False)
            }
            
            return MTFPhase2Status(
                is_running=self.is_running,
                scheduler_active=scheduler_status.get('is_running', False),
                cache_healthy=cache_stats.get('hit_rate', 0.0) > 0.5,
                total_signals_generated=self.stats['total_signals_processed'],
                active_symbols=list(self.active_symbols),
                last_update=self.stats['last_run_time'] or datetime.now(),
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting integration status: {e}")
            return MTFPhase2Status(
                is_running=False,
                scheduler_active=False,
                cache_healthy=False,
                total_signals_generated=0,
                active_symbols=[],
                last_update=datetime.now(),
                performance_metrics={}
            )
    
    async def add_symbol(self, symbol: str):
        """
        Add a new symbol to the integration
        """
        try:
            if symbol in self.active_symbols:
                logger.warning(f"⚠️ Symbol {symbol} is already active")
                return
            
            self.active_symbols.add(symbol)
            
            # Add to scheduler
            await self.mtf_scheduler.add_symbol(symbol)
            
            logger.info(f"➕ Added symbol {symbol} to MTF Phase 2 Integration")
            
        except Exception as e:
            logger.error(f"❌ Error adding symbol {symbol}: {e}")
            raise
    
    async def remove_symbol(self, symbol: str):
        """
        Remove a symbol from the integration
        """
        try:
            if symbol not in self.active_symbols:
                logger.warning(f"⚠️ Symbol {symbol} is not active")
                return
            
            self.active_symbols.remove(symbol)
            
            # Remove from scheduler
            await self.mtf_scheduler.remove_symbol(symbol)
            
            logger.info(f"➖ Removed symbol {symbol} from MTF Phase 2 Integration")
            
        except Exception as e:
            logger.error(f"❌ Error removing symbol {symbol}: {e}")
            raise
    
    async def get_signal_summary(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """
        Get a summary of generated signals
        """
        try:
            # Filter processing queue
            filtered_queue = self.processing_queue
            
            if symbol:
                filtered_queue = [entry for entry in filtered_queue if entry['symbol'] == symbol]
            
            if timeframe:
                filtered_queue = [entry for entry in filtered_queue if entry['timeframe'] == timeframe]
            
            # Extract signals
            signals = [entry['signal'] for entry in filtered_queue]
            
            # Get summary from signal generator
            summary = await self.mtf_signal_generator.get_signal_summary(signals)
            
            # Add integration-specific metrics
            summary['integration_metrics'] = {
                'total_processed': len(self.processing_queue),
                'active_symbols': len(self.active_symbols),
                'last_run_time': self.stats['last_run_time'].isoformat() if self.stats['last_run_time'] else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting signal summary: {e}")
            return {'error': str(e)}
    
    async def clear_cache(self):
        """
        Clear all caches
        """
        try:
            await self.cache_manager.clear_all_caches()
            await self.mtf_pattern_integrator.clear_cache()
            
            # Clear processing queue
            self.processing_queue.clear()
            
            logger.info("🧹 MTF Phase 2 Integration cache cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            raise
    
    async def run_performance_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Run a performance test of the Phase 2 integration
        """
        start_time = datetime.now()
        test_results = {
            'start_time': start_time,
            'end_time': None,
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'total_signals': 0,
            'processing_times': [],
            'errors': []
        }
        
        logger.info(f"🧪 Starting MTF Phase 2 performance test for {duration_seconds} seconds")
        
        try:
            while (datetime.now() - start_time).total_seconds() < duration_seconds:
                cycle_start = datetime.now()
                
                try:
                    # Test with a sample symbol and timeframe
                    test_symbol = "BTCUSDT"
                    test_timeframe = "1h"
                    
                    # Create mock data
                    test_data = self._create_test_data(test_symbol, test_timeframe)
                    
                    # Process with integration
                    signals = await self.process_symbol_timeframe(
                        test_symbol, test_timeframe, test_data
                    )
                    
                    # Update test results
                    test_results['total_cycles'] += 1
                    test_results['successful_cycles'] += 1
                    test_results['total_signals'] += len(signals)
                    
                    cycle_time = (datetime.now() - cycle_start).total_seconds()
                    test_results['processing_times'].append(cycle_time)
                    
                    # Wait a bit between cycles
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    test_results['total_cycles'] += 1
                    test_results['failed_cycles'] += 1
                    test_results['errors'].append(str(e))
                    logger.warning(f"⚠️ Performance test cycle failed: {e}")
            
            test_results['end_time'] = datetime.now()
            
            # Calculate performance metrics
            total_time = (test_results['end_time'] - test_results['start_time']).total_seconds()
            avg_processing_time = (
                sum(test_results['processing_times']) / len(test_results['processing_times'])
                if test_results['processing_times'] else 0.0
            )
            
            performance_summary = {
                'test_duration': total_time,
                'total_cycles': test_results['total_cycles'],
                'success_rate': test_results['successful_cycles'] / max(1, test_results['total_cycles']),
                'average_processing_time': avg_processing_time,
                'signals_per_cycle': test_results['total_signals'] / max(1, test_results['successful_cycles']),
                'cycles_per_second': test_results['total_cycles'] / total_time,
                'errors': len(test_results['errors'])
            }
            
            logger.info(f"✅ MTF Phase 2 performance test completed: {performance_summary}")
            
            return {
                'test_results': test_results,
                'performance_summary': performance_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            test_results['end_time'] = datetime.now()
            test_results['errors'].append(str(e))
            
            return {
                'test_results': test_results,
                'error': str(e)
            }
    
    def _create_test_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Create test data for performance testing
        """
        import numpy as np
        
        # Generate realistic test data
        np.random.seed(hash(f"{symbol}_{timeframe}_test") % 2**32)
        
        num_candles = 100
        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        
        prices = []
        current_price = base_price
        
        for i in range(num_candles):
            change = np.random.normal(0, 0.02)
            current_price *= (1 + change)
            
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            close_price = current_price
            
            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': np.random.uniform(1000, 10000)
            })
        
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=num_candles),
            periods=num_candles,
            freq='H'
        )
        
        return df
