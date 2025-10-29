"""
Signal Generation Scheduler for AlphaPulse
Coordinates signal generation across 100 symbols with round-robin scheduling
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
import time

from src.services.dynamic_symbol_manager import DynamicSymbolManager
from src.services.ai_model_integration_service import AIModelIntegrationService
from src.services.mtf_signal_storage import MTFSignalStorage

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result from symbol analysis"""
    symbol: str
    success: bool
    signal_generated: bool
    consensus_achieved: bool
    analysis_time_ms: float
    error_message: Optional[str] = None

class SignalGenerationScheduler:
    """
    Coordinates signal generation across multiple symbols
    Uses round-robin scheduling with parallel batch processing
    """
    
    def __init__(
        self,
        symbol_manager: DynamicSymbolManager,
        ai_service: AIModelIntegrationService,
        config: Dict[str, Any]
    ):
        self.symbol_manager = symbol_manager
        self.ai_service = ai_service
        self.config = config
        self.logger = logger
        
        # Initialize signal storage (will be initialized in initialize() method)
        self.signal_storage = None
        
        # Scheduling configuration
        self.symbols_per_batch = config.get('signal_generation', {}).get('symbols_per_batch', 10)
        self.analysis_interval = config.get('signal_generation', {}).get('analysis_interval_seconds', 60)
        self.min_data_candles = config.get('signal_generation', {}).get('min_data_candles', 200)
        
        # State
        self.is_running = False
        self.current_batch_index = 0
        self.symbol_queue = deque()
        self.analysis_history: Dict[str, datetime] = {}  # Track last analysis time per symbol
        
        # Statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'signals_generated': 0,
            'consensus_achieved_count': 0,
            'avg_analysis_time_ms': 0.0,
            'current_cycle': 0,
            'symbols_in_queue': 0,
            # MTF Storage stats
            'signals_stored': 0,
            'storage_failures': 0,
            'duplicates_skipped': 0,
            'mtf_refinements_succeeded': 0,
            'mtf_refinements_failed': 0
        }
        
        logger.info(f"✅ Signal Generation Scheduler initialized (batch_size={self.symbols_per_batch}, interval={self.analysis_interval}s)")
    
    async def initialize(self):
        """Initialize scheduler with symbol list"""
        try:
            # Initialize signal storage
            from src.database.connection import db_connection
            self.signal_storage = MTFSignalStorage(
                db_connection=db_connection,
                redis_url=self.config.get('redis', {}).get('url', 'redis://localhost:56379')
            )
            await self.signal_storage.initialize()
            logger.info("✅ MTF Signal Storage initialized")
            
            # Load all active symbols
            all_symbols = await self.symbol_manager.get_active_symbols()
            
            # Initialize queue
            self.symbol_queue = deque(all_symbols)
            self.stats['symbols_in_queue'] = len(self.symbol_queue)
            
            logger.info(f"✅ Scheduler initialized with {len(all_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"❌ Error initializing scheduler: {e}")
            raise
    
    async def start(self):
        """Start the signal generation scheduler"""
        if self.is_running:
            logger.warning("⚠️ Scheduler already running")
            return
        
        logger.info("🚀 Starting signal generation scheduler...")
        self.is_running = True
        
        # Start scheduling loop
        asyncio.create_task(self._scheduling_loop())
        
        logger.info("✅ Signal generation scheduler started")
    
    async def stop(self):
        """Stop the signal generation scheduler"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping signal generation scheduler...")
        self.is_running = False
        
        # Close signal storage
        if self.signal_storage:
            await self.signal_storage.close()
        
        logger.info("✅ Signal generation scheduler stopped")
    
    async def _scheduling_loop(self):
        """Main scheduling loop - processes symbols in round-robin fashion"""
        logger.info("🔄 Scheduling loop started")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Get next batch of symbols
                symbols_to_analyze = await self._get_next_symbol_batch()
                
                if symbols_to_analyze:
                    # Analyze batch in parallel
                    results = await self._analyze_symbol_batch(symbols_to_analyze)
                    
                    # Update statistics
                    await self._update_stats_from_results(results)
                    
                    # Log progress
                    logger.info(
                        f"📊 Cycle {self.stats['current_cycle']}: "
                        f"Analyzed {len(results)} symbols, "
                        f"{sum(1 for r in results if r.signal_generated)} signals generated"
                    )
                else:
                    logger.debug("⏸️ No symbols to analyze, waiting...")
                
                # Wait for next interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.analysis_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in scheduling loop: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _get_next_symbol_batch(self) -> List[str]:
        """Get next batch of symbols to analyze (round-robin)"""
        try:
            # Check if queue needs refresh
            if len(self.symbol_queue) < self.symbols_per_batch:
                # Reload symbol list
                all_symbols = await self.symbol_manager.get_active_symbols()
                self.symbol_queue = deque(all_symbols)
                self.stats['current_cycle'] += 1
                logger.info(f"🔄 Starting cycle {self.stats['current_cycle']} with {len(all_symbols)} symbols")
            
            # Get next batch
            batch = []
            for _ in range(min(self.symbols_per_batch, len(self.symbol_queue))):
                if self.symbol_queue:
                    symbol = self.symbol_queue.popleft()
                    batch.append(symbol)
            
            self.stats['symbols_in_queue'] = len(self.symbol_queue)
            
            return batch
            
        except Exception as e:
            logger.error(f"❌ Error getting next symbol batch: {e}")
            return []
    
    async def _analyze_symbol_batch(self, symbols: List[str]) -> List[AnalysisResult]:
        """Analyze a batch of symbols in parallel"""
        try:
            logger.debug(f"🔬 Analyzing batch of {len(symbols)} symbols...")
            
            # Create analysis tasks
            tasks = [
                self._analyze_single_symbol(symbol)
                for symbol in symbols
            ]
            
            # Execute in parallel with timeout
            timeout = self.config.get('signal_generation', {}).get('analysis_timeout_seconds', 30)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to AnalysisResult
            analysis_results = []
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Analysis failed for {symbol}: {result}")
                    analysis_results.append(AnalysisResult(
                        symbol=symbol,
                        success=False,
                        signal_generated=False,
                        consensus_achieved=False,
                        analysis_time_ms=0.0,
                        error_message=str(result)
                    ))
                elif result:
                    analysis_results.append(result)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing symbol batch: {e}")
            return []
    
    async def _analyze_single_symbol(self, symbol: str) -> AnalysisResult:
        """Analyze a single symbol and generate signal if consensus achieved"""
        start_time = time.time()
        
        try:
            # Generate AI signal with MTF entry refinement (runs all 9 model heads + consensus + entry refinement)
            signal = await self.ai_service.generate_ai_signal_with_mtf_entry(
                symbol=symbol,
                signal_timeframe='1h',  # Analyze trend on 1h
                entry_timeframe='15m'    # Find entry on 15m
            )
            
            analysis_time_ms = (time.time() - start_time) * 1000
            
            # Update last analysis time
            self.analysis_history[symbol] = datetime.now(timezone.utc)
            
            if signal:
                logger.info(
                    f"✅ {symbol}: Signal generated! "
                    f"Direction={signal.signal_direction}, "
                    f"Confidence={signal.confidence_score:.3f}, "
                    f"Agreeing heads={len(signal.agreeing_heads)}/9"
                )
                
                # Check for duplicate active signals
                duplicate_exists = await self.signal_storage.check_active_signal_exists(
                    symbol, signal.signal_direction
                )
                
                if duplicate_exists:
                    logger.info(
                        f"⏭️  Skipping {symbol} - active {signal.signal_direction} signal already exists"
                    )
                    self.stats['duplicates_skipped'] += 1
                    
                    return AnalysisResult(
                        symbol=symbol,
                        success=True,
                        signal_generated=False,  # Don't count as new signal
                        consensus_achieved=signal.consensus_achieved,
                        analysis_time_ms=analysis_time_ms
                    )
                
                # Store signal to database and cache
                stored = await self.signal_storage.store_mtf_signal(signal)
                
                if stored:
                    logger.info(f"💾 Stored MTF signal for {symbol} to database")
                    self.stats['signals_stored'] += 1
                    
                    # Track MTF refinement success
                    if signal.entry_strategy and signal.entry_strategy != 'MARKET_ENTRY':
                        self.stats['mtf_refinements_succeeded'] += 1
                    elif signal.entry_strategy == 'MARKET_ENTRY':
                        self.stats['mtf_refinements_failed'] += 1
                else:
                    logger.error(f"❌ Failed to store signal for {symbol}")
                    self.stats['storage_failures'] += 1
                
                return AnalysisResult(
                    symbol=symbol,
                    success=True,
                    signal_generated=True,
                    consensus_achieved=signal.consensus_achieved,
                    analysis_time_ms=analysis_time_ms
                )
            else:
                logger.debug(f"ℹ️ {symbol}: No consensus (no signal)")
                
                return AnalysisResult(
                    symbol=symbol,
                    success=True,
                    signal_generated=False,
                    consensus_achieved=False,
                    analysis_time_ms=analysis_time_ms
                )
                
        except Exception as e:
            analysis_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            
            return AnalysisResult(
                symbol=symbol,
                success=False,
                signal_generated=False,
                consensus_achieved=False,
                analysis_time_ms=analysis_time_ms,
                error_message=str(e)
            )
    
    async def _update_stats_from_results(self, results: List[AnalysisResult]):
        """Update statistics from analysis results"""
        for result in results:
            self.stats['total_analyses'] += 1
            
            if result.success:
                self.stats['successful_analyses'] += 1
            else:
                self.stats['failed_analyses'] += 1
            
            if result.signal_generated:
                self.stats['signals_generated'] += 1
            
            if result.consensus_achieved:
                self.stats['consensus_achieved_count'] += 1
        
        # Update average analysis time
        if results:
            avg_time = sum(r.analysis_time_ms for r in results) / len(results)
            # Exponential moving average
            if self.stats['avg_analysis_time_ms'] == 0:
                self.stats['avg_analysis_time_ms'] = avg_time
            else:
                alpha = 0.1
                self.stats['avg_analysis_time_ms'] = (
                    alpha * avg_time + (1 - alpha) * self.stats['avg_analysis_time_ms']
                )
    
    async def get_analysis_status(self, symbol: str) -> Dict[str, Any]:
        """Get analysis status for a specific symbol"""
        last_analysis = self.analysis_history.get(symbol)
        
        return {
            'symbol': symbol,
            'last_analysis': last_analysis.isoformat() if last_analysis else None,
            'minutes_since_analysis': (
                (datetime.now(timezone.utc) - last_analysis).total_seconds() / 60
            ) if last_analysis else None,
            'in_queue': symbol in self.symbol_queue
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        success_rate = (
            self.stats['successful_analyses'] / self.stats['total_analyses'] * 100
        ) if self.stats['total_analyses'] > 0 else 0
        
        signal_rate = (
            self.stats['signals_generated'] / self.stats['total_analyses'] * 100
        ) if self.stats['total_analyses'] > 0 else 0
        
        consensus_rate = (
            self.stats['consensus_achieved_count'] / self.stats['total_analyses'] * 100
        ) if self.stats['total_analyses'] > 0 else 0
        
        return {
            'stats': self.stats,
            'derived_metrics': {
                'success_rate': f"{success_rate:.1f}%",
                'signal_generation_rate': f"{signal_rate:.1f}%",
                'consensus_achievement_rate': f"{consensus_rate:.1f}%",
                'estimated_signals_per_day': (self.stats['signals_generated'] / max(self.stats['current_cycle'], 1)) * (24 * 60 / self.analysis_interval)
            },
            'queue_status': {
                'symbols_in_queue': len(self.symbol_queue),
                'current_cycle': self.stats['current_cycle'],
                'total_symbols_tracked': len(self.analysis_history)
            }
        }

