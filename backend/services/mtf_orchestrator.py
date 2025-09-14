import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

from .mtf_cache_manager import MTFCacheManager
from .time_sync_batcher import TimeSyncBatcher, CandleStatus
from .mtf_signal_merger import MTFSignalMerger, MTFSignal, MergedSignal, SignalType

logger = logging.getLogger(__name__)

@dataclass
class MTFProcessingResult:
    """Result of MTF processing for a symbol"""
    symbol: str
    merged_signals: List[MergedSignal]
    processing_time: float
    cache_stats: Dict[str, Any]
    candle_statuses: Dict[str, Dict]
    errors: List[str]

class MTFOrchestrator:
    """
    Multi-Timeframe Orchestrator
    Coordinates all MTF optimization components and implements hierarchical processing
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        # Initialize components
        self.cache_manager = MTFCacheManager(redis_url)
        self.time_batcher = TimeSyncBatcher()
        self.signal_merger = MTFSignalMerger()
        
        # Processing state
        self.is_running = False
        self.processing_stats = {
            'total_symbols_processed': 0,
            'total_signals_generated': 0,
            'total_processing_time': 0.0,
            'cache_hit_rate': 0.0,
            'errors': []
        }
        
        # Register callbacks
        self.time_batcher.register_candle_complete_callback(self._on_candle_complete)
        self.time_batcher.register_candle_form_callback(self._on_candle_form)
        
        # Processing queue
        self.processing_queue: List[Tuple[str, str, pd.DataFrame]] = []
        
        logger.info("🚀 MTF Orchestrator initialized")
    
    async def start(self):
        """Start the MTF orchestrator"""
        if self.is_running:
            logger.warning("MTF Orchestrator is already running")
            return
        
        self.is_running = True
        logger.info("🚀 MTF Orchestrator started")
        
        # Start background tasks
        asyncio.create_task(self._processing_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the MTF orchestrator"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 MTF Orchestrator stopped")
    
    async def process_symbol_hierarchically(
        self, 
        symbol: str, 
        timeframes: List[str] = None,
        market_data_service = None
    ) -> MTFProcessingResult:
        """
        Process a symbol using hierarchical MTF approach (4h → 1h → 15m)
        
        Args:
            symbol: Trading symbol to process
            timeframes: List of timeframes to process (defaults to ["4h", "1h", "15m"])
            market_data_service: Service to fetch market data
            
        Returns:
            MTFProcessingResult with merged signals and processing stats
        """
        start_time = datetime.utcnow()
        errors = []
        
        if timeframes is None:
            timeframes = ["4h", "1h", "15m"]
        
        try:
            logger.info(f"🔄 Starting hierarchical MTF processing for {symbol}")
            
            # Step 1: Process higher timeframes first
            higher_timeframe_signals = await self._process_higher_timeframes(
                symbol, timeframes, market_data_service
            )
            
            # Step 2: Process lower timeframes with higher TF context
            lower_timeframe_signals = await self._process_lower_timeframes(
                symbol, timeframes, higher_timeframe_signals, market_data_service
            )
            
            # Step 3: Merge signals across timeframes
            all_signals = higher_timeframe_signals + lower_timeframe_signals
            merged_signals = await self._merge_all_signals(symbol, all_signals)
            
            # Step 4: Calculate processing statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            cache_stats = self.cache_manager.get_cache_stats()
            candle_statuses = self.time_batcher.get_all_candle_statuses()
            
            # Update global stats
            self.processing_stats['total_symbols_processed'] += 1
            self.processing_stats['total_signals_generated'] += len(merged_signals)
            self.processing_stats['total_processing_time'] += processing_time
            
            result = MTFProcessingResult(
                symbol=symbol,
                merged_signals=merged_signals,
                processing_time=processing_time,
                cache_stats=cache_stats,
                candle_statuses=candle_statuses,
                errors=errors
            )
            
            logger.info(f"✅ Hierarchical MTF processing completed for {symbol}: "
                       f"{len(merged_signals)} signals in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {symbol}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return MTFProcessingResult(
                symbol=symbol,
                merged_signals=[],
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                cache_stats=self.cache_manager.get_cache_stats(),
                candle_statuses={},
                errors=errors
            )
    
    async def _process_higher_timeframes(
        self, 
        symbol: str, 
        timeframes: List[str], 
        market_data_service
    ) -> List[MTFSignal]:
        """Process higher timeframes first to establish market bias"""
        signals = []
        
        # Sort timeframes by hierarchy (higher first)
        sorted_timeframes = sorted(
            timeframes, 
            key=lambda tf: self.time_batcher.timeframe_hierarchy.index(tf) 
            if tf in self.time_batcher.timeframe_hierarchy else 999
        )
        
        for timeframe in sorted_timeframes:
            try:
                # Check if we should wait for this timeframe to complete
                if not self.time_batcher.is_candle_completed(symbol, timeframe):
                    logger.debug(f"⏳ Waiting for {timeframe} candle to complete for {symbol}")
                    continue
                
                # Get data from cache or fetch
                data = await self.cache_manager.get_mtf_data(symbol, timeframe)
                if data is None and market_data_service:
                    # Fetch from market data service
                    data = await market_data_service.get_historical_data(symbol, timeframe, limit=200)
                    if data is not None:
                        # Cache the data
                        await self.cache_manager.set_mtf_data(symbol, timeframe, data)
                
                if data is not None and len(data) > 0:
                    # Generate signal for this timeframe
                    signal = await self._generate_timeframe_signal(symbol, timeframe, data)
                    if signal:
                        signals.append(signal)
                        logger.info(f"📊 Generated {timeframe} signal for {symbol}: "
                                  f"{signal.signal_type.value} (confidence: {signal.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing {timeframe} for {symbol}: {e}")
        
        return signals
    
    async def _process_lower_timeframes(
        self, 
        symbol: str, 
        timeframes: List[str], 
        higher_signals: List[MTFSignal],
        market_data_service
    ) -> List[MTFSignal]:
        """Process lower timeframes with higher timeframe context"""
        signals = []
        
        # Get higher timeframe context
        higher_context = self._extract_higher_timeframe_context(higher_signals)
        
        # Process lower timeframes
        for timeframe in timeframes:
            try:
                # Skip if this is a higher timeframe (already processed)
                if any(s.timeframe == timeframe for s in higher_signals):
                    continue
                
                # Check if we should wait for higher timeframes
                if self.time_batcher.should_wait_for_higher_timeframes(symbol, timeframe):
                    logger.debug(f"⏳ Waiting for higher timeframes before processing {timeframe}")
                    continue
                
                # Get data
                data = await self.cache_manager.get_mtf_data(symbol, timeframe)
                if data is None and market_data_service:
                    data = await market_data_service.get_historical_data(symbol, timeframe, limit=200)
                    if data is not None:
                        await self.cache_manager.set_mtf_data(symbol, timeframe, data)
                
                if data is not None and len(data) > 0:
                    # Generate signal with higher timeframe context
                    signal = await self._generate_timeframe_signal(
                        symbol, timeframe, data, higher_context
                    )
                    if signal:
                        signals.append(signal)
                        logger.info(f"📊 Generated {timeframe} signal with context for {symbol}: "
                                  f"{signal.signal_type.value} (confidence: {signal.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing {timeframe} for {symbol}: {e}")
        
        return signals
    
    async def _generate_timeframe_signal(
        self, 
        symbol: str, 
        timeframe: str, 
        data: pd.DataFrame,
        higher_context: Dict = None
    ) -> Optional[MTFSignal]:
        """Generate a signal for a specific timeframe"""
        try:
            # Basic signal generation (this would integrate with your existing pattern detection)
            # For now, we'll create a mock signal
            
            # Calculate basic indicators
            if len(data) < 20:
                return None
            
            current_price = data['close'].iloc[-1]
            ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = data['close'].ewm(span=50).mean().iloc[-1]
            
            # Determine signal type
            if current_price > ema_20 > ema_50:
                signal_type = SignalType.BULLISH
                confidence = min((current_price - ema_50) / ema_50, 0.8)
            elif current_price < ema_20 < ema_50:
                signal_type = SignalType.BEARISH
                confidence = min((ema_50 - current_price) / ema_50, 0.8)
            else:
                signal_type = SignalType.NEUTRAL
                confidence = 0.3
            
            # Apply higher timeframe context if available
            if higher_context:
                context_boost = self._calculate_context_boost(signal_type, higher_context)
                confidence = min(confidence * (1 + context_boost), 1.0)
            
            # Create signal
            signal = MTFSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                patterns=["ema_crossover"],  # Mock pattern
                technical_indicators={
                    "ema_20": ema_20,
                    "ema_50": ema_50,
                    "current_price": current_price
                },
                market_context=higher_context or {}
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            return None
    
    def _extract_higher_timeframe_context(self, signals: List[MTFSignal]) -> Dict:
        """Extract context from higher timeframe signals"""
        context = {}
        
        for signal in signals:
            context[signal.timeframe] = {
                'trend': signal.signal_type.value,
                'confidence': signal.confidence,
                'indicators': signal.technical_indicators
            }
        
        return context
    
    def _calculate_context_boost(self, signal_type: SignalType, context: Dict) -> float:
        """Calculate confidence boost from higher timeframe context"""
        boost = 0.0
        
        for timeframe, ctx in context.items():
            if ctx['trend'] == signal_type.value:
                # Aligned with higher timeframe
                weight = self.signal_merger.calculate_higher_timeframe_weight(timeframe)
                boost += ctx['confidence'] * weight * 0.5  # 50% of the weight
        
        return boost
    
    async def _merge_all_signals(self, symbol: str, signals: List[MTFSignal]) -> List[MergedSignal]:
        """Merge all signals for a symbol"""
        if not signals:
            return []
        
        # Group signals by base timeframe
        merged_signals = []
        
        # Try different base timeframes
        for base_timeframe in ["15m", "1h", "4h"]:
            if any(s.timeframe == base_timeframe for s in signals):
                merged = self.signal_merger.merge_signals_across_timeframes(signals, base_timeframe)
                if merged:
                    merged_signals.append(merged)
        
        # Filter high confidence signals
        filtered_signals = self.signal_merger.filter_high_confidence_signals(merged_signals)
        
        return filtered_signals
    
    async def _on_candle_complete(self, symbol: str, timeframe: str, data: pd.DataFrame, candle_info):
        """Callback when a candle completes"""
        logger.info(f"🕯️ Candle completed: {symbol} {timeframe}")
        
        # Add to processing queue
        self.processing_queue.append((symbol, timeframe, data))
    
    async def _on_candle_form(self, symbol: str, timeframe: str, data: pd.DataFrame, candle_info):
        """Callback when a candle starts forming"""
        logger.debug(f"📊 Candle forming: {symbol} {timeframe}")
        
        # Track forming candle for early signals
        await self.cache_manager.set_mtf_data(symbol, timeframe, data, is_completed=False)
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Process queued items
                if self.processing_queue:
                    symbol, timeframe, data = self.processing_queue.pop(0)
                    
                    # Process the completed candle
                    await self.process_symbol_hierarchically(symbol, [timeframe])
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Cleanup loop for expired data"""
        while self.is_running:
            try:
                # Clean up expired cache entries
                await self.cache_manager.cleanup_expired_entries()
                
                # Clean up old candle trackers
                await self.time_batcher.cleanup_old_trackers()
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        # Add cache stats
        cache_stats = self.cache_manager.get_cache_stats()
        stats['cache_stats'] = cache_stats
        
        # Add candle statuses
        stats['candle_statuses'] = self.time_batcher.get_all_candle_statuses()
        
        # Calculate averages
        if stats['total_symbols_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_symbols_processed']
            stats['avg_signals_per_symbol'] = stats['total_signals_generated'] / stats['total_symbols_processed']
        
        return stats
    
    async def get_mtf_summary(self, symbol: str) -> Dict[str, Any]:
        """Get MTF summary for a symbol"""
        try:
            # Get cache stats
            cache_stats = self.cache_manager.get_cache_stats()
            
            # Get candle statuses
            candle_statuses = self.time_batcher.get_all_candle_statuses()
            symbol_candles = candle_statuses.get(symbol, {})
            
            # Get higher timeframe context
            higher_context = await self.cache_manager.get_higher_timeframe_context(symbol, "15m")
            
            return {
                'symbol': symbol,
                'cache_stats': cache_stats,
                'candle_statuses': symbol_candles,
                'higher_timeframe_context': higher_context,
                'processing_stats': self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting MTF summary for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
