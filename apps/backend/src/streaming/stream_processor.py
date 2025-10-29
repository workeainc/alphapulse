"""
Stream Processor for AlphaPulse
Main orchestrator for all streaming components with TimescaleDB integration
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import existing components
try:
    from .stream_buffer import StreamMessage, StreamBuffer, stream_buffer
    from .stream_normalizer import NormalizedData, StreamNormalizer, stream_normalizer
    from .candle_builder import Candle, CandleBuilder, candle_builder
    from .rolling_state_manager import RollingWindow, TechnicalIndicator, RollingStateManager, rolling_state_manager
    from ..src.database.connection import TimescaleDBConnection
    from ..src.core.config import settings
except ImportError:
    try:
        # Fallback for standalone testing
        from stream_buffer import StreamMessage, StreamBuffer, stream_buffer
        from stream_normalizer import NormalizedData, StreamNormalizer, stream_normalizer
        from candle_builder import Candle, CandleBuilder, candle_builder
        from rolling_state_manager import RollingWindow, TechnicalIndicator, RollingStateManager, rolling_state_manager
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
    except ImportError:
        # Final fallback - create minimal classes
        class StreamMessage:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class StreamBuffer:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
                self.metrics = type('obj', (object,), {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0
                })()
                self.is_connected = False
                self.is_running = False
                self.message_buffer = []
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
            
            def get_metrics(self):
                return {
                    'messages_received': 0,
                    'messages_processed': 0,
                    'messages_failed': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'buffer_size': 0,
                    'last_message_time': None,
                    'error_count': 0,
                    'reconnection_count': 0,
                    'is_connected': False,
                    'is_running': False,
                    'message_buffer_size': 0
                }
        
        stream_buffer = StreamBuffer()
        
        class NormalizedData:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class StreamNormalizer:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
            
            def get_metrics(self):
                return {
                    'messages_processed': 0,
                    'messages_validated': 0,
                    'messages_normalized': 0,
                    'duplicates_detected': 0,
                    'outliers_detected': 0,
                    'validation_errors': 0,
                    'avg_processing_time_ms': 0.0,
                    'throughput_mps': 0.0,
                    'last_processed_time': None
                }
        
        stream_normalizer = StreamNormalizer()
        
        class Candle:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class CandleBuilder:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
                self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
        
        candle_builder = CandleBuilder()
        
        class RollingWindow:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class TechnicalIndicator:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        class RollingStateManager:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
        
        rolling_state_manager = RollingStateManager()
        
        class TimescaleDBConnection:
            def __init__(self, config=None):
                self.config = config or {}
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
            
            async def shutdown(self):
                self.is_initialized = False
            
            async def close(self):
                self.is_initialized = False
        
        class settings:
            TIMESCALEDB_HOST = 'localhost'
            TIMESCALEDB_PORT = 5432
            TIMESCALEDB_DATABASE = 'alphapulse'
            TIMESCALEDB_USERNAME = 'alpha_emon'
            TIMESCALEDB_PASSWORD = 'Emon_@17711'

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of stream processing"""
    success: bool
    processing_time_ms: float
    components_processed: List[str]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamProcessorMetrics:
    """Stream processor performance metrics"""
    messages_processed: int = 0
    messages_successful: int = 0
    messages_failed: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_mps: float = 0.0
    last_processed_time: Optional[datetime] = None
    component_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class StreamProcessor:
    """
    Main stream processor that orchestrates all streaming components
    
    Features:
    - Orchestrates all streaming components
    - Handles stream routing and processing
    - Manages component lifecycle
    - Provides unified metrics and monitoring
    - TimescaleDB integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Processing settings
        self.enable_normalization = self.config.get('enable_normalization', True)
        self.enable_candle_building = self.config.get('enable_candle_building', True)
        self.enable_rolling_state = self.config.get('enable_rolling_state', True)
        self.enable_indicators = self.config.get('enable_indicators', True)
        
        # Performance settings
        self.max_concurrent_processing = self.config.get('max_concurrent_processing', 10)
        self.processing_timeout = self.config.get('processing_timeout', 5.0)  # seconds
        self.batch_size = self.config.get('batch_size', 100)
        
        # State management
        self.is_running = False
        self.metrics = StreamProcessorMetrics()
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_processing)
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.processing_tasks = set()
        
        # Component references
        self.stream_buffer = stream_buffer
        self.stream_normalizer = stream_normalizer
        self.candle_builder = candle_builder
        self.rolling_state_manager = rolling_state_manager
        
        # Processing callbacks
        self.pre_processing_callbacks = []
        self.post_processing_callbacks = []
        self.error_callbacks = []
        
        # TimescaleDB integration
        self.timescaledb = None
        
        logger.info("StreamProcessor initialized")
    
    async def initialize(self):
        """Initialize the stream processor and all components"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Initialize all components
            await self._initialize_components()
            
            # Start processing loop
            await self._start_processing_loop()
            
            self.is_running = True
            logger.info("✅ StreamProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StreamProcessor: {e}")
            raise
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        try:
            self.timescaledb = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 10,
                'max_overflow': 20
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized for stream processor")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _initialize_components(self):
        """Initialize all streaming components"""
        try:
            # Initialize stream buffer
            await self.stream_buffer.initialize()
            logger.info("✅ StreamBuffer initialized")
            
            # Initialize stream normalizer
            await self.stream_normalizer.initialize()
            logger.info("✅ StreamNormalizer initialized")
            
            # Initialize candle builder
            await self.candle_builder.initialize()
            logger.info("✅ CandleBuilder initialized")
            
            # Initialize rolling state manager
            await self.rolling_state_manager.initialize()
            logger.info("✅ RollingStateManager initialized")
            
            # Set up component callbacks
            await self._setup_component_callbacks()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def _setup_component_callbacks(self):
        """Set up callbacks between components"""
        try:
            # Set up candle builder callbacks for rolling state updates
            for timeframe in self.candle_builder.timeframes:
                self.candle_builder.add_completion_callback(timeframe, self._on_candle_completed)
            
            # Set up rolling state callbacks for indicator updates
            self.rolling_state_manager.add_indicator_callback('RSI', self._on_indicator_updated)
            self.rolling_state_manager.add_indicator_callback('MACD', self._on_indicator_updated)
            self.rolling_state_manager.add_indicator_callback('SMA', self._on_indicator_updated)
            
        except Exception as e:
            logger.error(f"Error setting up component callbacks: {e}")
    
    async def _start_processing_loop(self):
        """Start the main processing loop"""
        asyncio.create_task(self._processing_worker())
        logger.info("✅ Processing loop started")
    
    async def _processing_worker(self):
        """Main processing worker loop"""
        while self.is_running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process message
                async with self.processing_semaphore:
                    task = asyncio.create_task(self._process_message(message))
                    self.processing_tasks.add(task)
                    task.add_done_callback(self.processing_tasks.discard)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")
    
    async def _process_message(self, message: StreamMessage) -> ProcessingResult:
        """Process a single stream message through all components"""
        start_time = time.time()
        components_processed = []
        errors = []
        
        try:
            # Pre-processing callbacks
            await self._trigger_pre_processing_callbacks(message)
            
            # Step 1: Normalize message
            if self.enable_normalization:
                try:
                    normalized_data = await self.stream_normalizer.normalize_message(message)
                    components_processed.append('normalization')
                    
                    if normalized_data.validation_status != 'valid':
                        errors.append(f"Normalization failed: {normalized_data.validation_status}")
                        return ProcessingResult(
                            success=False,
                            processing_time_ms=(time.time() - start_time) * 1000,
                            components_processed=components_processed,
                            errors=errors
                        )
                    
                except Exception as e:
                    errors.append(f"Normalization error: {e}")
                    return ProcessingResult(
                        success=False,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        components_processed=components_processed,
                        errors=errors
                    )
            else:
                # Create dummy normalized data if normalization is disabled
                normalized_data = NormalizedData(
                    original_message=message,
                    normalized_data=message.data,
                    validation_status='valid',
                    confidence_score=1.0,
                    processing_time_ms=0.0
                )
            
            # Step 2: Build candles
            if self.enable_candle_building and message.data_type == 'tick':
                try:
                    candles = await self.candle_builder.process_tick(normalized_data)
                    components_processed.append('candle_building')
                    
                except Exception as e:
                    errors.append(f"Candle building error: {e}")
            
            # Step 3: Update rolling state
            if self.enable_rolling_state:
                try:
                    data = normalized_data.normalized_data
                    symbol = data.get('symbol', '').upper()
                    price = float(data.get('price', 0))
                    
                    if symbol and price > 0:
                        # Update price rolling window
                        await self.rolling_state_manager.update_rolling_window(
                            symbol, '1m', 'price', price
                        )
                        components_processed.append('rolling_state')
                        
                except Exception as e:
                    errors.append(f"Rolling state error: {e}")
            
            # Post-processing callbacks
            await self._trigger_post_processing_callbacks(message, normalized_data)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(True, processing_time, components_processed)
            
            return ProcessingResult(
                success=True,
                processing_time_ms=processing_time,
                components_processed=components_processed,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            errors.append(f"Processing error: {e}")
            
            # Trigger error callbacks
            await self._trigger_error_callbacks(message, e)
            
            # Update metrics
            self._update_metrics(False, processing_time, components_processed)
            
            return ProcessingResult(
                success=False,
                processing_time_ms=processing_time,
                components_processed=components_processed,
                errors=errors
            )
    
    async def _on_candle_completed(self, candle: Candle):
        """Callback when a candle is completed"""
        try:
            # Update rolling state with completed candle
            if self.enable_rolling_state:
                await self.rolling_state_manager.update_rolling_window(
                    candle.symbol, candle.timeframe, 'candle', candle
                )
            
            logger.debug(f"Candle completed: {candle.symbol} {candle.timeframe}")
            
        except Exception as e:
            logger.error(f"Error in candle completion callback: {e}")
    
    async def _on_indicator_updated(self, indicator: TechnicalIndicator):
        """Callback when an indicator is updated"""
        try:
            # Publish indicator update
            indicator_message = StreamMessage(
                id=f"indicator_{indicator.name}_{indicator.symbol}_{indicator.timeframe}",
                timestamp=indicator.timestamp,
                symbol=indicator.symbol,
                data_type='indicator',
                data={
                    'name': indicator.name,
                    'value': indicator.value,
                    'parameters': indicator.parameters,
                    'timeframe': indicator.timeframe
                },
                source='rolling_state_manager'
            )
            
            await self.stream_buffer.publish_message(indicator_message)
            
            logger.debug(f"Indicator updated: {indicator.name} {indicator.symbol} {indicator.timeframe}")
            
        except Exception as e:
            logger.error(f"Error in indicator update callback: {e}")
    
    async def _trigger_pre_processing_callbacks(self, message: StreamMessage):
        """Trigger pre-processing callbacks"""
        for callback in self.pre_processing_callbacks:
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Pre-processing callback error: {e}")
    
    async def _trigger_post_processing_callbacks(self, message: StreamMessage, normalized_data: NormalizedData):
        """Trigger post-processing callbacks"""
        for callback in self.post_processing_callbacks:
            try:
                await callback(message, normalized_data)
            except Exception as e:
                logger.error(f"Post-processing callback error: {e}")
    
    async def _trigger_error_callbacks(self, message: StreamMessage, error: Exception):
        """Trigger error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(message, error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    def _update_metrics(self, success: bool, processing_time_ms: float, components_processed: List[str]):
        """Update processing metrics"""
        self.metrics.messages_processed += 1
        
        if success:
            self.metrics.messages_successful += 1
        else:
            self.metrics.messages_failed += 1
        
        # Update average processing time
        if self.metrics.messages_processed > 0:
            self.metrics.avg_processing_time_ms = (
                (self.metrics.avg_processing_time_ms * (self.metrics.messages_processed - 1) + processing_time_ms) /
                self.metrics.messages_processed
            )
        
        self.metrics.last_processed_time = datetime.now(timezone.utc)
        
        # Update component metrics
        for component in components_processed:
            if component not in self.metrics.component_metrics:
                self.metrics.component_metrics[component] = {
                    'messages_processed': 0,
                    'avg_processing_time_ms': 0.0
                }
            
            self.metrics.component_metrics[component]['messages_processed'] += 1
    
    async def process_message(self, message: StreamMessage) -> ProcessingResult:
        """
        Process a stream message through the pipeline
        
        Args:
            message: StreamMessage to process
            
        Returns:
            ProcessingResult with processing details
        """
        try:
            # Add message to processing queue
            await self.processing_queue.put(message)
            
            # For immediate processing, we can also process directly
            return await self._process_message(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ProcessingResult(
                success=False,
                processing_time_ms=0.0,
                components_processed=[],
                errors=[str(e)]
            )
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[ProcessingResult]:
        """
        Process a batch of messages
        
        Args:
            messages: List of StreamMessages to process
            
        Returns:
            List of ProcessingResults
        """
        try:
            # Process messages concurrently
            tasks = [self.process_message(message) for message in messages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to ProcessingResult
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ProcessingResult(
                        success=False,
                        processing_time_ms=0.0,
                        components_processed=[],
                        errors=[str(result)]
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [ProcessingResult(
                success=False,
                processing_time_ms=0.0,
                components_processed=[],
                errors=[str(e)]
            ) for _ in messages]
    
    def add_pre_processing_callback(self, callback: Callable):
        """Add pre-processing callback"""
        self.pre_processing_callbacks.append(callback)
    
    def add_post_processing_callback(self, callback: Callable):
        """Add post-processing callback"""
        self.post_processing_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all components"""
        return {
            'stream_buffer': self.stream_buffer.get_metrics(),
            'stream_normalizer': self.stream_normalizer.get_metrics(),
            'candle_builder': self.candle_builder.get_metrics(),
            'rolling_state_manager': self.rolling_state_manager.get_metrics()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream processor metrics"""
        return {
            'is_running': self.is_running,
            'messages_processed': self.metrics.messages_processed,
            'messages_successful': self.metrics.messages_successful,
            'messages_failed': self.metrics.messages_failed,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_mps': self.metrics.throughput_mps,
            'last_processed_time': self.metrics.last_processed_time.isoformat() if self.metrics.last_processed_time else None,
            'queue_size': self.processing_queue.qsize(),
            'active_tasks': len(self.processing_tasks),
            'component_metrics': self.metrics.component_metrics
        }
    
    async def get_latest_signals(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get latest generated signals"""
        try:
            # Get signals from the signal buffer
            signals = []
            
            # If we have a signal buffer, get from there
            if hasattr(self, 'signal_buffer') and self.signal_buffer:
                buffer_signals = list(self.signal_buffer)[-50:]  # Get last 50 signals
                for signal in buffer_signals:
                    if isinstance(signal, dict):
                        if symbol is None or signal.get('symbol') == symbol:
                            signals.append(signal)
            else:
                # Fallback: return empty list
                signals = []
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        try:
            return {
                'status': 'active' if self.is_running else 'inactive',
                'initialized': self.is_initialized,
                'is_running': self.is_running,
                'messages_processed': self.metrics.messages_processed,
                'messages_failed': self.metrics.messages_failed,
                'queue_size': self.processing_queue.qsize(),
                'active_tasks': len(self.processing_tasks),
                'components': {
                    'stream_buffer': 'active' if self.stream_buffer else 'inactive',
                    'stream_normalizer': 'active' if self.stream_normalizer else 'inactive',
                    'candle_builder': 'active' if self.candle_builder else 'inactive',
                    'rolling_state_manager': 'active' if self.rolling_state_manager else 'inactive'
                }
            }
        except Exception as e:
            logger.error(f"Error getting processor status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown the stream processor and all components"""
        self.is_running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown all components
        await self.stream_buffer.shutdown()
        await self.stream_normalizer.shutdown()
        await self.candle_builder.shutdown()
        await self.rolling_state_manager.shutdown()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 StreamProcessor shutdown complete")

# Global instance
stream_processor = StreamProcessor()
