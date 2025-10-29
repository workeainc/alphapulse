"""
Rolling State Manager for AlphaPulse
In-memory rolling windows and technical indicator calculations with TimescaleDB integration
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Import existing components
try:
    from .stream_buffer import StreamMessage, StreamBuffer
    from .stream_normalizer import NormalizedData, StreamNormalizer
    from .candle_builder import Candle, CandleBuilder
    from ..src.database.connection import TimescaleDBConnection
    from ..src.core.config import settings
except ImportError:
    # Fallback for standalone testing
    try:
        from stream_buffer import StreamMessage, StreamBuffer
        from stream_normalizer import NormalizedData, StreamNormalizer
        from candle_builder import Candle, CandleBuilder
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
    except ImportError:
        # Minimal fallback classes for testing
        from dataclasses import dataclass
        from datetime import datetime, timezone
        from typing import Dict, Any, Optional
        from collections import deque
        
        @dataclass
        class StreamMessage:
            message_id: str
            symbol: str
            data_type: str
            source: str
            data: Dict[str, Any]
            timestamp: datetime
        
        @dataclass
        class StreamBuffer:
            def __init__(self, config=None): pass
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class NormalizedData:
            message_id: str
            symbol: str
            normalized_data: Dict[str, Any]
            validation_status: str
            timestamp: datetime
        
        @dataclass
        class StreamNormalizer:
            def __init__(self, config=None): pass
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class Candle:
            symbol: str
            timeframe: str
            open_time: datetime
            close_time: datetime
            open_price: float
            high_price: float
            low_price: float
            close_price: float
            volume: float
        
        @dataclass
        class CandleBuilder:
            def __init__(self, config=None): 
                self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            async def initialize(self): pass
            async def shutdown(self): pass
        
        @dataclass
        class TimescaleDBConnection:
            def __init__(self): pass
            async def initialize(self): pass
            async def close(self): pass
        
        class settings:
            DATABASE_HOST = 'localhost'
            DATABASE_PORT = 5432
            DATABASE_NAME = 'alphapulse'
            DATABASE_USER = 'alpha_emon'
            DATABASE_PASSWORD = 'Emon_@17711'
            TIMESCALEDB_HOST = 'localhost'
            TIMESCALEDB_PORT = 5432
            TIMESCALEDB_DATABASE = 'alphapulse'
            TIMESCALEDB_USERNAME = 'alpha_emon'
            TIMESCALEDB_PASSWORD = 'Emon_@17711'

logger = logging.getLogger(__name__)

@dataclass
class RollingWindow:
    """Rolling window data structure"""
    symbol: str
    timeframe: str
    window_size: int
    data_type: str  # 'price', 'volume', 'candle', 'indicator'
    data: deque
    last_update: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalIndicator:
    """Technical indicator result"""
    name: str
    value: float
    timestamp: datetime
    symbol: str
    timeframe: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RollingStateMetrics:
    """Rolling state performance metrics"""
    windows_created: int = 0
    windows_updated: int = 0
    indicators_calculated: int = 0
    avg_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    last_update_time: Optional[datetime] = None

class RollingStateManager:
    """
    Rolling state manager for in-memory rolling windows and technical indicators
    
    Features:
    - In-memory rolling windows for real-time data
    - Technical indicator calculations (SMA, EMA, RSI, MACD, etc.)
    - Pattern detection integration
    - Memory management and optimization
    - TimescaleDB integration for persistence
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Rolling window settings
        self.default_window_size = self.config.get('default_window_size', 100)
        self.max_window_size = self.config.get('max_window_size', 10000)
        self.memory_limit_mb = self.config.get('memory_limit_mb', 1024)  # 1GB default
        
        # Technical indicator settings
        self.enable_indicators = self.config.get('enable_indicators', True)
        self.indicator_periods = self.config.get('indicator_periods', {
            'sma': [20, 50, 200],
            'ema': [12, 26],
            'rsi': [14],
            'macd': [12, 26, 9],
            'bollinger': [20, 2]
        })
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.update_interval = self.config.get('update_interval', 0.1)  # seconds
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # seconds
        
        # State management
        self.is_running = False
        self.metrics = RollingStateMetrics()
        self.rolling_windows = defaultdict(lambda: defaultdict(dict))  # symbol -> timeframe -> data_type -> window
        self.technical_indicators = defaultdict(lambda: defaultdict(dict))  # symbol -> timeframe -> indicator -> value
        self.indicator_callbacks = defaultdict(list)  # indicator_name -> callbacks
        self.pattern_callbacks = defaultdict(list)  # pattern_type -> callbacks
        
        # TimescaleDB integration
        self.timescaledb = None
        self.db_batch = []
        self.db_flush_interval = self.config.get('db_flush_interval', 5.0)
        self.db_flush_task = None
        
        # Memory management
        self.memory_usage = 0.0
        self.last_cleanup = datetime.now(timezone.utc)
        
        logger.info("RollingStateManager initialized")
    
    async def initialize(self):
        """Initialize the rolling state manager"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("✅ RollingStateManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RollingStateManager: {e}")
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
                'pool_size': 5,
                'max_overflow': 10
            })
            
            await self.timescaledb.initialize()
            logger.info("✅ TimescaleDB connection initialized for rolling state manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start memory cleanup task
        asyncio.create_task(self._memory_cleanup_loop())
        
        # Start DB flush task
        if self.timescaledb:
            self.db_flush_task = asyncio.create_task(self._db_flush_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _memory_cleanup_loop(self):
        """Periodic memory cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_memory()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")
    
    async def _cleanup_memory(self):
        """Clean up memory usage"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Remove old windows
            for symbol in list(self.rolling_windows.keys()):
                for timeframe in list(self.rolling_windows[symbol].keys()):
                    for data_type in list(self.rolling_windows[symbol][timeframe].keys()):
                        window = self.rolling_windows[symbol][timeframe][data_type]
                        age = (current_time - window.last_update).total_seconds()
                        
                        if age > self.cleanup_interval:
                            del self.rolling_windows[symbol][timeframe][data_type]
            
            # Update memory usage
            self.metrics.memory_usage_mb = self._calculate_memory_usage()
            self.last_cleanup = current_time
            
            logger.debug(f"Memory cleanup completed. Usage: {self.metrics.memory_usage_mb:.2f}MB")
            
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in MB"""
        try:
            total_size = 0
            
            # Calculate window sizes
            for symbol_data in self.rolling_windows.values():
                for timeframe_data in symbol_data.values():
                    for window in timeframe_data.values():
                        total_size += len(window.data) * 64  # Approximate bytes per data point
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Memory calculation error: {e}")
            return 0.0
    
    async def _db_flush_loop(self):
        """Periodic database flush loop"""
        while self.is_running:
            try:
                if self.db_batch:
                    await self._flush_db_batch()
                await asyncio.sleep(self.db_flush_interval)
            except Exception as e:
                logger.error(f"DB flush error: {e}")
    
    async def _flush_db_batch(self):
        """Flush rolling state data to TimescaleDB"""
        if not self.db_batch or not self.timescaledb:
            return
        
        try:
            batch = self.db_batch.copy()
            self.db_batch.clear()
            
            # Insert batch into TimescaleDB
            async with self.timescaledb.async_session() as session:
                for data in batch:
                    await self._insert_rolling_data(session, data)
                await session.commit()
            
            logger.debug(f"Flushed {len(batch)} rolling state records to TimescaleDB")
            
        except Exception as e:
            logger.error(f"DB flush failed: {e}")
            # Re-add to batch for retry
            self.db_batch.extend(batch)
    
    async def _insert_rolling_data(self, session, data):
        """Insert rolling state data into TimescaleDB"""
        # Implementation depends on your rolling state table schema
        pass
    
    async def update_rolling_window(self, symbol: str, timeframe: str, data_type: str, value: Any, window_size: int = None) -> RollingWindow:
        """
        Update a rolling window with new data
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe (1m, 5m, etc.)
            data_type: Type of data ('price', 'volume', 'candle', 'indicator')
            value: Data value to add
            window_size: Window size (uses default if None)
            
        Returns:
            Updated RollingWindow
        """
        try:
            window_size = window_size or self.default_window_size
            
            # Get or create window
            if symbol not in self.rolling_windows:
                self.rolling_windows[symbol] = defaultdict(dict)
            if timeframe not in self.rolling_windows[symbol]:
                self.rolling_windows[symbol][timeframe] = {}
            if data_type not in self.rolling_windows[symbol][timeframe]:
                self.rolling_windows[symbol][timeframe][data_type] = RollingWindow(
                    symbol=symbol,
                    timeframe=timeframe,
                    window_size=window_size,
                    data_type=data_type,
                    data=deque(maxlen=window_size),
                    last_update=datetime.now(timezone.utc)
                )
                self.metrics.windows_created += 1
            
            window = self.rolling_windows[symbol][timeframe][data_type]
            
            # Add new value
            window.data.append(value)
            window.last_update = datetime.now(timezone.utc)
            
            # Update metrics
            self.metrics.windows_updated += 1
            self.metrics.last_update_time = datetime.now(timezone.utc)
            
            # Calculate indicators if enabled
            if self.enable_indicators and data_type == 'price':
                await self._calculate_indicators(symbol, timeframe, window)
            
            # Add to DB batch
            if self.timescaledb:
                self.db_batch.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_type': data_type,
                    'value': value,
                    'window_size': window_size,
                    'timestamp': datetime.now(timezone.utc)
                })
                
                if len(self.db_batch) >= self.batch_size:
                    await self._flush_db_batch()
            
            return window
            
        except Exception as e:
            logger.error(f"Error updating rolling window: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, timeframe: str, window: RollingWindow):
        """Calculate technical indicators for the given window"""
        try:
            if len(window.data) < 2:
                return
            
            prices = list(window.data)
            
            # Calculate SMA
            for period in self.indicator_periods.get('sma', []):
                if len(prices) >= period:
                    sma_value = np.mean(prices[-period:])
                    indicator = TechnicalIndicator(
                        name='SMA',
                        value=sma_value,
                        timestamp=window.last_update,
                        symbol=symbol,
                        timeframe=timeframe,
                        parameters={'period': period}
                    )
                    await self._store_indicator(indicator)
            
            # Calculate EMA
            for period in self.indicator_periods.get('ema', []):
                if len(prices) >= period:
                    ema_value = self._calculate_ema(prices, period)
                    indicator = TechnicalIndicator(
                        name='EMA',
                        value=ema_value,
                        timestamp=window.last_update,
                        symbol=symbol,
                        timeframe=timeframe,
                        parameters={'period': period}
                    )
                    await self._store_indicator(indicator)
            
            # Calculate RSI
            for period in self.indicator_periods.get('rsi', []):
                if len(prices) >= period + 1:
                    rsi_value = self._calculate_rsi(prices, period)
                    indicator = TechnicalIndicator(
                        name='RSI',
                        value=rsi_value,
                        timestamp=window.last_update,
                        symbol=symbol,
                        timeframe=timeframe,
                        parameters={'period': period}
                    )
                    await self._store_indicator(indicator)
            
            # Calculate MACD
            macd_periods = self.indicator_periods.get('macd', [12, 26, 9])
            if len(macd_periods) == 3 and len(prices) >= max(macd_periods):
                macd_line, signal_line, histogram = self._calculate_macd(prices, macd_periods[0], macd_periods[1], macd_periods[2])
                
                # Store MACD line
                indicator = TechnicalIndicator(
                    name='MACD',
                    value=macd_line,
                    timestamp=window.last_update,
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters={'fast': macd_periods[0], 'slow': macd_periods[1], 'signal': macd_periods[2]}
                )
                await self._store_indicator(indicator)
                
                # Store signal line
                indicator = TechnicalIndicator(
                    name='MACD_SIGNAL',
                    value=signal_line,
                    timestamp=window.last_update,
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters={'fast': macd_periods[0], 'slow': macd_periods[1], 'signal': macd_periods[2]}
                )
                await self._store_indicator(indicator)
            
            # Calculate Bollinger Bands
            bb_periods = self.indicator_periods.get('bollinger', [20, 2])
            if len(bb_periods) == 2 and len(prices) >= bb_periods[0]:
                upper, middle, lower = self._calculate_bollinger_bands(prices, bb_periods[0], bb_periods[1])
                
                # Store upper band
                indicator = TechnicalIndicator(
                    name='BB_UPPER',
                    value=upper,
                    timestamp=window.last_update,
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters={'period': bb_periods[0], 'std_dev': bb_periods[1]}
                )
                await self._store_indicator(indicator)
                
                # Store middle band (SMA)
                indicator = TechnicalIndicator(
                    name='BB_MIDDLE',
                    value=middle,
                    timestamp=window.last_update,
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters={'period': bb_periods[0], 'std_dev': bb_periods[1]}
                )
                await self._store_indicator(indicator)
                
                # Store lower band
                indicator = TechnicalIndicator(
                    name='BB_LOWER',
                    value=lower,
                    timestamp=window.last_update,
                    symbol=symbol,
                    timeframe=timeframe,
                    parameters={'period': bb_periods[0], 'std_dev': bb_periods[1]}
                )
                await self._store_indicator(indicator)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float], fast_period: int, slow_period: int, signal_period: int) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < max(fast_period, slow_period):
            return prices[-1], prices[-1], 0.0
        
        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD line)
        # For simplicity, we'll use the current MACD value
        signal_line = macd_line
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    async def _store_indicator(self, indicator: TechnicalIndicator):
        """Store technical indicator result"""
        try:
            # Store in memory
            key = f"{indicator.name}_{'_'.join(str(v) for v in indicator.parameters.values())}"
            self.technical_indicators[indicator.symbol][indicator.timeframe][key] = indicator
            
            # Update metrics
            self.metrics.indicators_calculated += 1
            
            # Trigger callbacks
            await self._trigger_indicator_callbacks(indicator)
            
        except Exception as e:
            logger.error(f"Error storing indicator: {e}")
    
    async def _trigger_indicator_callbacks(self, indicator: TechnicalIndicator):
        """Trigger registered indicator callbacks"""
        callbacks = self.indicator_callbacks.get(indicator.name, [])
        for callback in callbacks:
            try:
                await callback(indicator)
            except Exception as e:
                logger.error(f"Indicator callback error: {e}")
    
    def get_rolling_window(self, symbol: str, timeframe: str, data_type: str) -> Optional[RollingWindow]:
        """Get a rolling window"""
        return self.rolling_windows.get(symbol, {}).get(timeframe, {}).get(data_type)
    
    def get_indicator_value(self, symbol: str, timeframe: str, indicator_name: str, parameters: Dict[str, Any] = None) -> Optional[float]:
        """Get the latest value of a technical indicator"""
        try:
            key = f"{indicator_name}_{'_'.join(str(v) for v in (parameters or {}).values())}"
            indicator = self.technical_indicators.get(symbol, {}).get(timeframe, {}).get(key)
            return indicator.value if indicator else None
        except Exception as e:
            logger.error(f"Error getting indicator value: {e}")
            return None
    
    def add_indicator_callback(self, indicator_name: str, callback: Callable):
        """Add callback for indicator updates"""
        self.indicator_callbacks[indicator_name].append(callback)
    
    def add_pattern_callback(self, pattern_type: str, callback: Callable):
        """Add callback for pattern detection"""
        self.pattern_callbacks[pattern_type].append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rolling state metrics"""
        return {
            'is_running': self.is_running,
            'windows_created': self.metrics.windows_created,
            'windows_updated': self.metrics.windows_updated,
            'indicators_calculated': self.metrics.indicators_calculated,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'memory_usage_mb': self.metrics.memory_usage_mb,
            'last_update_time': self.metrics.last_update_time.isoformat() if self.metrics.last_update_time else None,
            'active_windows': sum(len(timeframes) for timeframes in self.rolling_windows.values()),
            'active_indicators': sum(len(indicators) for timeframes in self.technical_indicators.values() for indicators in timeframes.values()),
            'db_batch_size': len(self.db_batch)
        }
    
    async def shutdown(self):
        """Shutdown the rolling state manager"""
        self.is_running = False
        
        # Cancel background tasks
        if self.db_flush_task:
            self.db_flush_task.cancel()
        
        # Flush remaining data
        if self.db_batch and self.timescaledb:
            await self._flush_db_batch()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 RollingStateManager shutdown complete")

# Global instance
rolling_state_manager = RollingStateManager()
