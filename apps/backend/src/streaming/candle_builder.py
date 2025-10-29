"""
Real-time Candle Builder for AlphaPulse
Multi-timeframe OHLCV candle construction with TimescaleDB integration
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

# Import existing components
try:
    from .stream_buffer import StreamMessage, StreamBuffer
    from .stream_normalizer import NormalizedData, StreamNormalizer
    from ..src.database.connection import TimescaleDBConnection
    from ..src.core.config import settings
except ImportError:
    # Fallback for standalone testing
    try:
        from stream_buffer import StreamMessage, StreamBuffer
        from stream_normalizer import NormalizedData, StreamNormalizer
        from src.database.connection import TimescaleDBConnection
        from src.core.config import settings
    except ImportError:
        # Minimal fallback classes for testing
        from dataclasses import dataclass
        from datetime import datetime, timezone
        from typing import Dict, Any, Optional
        
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
class Candle:
    """OHLCV candle structure"""
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    trade_count: int = 0
    vwap: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandleMetrics:
    """Candle building performance metrics"""
    candles_created: int = 0
    candles_closed: int = 0
    candles_updated: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_cps: float = 0.0  # candles per second
    last_candle_time: Optional[datetime] = None
    active_candles: int = 0

class CandleBuilder:
    """
    Real-time candle builder with multi-timeframe support
    
    Features:
    - Multi-timeframe candle construction (1m, 5m, 15m, 1h, 4h, 1d)
    - Exact close semantics for precise timing
    - Volume-weighted average price (VWAP) calculation
    - Trade count tracking
    - TimescaleDB integration for persistence
    - Real-time candle updates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Timeframe configuration
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        # Candle settings
        self.exact_close = self.config.get('exact_close', True)
        self.enable_vwap = self.config.get('enable_vwap', True)
        self.enable_trade_count = self.config.get('enable_trade_count', True)
        self.max_candle_age = self.config.get('max_candle_age', 3600)  # seconds
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.flush_interval = self.config.get('flush_interval', 1.0)  # seconds
        self.processing_timeout = self.config.get('processing_timeout', 5.0)  # seconds
        
        # State management
        self.is_running = False
        self.metrics = CandleMetrics()
        self.active_candles = defaultdict(lambda: defaultdict(dict))  # symbol -> timeframe -> candle
        self.candle_history = defaultdict(lambda: defaultdict(deque))  # symbol -> timeframe -> history
        self.price_cache = defaultdict(lambda: defaultdict(deque))  # symbol -> timeframe -> prices
        
        # TimescaleDB integration
        self.timescaledb = None
        self.db_batch = []
        self.db_flush_interval = self.config.get('db_flush_interval', 5.0)
        self.db_flush_task = None
        
        # Processing callbacks
        self.candle_callbacks = defaultdict(list)  # timeframe -> callbacks
        self.completion_callbacks = defaultdict(list)  # timeframe -> callbacks
        
        logger.info("CandleBuilder initialized")
    
    async def initialize(self):
        """Initialize the candle builder"""
        try:
            # Initialize TimescaleDB connection
            await self._initialize_timescaledb()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("✅ CandleBuilder initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CandleBuilder: {e}")
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
            logger.info("✅ TimescaleDB connection initialized for candle builder")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
            self.timescaledb = None
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start candle cleanup task
        asyncio.create_task(self._candle_cleanup_loop())
        
        # Start DB flush task
        if self.timescaledb:
            self.db_flush_task = asyncio.create_task(self._db_flush_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _candle_cleanup_loop(self):
        """Periodic cleanup of old candles"""
        while self.is_running:
            try:
                await self._cleanup_old_candles()
                await asyncio.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Candle cleanup error: {e}")
    
    async def _cleanup_old_candles(self):
        """Clean up old candles from memory"""
        current_time = datetime.now(timezone.utc)
        
        for symbol in list(self.active_candles.keys()):
            for timeframe in list(self.active_candles[symbol].keys()):
                candle = self.active_candles[symbol][timeframe]
                if candle:
                    age = (current_time - candle.open_time).total_seconds()
                    if age > self.max_candle_age:
                        # Close and store old candle
                        await self._close_candle(symbol, timeframe, current_time)
                        del self.active_candles[symbol][timeframe]
    
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
        """Flush candles to TimescaleDB"""
        if not self.db_batch or not self.timescaledb:
            return
        
        try:
            batch = self.db_batch.copy()
            self.db_batch.clear()
            
            # Insert batch into TimescaleDB
            async with self.timescaledb.async_session() as session:
                for candle in batch:
                    await self._insert_candle_data(session, candle)
                await session.commit()
            
            logger.debug(f"Flushed {len(batch)} candles to TimescaleDB")
            
        except Exception as e:
            logger.error(f"DB flush failed: {e}")
            # Re-add to batch for retry
            self.db_batch.extend(batch)
    
    async def _insert_candle_data(self, session, candle: Candle):
        """Insert candle data into TimescaleDB"""
        # Implementation depends on your candle table schema
        # This would typically go into a candles table
        pass
    
    async def process_tick(self, normalized_data: NormalizedData) -> List[Candle]:
        """
        Process a normalized tick to build candles
        
        Args:
            normalized_data: Normalized tick data
            
        Returns:
            List of updated/closed candles
        """
        if normalized_data.validation_status != 'valid':
            return []
        
        start_time = asyncio.get_event_loop().time()
        updated_candles = []
        
        try:
            data = normalized_data.normalized_data
            symbol = data.get('symbol', '').upper()
            price = float(data.get('price', 0))
            volume = float(data.get('volume', 0))
            timestamp = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
            
            if not symbol or price <= 0:
                return []
            
            # Process each timeframe
            for timeframe in self.timeframes:
                candle = await self._update_candle(symbol, timeframe, price, volume, timestamp)
                if candle:
                    updated_candles.append(candle)
                    
                    # Check if candle should be closed
                    if await self._should_close_candle(symbol, timeframe, timestamp):
                        closed_candle = await self._close_candle(symbol, timeframe, timestamp)
                        if closed_candle:
                            updated_candles.append(closed_candle)
            
            # Update metrics
            self.metrics.candles_updated += len(updated_candles)
            self.metrics.last_candle_time = datetime.now(timezone.utc)
            self.metrics.active_candles = sum(len(timeframes) for timeframes in self.active_candles.values())
            
            # Add to DB batch
            if self.timescaledb and updated_candles:
                self.db_batch.extend(updated_candles)
                if len(self.db_batch) >= self.batch_size:
                    await self._flush_db_batch()
            
            # Trigger callbacks
            for candle in updated_candles:
                await self._trigger_callbacks(candle)
            
            return updated_candles
            
        except Exception as e:
            logger.error(f"Candle processing error: {e}")
            return []
        
        finally:
            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update average processing time
            if self.metrics.candles_updated > 0:
                self.metrics.avg_processing_time_ms = (
                    (self.metrics.avg_processing_time_ms * (self.metrics.candles_updated - 1) + processing_time) /
                    self.metrics.candles_updated
                )
    
    async def _update_candle(self, symbol: str, timeframe: str, price: float, volume: float, timestamp: datetime) -> Optional[Candle]:
        """Update or create a candle for the given timeframe"""
        try:
            # Get or create active candle
            if symbol not in self.active_candles or timeframe not in self.active_candles[symbol]:
                await self._create_new_candle(symbol, timeframe, price, volume, timestamp)
                return None
            
            candle = self.active_candles[symbol][timeframe]
            
            # Update candle data
            candle.high_price = max(candle.high_price, price)
            candle.low_price = min(candle.low_price, price)
            candle.close_price = price
            candle.volume += volume
            candle.trade_count += 1
            
            # Update VWAP
            if self.enable_vwap:
                candle.vwap = await self._calculate_vwap(symbol, timeframe, price, volume)
            
            # Update timestamp
            candle.timestamp = timestamp
            
            return candle
            
        except Exception as e:
            logger.error(f"Error updating candle: {e}")
            return None
    
    async def _create_new_candle(self, symbol: str, timeframe: str, price: float, volume: float, timestamp: datetime):
        """Create a new candle for the given timeframe"""
        try:
            # Calculate candle boundaries
            seconds = self.timeframe_seconds[timeframe]
            open_time = self._get_candle_open_time(timestamp, seconds)
            close_time = open_time + timedelta(seconds=seconds)
            
            # Create new candle
            candle = Candle(
                symbol=symbol,
                timeframe=timeframe,
                open_time=open_time,
                close_time=close_time,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                volume=volume,
                trade_count=1,
                vwap=price if self.enable_vwap else 0.0,
                timestamp=timestamp
            )
            
            # Store active candle
            self.active_candles[symbol][timeframe] = candle
            
            # Update metrics
            self.metrics.candles_created += 1
            
            logger.debug(f"Created new {timeframe} candle for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating new candle: {e}")
    
    def _get_candle_open_time(self, timestamp: datetime, seconds: int) -> datetime:
        """Calculate the open time for a candle based on timestamp and timeframe"""
        # Round down to the nearest timeframe boundary
        timestamp_seconds = int(timestamp.timestamp())
        open_seconds = (timestamp_seconds // seconds) * seconds
        return datetime.fromtimestamp(open_seconds, tz=timezone.utc)
    
    async def _should_close_candle(self, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        """Check if a candle should be closed"""
        if symbol not in self.active_candles or timeframe not in self.active_candles[symbol]:
            return False
        
        candle = self.active_candles[symbol][timeframe]
        return timestamp >= candle.close_time
    
    async def _close_candle(self, symbol: str, timeframe: str, timestamp: datetime) -> Optional[Candle]:
        """Close a candle and store it"""
        try:
            if symbol not in self.active_candles or timeframe not in self.active_candles[symbol]:
                return None
            
            candle = self.active_candles[symbol][timeframe]
            
            # Update close time if needed
            if timestamp > candle.close_time:
                candle.close_time = timestamp
            
            # Store in history
            self.candle_history[symbol][timeframe].append(candle)
            
            # Keep only recent history in memory
            max_history = 1000
            if len(self.candle_history[symbol][timeframe]) > max_history:
                self.candle_history[symbol][timeframe].popleft()
            
            # Remove from active candles
            del self.active_candles[symbol][timeframe]
            
            # Update metrics
            self.metrics.candles_closed += 1
            
            # Trigger completion callbacks
            await self._trigger_completion_callbacks(candle)
            
            logger.debug(f"Closed {timeframe} candle for {symbol}")
            return candle
            
        except Exception as e:
            logger.error(f"Error closing candle: {e}")
            return None
    
    async def _calculate_vwap(self, symbol: str, timeframe: str, price: float, volume: float) -> float:
        """Calculate Volume Weighted Average Price using standard formula"""
        try:
            # Get price cache for this symbol/timeframe
            cache_key = f"{symbol}_{timeframe}"
            if cache_key not in self.price_cache:
                self.price_cache[cache_key] = deque(maxlen=1000)
            
            # Add current price-volume pair
            self.price_cache[cache_key].append((price, volume))
            
            # Standard VWAP formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
            # Where Typical Price = (High + Low + Close) / 3
            # For real-time, we use current price as close, and estimate high/low
            typical_price = price  # Simplified for real-time calculation
            
            # Calculate VWAP
            total_pv = sum(tp * v for tp, v in self.price_cache[cache_key])
            total_volume = sum(v for _, v in self.price_cache[cache_key])
            
            return total_pv / total_volume if total_volume > 0 else price
            
        except Exception as e:
            logger.error(f"VWAP calculation error: {e}")
            return price
    
    async def _trigger_callbacks(self, candle: Candle):
        """Trigger registered callbacks"""
        callbacks = self.candle_callbacks.get(candle.timeframe, [])
        for callback in callbacks:
            try:
                await callback(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")
    
    async def _trigger_completion_callbacks(self, candle: Candle):
        """Trigger completion callbacks for closed candles"""
        callbacks = self.completion_callbacks.get(candle.timeframe, [])
        for callback in callbacks:
            try:
                await callback(candle)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
    
    def add_candle_callback(self, timeframe: str, callback):
        """Add callback for candle updates"""
        self.candle_callbacks[timeframe].append(callback)
    
    def add_completion_callback(self, timeframe: str, callback):
        """Add callback for candle completions"""
        self.completion_callbacks[timeframe].append(callback)
    
    def get_active_candle(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Get the currently active candle for a symbol and timeframe"""
        return self.active_candles.get(symbol, {}).get(timeframe)
    
    def get_candle_history(self, symbol: str, timeframe: str, limit: int = 100) -> List[Candle]:
        """Get recent candle history for a symbol and timeframe"""
        history = self.candle_history.get(symbol, {}).get(timeframe, [])
        return list(history)[-limit:] if history else []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get candle building metrics"""
        return {
            'is_running': self.is_running,
            'candles_created': self.metrics.candles_created,
            'candles_closed': self.metrics.candles_closed,
            'candles_updated': self.metrics.candles_updated,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'throughput_cps': self.metrics.throughput_cps,
            'last_candle_time': self.metrics.last_candle_time.isoformat() if self.metrics.last_candle_time else None,
            'active_candles': self.metrics.active_candles,
            'timeframes': self.timeframes,
            'db_batch_size': len(self.db_batch)
        }
    
    async def get_latest_candles(self, symbol: str = None, timeframe: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest candles for a symbol and timeframe"""
        try:
            candles = []
            
            if symbol and timeframe:
                # Get specific symbol/timeframe candles
                history = self.candle_history.get(symbol, {}).get(timeframe, [])
                recent_candles = list(history)[-limit:] if history else []
                
                for candle in recent_candles:
                    candles.append({
                        'symbol': candle.symbol,
                        'timeframe': candle.timeframe,
                        'open': candle.open,
                        'high': candle.high,
                        'low': candle.low,
                        'close': candle.close,
                        'volume': candle.volume,
                        'timestamp': candle.timestamp.isoformat(),
                        'vwap': candle.vwap,
                        'is_complete': candle.is_complete
                    })
                    
            elif symbol:
                # Get all timeframes for a symbol
                symbol_candles = self.candle_history.get(symbol, {})
                for tf, history in symbol_candles.items():
                    recent_candles = list(history)[-limit:] if history else []
                    for candle in recent_candles:
                        candles.append({
                            'symbol': candle.symbol,
                            'timeframe': candle.timeframe,
                            'open': candle.open,
                            'high': candle.high,
                            'low': candle.low,
                            'close': candle.close,
                            'volume': candle.volume,
                            'timestamp': candle.timestamp.isoformat(),
                            'vwap': candle.vwap,
                            'is_complete': candle.is_complete
                        })
                        
            else:
                # Get latest candles from all symbols/timeframes
                for sym, timeframes in self.candle_history.items():
                    for tf, history in timeframes.items():
                        recent_candles = list(history)[-10:] if history else []  # Limit per timeframe
                        for candle in recent_candles:
                            candles.append({
                                'symbol': candle.symbol,
                                'timeframe': candle.timeframe,
                                'open': candle.open,
                                'high': candle.high,
                                'low': candle.low,
                                'close': candle.close,
                                'volume': candle.volume,
                                'timestamp': candle.timestamp.isoformat(),
                                'vwap': candle.vwap,
                                'is_complete': candle.is_complete
                            })
            
            # Sort by timestamp descending and limit
            candles.sort(key=lambda x: x['timestamp'], reverse=True)
            return candles[:limit]
            
        except Exception as e:
            logger.error(f"Error getting latest candles: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the candle builder"""
        self.is_running = False
        
        # Close all active candles
        current_time = datetime.now(timezone.utc)
        for symbol in list(self.active_candles.keys()):
            for timeframe in list(self.active_candles[symbol].keys()):
                await self._close_candle(symbol, timeframe, current_time)
        
        # Cancel background tasks
        if self.db_flush_task:
            self.db_flush_task.cancel()
        
        # Flush remaining data
        if self.db_batch and self.timescaledb:
            await self._flush_db_batch()
        
        # Close TimescaleDB connection
        if self.timescaledb:
            await self.timescaledb.close()
        
        logger.info("🛑 CandleBuilder shutdown complete")

# Global instance
candle_builder = CandleBuilder()
