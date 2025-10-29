"""
Sliding Window Buffer Service for AlphaPlus
Provides ultra-fast access to recent candlestick data with intelligent caching
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
import asyncio
import json

logger = logging.getLogger(__name__)

class SlidingWindowBuffer:
    """Advanced sliding window buffer with intelligent caching and memory management"""
    
    def __init__(self, max_size: int = 1000, enable_compression: bool = True):
        self.max_size = max_size
        self.enable_compression = enable_compression
        self.buffers = {}  # symbol_timeframe -> buffer
        self.timestamps = {}  # symbol_timeframe -> timestamps
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self.memory_usage = 0
        self.last_cleanup = datetime.now()
        
        logger.info(f"🔄 Sliding window buffer initialized with max_size={max_size}")
    
    def add_candle(self, symbol: str, timeframe: str, candle_data: Dict) -> bool:
        """Add new candle to buffer with automatic size management"""
        try:
            key = f"{symbol}_{timeframe}"
            
            # Initialize buffer if needed
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.max_size)
                self.timestamps[key] = deque(maxlen=self.max_size)
            
            # Add new data
            self.buffers[key].append(candle_data)
            self.timestamps[key].append(candle_data['timestamp'])
            
            # Update memory usage
            self._update_memory_usage()
            
            # Periodic cleanup
            self._periodic_cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding candle to buffer: {e}")
            return False
    
    def get_recent_candles(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Get recent candles from buffer with cache statistics"""
        try:
            key = f"{symbol}_{timeframe}"
            self.cache_stats['total_requests'] += 1
            
            if key in self.buffers:
                self.cache_stats['hits'] += 1
                buffer_data = list(self.buffers[key])
                return buffer_data[-count:] if count <= len(buffer_data) else buffer_data
            else:
                self.cache_stats['misses'] += 1
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent candles: {e}")
            return []
    
    def get_all_candles(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get all candles from buffer"""
        try:
            key = f"{symbol}_{timeframe}"
            if key in self.buffers:
                return list(self.buffers[key])
            return []
            
        except Exception as e:
            logger.error(f"Error getting all candles: {e}")
            return []
    
    def get_candles_in_range(self, symbol: str, timeframe: str, 
                           start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get candles within a specific time range"""
        try:
            key = f"{symbol}_{timeframe}"
            if key not in self.buffers:
                return []
            
            candles = []
            timestamps = list(self.timestamps[key])
            buffer_data = list(self.buffers[key])
            
            for i, timestamp in enumerate(timestamps):
                if start_time <= timestamp <= end_time:
                    candles.append(buffer_data[i])
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting candles in range: {e}")
            return []
    
    def get_ohlcv_arrays(self, symbol: str, timeframe: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get OHLCV arrays for vectorized operations"""
        try:
            candles = self.get_all_candles(symbol, timeframe)
            if not candles:
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
            opens = np.array([c['open'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            closes = np.array([c['close'] for c in candles])
            volumes = np.array([c.get('volume', 0) for c in candles])
            
            return opens, highs, lows, closes, volumes
            
        except Exception as e:
            logger.error(f"Error getting OHLCV arrays: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the most recent candle"""
        try:
            key = f"{symbol}_{timeframe}"
            if key in self.buffers and self.buffers[key]:
                return self.buffers[key][-1]
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest candle: {e}")
            return None
    
    def has_sufficient_data(self, symbol: str, timeframe: str, min_count: int = 5) -> bool:
        """Check if buffer has sufficient data for pattern detection"""
        try:
            key = f"{symbol}_{timeframe}"
            return key in self.buffers and len(self.buffers[key]) >= min_count
            
        except Exception as e:
            logger.error(f"Error checking sufficient data: {e}")
            return False
    
    def get_buffer_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get statistics for a specific buffer"""
        try:
            key = f"{symbol}_{timeframe}"
            if key not in self.buffers:
                return {
                    'size': 0,
                    'oldest_timestamp': None,
                    'newest_timestamp': None,
                    'time_span_hours': 0
                }
            
            buffer_data = list(self.buffers[key])
            timestamps = list(self.timestamps[key])
            
            if not buffer_data:
                return {'size': 0, 'oldest_timestamp': None, 'newest_timestamp': None, 'time_span_hours': 0}
            
            oldest = timestamps[0]
            newest = timestamps[-1]
            time_span = (newest - oldest).total_seconds() / 3600 if oldest and newest else 0
            
            return {
                'size': len(buffer_data),
                'oldest_timestamp': oldest,
                'newest_timestamp': newest,
                'time_span_hours': time_span
            }
            
        except Exception as e:
            logger.error(f"Error getting buffer stats: {e}")
            return {'size': 0, 'oldest_timestamp': None, 'newest_timestamp': None, 'time_span_hours': 0}
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global buffer statistics"""
        try:
            total_buffers = len(self.buffers)
            total_candles = sum(len(buffer) for buffer in self.buffers.values())
            
            cache_hit_rate = (
                self.cache_stats['hits'] / self.cache_stats['total_requests']
                if self.cache_stats['total_requests'] > 0 else 0.0
            )
            
            return {
                'total_buffers': total_buffers,
                'total_candles': total_candles,
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'cache_hit_rate': cache_hit_rate,
                'cache_hits': self.cache_stats['hits'],
                'cache_misses': self.cache_stats['misses'],
                'cache_evictions': self.cache_stats['evictions'],
                'total_requests': self.cache_stats['total_requests']
            }
            
        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {}
    
    def clear_buffer(self, symbol: str, timeframe: str) -> bool:
        """Clear a specific buffer"""
        try:
            key = f"{symbol}_{timeframe}"
            if key in self.buffers:
                del self.buffers[key]
                del self.timestamps[key]
                self._update_memory_usage()
                logger.info(f"Cleared buffer for {key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing buffer: {e}")
            return False
    
    def clear_all_buffers(self):
        """Clear all buffers"""
        try:
            self.buffers.clear()
            self.timestamps.clear()
            self.memory_usage = 0
            logger.info("Cleared all buffers")
            
        except Exception as e:
            logger.error(f"Error clearing all buffers: {e}")
    
    def _update_memory_usage(self):
        """Update memory usage estimation"""
        try:
            total_size = 0
            for buffer_data in self.buffers.values():
                # Estimate memory usage (rough calculation)
                total_size += len(buffer_data) * 200  # ~200 bytes per candle
            
            self.memory_usage = total_size
            
        except Exception as e:
            logger.error(f"Error updating memory usage: {e}")
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        try:
            now = datetime.now()
            if (now - self.last_cleanup).total_seconds() > 300:  # Every 5 minutes
                self._cleanup_old_data()
                self.last_cleanup = now
                
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to free memory"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep last 24 hours
            evicted_count = 0
            
            for key in list(self.buffers.keys()):
                if key in self.timestamps:
                    # Remove old timestamps and corresponding data
                    old_indices = []
                    for i, timestamp in enumerate(self.timestamps[key]):
                        if timestamp < cutoff_time:
                            old_indices.append(i)
                    
                    # Remove from end to avoid index shifting issues
                    for i in reversed(old_indices):
                        if i < len(self.buffers[key]):
                            self.buffers[key].popleft()
                            self.timestamps[key].popleft()
                            evicted_count += 1
                    
                    # Remove empty buffers
                    if len(self.buffers[key]) == 0:
                        del self.buffers[key]
                        del self.timestamps[key]
            
            if evicted_count > 0:
                self.cache_stats['evictions'] += evicted_count
                self._update_memory_usage()
                logger.info(f"Cleaned up {evicted_count} old candles")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

class AsyncSlidingWindowBuffer:
    """Async wrapper for sliding window buffer with concurrent operations"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = SlidingWindowBuffer(max_size)
        self.lock = asyncio.Lock()
    
    async def add_candle_async(self, symbol: str, timeframe: str, candle_data: Dict) -> bool:
        """Async add candle to buffer"""
        async with self.lock:
            return self.buffer.add_candle(symbol, timeframe, candle_data)
    
    async def get_recent_candles_async(self, symbol: str, timeframe: str, count: int) -> List[Dict]:
        """Async get recent candles"""
        async with self.lock:
            return self.buffer.get_recent_candles(symbol, timeframe, count)
    
    async def get_ohlcv_arrays_async(self, symbol: str, timeframe: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Async get OHLCV arrays"""
        async with self.lock:
            return self.buffer.get_ohlcv_arrays(symbol, timeframe)
    
    async def get_buffer_stats_async(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Async get buffer stats"""
        async with self.lock:
            return self.buffer.get_buffer_stats(symbol, timeframe)
    
    async def get_global_stats_async(self) -> Dict[str, Any]:
        """Async get global stats"""
        async with self.lock:
            return self.buffer.get_global_stats()
    
    async def clear_buffer_async(self, symbol: str, timeframe: str) -> bool:
        """Async clear buffer"""
        async with self.lock:
            return self.buffer.clear_buffer(symbol, timeframe)
    
    async def clear_all_buffers_async(self):
        """Async clear all buffers"""
        async with self.lock:
            self.buffer.clear_all_buffers()

