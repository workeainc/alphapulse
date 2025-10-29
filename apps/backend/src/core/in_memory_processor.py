"""
In-Memory Processing Layer for AlphaPlus
Ultra-low latency processing with shared ring buffers and vectorized operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

logger = logging.getLogger(__name__)

@dataclass
class InMemoryCandle:
    """In-memory candle structure optimized for speed"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool = False
    indicators: Dict[str, float] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}

@dataclass
class RingBuffer:
    """Thread-safe ring buffer for ultra-fast data access"""
    max_size: int
    data: deque
    lock: threading.Lock
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def append(self, item: Any):
        """Thread-safe append"""
        with self.lock:
            self.data.append(item)
    
    def get_latest(self, count: int = None) -> List[Any]:
        """Get latest items thread-safely"""
        with self.lock:
            if count is None:
                return list(self.data)
            return list(self.data)[-count:]
    
    def get_slice(self, start_idx: int, end_idx: int) -> List[Any]:
        """Get slice of data thread-safely"""
        with self.lock:
            data_list = list(self.data)
            return data_list[start_idx:end_idx]

class InMemoryProcessor:
    """
    Ultra-low latency in-memory processor with shared ring buffers
    Processes data in RAM before any database writes
    """
    
    def __init__(self, max_buffer_size: int = 1000, max_workers: int = 4):
        self.max_buffer_size = max_buffer_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Shared ring buffers per symbol/timeframe
        self.candle_buffers: Dict[str, RingBuffer] = {}
        self.indicator_buffers: Dict[str, RingBuffer] = {}
        self.signal_buffers: Dict[str, RingBuffer] = {}
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'avg_processing_time_ms': 0.0,
            'buffer_hits': 0,
            'buffer_misses': 0,
            'memory_usage_mb': 0.0
        }
        
        # Thread safety
        self.buffer_locks: Dict[str, threading.Lock] = {}
        
        logger.info(f"In-Memory Processor initialized with {max_workers} workers")
    
    def get_buffer_key(self, symbol: str, timeframe: str) -> str:
        """Generate buffer key for symbol/timeframe combination"""
        return f"{symbol}_{timeframe}"
    
    def ensure_buffer_exists(self, symbol: str, timeframe: str) -> str:
        """Ensure ring buffer exists for symbol/timeframe"""
        buffer_key = self.get_buffer_key(symbol, timeframe)
        
        if buffer_key not in self.candle_buffers:
            with threading.Lock():
                if buffer_key not in self.candle_buffers:
                    self.candle_buffers[buffer_key] = RingBuffer(self.max_buffer_size)
                    self.indicator_buffers[buffer_key] = RingBuffer(self.max_buffer_size)
                    self.signal_buffers[buffer_key] = RingBuffer(self.max_buffer_size)
                    self.buffer_locks[buffer_key] = threading.Lock()
        
        return buffer_key
    
    async def process_candle_in_memory(self, 
                                     symbol: str, 
                                     timeframe: str, 
                                     candle_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process candle data entirely in memory for ultra-low latency
        Returns processed result without any database writes
        """
        start_time = time.time()
        
        try:
            # Ensure buffer exists
            buffer_key = self.ensure_buffer_exists(symbol, timeframe)
            
            # Create in-memory candle
            candle = InMemoryCandle(
                timestamp=datetime.fromtimestamp(candle_data['timestamp'] / 1000, tz=timezone.utc),
                open=float(candle_data['open']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                close=float(candle_data['close']),
                volume=float(candle_data['volume']),
                is_complete=candle_data.get('is_complete', True)
            )
            
            # Add to ring buffer
            self.candle_buffers[buffer_key].append(candle)
            
            # Calculate indicators in memory (vectorized)
            indicators = await self._calculate_indicators_vectorized(buffer_key)
            
            # Update candle with indicators
            candle.indicators = indicators
            
            # Add to indicator buffer
            self.indicator_buffers[buffer_key].append(indicators)
            
            # Generate signals in memory
            signals = await self._generate_signals_in_memory(buffer_key, candle)
            
            # Add to signal buffer
            if signals:
                self.signal_buffers[buffer_key].append(signals)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time)
            
            # Return processed result
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'candle': candle,
                'indicators': indicators,
                'signals': signals,
                'processing_time_ms': processing_time,
                'buffer_size': len(self.candle_buffers[buffer_key].data)
            }
            
            logger.debug(f"⚡ In-memory processing: {symbol} {timeframe} in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in in-memory processing: {e}")
            return None
    
    async def _calculate_indicators_vectorized(self, buffer_key: str) -> Dict[str, float]:
        """Calculate indicators using vectorized operations"""
        try:
            # Get latest candles
            candles = self.candle_buffers[buffer_key].get_latest(100)
            
            if len(candles) < 20:
                return {}
            
            # Convert to numpy arrays for vectorized operations
            closes = np.array([c.close for c in candles])
            highs = np.array([c.high for c in candles])
            lows = np.array([c.low for c in candles])
            volumes = np.array([c.volume for c in candles])
            
            # Vectorized indicator calculations
            indicators = {}
            
            # RSI (14-period)
            if len(closes) >= 14:
                delta = np.diff(closes)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))
                else:
                    indicators['rsi'] = 100
            
            # SMA (20-period)
            if len(closes) >= 20:
                indicators['sma_20'] = np.mean(closes[-20:])
            
            # ATR (14-period)
            if len(closes) >= 14:
                tr = np.maximum(
                    highs - lows,
                    np.maximum(
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1))
                    )
                )
                indicators['atr'] = np.mean(tr[-14:])
            
            # Volume SMA
            if len(volumes) >= 20:
                indicators['volume_sma'] = np.mean(volumes[-20:])
                indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    async def _generate_signals_in_memory(self, buffer_key: str, candle: InMemoryCandle) -> Optional[Dict[str, Any]]:
        """Generate trading signals entirely in memory"""
        try:
            # Get latest indicators
            indicators = self.indicator_buffers[buffer_key].get_latest(10)
            
            if not indicators:
                return None
            
            latest_indicators = indicators[-1]
            
            # Simple signal generation logic (can be enhanced)
            signals = {}
            
            # RSI signals
            if 'rsi' in latest_indicators:
                rsi = latest_indicators['rsi']
                if rsi < 30:
                    signals['rsi_oversold'] = {'direction': 'buy', 'confidence': 0.7}
                elif rsi > 70:
                    signals['rsi_overbought'] = {'direction': 'sell', 'confidence': 0.7}
            
            # Moving average signals
            if 'sma_20' in latest_indicators and len(indicators) > 1:
                prev_sma = indicators[-2].get('sma_20', 0)
                curr_sma = latest_indicators['sma_20']
                
                if candle.close > curr_sma and curr_sma > prev_sma:
                    signals['ma_uptrend'] = {'direction': 'buy', 'confidence': 0.6}
                elif candle.close < curr_sma and curr_sma < prev_sma:
                    signals['ma_downtrend'] = {'direction': 'sell', 'confidence': 0.6}
            
            return signals if signals else None
            
        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            return None
    
    def _update_stats(self, processing_time_ms: float):
        """Update performance statistics"""
        self.stats['total_processed'] += 1
        self.stats['avg_processing_time_ms'] = (
            (self.stats['avg_processing_time_ms'] * (self.stats['total_processed'] - 1) + processing_time_ms) /
            self.stats['total_processed']
        )
        
        # Update memory usage
        process = psutil.Process()
        self.stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        stats = {}
        
        for buffer_key in self.candle_buffers:
            symbol, timeframe = buffer_key.split('_', 1)
            stats[buffer_key] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'candle_count': len(self.candle_buffers[buffer_key].data),
                'indicator_count': len(self.indicator_buffers[buffer_key].data),
                'signal_count': len(self.signal_buffers[buffer_key].data)
            }
        
        return stats
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old data from buffers"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        for buffer_key in self.candle_buffers:
            candles = self.candle_buffers[buffer_key].get_latest()
            
            # Remove old candles
            with self.buffer_locks[buffer_key]:
                while candles and candles[0].timestamp < cutoff_time:
                    self.candle_buffers[buffer_key].data.popleft()
                    candles = candles[1:]
        
        logger.info(f"🧹 Cleaned up data older than {max_age_hours} hours")
    
    def shutdown(self):
        """Shutdown the processor"""
        self.executor.shutdown(wait=True)
        logger.info("In-Memory Processor shutdown complete")
