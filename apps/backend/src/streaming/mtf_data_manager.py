"""
Multi-Timeframe Data Manager
Efficiently manages candle data across all timeframes
Aggregates 1m candles to 5m, 15m, 1h, 4h, 1d automatically
"""

import logging
from typing import Dict, List, Callable, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class MTFDataManager:
    """
    Streams and aggregates candle data across multiple timeframes
    Base: 1m candles, aggregates to all higher timeframes
    """
    
    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        
        # Timeframes to manage
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Buffers: {symbol_tf: deque(candles)}
        self.buffers = {}
        
        # Candle counters for aggregation: {symbol: count}
        self.candle_counters = {}
        
        # Callbacks: {timeframe: [callbacks]}
        self.callbacks = {tf: [] for tf in self.timeframes}
        
        logger.info(f"MTF Data Manager initialized with {len(self.timeframes)} timeframes")
    
    def register_callback(self, timeframe: str, callback: Callable):
        """Register callback for timeframe candle completion"""
        if timeframe in self.callbacks:
            self.callbacks[timeframe].append(callback)
    
    async def on_1m_candle(self, symbol: str, candle: Dict):
        """
        Process new 1m candle
        Aggregates to higher timeframes automatically
        """
        
        # Initialize if first candle
        if symbol not in self.candle_counters:
            self.candle_counters[symbol] = 0
            for tf in self.timeframes:
                self.buffers[f"{symbol}_{tf}"] = deque(maxlen=self.buffer_size)
        
        # Store 1m candle
        self.buffers[f"{symbol}_1m"].append(candle)
        self.candle_counters[symbol] += 1
        
        count = self.candle_counters[symbol]
        
        # Trigger 1m callbacks
        await self._trigger_callbacks('1m', symbol, candle)
        
        # === AGGREGATE TO HIGHER TIMEFRAMES ===
        
        # Every 5 candles → 5m candle
        if count % 5 == 0:
            candle_5m = self._aggregate_candles(symbol, '1m', 5)
            if candle_5m:
                self.buffers[f"{symbol}_5m"].append(candle_5m)
                await self._trigger_callbacks('5m', symbol, candle_5m)
        
        # Every 15 candles → 15m candle
        if count % 15 == 0:
            candle_15m = self._aggregate_candles(symbol, '1m', 15)
            if candle_15m:
                self.buffers[f"{symbol}_15m"].append(candle_15m)
                await self._trigger_callbacks('15m', symbol, candle_15m)
        
        # Every 60 candles → 1h candle
        if count % 60 == 0:
            candle_1h = self._aggregate_candles(symbol, '1m', 60)
            if candle_1h:
                self.buffers[f"{symbol}_1h"].append(candle_1h)
                await self._trigger_callbacks('1h', symbol, candle_1h)
        
        # Every 240 candles → 4h candle
        if count % 240 == 0:
            candle_4h = self._aggregate_candles(symbol, '1m', 240)
            if candle_4h:
                self.buffers[f"{symbol}_4h"].append(candle_4h)
                await self._trigger_callbacks('4h', symbol, candle_4h)
        
        # Every 1440 candles → 1d candle
        if count % 1440 == 0:
            candle_1d = self._aggregate_candles(symbol, '1m', 1440)
            if candle_1d:
                self.buffers[f"{symbol}_1d"].append(candle_1d)
                await self._trigger_callbacks('1d', symbol, candle_1d)
    
    def _aggregate_candles(self, symbol: str, base_tf: str, count: int) -> Optional[Dict]:
        """Aggregate N candles into one higher timeframe candle"""
        
        buffer_key = f"{symbol}_{base_tf}"
        
        if buffer_key not in self.buffers:
            return None
        
        # Get last N candles
        candles = list(self.buffers[buffer_key])[-count:]
        
        if len(candles) < count:
            return None  # Not enough candles yet
        
        # Aggregate OHLCV
        aggregated = {
            'timestamp': candles[0]['timestamp'],  # Start time
            'open': candles[0]['open'],
            'high': max(c['high'] for c in candles),
            'low': min(c['low'] for c in candles),
            'close': candles[-1]['close'],
            'volume': sum(c['volume'] for c in candles),
            'symbol': symbol
        }
        
        return aggregated
    
    async def _trigger_callbacks(self, timeframe: str, symbol: str, candle: Dict):
        """Trigger all registered callbacks for this timeframe"""
        
        if timeframe not in self.callbacks:
            return
        
        for callback in self.callbacks[timeframe]:
            try:
                await callback(symbol, timeframe, candle)
            except Exception as e:
                logger.error(f"Error in callback for {timeframe}: {e}")
    
    def get_buffer(self, symbol: str, timeframe: str) -> List[Dict]:
        """Get candle buffer for symbol/timeframe"""
        buffer_key = f"{symbol}_{timeframe}"
        return list(self.buffers.get(buffer_key, []))
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get latest candle for symbol/timeframe"""
        buffer = self.get_buffer(symbol, timeframe)
        return buffer[-1] if buffer else None

