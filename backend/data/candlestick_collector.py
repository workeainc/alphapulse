#!/usr/bin/env python3
"""
Candlestick & Technical Indicator Data Collector
AlphaPulse Trading Bot - Real-time OHLCV collection with indicator calculation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .websocket_client import BinanceWebSocketClient
from .exchange_connector import ExchangeConnector
from .storage import DataStorage
# from ..strategies.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class CandlestickData:
    """Candlestick data with technical indicators"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    indicators: Dict
    patterns: List[str]
    market_cap: Optional[float] = None
    btc_dominance: Optional[float] = None

class CandlestickCollector:
    """
    Collects candlestick data from multiple sources and calculates technical indicators
    on ingestion for real-time analysis
    """
    
    def __init__(self, config: Dict = None):
        """Initialize candlestick collector"""
        self.config = config or {}
        
        # Initialize components
        self.websocket_client = BinanceWebSocketClient()
        self.exchange_connector = ExchangeConnector("binance")  # Default to Binance
        self.storage = DataStorage(self.config.get('storage_path', 'data'))
        # self.indicators_calc = TechnicalIndicators()
        
        # Data storage
        self.candlestick_buffer = {}  # symbol -> timeframe -> buffer
        self.indicators_buffer = {}   # symbol -> timeframe -> indicators
        
        # Configuration
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.indicator_periods = self.config.get('indicator_periods', {
            'ema': [9, 21, 50, 200],
            'sma': [20, 50, 100, 200],
            'rsi': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': 20,
            'atr': 14
        })
        
        # Performance tracking
        self.stats = {
            'candlesticks_collected': 0,
            'indicators_calculated': 0,
            'storage_operations': 0,
            'last_update': None
        }
        
        # Callbacks
        self.data_callbacks = []
        self.indicator_callbacks = []
        
        logger.info("🚀 Candlestick Collector initialized")
    
    async def start_collection(self):
        """Start collecting candlestick data"""
        try:
            # Connect to WebSocket
            await self.websocket_client.connect()
            
            # Subscribe to all symbols and timeframes
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    await self.websocket_client.subscribe_candlesticks(
                        symbol, timeframe, self._handle_candlestick
                    )
            
            # Start WebSocket listener
            asyncio.create_task(self.websocket_client.listen())
            
            # Start historical data collection
            await self._collect_historical_data()
            
            logger.info("✅ Candlestick collection started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start candlestick collection: {e}")
            raise
    
    async def stop_collection(self):
        """Stop collecting candlestick data"""
        try:
            await self.websocket_client.disconnect()
            logger.info("🛑 Candlestick collection stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping collection: {e}")
    
    async def _handle_candlestick(self, candlestick_data: Dict):
        """Handle incoming WebSocket candlestick data"""
        try:
            # Parse candlestick data
            parsed_data = self._parse_websocket_candlestick(candlestick_data)
            if not parsed_data:
                return
            
            symbol = parsed_data['symbol']
            timeframe = parsed_data['timeframe']
            
            # Store in buffer
            self._store_candlestick(symbol, timeframe, parsed_data)
            
            # Calculate indicators on ingestion
            indicators = await self._calculate_indicators(symbol, timeframe)
            
            # Create complete candlestick data
            complete_data = CandlestickData(
                symbol=symbol,
                timestamp=parsed_data['timestamp'],
                open=parsed_data['open'],
                high=parsed_data['high'],
                low=parsed_data['low'],
                close=parsed_data['close'],
                volume=parsed_data['volume'],
                timeframe=timeframe,
                indicators=indicators,
                patterns=[]  # Will be filled by pattern detector
            )
            
            # Store in database
            await self._store_candlestick_data(complete_data)
            
            # Notify callbacks
            await self._notify_data_callbacks(complete_data)
            await self._notify_indicator_callbacks(complete_data)
            
            # Update stats
            self._update_stats()
            
        except Exception as e:
            logger.error(f"❌ Error handling candlestick: {e}")
    
    def _parse_websocket_candlestick(self, data: Dict) -> Optional[Dict]:
        """Parse WebSocket candlestick data"""
        try:
            if 'k' not in data:
                return None
            
            k = data['k']
            return {
                'symbol': data['s'],
                'timestamp': datetime.fromtimestamp(k['t'] / 1000),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'timeframe': k['i'],
                'is_closed': k['x']
            }
        except Exception as e:
            logger.error(f"❌ Error parsing candlestick: {e}")
            return None
    
    def _store_candlestick(self, symbol: str, timeframe: str, data: Dict):
        """Store candlestick in buffer"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.candlestick_buffer:
            self.candlestick_buffer[key] = []
        
        self.candlestick_buffer[key].append(data)
        
        # Maintain buffer size
        if len(self.candlestick_buffer[key]) > self.buffer_size:
            self.candlestick_buffer[key] = self.candlestick_buffer[key][-self.buffer_size:]
    
    async def _calculate_indicators(self, symbol: str, timeframe: str) -> Dict:
        """Calculate technical indicators for the symbol/timeframe"""
        try:
            key = f"{symbol}_{timeframe}"
            if key not in self.candlestick_buffer:
                return {}
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.candlestick_buffer[key])
            if len(df) < 50:  # Need minimum data for indicators
                return {}
            
            # Calculate indicators
            indicators = {}
            
            # EMAs
            for period in self.indicator_periods['ema']:
                if len(df) >= period:
                    indicators[f'ema_{period}'] = self.indicators_calc.ema(df['close'], period).iloc[-1]
            
            # SMAs
            for period in self.indicator_periods['sma']:
                if len(df) >= period:
                    indicators[f'sma_{period}'] = self.indicators_calc.sma(df['close'], period).iloc[-1]
            
            # RSI
            if len(df) >= self.indicator_periods['rsi']:
                indicators['rsi'] = self.indicators_calc.rsi(df['close'], self.indicator_periods['rsi']).iloc[-1]
            
            # MACD
            macd_config = self.indicator_periods['macd']
            if len(df) >= max(macd_config['fast'], macd_config['slow']):
                macd_result = self.indicators_calc.macd(
                    df['close'], 
                    macd_config['fast'], 
                    macd_config['slow'], 
                    macd_config['signal']
                )
                indicators['macd'] = macd_result['macd'].iloc[-1]
                indicators['macd_signal'] = macd_result['signal'].iloc[-1]
                indicators['macd_histogram'] = macd_result['histogram'].iloc[-1]
            
            # Bollinger Bands
            if len(df) >= self.indicator_periods['bollinger']:
                bb_result = self.indicators_calc.bollinger_bands(
                    df['close'], 
                    self.indicator_periods['bollinger']
                )
                indicators['bb_upper'] = bb_result['upper'].iloc[-1]
                indicators['bb_middle'] = bb_result['middle'].iloc[-1]
                indicators['bb_lower'] = bb_result['lower'].iloc[-1]
            
            # ATR
            if len(df) >= self.indicator_periods['atr']:
                indicators['atr'] = self.indicators_calc.atr(
                    df, 
                    self.indicator_periods['atr']
                ).iloc[-1]
            
            self.stats['indicators_calculated'] += 1
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    async def _collect_historical_data(self):
        """Collect historical data for all symbols and timeframes"""
        try:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    logger.info(f"📊 Collecting historical data for {symbol} {timeframe}")
                    
                    # Get historical data from exchange
                    historical_data = await self.exchange_connector.get_historical_candlesticks(
                        symbol, timeframe, limit=1000
                    )
                    
                    if historical_data:
                        # Store historical data
                        for candle in historical_data:
                            self._store_candlestick(symbol, timeframe, candle)
                        
                        logger.info(f"✅ Collected {len(historical_data)} historical candles for {symbol} {timeframe}")
                    
        except Exception as e:
            logger.error(f"❌ Error collecting historical data: {e}")
    
    async def _store_candlestick_data(self, data: CandlestickData):
        """Store complete candlestick data in database"""
        try:
            # Store in TimescaleDB
            await self.storage.store_candlestick(
                symbol=data.symbol,
                timestamp=data.timestamp,
                open=data.open,
                high=data.high,
                low=data.low,
                close=data.close,
                volume=data.volume,
                timeframe=data.timeframe,
                indicators=data.indicators,
                patterns=data.patterns
            )
            
            self.stats['storage_operations'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error storing candlestick data: {e}")
    
    async def _notify_data_callbacks(self, data: CandlestickData):
        """Notify data callbacks"""
        for callback in self.data_callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"❌ Error in data callback: {e}")
    
    async def _notify_indicator_callbacks(self, data: CandlestickData):
        """Notify indicator callbacks"""
        for callback in self.indicator_callbacks:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"❌ Error in indicator callback: {e}")
    
    def _update_stats(self):
        """Update collection statistics"""
        self.stats['candlesticks_collected'] += 1
        self.stats['last_update'] = datetime.now()
    
    def add_data_callback(self, callback: Callable):
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def add_indicator_callback(self, callback: Callable):
        """Add indicator callback"""
        self.indicator_callbacks.append(callback)
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return self.stats.copy()
    
    def get_symbol_data(self, symbol: str, timeframe: str = None) -> Dict:
        """Get collected data for a symbol"""
        if timeframe:
            key = f"{symbol}_{timeframe}"
            return {
                'candlesticks': self.candlestick_buffer.get(key, []),
                'indicators': self.indicators_buffer.get(key, {})
            }
        else:
            result = {}
            for tf in self.timeframes:
                key = f"{symbol}_{tf}"
                result[tf] = {
                    'candlesticks': self.candlestick_buffer.get(key, []),
                    'indicators': self.indicators_buffer.get(key, {})
                }
            return result

async def test_candlestick_collector():
    """Test the candlestick collector"""
    config = {
        'symbols': ['BTCUSDT'],
        'timeframes': ['1m', '5m'],
        'buffer_size': 100,
        'storage_path': 'test_data'
    }
    
    collector = CandlestickCollector(config)
    
    try:
        await collector.start_collection()
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Get stats
        stats = collector.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Get data
        data = collector.get_symbol_data('BTCUSDT', '1m')
        print(f"BTCUSDT 1m data: {len(data['candlesticks'])} candles")
        
    finally:
        await collector.stop_collection()

if __name__ == "__main__":
    asyncio.run(test_candlestick_collector())
