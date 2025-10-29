"""
Live Market Data Connector
Connects to Binance WebSocket for real-time price data
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Callable, Optional
import websockets

logger = logging.getLogger(__name__)

class LiveMarketConnector:
    """Real-time market data connector for Binance"""
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        self.ws = None
        self.is_running = False
        self.callbacks = []
        self.current_prices = {}
        
        # Binance WebSocket URL
        self.ws_url = "wss://stream.binance.com:9443/ws"
        
    def add_callback(self, callback: Callable):
        """Add callback for new candle data"""
        self.callbacks.append(callback)
    
    async def start(self):
        """Start WebSocket connection"""
        self.is_running = True
        
        # Create stream names for all symbol/timeframe combinations
        streams = []
        for symbol in self.symbols:
            # Kline (candlestick) streams
            for tf in self.timeframes:
                stream_name = f"{symbol.lower()}@kline_{tf}"
                streams.append(stream_name)
            
            # Mini ticker for current price
            streams.append(f"{symbol.lower()}@miniTicker")
        
        # Combine streams - Binance uses /stream?streams= for multiple streams
        stream_str = "/".join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={stream_str}"
        
        logger.info(f"📡 WebSocket URL: {url[:80]}...")
        
        logger.info(f"Connecting to Binance WebSocket for {len(self.symbols)} symbols...")
        
        while self.is_running:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info(f"Connected to Binance! Streaming {len(streams)} data feeds")
                    
                    while self.is_running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # DEBUG: Log every message type
                        stream = data.get('stream', 'NO_STREAM')
                        if '@kline_' in stream:
                            event_data = data.get('data', {})
                            kline = event_data.get('k', {})
                            is_closed = kline.get('x', False)
                            logger.info(f"📩 Received kline: {event_data.get('s', 'UNKNOWN')} closed={is_closed}")
                        
                        await self._process_message(data)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)  # Wait before reconnect
                    logger.info("Reconnecting...")
    
    async def _process_message(self, data: Dict):
        """Process incoming WebSocket message"""
        try:
            stream = data.get('stream', '')
            event_data = data.get('data', {})
            
            # Handle kline (candlestick) data
            if '@kline_' in stream:
                kline = event_data.get('k', {})
                
                # DEBUG: Log all kline events
                symbol = event_data.get('s', 'UNKNOWN')
                is_closed = kline.get('x', False)
                logger.debug(f"📊 Kline event: {symbol} closed={is_closed}")
                
                if kline.get('x'):  # Candle closed
                    symbol = event_data['s']
                    timeframe = kline['i']
                    
                    logger.info(f"🕯️ Candle closed: {symbol} {timeframe} @ {kline['t']}")
                    
                    candle_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'is_closed': True
                    }
                    
                    # Trigger all callbacks
                    logger.info(f"🔔 Triggering {len(self.callbacks)} callbacks for {symbol} {timeframe}")
                    for callback in self.callbacks:
                        try:
                            await callback(candle_data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
            
            # Handle mini ticker (current price)
            elif '@miniTicker' in stream:
                symbol = event_data.get('s')
                price = float(event_data.get('c', 0))
                
                if symbol and price > 0:
                    self.current_prices[symbol] = price
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        return self.current_prices.get(symbol)
    
    async def stop(self):
        """Stop WebSocket connection"""
        self.is_running = False
        logger.info("Market connector stopped")

