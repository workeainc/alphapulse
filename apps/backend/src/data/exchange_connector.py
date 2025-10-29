#!/usr/bin/env python3
"""
Exchange Data Connector for AlphaPulse
Handles data fetching from various exchanges using public APIs
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CandlestickData:
    """Standardized candlestick data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_base: Optional[float] = None
    taker_buy_quote: Optional[float] = None

class ExchangeConnector:
    """Base class for exchange data connectors"""
    
    def __init__(self, exchange_name: str, api_key: str = None, secret: str = None):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.secret = secret
        self.session = None
        self.base_url = ""
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> List[CandlestickData]:
        """Fetch OHLCV data from exchange"""
        raise NotImplementedError("Subclasses must implement fetch_ohlcv")
    
    async def stream_ohlcv(self, symbol: str, timeframe: str):
        """Stream real-time OHLCV data"""
        raise NotImplementedError("Subclasses must implement stream_ohlcv")
    
    def normalize_ohlcv(self, raw_data: List) -> List[CandlestickData]:
        """Normalize raw OHLCV data to standard format"""
        raise NotImplementedError("Subclasses must implement normalize_ohlcv")

class BinancePublicAPI(ExchangeConnector):
    """Binance public API connector for candlestick data"""
    
    def __init__(self):
        super().__init__("binance")
        self.base_url = "https://data-api.binance.vision"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers={
            'User-Agent': 'AlphaPulse/1.0'
        })
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[CandlestickData]:
        """
        Fetch historical candlestick data from Binance public API
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Maximum number of candles to fetch
            start_time: Start time for data range
            end_time: End time for data range
        
        Returns:
            List of CandlestickData objects
        """
        endpoint = f"{self.base_url}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': timeframe,
            'limit': min(limit, 1000)  # Binance max is 1000
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Convert to standardized format
                return self.normalize_ohlcv(data)
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching data: {e}")
            return []
    
    async def fetch_recent_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100
    ) -> List[CandlestickData]:
        """Fetch recent candlestick data"""
        return await self.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    async def fetch_exchange_info(self) -> Dict:
        """Fetch exchange information including symbols and limits"""
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"
        
        try:
            async with self.session.get(endpoint) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching exchange info: {e}")
            return {}
    
    async def get_symbols_with_candlestick_data(self) -> List[str]:
        """Get list of symbols that have candlestick data available"""
        exchange_info = await self.fetch_exchange_info()
        
        if not exchange_info or 'symbols' not in exchange_info:
            return []
        
        # Filter for symbols with TRADING status
        trading_symbols = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['status'] == 'TRADING'
        ]
        
        return trading_symbols
    
    def normalize_ohlcv(self, raw_data: List) -> List[CandlestickData]:
        """Normalize raw Binance OHLCV data to standard format"""
        normalized_data = []
        
        for candle in raw_data:
            try:
                # Binance klines format: [open_time, open, high, low, close, volume, close_time, quote_volume, trades_count, taker_buy_base, taker_buy_quote, ignore]
                normalized_candle = CandlestickData(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    quote_volume=float(candle[7]) if len(candle) > 7 else None,
                    trades_count=int(candle[8]) if len(candle) > 8 else None,
                    taker_buy_base=float(candle[9]) if len(candle) > 9 else None,
                    taker_buy_quote=float(candle[10]) if len(candle) > 10 else None
                )
                normalized_data.append(normalized_candle)
                
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error normalizing candle data: {e}, data: {candle}")
                continue
        
        return normalized_data
    
    def to_dataframe(self, candlesticks: List[CandlestickData]) -> pd.DataFrame:
        """Convert list of CandlestickData to pandas DataFrame"""
        if not candlesticks:
            return pd.DataFrame()
        
        data = []
        for candle in candlesticks:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'quote_volume': candle.quote_volume,
                'trades_count': candle.trades_count,
                'taker_buy_base': candle.taker_buy_base,
                'taker_buy_quote': candle.taker_buy_quote
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

class CoinbasePublicAPI(ExchangeConnector):
    """Coinbase public API connector for candlestick data"""
    
    def __init__(self):
        super().__init__("coinbase")
        self.base_url = "https://api.exchange.coinbase.com"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers={
            'User-Agent': 'AlphaPulse/1.0'
        })
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[CandlestickData]:
        """
        Fetch historical candlestick data from Coinbase public API
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Maximum number of candles to fetch
            start_time: Start time for data range
            end_time: End time for data range
        
        Returns:
            List of CandlestickData objects
        """
        # Convert timeframe to Coinbase format
        timeframe_map = {
            '1m': '60',
            '5m': '300',
            '15m': '900',
            '1h': '3600',
            '4h': '14400',
            '1d': '86400'
        }
        
        granularity = timeframe_map.get(timeframe, '3600')
        endpoint = f"{self.base_url}/products/{symbol}/candles"
        
        params = {
            'granularity': granularity
        }
        
        if start_time:
            params['start'] = start_time.isoformat()
        if end_time:
            params['end'] = end_time.isoformat()
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Limit results if needed
                if limit and len(data) > limit:
                    data = data[-limit:]
                
                # Convert to standardized format
                return self.normalize_ohlcv(data)
                
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching Coinbase data: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching Coinbase data: {e}")
            return []
    
    def normalize_ohlcv(self, raw_data: List) -> List[CandlestickData]:
        """Normalize raw Coinbase OHLCV data to standard format"""
        normalized_data = []
        
        for candle in raw_data:
            try:
                # Coinbase candles format: [timestamp, open, high, low, close, volume]
                normalized_candle = CandlestickData(
                    timestamp=datetime.fromtimestamp(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                )
                normalized_data.append(normalized_candle)
                
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error normalizing Coinbase candle data: {e}, data: {candle}")
                continue
        
        return normalized_data

# Factory function for creating exchange connectors
def create_exchange_connector(exchange_name: str, **kwargs) -> ExchangeConnector:
    """Create an exchange connector instance"""
    exchange_map = {
        'binance': BinancePublicAPI,
        'coinbase': CoinbasePublicAPI
    }
    
    connector_class = exchange_map.get(exchange_name.lower())
    if not connector_class:
        raise ValueError(f"Unsupported exchange: {exchange_name}")
    
    return connector_class(**kwargs)

# Example usage and testing
async def test_exchange_connector():
    """Test the exchange connector functionality"""
    async with BinancePublicAPI() as api:
        # Test fetching recent data
        candlesticks = await api.fetch_recent_ohlcv('BTCUSDT', '1h', limit=10)
        print(f"Fetched {len(candlesticks)} candlesticks")
        
        if candlesticks:
            # Convert to DataFrame
            df = api.to_dataframe(candlesticks)
            print("\nDataFrame:")
            print(df.head())
            
            # Test exchange info
            symbols = await api.get_symbols_with_candlestick_data()
            print(f"\nAvailable symbols: {len(symbols)}")
            print(f"First 10 symbols: {symbols[:10]}")

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_exchange_connector())
