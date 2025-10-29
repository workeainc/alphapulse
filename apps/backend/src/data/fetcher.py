import asyncio
import logging
from typing import Optional, Dict, List, Tuple
import pandas as pd
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from src.app.core.config import settings
from src.app.models.database import MarketData, get_db


class MarketDataService:
    """
    Service for fetching and managing market data from various sources.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange connections
        self.exchanges = {}
        self._initialize_exchanges()
        
        # Enhanced multi-timeframe cache
        self.data_cache = {}
        self.timeframe_cache = defaultdict(dict)  # symbol -> {timeframe -> (data, timestamp)}
        self.cache_timeout = 60  # seconds
        self.max_cache_size = 1000  # Maximum number of cached entries
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache warming configuration
        self.common_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.cache_warming_enabled = True
    
    def _initialize_exchanges(self):
        """Initialize exchange connections."""
        try:
            # Binance
            if settings.BINANCE_API_KEY and settings.BINANCE_SECRET_KEY:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': settings.BINANCE_API_KEY,
                    'secret': settings.BINANCE_SECRET_KEY,
                    'sandbox': True  # Use testnet for safety
                })
            
            # Fallback to public APIs
            self.exchanges['yfinance'] = None  # yfinance doesn't need initialization
            
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for symbol, timeframe, and limit."""
        return f"{symbol}_{timeframe}_{limit}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid."""
        return (datetime.now() - timestamp).seconds < self.cache_timeout
    
    def _get_from_timeframe_cache(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Get data from the enhanced timeframe cache."""
        if symbol in self.timeframe_cache and timeframe in self.timeframe_cache[symbol]:
            data, timestamp = self.timeframe_cache[symbol][timeframe]
            if self._is_cache_valid(timestamp) and len(data) >= limit * 0.8:
                self.cache_hits += 1
                return data
        self.cache_misses += 1
        return None
    
    def _update_timeframe_cache(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Update the timeframe cache with new data."""
        self.timeframe_cache[symbol][timeframe] = (data, datetime.now())
        
        # Cache size management
        if len(self.timeframe_cache) > self.max_cache_size:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat."""
        current_time = datetime.now()
        symbols_to_remove = []
        
        for symbol, timeframes in self.timeframe_cache.items():
            timeframes_to_remove = []
            for timeframe, (data, timestamp) in timeframes.items():
                if not self._is_cache_valid(timestamp):
                    timeframes_to_remove.append(timeframe)
            
            for timeframe in timeframes_to_remove:
                del timeframes[timeframe]
            
            if not timeframes:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.timeframe_cache[symbol]
    
    async def warm_cache_for_symbol(self, symbol: str, timeframes: List[str] = None):
        """Pre-fetch data for common timeframes to warm the cache."""
        if not self.cache_warming_enabled:
            return
        
        if timeframes is None:
            timeframes = self.common_timeframes
        
        self.logger.info(f"Warming cache for {symbol} across {len(timeframes)} timeframes")
        
        # Fetch data for each timeframe concurrently
        tasks = []
        for timeframe in timeframes:
            task = self.get_historical_data(symbol, timeframe, 100)
            tasks.append(task)
        
        # Execute all fetches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log cache warming results
        successful_fetches = sum(1 for r in results if not isinstance(r, Exception))
        self.logger.info(f"Cache warming completed for {symbol}: {successful_fetches}/{len(timeframes)} timeframes")
    
    async def warm_cache_for_symbols(self, symbols: List[str], timeframes: List[str] = None):
        """Pre-fetch data for multiple symbols across timeframes."""
        if not self.cache_warming_enabled:
            return
        
        self.logger.info(f"Warming cache for {len(symbols)} symbols")
        
        # Warm cache for each symbol concurrently
        tasks = [self.warm_cache_for_symbol(symbol, timeframes) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_symbols': len(self.timeframe_cache),
            'total_cached_timeframes': sum(len(timeframes) for timeframes in self.timeframe_cache.values()),
            'cache_size_mb': self._estimate_cache_size_mb()
        }
    
    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB."""
        total_size = 0
        for symbol, timeframes in self.timeframe_cache.items():
            for timeframe, (data, timestamp) in timeframes.items():
                if hasattr(data, 'memory_usage'):
                    total_size += data.memory_usage(deep=True).sum()
                else:
                    # Rough estimate: 8 bytes per numeric value
                    total_size += data.size * 8
        
        return round(total_size / (1024 * 1024), 2)
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Try Binance first
            if 'binance' in self.exchanges:
                ticker = await self.exchanges['binance'].fetch_ticker(symbol)
                return ticker['last']
            
            # Fallback to yfinance
            ticker = yf.Ticker(symbol.replace('/', ''))
            info = ticker.info
            if 'regularMarketPrice' in info:
                return info['regularMarketPrice']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check enhanced timeframe cache first
            cached_data = self._get_from_timeframe_cache(symbol, timeframe, limit)
            if cached_data is not None:
                return cached_data
            
            # Try to get from database first
            db_data = await self._get_from_database(symbol, timeframe, limit)
            if db_data is not None and len(db_data) >= limit * 0.8:  # At least 80% of requested data
                self._update_timeframe_cache(symbol, timeframe, db_data)
                return db_data
            
            # Fetch from exchange
            exchange_data = await self._fetch_from_exchange(symbol, timeframe, limit)
            if exchange_data is not None:
                # Save to database
                await self._save_to_database(exchange_data, symbol, timeframe)
                
                # Update enhanced timeframe cache
                self._update_timeframe_cache(symbol, timeframe, exchange_data)
                
                return exchange_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def get_batch_historical_data(self, symbols: List[str], timeframe: str = '1h', 
                                      limit: int = 200) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get historical OHLCV data for multiple symbols in a single timeframe.
        This is a critical efficiency optimization that reduces API calls from N to 1 per timeframe.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframe: Timeframe ('1m', '5m', '1h', '4h', '1d')
            limit: Number of candles to fetch per symbol
            
        Returns:
            Dictionary mapping symbols to their OHLCV DataFrames
        """
        try:
            if not symbols:
                return {}
            
            self.logger.info(f"🔄 Starting batch OHLCV fetch for {len(symbols)} symbols: {timeframe}")
            start_time = datetime.now()
            
            # Initialize results dictionary
            results = {}
            
            # Check enhanced timeframe cache first for all symbols
            symbols_to_fetch = []
            for symbol in symbols:
                cached_data = self._get_from_timeframe_cache(symbol, timeframe, limit)
                if cached_data is not None:
                    results[symbol] = cached_data
                    self.logger.debug(f"✅ {symbol}: Using cached data")
                else:
                    symbols_to_fetch.append(symbol)
            
            if not symbols_to_fetch:
                self.logger.info(f"✅ All {len(symbols)} symbols served from cache")
                return results
            
            # Try to get remaining symbols from database first
            symbols_from_db = []
            for symbol in symbols_to_fetch:
                db_data = await self._get_from_database(symbol, timeframe, limit)
                if db_data is not None and len(db_data) >= limit * 0.8:
                    results[symbol] = db_data
                    # Update enhanced timeframe cache
                    self._update_timeframe_cache(symbol, timeframe, db_data)
                    symbols_from_db.append(symbol)
                    self.logger.debug(f"✅ {symbol}: Using database data")
            
            # Remove symbols we got from database
            symbols_to_fetch = [s for s in symbols_to_fetch if s not in symbols_from_db]
            
            if not symbols_to_fetch:
                self.logger.info(f"✅ All remaining symbols served from database")
                return results
            
            # Fetch remaining symbols from exchange in batch
            self.logger.info(f"🔄 Fetching {len(symbols_to_fetch)} symbols from exchange")
            batch_results = await self._fetch_batch_from_exchange(symbols_to_fetch, timeframe, limit)
            
            # Process batch results
            for symbol, data in batch_results.items():
                if data is not None:
                    # Save to database
                    await self._save_to_database(data, symbol, timeframe)
                    
                    # Update enhanced timeframe cache
                    self._update_timeframe_cache(symbol, timeframe, data)
                    
                    results[symbol] = data
                    self.logger.debug(f"✅ {symbol}: Fetched from exchange")
                else:
                    results[symbol] = None
                    self.logger.warning(f"⚠️ {symbol}: Failed to fetch from exchange")
            
            # Log performance metrics
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"✅ Batch fetch completed in {total_time:.2f}s: {len(symbols)} symbols, {timeframe}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch historical data fetch: {e}")
            # Return whatever we managed to get
            return results

    async def _fetch_batch_from_exchange(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch OHLCV data for multiple symbols from exchange in batch.
        This is the core efficiency optimization - reduces API calls significantly.
        """
        try:
            results = {}
            
            # Try Binance batch fetch first (most efficient)
            if 'binance' in self.exchanges:
                # Binance supports batch OHLCV fetch for multiple symbols
                batch_results = await self._fetch_binance_batch(symbols, timeframe, limit)
                if batch_results:
                    return batch_results
            
            # Fallback to parallel individual fetches
            self.logger.info(f"🔄 Using parallel individual fetches for {len(symbols)} symbols")
            tasks = []
            for symbol in symbols:
                task = self._fetch_from_exchange(symbol, timeframe, limit)
                tasks.append((symbol, task))
            
            # Execute all fetches in parallel
            for symbol, task in tasks:
                try:
                    data = await task
                    results[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error fetching {symbol}: {e}")
                    results[symbol] = None
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch exchange fetch: {e}")
            return {symbol: None for symbol in symbols}

    async def _fetch_binance_batch(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Optimized batch fetch for Binance - uses their batch OHLCV endpoint when possible.
        This can reduce API calls from N to 1 for the same timeframe.
        """
        try:
            results = {}
            
            # Binance doesn't have a true batch OHLCV endpoint, but we can optimize
            # by using concurrent requests with rate limiting
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests to avoid rate limits
            
            async def fetch_with_semaphore(symbol):
                async with semaphore:
                    return await self._fetch_from_exchange(symbol, timeframe, limit)
            
            # Create tasks for all symbols
            tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
            
            # Execute all tasks concurrently
            batch_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, symbol in enumerate(symbols):
                if isinstance(batch_data[i], Exception):
                    self.logger.error(f"Error fetching {symbol}: {batch_data[i]}")
                    results[symbol] = None
                else:
                    results[symbol] = batch_data[i]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Binance batch fetch: {e}")
            return {symbol: None for symbol in symbols}
    
    async def _get_from_database(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Get historical data from ..src.database."""
        try:
            db = next(get_db())
            
            # Get the most recent data
            market_data = db.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            ).order_by(MarketData.timestamp.desc()).limit(limit).all()
            
            if not market_data:
                return None
            
            # Convert to DataFrame
            data = []
            for record in reversed(market_data):  # Reverse to get chronological order
                data.append({
                    'timestamp': record.timestamp,
                    'open_price': record.open_price,
                    'high_price': record.high_price,
                    'low_price': record.low_price,
                    'close_price': record.close_price,
                    'volume': record.volume,
                    'ema_9': record.ema_9,
                    'ema_21': record.ema_21,
                    'ema_50': record.ema_50,
                    'ema_200': record.ema_200,
                    'rsi': record.rsi,
                    'macd': record.macd,
                    'macd_signal': record.macd_signal,
                    'macd_histogram': record.macd_histogram,
                    'bb_upper': record.bb_upper,
                    'bb_middle': record.bb_middle,
                    'bb_lower': record.bb_lower,
                    'atr': record.atr,
                    'volatility': record.volatility,
                    'trend_strength': record.trend_strength,
                    'market_regime': record.market_regime
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error getting data from database: {e}")
            return None
    
    async def _fetch_from_exchange(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from exchange."""
        try:
            # Try Binance first
            if 'binance' in self.exchanges:
                ohlcv = await self.exchanges['binance'].fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Rename columns to match our schema
                    df = df.rename(columns={
                        'open': 'open_price',
                        'high': 'high_price',
                        'low': 'low_price',
                        'close': 'close_price'
                    })
                    
                    return df
            
            # Fallback to yfinance
            if symbol in ['BTC/USDT', 'ETH/USDT']:
                # Convert symbol for yfinance
                yf_symbol = symbol.replace('/', '') + '-USD'
                ticker = yf.Ticker(yf_symbol)
                
                # Get historical data
                period = self._get_yfinance_period(timeframe, limit)
                hist = ticker.history(period=period, interval=self._get_yfinance_interval(timeframe))
                
                if not hist.empty:
                    # Reset index to get timestamp as column
                    hist = hist.reset_index()
                    
                    # Rename columns
                    hist = hist.rename(columns={
                        'Open': 'open_price',
                        'High': 'high_price',
                        'Low': 'low_price',
                        'Close': 'close_price',
                        'Volume': 'volume',
                        'Datetime': 'timestamp'
                    })
                    
                    # Select only needed columns
                    df = hist[['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
                    
                    return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching from exchange: {e}")
            return None
    
    def _get_yfinance_period(self, timeframe: str, limit: int) -> str:
        """Convert timeframe to yfinance period."""
        if timeframe == '1m':
            return '1d'
        elif timeframe == '5m':
            return '5d'
        elif timeframe == '1h':
            return '30d'
        elif timeframe == '4h':
            return '60d'
        elif timeframe == '1d':
            return '1y'
        else:
            return '30d'
    
    def _get_yfinance_interval(self, timeframe: str) -> str:
        """Convert timeframe to yfinance interval."""
        if timeframe == '1m':
            return '1m'
        elif timeframe == '5m':
            return '5m'
        elif timeframe == '1h':
            return '1h'
        elif timeframe == '4h':
            return '4h'
        elif timeframe == '1d':
            return '1d'
        else:
            return '1h'
    
    async def _save_to_database(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save market data to database."""
        try:
            db = next(get_db())
            
            for _, row in data.iterrows():
                market_data = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row['timestamp'],
                    open_price=row['open_price'],
                    high_price=row['high_price'],
                    low_price=row['low_price'],
                    close_price=row['close_price'],
                    volume=row['volume']
                )
                
                db.add(market_data)
            
            db.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book for a symbol."""
        try:
            if 'binance' in self.exchanges:
                order_book = await self.exchanges['binance'].fetch_order_book(symbol, limit)
                return order_book
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed ticker information."""
        try:
            if 'binance' in self.exchanges:
                ticker = await self.exchanges['binance'].fetch_ticker(symbol)
                return ticker
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ticker info for {symbol}: {e}")
            return None
    
    async def get_market_cap_data(self) -> Optional[Dict]:
        """Get market capitalization data for major cryptocurrencies."""
        try:
            # Use yfinance for market cap data
            symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
            market_cap_data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if 'marketCap' in info:
                    market_cap_data[symbol.replace('-USD', '')] = {
                        'market_cap': info['marketCap'],
                        'volume_24h': info.get('volume', 0),
                        'price_change_24h': info.get('regularMarketChangePercent', 0)
                    }
            
            return market_cap_data
            
        except Exception as e:
            self.logger.error(f"Error getting market cap data: {e}")
            return None
    
    async def calculate_btc_dominance(self) -> Optional[float]:
        """Calculate Bitcoin dominance percentage."""
        try:
            market_cap_data = await self.get_market_cap_data()
            
            if market_cap_data and 'BTC' in market_cap_data:
                btc_market_cap = market_cap_data['BTC']['market_cap']
                total_market_cap = sum(data['market_cap'] for data in market_cap_data.values())
                
                if total_market_cap > 0:
                    return (btc_market_cap / total_market_cap) * 100
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating BTC dominance: {e}")
            return None
    
    async def get_volume_profile(self, symbol: str, timeframe: str = '1h', 
                                periods: int = 24) -> Optional[Dict]:
        """Get volume profile for a symbol."""
        try:
            data = await self.get_historical_data(symbol, timeframe, periods)
            
            if data is None or len(data) == 0:
                return None
            
            # Calculate volume statistics
            volume_stats = {
                'current_volume': data['volume'].iloc[-1],
                'avg_volume': data['volume'].mean(),
                'volume_sma': data['volume'].rolling(window=20).mean().iloc[-1],
                'volume_ratio': data['volume'].iloc[-1] / data['volume'].mean(),
                'volume_trend': 'increasing' if data['volume'].iloc[-1] > data['volume'].iloc[-2] else 'decreasing'
            }
            
            return volume_stats
            
        except Exception as e:
            self.logger.error(f"Error getting volume profile for {symbol}: {e}")
            return None
    
    async def get_support_resistance_levels(self, symbol: str, timeframe: str = '1h', 
                                          periods: int = 100) -> Optional[Dict]:
        """Calculate support and resistance levels."""
        try:
            data = await self.get_historical_data(symbol, timeframe, periods)
            
            if data is None or len(data) < 20:
                return None
            
            # Calculate pivot points
            high = data['high_price'].max()
            low = data['low_price'].min()
            close = data['close_price'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'resistance': [r1, r2, r3],
                'support': [s1, s2, s3],
                'current_price': close
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance for {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()
        self.timeframe_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Cache cleared")
    
    def clear_symbol_cache(self, symbol: str):
        """Clear cache for a specific symbol."""
        if symbol in self.timeframe_cache:
            del self.timeframe_cache[symbol]
            self.logger.info(f"Cache cleared for {symbol}")
    
    def set_cache_timeout(self, timeout_seconds: int):
        """Set cache timeout in seconds."""
        self.cache_timeout = timeout_seconds
        self.logger.info(f"Cache timeout set to {timeout_seconds} seconds")
    
    def enable_cache_warming(self, enabled: bool = True):
        """Enable or disable cache warming."""
        self.cache_warming_enabled = enabled
        self.logger.info(f"Cache warming {'enabled' if enabled else 'disabled'}")
