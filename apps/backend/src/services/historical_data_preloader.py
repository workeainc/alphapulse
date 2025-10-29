#!/usr/bin/env python3
"""
Historical Data Preloader for AlphaPlus
Preloads historical data for algorithms before WebSocket starts
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json
import time
import ccxt

logger = logging.getLogger(__name__)

@dataclass
class PreloadConfig:
    """Configuration for historical data preloading"""
    symbols: List[str]
    timeframes: List[str]
    lookback_days: int = 7
    min_candles: int = 200
    batch_size: int = 1000
    rate_limit_delay: float = 0.1
    enable_ccxt: bool = True
    enable_database_cache: bool = True

@dataclass
class PreloadResult:
    """Result of historical data preloading"""
    symbol: str
    timeframe: str
    candles_loaded: int
    data_quality: float
    preload_time: float
    success: bool
    error_message: Optional[str] = None

class HistoricalDataPreloader:
    """Preloads historical data for algorithm initialization"""
    
    def __init__(self, db_url: str = None, config: PreloadConfig = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.config = config or PreloadConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT'],
            timeframes=['1m', '5m', '15m', '1h'],
            lookback_days=7,
            min_candles=200
        )
        self.logger = logger
        self.db_pool = None
        self.exchange = None
        
        # Statistics
        self.stats = {
            'total_preloads': 0,
            'successful_preloads': 0,
            'failed_preloads': 0,
            'total_candles_loaded': 0,
            'avg_preload_time': 0.0,
            'last_preload': None
        }
        
        logger.info("🔧 Historical Data Preloader initialized")
    
    async def initialize(self):
        """Initialize database connection and exchange"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10
            )
            self.logger.info("✅ Database connection pool created")
            
            # Initialize exchange for data fetching
            if self.config.enable_ccxt:
                try:
                    # Use async version of binance exchange
                    import ccxt.async_support as ccxt_async
                    self.exchange = ccxt_async.binance({
                        'sandbox': False,
                        'enableRateLimit': True,
                        'rateLimit': 1200,  # 1200ms between requests
                        'timeout': 30000,   # 30 second timeout
                        'options': {
                            'defaultType': 'future'  # Use futures API
                        }
                    })
                    
                    # Test connection with a simple call
                    await self.exchange.load_markets()
                    self.logger.info("✅ Binance exchange initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ CCXT not available, using database only: {e}")
                    self.exchange = None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize preloader: {e}")
            raise
    
    async def close(self):
        """Close database connection pool and exchange"""
        try:
            if self.db_pool:
                await self.db_pool.close()
                self.logger.info("🔌 Database connection pool closed")
        except Exception as e:
            self.logger.warning(f"⚠️ Error closing database pool: {e}")
        
        try:
            if self.exchange:
                await self.exchange.close()
                self.logger.info("🔌 Exchange connection closed")
        except Exception as e:
            self.logger.warning(f"⚠️ Error closing exchange: {e}")
    
    async def preload_all_symbols(self) -> Dict[str, List[PreloadResult]]:
        """Preload historical data for all configured symbols and timeframes"""
        start_time = time.time()
        results = {}
        
        try:
            self.logger.info(f"🚀 Starting historical data preload for {len(self.config.symbols)} symbols")
            
            for symbol in self.config.symbols:
                symbol_results = []
                
                for timeframe in self.config.timeframes:
                    try:
                        result = await self.preload_symbol_timeframe(symbol, timeframe)
                        symbol_results.append(result)
                        
                        if result.success:
                            self.stats['successful_preloads'] += 1
                            self.stats['total_candles_loaded'] += result.candles_loaded
                        else:
                            self.stats['failed_preloads'] += 1
                        
                        self.stats['total_preloads'] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(self.config.rate_limit_delay)
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error preloading {symbol} {timeframe}: {e}")
                        symbol_results.append(PreloadResult(
                            symbol=symbol,
                            timeframe=timeframe,
                            candles_loaded=0,
                            data_quality=0.0,
                            preload_time=0.0,
                            success=False,
                            error_message=str(e)
                        ))
                        self.stats['failed_preloads'] += 1
                        self.stats['total_preloads'] += 1
                
                results[symbol] = symbol_results
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats['avg_preload_time'] = total_time / self.stats['total_preloads'] if self.stats['total_preloads'] > 0 else 0
            self.stats['last_preload'] = datetime.now()
            
            self.logger.info(f"✅ Historical data preload completed in {total_time:.2f}s")
            self.logger.info(f"📊 Loaded {self.stats['total_candles_loaded']} candles across {self.stats['successful_preloads']} successful preloads")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in preload_all_symbols: {e}")
            raise
    
    async def preload_symbol_timeframe(self, symbol: str, timeframe: str) -> PreloadResult:
        """Preload historical data for a specific symbol and timeframe"""
        start_time = time.time()
        
        try:
            self.logger.info(f"📈 Preloading {symbol} {timeframe}...")
            
            # Check if we have enough data in database
            existing_candles = await self._get_existing_candles_count(symbol, timeframe)
            
            if existing_candles >= self.config.min_candles:
                self.logger.info(f"✅ Sufficient data exists in database: {existing_candles} candles")
                return PreloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles_loaded=existing_candles,
                    data_quality=1.0,
                    preload_time=time.time() - start_time,
                    success=True
                )
            
            # Fetch additional data from exchange
            needed_candles = self.config.min_candles - existing_candles
            new_candles = await self._fetch_from_exchange(symbol, timeframe, needed_candles)
            
            if new_candles:
                # Store new candles in database
                stored_count = await self._store_candles_in_database(symbol, timeframe, new_candles)
                
                # Calculate data quality
                data_quality = await self._calculate_data_quality(symbol, timeframe)
                
                preload_time = time.time() - start_time
                
                self.logger.info(f"✅ Preloaded {symbol} {timeframe}: {stored_count} new candles")
                
                return PreloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles_loaded=existing_candles + stored_count,
                    data_quality=data_quality,
                    preload_time=preload_time,
                    success=True
                )
            else:
                return PreloadResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles_loaded=existing_candles,
                    data_quality=0.5,
                    preload_time=time.time() - start_time,
                    success=False,
                    error_message="Failed to fetch data from exchange"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error preloading {symbol} {timeframe}: {e}")
            return PreloadResult(
                symbol=symbol,
                timeframe=timeframe,
                candles_loaded=0,
                data_quality=0.0,
                preload_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def _get_existing_candles_count(self, symbol: str, timeframe: str) -> int:
        """Get count of existing candles in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT COUNT(*) as count
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '%s days'
                """ % self.config.lookback_days
                
                result = await conn.fetchval(query, symbol, timeframe)
                return result or 0
                
        except Exception as e:
            self.logger.error(f"❌ Error getting existing candles count: {e}")
            return 0
    
    async def _fetch_from_exchange(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Fetch historical data from exchange"""
        if not self.exchange:
            self.logger.warning("⚠️ No exchange available for fetching data")
            return []
        
        try:
            # Calculate start time based on timeframe and limit
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            start_time = datetime.now() - timedelta(minutes=timeframe_minutes * limit)
            
            # Fetch klines from exchange with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    klines = await self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=int(start_time.timestamp() * 1000),
                        limit=limit
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Exchange fetch attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
            
            # Convert to our format
            candles = []
            for kline in klines:
                candle = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'source': 'historical_preload'
                }
                candles.append(candle)
            
            self.logger.info(f"📊 Fetched {len(candles)} candles from exchange for {symbol} {timeframe}")
            return candles
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching from exchange: {e}")
            return []
    
    async def _store_candles_in_database(self, symbol: str, timeframe: str, candles: List[Dict]) -> int:
        """Store candles in database"""
        if not candles:
            return 0
        
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for candle in candles:
                    values.append((
                        candle['symbol'],
                        candle['timeframe'],
                        candle['timestamp'],
                        candle['open'],
                        candle['high'],
                        candle['low'],
                        candle['close'],
                        candle['volume'],
                        candle['source']
                    ))
                
                # Batch insert without conflict resolution (temporary fix)
                query = """
                    INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                
                try:
                    result = await conn.executemany(query, values)
                    stored_count = len(candles)
                    self.logger.info(f"💾 Stored {stored_count} candles in database for {symbol} {timeframe}")
                except asyncpg.UniqueViolationError:
                    # Handle duplicate entries gracefully
                    self.logger.warning(f"⚠️ Duplicate candles detected for {symbol} {timeframe}, skipping duplicates")
                    stored_count = 0
                except Exception as e:
                    self.logger.error(f"❌ Error storing candles: {e}")
                    stored_count = 0
                return stored_count
                
        except Exception as e:
            self.logger.error(f"❌ Error storing candles in database: {e}")
            return 0
    
    async def _calculate_data_quality(self, symbol: str, timeframe: str) -> float:
        """Calculate data quality score"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_candles,
                        COUNT(CASE WHEN volume > 0 THEN 1 END) as valid_volume,
                        COUNT(CASE WHEN high >= low AND high >= open AND high >= close AND low <= open AND low <= close THEN 1 END) as valid_ohlc
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '%s days'
                """ % self.config.lookback_days
                
                result = await conn.fetchrow(query, symbol, timeframe)
                
                if result['total_candles'] == 0:
                    return 0.0
                
                volume_quality = result['valid_volume'] / result['total_candles']
                ohlc_quality = result['valid_ohlc'] / result['total_candles']
                
                return (volume_quality + ohlc_quality) / 2
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating data quality: {e}")
            return 0.0
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 60)
    
    async def get_preload_status(self) -> Dict[str, Any]:
        """Get current preload status and statistics"""
        return {
            'stats': self.stats,
            'config': {
                'symbols': self.config.symbols,
                'timeframes': self.config.timeframes,
                'lookback_days': self.config.lookback_days,
                'min_candles': self.config.min_candles
            },
            'exchange_available': self.exchange is not None,
            'database_connected': self.db_pool is not None
        }

# Example usage and testing
async def main():
    """Test the historical data preloader"""
    preloader = HistoricalDataPreloader()
    
    try:
        await preloader.initialize()
        
        # Preload data for all symbols
        results = await preloader.preload_all_symbols()
        
        # Print results
        for symbol, symbol_results in results.items():
            print(f"\n📊 {symbol} Results:")
            for result in symbol_results:
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.timeframe}: {result.candles_loaded} candles, quality: {result.data_quality:.2f}")
        
        # Print statistics
        status = await preloader.get_preload_status()
        print(f"\n📈 Preload Statistics:")
        print(f"  Total preloads: {status['stats']['total_preloads']}")
        print(f"  Successful: {status['stats']['successful_preloads']}")
        print(f"  Failed: {status['stats']['failed_preloads']}")
        print(f"  Total candles: {status['stats']['total_candles_loaded']}")
        
    finally:
        await preloader.close()

if __name__ == "__main__":
    asyncio.run(main())
