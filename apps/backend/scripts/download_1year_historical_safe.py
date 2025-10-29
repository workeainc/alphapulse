#!/usr/bin/env python3
"""
Windows-safe version of historical data downloader
No emojis, better error handling, progress tracking
"""

import asyncio
import logging
import asyncpg
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import ccxt.async_support as ccxt_async
import time
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('historical_download.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
TIMEFRAMES = ['1m', '5m', '15m', '1h']

TIMEFRAME_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
    '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720, '1d': 1440
}

class SafeDownloader:
    def __init__(self):
        self.db_pool = None
        self.exchange = None
        self.stats = {'total_candles': 0, 'total_stored': 0, 'errors': []}
    
    async def initialize(self):
        try:
            logger.info("Connecting to database...")
            self.db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10)
            logger.info("[OK] Database connected")
            
            logger.info("Initializing Binance API (this may take 10-30 seconds)...")
            logger.info("  Please wait, loading market data...")
            
            self.exchange = ccxt_async.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'rateLimit': 500,
                'timeout': 60000,
                'options': {'defaultType': 'spot'}
            })
            
            # Add progress indicator
            try:
                logger.info("  Connecting to Binance...")
                markets_task = asyncio.create_task(self.exchange.load_markets())
                
                # Wait with progress updates
                for i in range(30):  # Max 30 seconds
                    await asyncio.sleep(1)
                    if markets_task.done():
                        break
                    if i % 5 == 0 and i > 0:
                        logger.info(f"  Still loading... ({i}s)")
                
                if not markets_task.done():
                    logger.warning("  Taking longer than expected, continuing...")
                    await asyncio.wait_for(markets_task, timeout=10.0)
                
                logger.info("[OK] Binance API initialized")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load Binance markets: {e}")
                logger.error("  Check your internet connection and try again")
                raise
                
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise
    
    async def close(self):
        if self.db_pool:
            await self.db_pool.close()
        if self.exchange:
            await self.exchange.close()
    
    async def download_symbol_timeframe(self, symbol, timeframe, days=365):
        start_time = time.time()
        try:
            logger.info(f"\n[{symbol} {timeframe}] Starting download...")
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            all_candles = []
            current_start = start_date
            chunk_count = 0
            
            while current_start < end_date:
                chunk_count += 1
                if chunk_count % 10 == 0:
                    logger.info(f"  [{symbol} {timeframe}] Downloaded {len(all_candles):,} candles so far...")
                
                minutes_between = (end_date - current_start).total_seconds() / 60
                timeframe_minutes = TIMEFRAME_MINUTES[timeframe]
                candles_needed = int(minutes_between / timeframe_minutes)
                fetch_limit = min(1000, candles_needed)
                
                try:
                    since_ms = int(current_start.timestamp() * 1000)
                    klines = await asyncio.wait_for(
                        self.exchange.fetch_ohlcv(symbol, timeframe, since_ms, fetch_limit),
                        timeout=30.0
                    )
                    
                    if not klines:
                        break
                    
                    for kline in klines:
                        all_candles.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'timestamp': datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'source': 'historical_1year'
                        })
                    
                    if klines:
                        last_timestamp = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc)
                        current_start = last_timestamp + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
                    else:
                        break
                    
                    await asyncio.sleep(0.5)  # Rate limit
                    
                except Exception as e:
                    logger.error(f"  [ERROR] Chunk {chunk_count} failed: {e}")
                    break
            
            # Store in database
            if all_candles:
                stored = await self.store_candles(all_candles)
                elapsed = time.time() - start_time
                logger.info(f"[OK] {symbol} {timeframe}] {len(all_candles):,)} downloaded, {stored:,} stored in {elapsed:.1f}s")
                self.stats['total_candles'] += len(all_candles)
                self.stats['total_stored'] += stored
                return {'success': True, 'count': stored}
            else:
                return {'success': False, 'error': 'No data'}
                
        except Exception as e:
            logger.error(f"[ERROR] {symbol} {timeframe}: {e}")
            self.stats['errors'].append(f"{symbol} {timeframe}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def store_candles(self, candles):
        if not candles:
            return 0
        
        try:
            async with self.db_pool.acquire() as conn:
                values = [
                    (c['symbol'], c['timeframe'], c['timestamp'], 
                     c['open'], c['high'], c['low'], c['close'], 
                     c['volume'], c['source'])
                    for c in candles
                ]
                
                query = """
                    INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """
                
                await conn.executemany(query, values)
                return len(candles)  # Approximate
        except Exception as e:
            logger.error(f"[ERROR] Storage failed: {e}")
            return 0
    
    async def download_all(self):
        logger.info("=" * 70)
        logger.info("HISTORICAL DATA DOWNLOAD - STARTING")
        logger.info("=" * 70)
        logger.info(f"Symbols: {', '.join(SYMBOLS)}")
        logger.info(f"Timeframes: {', '.join(TIMEFRAMES)}")
        logger.info("Estimated time: ~25 minutes")
        logger.info("=" * 70)
        
        start_time = time.time()
        results = []
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                result = await self.download_symbol_timeframe(symbol, timeframe)
                results.append(result)
                await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success'))
        
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Total stored: {self.stats['total_stored']:,}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        if self.stats['errors']:
            logger.info(f"Errors: {len(self.stats['errors'])}")
        
        return results

async def main():
    downloader = SafeDownloader()
    try:
        await downloader.initialize()
        await downloader.download_all()
    except KeyboardInterrupt:
        logger.warning("\n[WARN] Interrupted by user")
    except Exception as e:
        logger.error(f"\n[ERROR] Fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())

