#!/usr/bin/env python3
"""
Direct historical data downloader - skips load_markets() to avoid timeout
Uses direct API calls without loading full market list
"""

import asyncio
import logging
import asyncpg
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import aiohttp
import json
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
    '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
}

BINANCE_BASE_URL = 'https://api.binance.com/api/v3/klines'

class DirectDownloader:
    def __init__(self):
        self.db_pool = None
        self.session = None
        self.stats = {'total_candles': 0, 'total_stored': 0, 'errors': [], 'batches_stored': 0}
    
    async def initialize(self):
        try:
            logger.info("Connecting to database...")
            self.db_pool = await asyncpg.create_pool(**DB_CONFIG, min_size=2, max_size=10)
            logger.info("[OK] Database connected")
            
            logger.info("Testing Binance API connectivity...")
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(ssl=False)
            )
            
            # Test connection with a simple ping
            try:
                async with self.session.get('https://api.binance.com/api/v3/ping', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info("[OK] Binance API is reachable")
                    else:
                        logger.warning(f"[WARN] Binance API returned status {resp.status}")
            except Exception as e:
                logger.warning(f"[WARN] Could not ping Binance API: {e}")
                logger.info("  Attempting to continue anyway...")
                
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise
    
    async def close(self):
        if self.db_pool:
            await self.db_pool.close()
        if self.session:
            await self.session.close()
    
    async def fetch_klines_direct(self, symbol: str, interval: str, start_time: int, end_time: int = None, limit: int = 1000):
        """Fetch klines directly from Binance API without CCXT"""
        url = BINANCE_BASE_URL
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'limit': limit
        }
        
        if end_time:
            params['endTime'] = end_time
        
        try:
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    raise Exception(f"Binance API error {response.status}: {error_text}")
        except asyncio.TimeoutError:
            raise Exception(f"Timeout fetching {symbol} {interval}")
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    async def download_symbol_timeframe(self, symbol: str, timeframe: str, days: int = 365):
        start_time = time.time()
        all_candles = []
        
        try:
            logger.info(f"\n[{symbol} {timeframe}] Starting download...")
            
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            current_start = start_date
            
            chunk_count = 0
            max_chunks = 1000  # Safety limit
            consecutive_errors = 0
            
            while current_start < end_date and chunk_count < max_chunks:
                chunk_count += 1
                
                if chunk_count % 20 == 0:
                    logger.info(f"  [{symbol} {timeframe}] Progress: {len(all_candles):,} candles...")
                
                # Calculate fetch window
                minutes_between = (end_date - current_start).total_seconds() / 60
                timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
                candles_needed = int(minutes_between / timeframe_minutes)
                fetch_limit = min(1000, max(1, candles_needed))
                
                since_ms = int(current_start.timestamp() * 1000)
                
                try:
                    klines = await self.fetch_klines_direct(symbol, timeframe, since_ms, limit=fetch_limit)
                    
                    if not klines or len(klines) == 0:
                        logger.debug(f"  No more data for {symbol} {timeframe}")
                        break
                    
                    # Convert Binance format to our format
                    chunk_candles = []
                    for kline in klines:
                        chunk_candles.append({
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
                    
                    # Add new chunk to accumulator
                    all_candles.extend(chunk_candles)
                    
                    # Store in batches every 25,000 candles (smaller batches to avoid timeout)
                    if len(all_candles) >= 25000:
                        logger.info(f"  [{symbol} {timeframe}] BATCH STORAGE TRIGGERED at {len(all_candles):,} candles")
                        
                        # Calculate how many to store (keep recent chunk in memory)
                        chunk_size = len(chunk_candles)
                        batch_to_store_size = len(all_candles) - chunk_size
                        
                        if batch_to_store_size > 0:
                            batch_to_store = all_candles[:batch_to_store_size]
                            
                            try:
                                logger.info(f"  [{symbol} {timeframe}] Storing {len(batch_to_store):,} candles in database...")
                                stored = await self.store_candles(batch_to_store)
                                logger.info(f"  [{symbol} {timeframe}] Successfully stored {stored:,} candles in database")
                                self.stats['total_stored'] += stored
                                self.stats['batches_stored'] += 1
                                
                                # Keep only the most recent chunk(s) in memory
                                all_candles = all_candles[batch_to_store_size:]
                                
                            except Exception as storage_error:
                                logger.error(f"  [ERROR] Storage failed: {storage_error}")
                                import traceback
                                logger.error(traceback.format_exc())
                                logger.warning(f"  [{symbol} {timeframe}] Continuing download, will retry storage later...")
                                # Limit memory if storage keeps failing
                                if len(all_candles) > 100000:
                                    logger.warning(f"  [{symbol} {timeframe}] Memory limit reached, forcing storage of oldest 50k...")
                                    try:
                                        oldest_batch = all_candles[:50000]
                                        await self.store_candles(oldest_batch)
                                        all_candles = all_candles[50000:]
                                    except:
                                        # Last resort: clear some data
                                        all_candles = all_candles[-50000:]
                        else:
                            logger.debug(f"  [{symbol} {timeframe}] Batch size too small, waiting for more data")
                    
                    # Update current_start to next position
                    if klines:
                        last_timestamp_ms = klines[-1][0]
                        last_timestamp = datetime.fromtimestamp(last_timestamp_ms / 1000, tz=timezone.utc)
                        current_start = last_timestamp + timedelta(minutes=timeframe_minutes)
                    else:
                        break
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    consecutive_errors += 1
                    error_msg = str(e)
                    logger.warning(f"  [WARN] Chunk {chunk_count} failed (error #{consecutive_errors}): {error_msg}")
                    
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        logger.info("  Rate limit hit, waiting 10 seconds...")
                        await asyncio.sleep(10)
                        continue
                    elif "404" in error_msg or "Invalid symbol" in error_msg:
                        logger.error(f"  [ERROR] Invalid symbol or endpoint: {error_msg}")
                        break
                    elif "400" in error_msg:
                        logger.warning(f"  [WARN] Bad request (possibly reached data limit)")
                        # Check if this is likely the end of data
                        if len(all_candles) > 500000:  # We have substantial data already
                            logger.info(f"  [{symbol} {timeframe}] Likely reached end of available historical data")
                            break
                    elif consecutive_errors >= 5:  # After 5 consecutive errors, give up
                        logger.error(f"  [ERROR] Too many consecutive failures ({consecutive_errors}), stopping {symbol} {timeframe}")
                        logger.info(f"  [{symbol} {timeframe}] Downloaded {len(all_candles):,} candles before stopping")
                        break
                    
                    # Retry with exponential backoff
                    wait_time = min(15, 2 ** min(consecutive_errors, 4))
                    logger.info(f"  Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
            
            # Store remaining candles in database (final batch)
            if all_candles:
                logger.info(f"  [{symbol} {timeframe}] Storing final batch: {len(all_candles):,} candles...")
                try:
                    stored = await self.store_candles(all_candles)
                    logger.info(f"  [{symbol} {timeframe}] Successfully stored final batch: {stored:,} candles")
                except Exception as e:
                    logger.error(f"  [ERROR] Final storage failed: {e}")
                    stored = 0
                
                elapsed = time.time() - start_time
                logger.info(f"[OK] {symbol} {timeframe}: Completed in {elapsed/60:.1f} minutes")
                # Note: total_stored is tracked during batch storage
                return {'success': True, 'count': stored, 'total_in_session': len(all_candles)}
            else:
                logger.warning(f"[WARN] {symbol} {timeframe}: No data downloaded")
                return {'success': False, 'error': 'No data'}
                
        except Exception as e:
            logger.error(f"[ERROR] {symbol} {timeframe}: {e}")
            self.stats['errors'].append(f"{symbol} {timeframe}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    async def store_candles(self, candles):
        if not candles:
            return 0
        
        symbol = candles[0]['symbol']
        timeframe = candles[0]['timeframe']
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get count before
                count_before = await conn.fetchval(
                    "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = $1 AND timeframe = $2",
                    symbol, timeframe
                ) or 0
                
                # Use TEMPORARY TABLE approach for faster bulk inserts
                # This avoids TimescaleDB chunk checking overhead during bulk load
                logger.info(f"  [{symbol} {timeframe}] Using temp table for fast bulk insert of {len(candles):,} candles...")
                
                try:
                    # Create temporary table with same structure
                    await conn.execute("""
                        CREATE TEMPORARY TABLE temp_ohlcv_insert (
                            symbol TEXT,
                            timeframe TEXT,
                            timestamp TIMESTAMPTZ,
                            open NUMERIC,
                            high NUMERIC,
                            low NUMERIC,
                            close NUMERIC,
                            volume NUMERIC,
                            source TEXT
                        ) ON COMMIT DROP
                    """)
                    
                    # Bulk insert into temp table (fast, no constraint checking)
                    batch_size = 1000  # Larger batches for temp table (no constraint overhead)
                    total_inserted = 0
                    num_batches = (len(candles) + batch_size - 1) // batch_size
                    
                    for i in range(0, len(candles), batch_size):
                        batch = candles[i:i + batch_size]
                        batch_num = (i // batch_size) + 1
                        
                        if batch_num % 10 == 0 or batch_num == num_batches:
                            logger.info(f"  [{symbol} {timeframe}] Temp table insert: {batch_num}/{num_batches} batches ({i + len(batch):,}/{len(candles):,} candles)")
                        
                        values = [
                            (c['symbol'], c['timeframe'], c['timestamp'], 
                             c['open'], c['high'], c['low'], c['close'], 
                             c['volume'], c['source'])
                            for c in batch
                        ]
                        
                        try:
                            await conn.executemany("""
                                INSERT INTO temp_ohlcv_insert (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            """, values)
                            total_inserted += len(batch)
                        except Exception as e:
                            logger.warning(f"  [WARN] Temp insert batch {batch_num} failed: {e}")
                    
                    logger.info(f"  [{symbol} {timeframe}] Inserted {total_inserted:,} rows into temp table, now copying to main table...")
                    
                    # Copy from temp table to main table (single operation, faster conflict check)
                    await conn.execute("""
                        INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                        SELECT symbol, timeframe, timestamp, open, high, low, close, volume, source
                        FROM temp_ohlcv_insert
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """)
                    
                    # Get count after insert
                    count_after = await conn.fetchval(
                        "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = $1 AND timeframe = $2",
                        symbol, timeframe
                    ) or 0
                    
                    stored = count_after - count_before
                    logger.info(f"  [{symbol} {timeframe}] Copied {stored:,} new rows from temp table to ohlcv_data")
                    
                except Exception as temp_error:
                    logger.warning(f"  [WARN] Temp table method failed: {temp_error}, trying row-by-row inserts...")
                    
                    # Fallback: row-by-row inserts
                    stored = 0
                    for i, c in enumerate(candles):
                        if (i + 1) % 1000 == 0:
                            logger.info(f"  [{symbol} {timeframe}] Row-by-row progress: {i + 1:,}/{len(candles):,})")
                        
                        try:
                            await asyncio.wait_for(
                                conn.execute("""
                                    INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                                """, c['symbol'], c['timeframe'], c['timestamp'],
                                c['open'], c['high'], c['low'], c['close'], c['volume'], c['source']),
                                timeout=5.0
                            )
                            stored += 1
                        except:
                            pass
                    
                    # Recalculate stored count
                    count_after = await conn.fetchval(
                        "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = $1 AND timeframe = $2",
                        symbol, timeframe
                    ) or 0
                    stored = count_after - count_before
                
                # stored and count_after are already set in the try/except blocks above
                logger.info(f"  [{symbol} {timeframe}] Stored: {stored:,} new candles (attempted: {len(candles):,})")
                
                if stored < len(candles) * 0.8:  # If less than 80% stored, warn
                    logger.warning(f"  [WARN] Only {stored:,} of {len(candles):,} candles stored (may be duplicates or timeouts)")
                
                return stored
                
        except Exception as e:
            logger.error(f"[ERROR] Storage failed for {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Re-raise to trigger retry logic
    
    async def download_all(self):
        logger.info("=" * 70)
        logger.info("HISTORICAL DATA DOWNLOAD - DIRECT API METHOD")
        logger.info("=" * 70)
        logger.info(f"Symbols: {', '.join(SYMBOLS)}")
        logger.info(f"Timeframes: {', '.join(TIMEFRAMES)}")
        logger.info(f"Total tasks: {len(SYMBOLS) * len(TIMEFRAMES)}")
        logger.info("Estimated time: ~25 minutes")
        logger.info("=" * 70)
        
        start_time = time.time()
        results = []
        task_num = 0
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                task_num += 1
                logger.info(f"\n[TASK {task_num}/{len(SYMBOLS) * len(TIMEFRAMES)}] {symbol} {timeframe}")
                
                result = await self.download_symbol_timeframe(symbol, timeframe)
                results.append(result)
                
                # Small delay between tasks
                await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success'))
        
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Total candles downloaded: {self.stats['total_candles']:,}")
        logger.info(f"Total candles stored: {self.stats['total_stored']:,}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        
        if self.stats['errors']:
            logger.info(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10
                logger.info(f"  - {error}")
        
        logger.info("=" * 70)
        
        return results

async def main():
    downloader = DirectDownloader()
    try:
        await downloader.initialize()
        await downloader.download_all()
        logger.info("\n[SUCCESS] Download complete!")
    except KeyboardInterrupt:
        logger.warning("\n[WARN] Interrupted by user")
    except Exception as e:
        logger.error(f"\n[ERROR] Fatal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await downloader.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted")

