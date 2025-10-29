#!/usr/bin/env python3
"""
Download 1 Year of Historical Binance Data for AlphaPulse
Downloads OHLCV data for 5 symbols with multiple timeframes
Stores in existing ohlcv_data table with proper conflict handling
"""

import asyncio
import logging
import asyncpg
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import ccxt.async_support as ccxt_async
import time
import sys

# Configure logging (Windows-compatible, no emojis)
import sys
# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('historical_download.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Database configuration (matching main.py)
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

# Symbols to download (matching main.py)
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

# Timeframes to download (matching main.py MTF aggregation)
TIMEFRAMES = ['1m', '5m', '15m', '1h']  # Starting with all timeframes

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
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

class HistoricalDataDownloader:
    """Download 1 year of historical data from Binance"""
    
    def __init__(self):
        self.db_pool = None
        self.exchange = None
        self.stats = {
            'total_candles': 0,
            'total_stored': 0,
            'total_skipped': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    async def initialize(self):
        """Initialize database and exchange connections"""
        try:
            # Initialize database
            self.db_pool = await asyncpg.create_pool(
                **DB_CONFIG,
                min_size=2,
                max_size=10
            )
            logger.info("[OK] Database connection pool created")
            
            # Verify table exists and has unique constraint
            async with self.db_pool.acquire() as conn:
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'ohlcv_data'
                    )
                """)
                
                if not table_exists:
                    raise Exception("ohlcv_data table does not exist! Run schema verification first.")
                
                logger.info("[OK] Verified ohlcv_data table exists")
            
            # Initialize Binance exchange
            self.exchange = ccxt_async.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'rateLimit': 500,  # 500ms = 120 requests per minute (under Binance limit)
                'timeout': 60000,  # Increased to 60 seconds
                'options': {
                    'defaultType': 'spot'  # Use spot market
                }
            })
            
            logger.info("Loading Binance markets...")
            try:
                await asyncio.wait_for(self.exchange.load_markets(), timeout=30.0)
                logger.info("[OK] Binance exchange initialized")
            except asyncio.TimeoutError:
                logger.error("[ERROR] Timeout loading Binance markets. Check internet connection.")
                raise
            except Exception as e:
                logger.error(f"[ERROR] Error initializing Binance exchange: {e}")
                raise
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize: {e}")
            raise
    
    async def close(self):
        """Close connections"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Database connection pool closed")
        if self.exchange:
            await self.exchange.close()
            logger.info("🔌 Exchange connection closed")
    
    async def fetch_candles_chunked(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict]:
        """Fetch candles in chunks (Binance limit: 1000 per request)"""
        all_candles = []
        current_start = start_date
        chunk_size = 1000  # Binance maximum per request
        
        logger.info(f"  📊 Fetching {symbol} {timeframe} from {start_date.date()} to {end_date.date()}")
        
        attempt = 0
        max_attempts = 3
        
        while current_start < end_date:
            try:
                attempt += 1
                
                # Calculate how many candles to fetch in this chunk
                minutes_between = (end_date - current_start).total_seconds() / 60
                timeframe_minutes = TIMEFRAME_MINUTES[timeframe]
                candles_needed = int(minutes_between / timeframe_minutes)
                fetch_limit = min(chunk_size, candles_needed)
                
                # Fetch chunk
                since_ms = int(current_start.timestamp() * 1000)
                
                try:
                    logger.debug(f"    Fetching chunk: {symbol} {timeframe} from {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)}")
                    
                    # Add timeout to API call
                    klines = await asyncio.wait_for(
                        self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=since_ms,
                            limit=fetch_limit
                        ),
                        timeout=30.0  # 30 second timeout per API call
                    )
                    
                    if not klines:
                        logger.warning(f"    ⚠️ Empty response for {symbol} {timeframe}")
                        break
                        
                except asyncio.TimeoutError:
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt
                        logger.warning(f"    ⚠️ Timeout (attempt {attempt}/{max_attempts}), retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"    ❌ Timeout after {max_attempts} attempts")
                        raise
                except Exception as api_error:
                    if attempt < max_attempts:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"    ⚠️ API error (attempt {attempt}/{max_attempts}), retrying in {wait_time}s: {api_error}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"    ❌ API error after {max_attempts} attempts: {api_error}")
                        raise
                
                if not klines:
                    logger.warning(f"    ⚠️ No data returned for {symbol} {timeframe} at {current_start}")
                    break
                
                # Convert to our format
                chunk_candles = []
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
                        'source': 'historical_1year'
                    }
                    chunk_candles.append(candle)
                
                all_candles.extend(chunk_candles)
                
                # Update start time for next chunk (use last candle timestamp + 1 interval)
                if chunk_candles:
                    last_timestamp = chunk_candles[-1]['timestamp']
                    current_start = last_timestamp + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
                else:
                    break
                
                # Reset attempt counter on success
                attempt = 0
                
                # Rate limiting (be nice to Binance API)
                await asyncio.sleep(0.5)  # 500ms between requests
                
                # Log progress every 5000 candles
                if len(all_candles) % 5000 == 0 and len(all_candles) > 0:
                    logger.info(f"    📈 Progress: {len(all_candles):,} candles downloaded...")
                
            except Exception as e:
                logger.error(f"    ❌ Error fetching chunk for {symbol} {timeframe}: {e}")
                self.stats['errors'].append(f"{symbol} {timeframe}: {str(e)}")
                break
        
        logger.info(f"  ✅ Downloaded {len(all_candles):,} candles for {symbol} {timeframe}")
        return all_candles
    
    async def store_candles(self, candles: List[Dict]) -> tuple[int, int]:
        """Store candles in database with conflict handling
        
        Returns:
            (stored_count, skipped_count)
        """
        if not candles:
            return (0, 0)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get count before insert
                symbol = candles[0]['symbol']
                timeframe = candles[0]['timeframe']
                count_before = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                
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
                
                # Use ON CONFLICT to handle duplicates gracefully
                # Try column list first (works with unique index)
                query = """
                    INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """
                
                await conn.executemany(query, values)
                
                # Get count after insert
                count_after = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                
                stored_count = (count_after or 0) - (count_before or 0)
                skipped_count = len(candles) - stored_count
                
                logger.debug(f"💾 Attempted: {len(candles):,}, Stored: {stored_count:,}, Skipped: {skipped_count:,}")
                
        except Exception as e:
            logger.error(f"❌ Error storing candles: {e}")
            raise
        
        return (stored_count, skipped_count)
    
    async def check_existing_data(self, symbol: str, timeframe: str) -> int:
        """Check how many candles already exist"""
        try:
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = $2
                """, symbol, timeframe)
                return count or 0
        except Exception as e:
            logger.error(f"❌ Error checking existing data: {e}")
            return 0
    
    async def download_symbol_timeframe(
        self, 
        symbol: str, 
        timeframe: str, 
        days: int = 365
    ) -> Dict:
        """Download historical data for one symbol/timeframe combination"""
        start_time = time.time()
        
        try:
            # Check existing data
            existing = await self.check_existing_data(symbol, timeframe)
            if existing > 0:
                logger.info(f"📊 {symbol} {timeframe}: {existing:,} candles already exist")
            
            # Calculate date range (1 year ago to now)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Fetch all candles
            candles = await self.fetch_candles_chunked(symbol, timeframe, start_date, end_date)
            
            if not candles:
                logger.warning(f"⚠️ No candles fetched for {symbol} {timeframe}")
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'success': False,
                    'candles_downloaded': 0,
                    'candles_stored': 0,
                    'elapsed_seconds': time.time() - start_time
                }
            
            # Store in database
            stored, skipped = await self.store_candles(candles)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ {symbol} {timeframe}: Downloaded {len(candles):,}, Stored {stored:,} in {elapsed:.1f}s")
            
            self.stats['total_candles'] += len(candles)
            self.stats['total_stored'] += stored
            self.stats['total_skipped'] += skipped
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'candles_downloaded': len(candles),
                'candles_stored': stored,
                'candles_skipped': skipped,
                'elapsed_seconds': elapsed
            }
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol} {timeframe}: {e}")
            self.stats['errors'].append(f"{symbol} {timeframe}: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': str(e),
                'elapsed_seconds': time.time() - start_time
            }
    
    async def download_all(self, symbols: List[str], timeframes: List[str], days: int = 365):
        """Download historical data for all symbols and timeframes"""
        self.stats['start_time'] = time.time()
        
        results = []
        
        logger.info("=" * 80)
        logger.info("🚀 STARTING 1-YEAR HISTORICAL DATA DOWNLOAD")
        logger.info("=" * 80)
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Timeframes: {', '.join(timeframes)}")
        logger.info(f"Period: {days} days (1 year)")
        logger.info("=" * 80)
        
        total_tasks = len(symbols) * len(timeframes)
        completed = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                completed += 1
                logger.info(f"\n[{completed}/{total_tasks}] Processing {symbol} {timeframe}...")
                
                result = await self.download_symbol_timeframe(symbol, timeframe, days)
                results.append(result)
                
                # Small delay between symbol/timeframe combinations
                await asyncio.sleep(1)
        
        # Summary
        self.stats['end_time'] = time.time()
        total_time = self.stats['end_time'] - self.stats['start_time']
        successful = sum(1 for r in results if r.get('success'))
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Successful: {successful}/{total_tasks}")
        logger.info(f"📈 Total Candles Downloaded: {self.stats['total_candles']:,}")
        logger.info(f"💾 Total Candles Stored: {self.stats['total_stored']:,}")
        logger.info(f"⏭️  Total Candles Skipped (duplicates): {self.stats['total_skipped']:,}")
        logger.info(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        if self.stats['errors']:
            logger.info(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                logger.info(f"  - {error}")
        
        logger.info("=" * 80)
        
        return results

async def main():
    """Main function"""
    downloader = HistoricalDataDownloader()
    
    try:
        # Initialize
        await downloader.initialize()
        
        # Download all data
        results = await downloader.download_all(SYMBOLS, TIMEFRAMES, days=365)
        
        # Print detailed results
        logger.info("\n📋 DETAILED RESULTS:")
        for r in results:
            if r.get('success'):
                logger.info(
                    f"  ✅ {r['symbol']} {r['timeframe']}: "
                    f"Downloaded {r['candles_downloaded']:,}, "
                    f"Stored {r['candles_stored']:,} "
                    f"({r['elapsed_seconds']:.1f}s)"
                )
            else:
                logger.info(
                    f"  ❌ {r['symbol']} {r['timeframe']}: "
                    f"Failed - {r.get('error', 'Unknown error')}"
                )
        
        logger.info("\n✅ Download complete! Historical data is now available for indicator calculations.")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Download interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())

