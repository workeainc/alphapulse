#!/usr/bin/env python3
"""
Startup Gap Detection & Backfill Service
Automatically detects and fills data gaps when system restarts
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import asyncpg
import ccxt
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GapInfo:
    """Information about detected gap"""
    symbol: str
    timeframe: str
    last_candle_time: datetime
    current_time: datetime
    gap_duration_minutes: int
    estimated_missing_candles: int

class StartupGapBackfillService:
    """
    Detects and fills gaps in OHLCV data on system startup
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        
        # Configuration
        self.base_timeframe = '1m'  # Only backfill 1m data
        self.max_gap_days = 30  # Don't backfill gaps older than 30 days (Binance limit)
        self.batch_size = 1000  # Binance limit per request
        self.rate_limit_delay = 0.5  # 500ms between requests (safe for 1200/min limit)
        
        # Statistics
        self.stats = {
            'gaps_detected': 0,
            'gaps_filled': 0,
            'candles_fetched': 0,
            'candles_stored': 0,
            'errors': 0,
            'symbols_processed': 0
        }
        
        logger.info("🔧 Startup Gap Backfill Service initialized")
    
    async def detect_and_fill_all_gaps(self, symbols: List[str]) -> Dict:
        """
        Main entry point: Detect and fill gaps for all symbols
        
        Args:
            symbols: List of symbols to check (e.g., ['BTCUSDT', 'ETHUSDT'])
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"🔍 Starting gap detection for {len(symbols)} symbols...")
        start_time = datetime.now()
        
        gaps = []
        
        # 1. Detect all gaps
        for symbol in symbols:
            gap = await self._detect_gap(symbol)
            if gap:
                gaps.append(gap)
                self.stats['gaps_detected'] += 1
        
        if not gaps:
            logger.info("✅ No gaps detected! System is up to date.")
            return self.stats
        
        # 2. Log detected gaps
        logger.warning(f"⚠️ Detected {len(gaps)} gaps:")
        for gap in gaps:
            logger.warning(
                f"   {gap.symbol}: {gap.gap_duration_minutes} minutes "
                f"({gap.estimated_missing_candles} candles) - "
                f"Last: {gap.last_candle_time.strftime('%Y-%m-%d %H:%M')}"
            )
        
        # 3. Fill all gaps
        for gap in gaps:
            try:
                await self._fill_gap(gap)
                self.stats['gaps_filled'] += 1
                self.stats['symbols_processed'] += 1
            except Exception as e:
                logger.error(f"❌ Error filling gap for {gap.symbol}: {e}")
                self.stats['errors'] += 1
        
        # 4. Report results
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 GAP BACKFILL COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Gaps Detected: {self.stats['gaps_detected']}")
        logger.info(f"Gaps Filled: {self.stats['gaps_filled']}")
        logger.info(f"Candles Fetched: {self.stats['candles_fetched']}")
        logger.info(f"Candles Stored: {self.stats['candles_stored']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*80}\n")
        
        return self.stats
    
    async def _detect_gap(self, symbol: str) -> Optional[GapInfo]:
        """
        Detect if there's a gap for a symbol
        
        Returns:
            GapInfo if gap exists, None if data is current
        """
        try:
            # Get the last candle from database
            query = """
                SELECT timestamp
                FROM ohlcv_data
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            result = await self.db_pool.fetchrow(
                query, 
                symbol, 
                self.base_timeframe
            )
            
            if not result:
                # No data exists - need initial load
                logger.info(f"📭 {symbol}: No data in database, need initial load")
                current_time = datetime.now(timezone.utc)
                # Get last 7 days of data for initial load
                initial_days = min(7, self.max_gap_days)
                last_candle_time = current_time - timedelta(days=initial_days)
                gap_minutes = initial_days * 24 * 60
                
                return GapInfo(
                    symbol=symbol,
                    timeframe=self.base_timeframe,
                    last_candle_time=last_candle_time,
                    current_time=current_time,
                    gap_duration_minutes=gap_minutes,
                    estimated_missing_candles=gap_minutes
                )
            
            last_candle_time = result['timestamp']
            current_time = datetime.now(timezone.utc)
            
            # Calculate gap
            gap_duration = current_time - last_candle_time
            gap_minutes = int(gap_duration.total_seconds() / 60)
            
            # If gap is more than 2 minutes (allowing 1 minute for current candle formation)
            if gap_minutes > 2:
                # Check if gap is within acceptable range
                if gap_minutes > self.max_gap_days * 24 * 60:
                    logger.warning(
                        f"⚠️ {symbol}: Gap is {gap_minutes/1440:.1f} days, "
                        f"only filling last {self.max_gap_days} days"
                    )
                    gap_minutes = self.max_gap_days * 24 * 60
                    last_candle_time = current_time - timedelta(minutes=gap_minutes)
                
                return GapInfo(
                    symbol=symbol,
                    timeframe=self.base_timeframe,
                    last_candle_time=last_candle_time,
                    current_time=current_time,
                    gap_duration_minutes=gap_minutes,
                    estimated_missing_candles=gap_minutes
                )
            
            # No significant gap
            return None
            
        except Exception as e:
            logger.error(f"❌ Error detecting gap for {symbol}: {e}")
            return None
    
    async def _fill_gap(self, gap: GapInfo):
        """
        Fill a detected gap by fetching from Binance and storing in DB
        """
        logger.info(
            f"🔄 Filling gap for {gap.symbol}: "
            f"{gap.estimated_missing_candles} candles, "
            f"{gap.gap_duration_minutes/60:.1f} hours"
        )
        
        # Calculate number of batches needed (Binance limit: 1000 candles per request)
        num_batches = (gap.estimated_missing_candles // self.batch_size) + 1
        
        total_stored = 0
        start_time = gap.last_candle_time
        
        for batch_num in range(num_batches):
            try:
                # Calculate time range for this batch
                batch_start = start_time + timedelta(minutes=batch_num * self.batch_size)
                batch_end = min(
                    batch_start + timedelta(minutes=self.batch_size),
                    gap.current_time
                )
                
                # Stop if we've reached current time
                if batch_start >= gap.current_time:
                    break
                
                # Fetch from Binance
                candles = await self._fetch_from_binance(
                    symbol=gap.symbol,
                    start_time=batch_start,
                    end_time=batch_end
                )
                
                if candles:
                    # Store in database
                    stored = await self._store_candles(candles)
                    total_stored += stored
                    self.stats['candles_fetched'] += len(candles)
                    self.stats['candles_stored'] += stored
                    
                    logger.info(
                        f"   Batch {batch_num + 1}/{num_batches}: "
                        f"Fetched {len(candles)}, Stored {stored} candles"
                    )
                
                # Rate limiting - respect Binance limits
                if batch_num < num_batches - 1:
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_num} for {gap.symbol}: {e}")
                # Continue with next batch even if one fails
                continue
        
        logger.info(
            f"✅ {gap.symbol}: Filled {total_stored} candles "
            f"({total_stored/60:.1f} hours of data)"
        )
    
    async def _fetch_from_binance(
        self, 
        symbol: str, 
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Fetch OHLCV data from Binance
        """
        try:
            # Convert to milliseconds timestamp
            since = int(start_time.timestamp() * 1000)
            
            # Fetch with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=self.base_timeframe,
                        since=since,
                        limit=self.batch_size
                    )
                    
                    # Convert to our format
                    candles = []
                    for kline in ohlcv:
                        # Filter to only include candles in our time range
                        candle_time = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc)
                        if start_time <= candle_time < end_time:
                            candles.append({
                                'symbol': symbol,
                                'timeframe': self.base_timeframe,
                                'timestamp': candle_time,
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'source': 'backfill'
                            })
                    
                    return candles
                    
                except ccxt.RateLimitExceeded as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            f"⚠️ Rate limit hit for {symbol}, "
                            f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Fetch attempt {attempt + 1} failed for {symbol}: {e}"
                        )
                        await asyncio.sleep(1)
                    else:
                        raise e
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error fetching from Binance for {symbol}: {e}")
            return []
    
    async def _store_candles(self, candles: List[Dict]) -> int:
        """
        Store candles in TimescaleDB
        Uses INSERT ... ON CONFLICT DO NOTHING to avoid duplicates
        """
        if not candles:
            return 0
        
        try:
            query = """
                INSERT INTO ohlcv_data (
                    symbol, timeframe, timestamp, 
                    open, high, low, close, volume, source
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
            """
            
            stored_count = 0
            async with self.db_pool.acquire() as conn:
                for candle in candles:
                    try:
                        result = await conn.execute(
                            query,
                            candle['symbol'],
                            candle['timeframe'],
                            candle['timestamp'],
                            candle['open'],
                            candle['high'],
                            candle['low'],
                            candle['close'],
                            candle['volume'],
                            candle['source']
                        )
                        # If INSERT was successful (not a duplicate)
                        if result == "INSERT 0 1":
                            stored_count += 1
                    except Exception as e:
                        # Skip duplicates silently
                        continue
            
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error storing candles: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get backfill statistics"""
        return self.stats.copy()


# Standalone script for manual gap filling
async def run_manual_backfill():
    """
    Run gap backfill manually (useful for testing)
    """
    import ccxt
    
    # Database config
    db_pool = await asyncpg.create_pool(
        host='localhost',
        port=55433,
        database='alphapulse',
        user='alpha_emon',
        password='Emon_@17711',
        min_size=2,
        max_size=5
    )
    
    # Initialize Binance
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Symbols to backfill
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
    
    # Run backfill
    service = StartupGapBackfillService(db_pool, exchange)
    stats = await service.detect_and_fill_all_gaps(symbols)
    
    await db_pool.close()
    
    return stats


if __name__ == "__main__":
    # Run manual backfill
    asyncio.run(run_manual_backfill())

