# Storage Optimization Notes

## Problem Identified

TimescaleDB hypertable with many chunks (1000+ chunks) causes extremely slow `ON CONFLICT` checks. Each INSERT has to check the unique constraint across ALL chunks, leading to:

- Batch inserts timing out after 30+ seconds
- Queries getting stuck for HOURS
- Database becoming unresponsive

## Solution Implemented

1. **Killed Stuck Queries**: Terminated 4 long-running INSERT queries (5+ hours old)

2. **Row-by-Row Inserts**: 
   - Changed from batch inserts (500 rows) to row-by-row inserts
   - Batch size: 10 rows at a time
   - Timeout: 5 seconds per row (instead of 30s per batch)
   - Skips failed rows and continues

3. **More Frequent Storage**:
   - Storage triggers at 25,000 candles (reduced from 50k)
   - Prevents memory buildup
   - Ensures progress is saved incrementally

## Performance Trade-offs

**Slower but Safe:**
- Row-by-row is slower but won't hang
- Each row completes in < 5 seconds (or times out)
- No more stuck queries blocking database

**Expected Speed:**
- ~1-2 rows per second (with conflict checking)
- 3.37M candles ÷ 2 rows/sec = ~20 days (too slow!)

## Better Solutions (Future)

1. **Disable Unique Constraint Temporarily**:
   ```sql
   ALTER TABLE ohlcv_data DROP CONSTRAINT IF EXISTS idx_ohlcv_unique;
   -- Do bulk insert
   -- Re-add constraint: CREATE UNIQUE INDEX idx_ohlcv_unique ON ohlcv_data (symbol, timeframe, timestamp);
   ```

2. **Use COPY FROM** (asyncpg supports this):
   - Much faster bulk inserts
   - May bypass some constraint checking overhead
   - Requires handling duplicates differently

3. **Direct Chunk Insertion**:
   - Insert directly to TimescaleDB chunks
   - Bypass hypertable routing
   - Complex to implement correctly

4. **Temporary Table Approach**:
   - Bulk insert to temp table
   - `INSERT INTO ohlcv_data SELECT * FROM temp WHERE NOT EXISTS...`
   - Faster conflict checking

## Current Status

The script now uses row-by-row inserts with proper timeouts. It will be slow but reliable. Consider implementing one of the "Better Solutions" above for faster bulk loading.

