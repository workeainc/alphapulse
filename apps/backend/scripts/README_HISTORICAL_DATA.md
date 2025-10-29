# Historical Data Download Setup Guide

This guide explains how to download 1 year of historical Binance data for AlphaPulse indicator calculations.

## Overview

Without historical data, advanced indicators (CVD, OBV, VWAP, Volume Profile) cannot calculate properly. This setup downloads:
- **5 symbols**: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT
- **4 timeframes**: 1m, 5m, 15m, 1h
- **1 year** of OHLCV data (~3.37 million candles)
- **Estimated time**: ~25 minutes

## Quick Start

### Option 1: Automated Full Setup (Recommended)

Run the master setup script that does everything automatically:

```bash
cd apps/backend
python scripts/run_full_setup.py
```

This will:
1. ✅ Verify database schema
2. ✅ Mark existing signals as test data
3. ✅ Download 1 year of historical data (~25 minutes)
4. ✅ Verify data integrity
5. ✅ Run automated tests

### Option 2: Manual Step-by-Step

If you prefer to run steps individually:

#### Step 1: Verify Database Schema

```bash
cd apps/backend
python scripts/verify_database_schema.py
```

This verifies:
- Database connection works
- `ohlcv_data` table exists with correct structure
- Indexes are present for performance
- TimescaleDB hypertable is configured

#### Step 2: Mark Test Signals (Optional)

Mark existing signals so you can distinguish test data from real signals:

```bash
python scripts/mark_test_signals.py
```

This:
- Adds `is_test_data` column if it doesn't exist
- Marks all existing signals as test data
- Preserves all data (nothing is deleted)

#### Step 3: Download Historical Data

```bash
python scripts/download_1year_historical.py
```

This will:
- Download 1 year of data from Binance
- Store in `ohlcv_data` table
- Handle duplicates gracefully
- Show progress for each symbol/timeframe
- Take ~25 minutes to complete

**Important**: This script respects Binance API rate limits and includes:
- Pagination (1000 candles per API call)
- Rate limiting (500ms between requests)
- Retry logic for failed requests
- Progress tracking

#### Step 4: Verify Data Downloaded

Run the verification script again:

```bash
python scripts/verify_database_schema.py
```

Check the output for:
- Total candles count (should be ~3.37M)
- Data per symbol/timeframe
- Date range coverage (should be ~365 days)

#### Step 5: Run Tests (Optional)

```bash
python tests/test_historical_integration.py
```

Or using pytest:

```bash
pytest tests/test_historical_integration.py -v
```

## Script Details

### `verify_database_schema.py`

Verifies database connection and table structure.

**Checks:**
- Database connection
- `ohlcv_data` table existence
- Required columns present
- Indexes exist
- TimescaleDB hypertable status
- Current data counts

**Exit Codes:**
- `0`: Success - all checks passed
- `1`: Failure - issues found

### `mark_test_signals.py`

Marks existing signals as test data without deletion.

**Actions:**
- Creates `is_test_data` BOOLEAN column if missing
- Sets `is_test_data = TRUE` for all existing signals
- Creates metadata entry for tracking

**Safe to run multiple times**: Yes (idempotent)

### `download_1year_historical.py`

Downloads historical OHLCV data from Binance.

**Features:**
- Paginated download (handles Binance 1000 candle limit)
- Rate limiting (respects API limits)
- Duplicate handling (ON CONFLICT DO NOTHING)
- Progress tracking
- Error recovery with retries
- Comprehensive logging

**Configuration:**
- Symbols: `SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']`
- Timeframes: `TIMEFRAMES = ['1m', '5m', '15m', '1h']`
- Days: `days = 365`

**Logs:**
- Progress to console
- Detailed log to `historical_download.log`

**Estimated Times:**
- **1m timeframe**: ~17.5 minutes per symbol
- **5m timeframe**: ~3.5 minutes per symbol
- **15m timeframe**: ~72 seconds per symbol
- **1h timeframe**: ~18 seconds per symbol
- **Total**: ~25 minutes for all symbols/timeframes

### `run_full_setup.py`

Master script that orchestrates all setup steps.

**Usage:**
```bash
python scripts/run_full_setup.py
```

**Output:**
- Step-by-step execution
- Progress logging
- Summary report at end
- Log file: `setup_YYYYMMDD_HHMMSS.log`

**Exit Codes:**
- `0`: Success
- `1`: Failure (check log for details)

## Backend Integration

After downloading historical data, the backend (`main.py`) will:

1. **On Startup**: Load recent historical data into indicator buffers
2. **Indicator Calculation**: Use historical context immediately
3. **Signal Generation**: Generate signals with full indicator data from day 1

### How It Works

The `load_historical_data_to_buffers()` function in `main.py`:
- Loads last 500 candles per symbol/timeframe from database
- Populates `RealtimeIndicatorCalculator` buffers
- Populates `MTFDataManager` buffers
- **Does NOT trigger callbacks** (prevents signals from old data)

This means indicators can calculate immediately without waiting for WebSocket data accumulation.

## Verification

### Check Data Counts

Connect to PostgreSQL and run:

```sql
-- Total candles
SELECT COUNT(*) FROM ohlcv_data;

-- By symbol
SELECT symbol, COUNT(*) as count
FROM ohlcv_data
GROUP BY symbol
ORDER BY count DESC;

-- By timeframe
SELECT timeframe, COUNT(*) as count
FROM ohlcv_data
GROUP BY timeframe
ORDER BY count DESC;

-- Date range
SELECT 
    MIN(timestamp) as oldest,
    MAX(timestamp) as newest,
    MAX(timestamp) - MIN(timestamp) as duration
FROM ohlcv_data
WHERE source = 'historical_1year';
```

### Expected Results

For 5 symbols × 4 timeframes × 365 days:
- **1m**: ~2,628,000 candles (525,600 per symbol)
- **5m**: ~525,600 candles (105,120 per symbol)
- **15m**: ~175,200 candles (35,040 per symbol)
- **1h**: ~43,800 candles (8,760 per symbol)
- **Total**: ~3,372,600 candles

### Test Signals

Check that old signals are marked:

```sql
SELECT 
    COUNT(*) FILTER (WHERE is_test_data = TRUE) as test_signals,
    COUNT(*) FILTER (WHERE is_test_data = FALSE OR is_test_data IS NULL) as real_signals
FROM live_signals;
```

## Troubleshooting

### Database Connection Error

**Error**: `Could not connect to database`

**Solution**:
1. Verify PostgreSQL is running
2. Check connection settings in `DB_CONFIG`
3. Verify port 55433 is correct (TimescaleDB default)

### Binance API Errors

**Error**: `API error` or rate limit exceeded

**Solution**:
- The script includes retry logic
- Rate limiting is built in (500ms between requests)
- If persistent, wait 5 minutes and retry

### Insufficient Data

**Error**: Tests fail with "insufficient data"

**Solution**:
- Check if download completed successfully
- Verify `ohlcv_data` table has data
- Re-run download script (idempotent, will skip duplicates)

### Duplicate Key Errors

**Error**: `ON CONFLICT` not working

**Solution**:
- Verify unique index exists: `idx_ohlcv_unique`
- Check schema: `\d ohlcv_data` in psql
- Run verification script to check schema

## Performance

### Database Size

After downloading 1 year of data:
- **Estimated size**: ~500MB - 1GB (compressed by TimescaleDB)
- **With compression**: ~200-400MB after 1 month

### Query Performance

With proper indexes:
- Single symbol/timeframe query: < 100ms
- 500 candle load: < 200ms
- Full indicator calculation: < 1s

## Maintenance

### Adding More Symbols

Edit `download_1year_historical.py`:
```python
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'NEWSYMBOL']
```

### Adding More Timeframes

Edit `download_1year_historical.py`:
```python
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
```

### Updating Historical Data

The download script is idempotent:
- Re-running will skip duplicates
- Only new data will be downloaded
- Safe to run daily/weekly to backfill gaps

## Next Steps

After successful setup:

1. **Start Backend**:
   ```bash
   python main.py
   # or
   ./start_backend.sh
   ```

2. **Verify Indicators**:
   - Check logs for "Historical data loaded into buffers"
   - Indicators should calculate immediately
   - Signals should include full SDE analysis

3. **Frontend Verification**:
   - Frontend should display signals
   - WebSocket should stream real-time updates
   - API endpoints should return signals with indicators

## Support

For issues:
1. Check log files: `historical_download.log`, `setup_*.log`
2. Run verification script: `python scripts/verify_database_schema.py`
3. Check database directly with SQL queries above

## Script Files

- `verify_database_schema.py` - Database verification
- `mark_test_signals.py` - Mark existing signals as test
- `download_1year_historical.py` - Download historical data
- `run_full_setup.py` - Master execution script
- `test_historical_integration.py` - Automated test suite

