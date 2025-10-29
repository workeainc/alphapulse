#!/usr/bin/env python3
"""
Manual Backfill Script
Run this manually to backfill data without starting the full system
"""

import asyncio
import asyncpg
import ccxt
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.startup_gap_backfill_service import StartupGapBackfillService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run manual backfill"""
    
    logger.info("\n" + "="*80)
    logger.info("🔄 MANUAL GAP BACKFILL")
    logger.info("="*80 + "\n")
    
    try:
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
        logger.info("✓ Database connected")
        
        # Initialize Binance
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        logger.info("✓ Binance exchange initialized")
        
        # Symbols to backfill
        symbols = [
            'BTCUSDT', 
            'ETHUSDT', 
            'BNBUSDT', 
            'SOLUSDT', 
            'ADAUSDT', 
            'XRPUSDT'
        ]
        
        logger.info(f"📊 Checking {len(symbols)} symbols for gaps...\n")
        
        # Run backfill
        service = StartupGapBackfillService(db_pool, exchange)
        stats = await service.detect_and_fill_all_gaps(symbols)
        
        # Close database
        await db_pool.close()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("✅ BACKFILL COMPLETE")
        logger.info("="*80)
        logger.info(f"Symbols processed: {stats['symbols_processed']}")
        logger.info(f"Gaps detected: {stats['gaps_detected']}")
        logger.info(f"Gaps filled: {stats['gaps_filled']}")
        logger.info(f"Candles fetched: {stats['candles_fetched']:,}")
        logger.info(f"Candles stored: {stats['candles_stored']:,}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

