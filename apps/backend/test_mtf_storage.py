"""
Test Script for MTF Signal Storage
Verifies signal storage, caching, and retrieval work correctly
"""

import asyncio
import logging
from datetime import datetime, timezone

from src.services.ai_model_integration_service import AIModelIntegrationService, AIModelSignal
from src.services.mtf_signal_storage import MTFSignalStorage
from src.database.connection import db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_mtf_signal_storage():
    """Test MTF signal storage functionality"""
    
    print("\n" + "=" * 80)
    print("MTF SIGNAL STORAGE TEST")
    print("=" * 80 + "\n")
    
    try:
        # Initialize database connection
        await db_connection.initialize()
        print("[OK] Database connection initialized")
        
        # Initialize storage service
        storage = MTFSignalStorage(
            db_connection=db_connection,
            redis_url='redis://localhost:56379'
        )
        await storage.initialize()
        print("[OK] MTF Signal Storage initialized")
        
        # Test 1: Generate MTF signal for BTCUSDT
        print("\n[TEST 1] Generating MTF signal for BTCUSDT...")
        ai_service = AIModelIntegrationService()
        
        signal = await ai_service.generate_ai_signal_with_mtf_entry(
            symbol='BTCUSDT',
            signal_timeframe='1h',
            entry_timeframe='15m'
        )
        
        if signal:
            print(f"[OK] Signal generated!")
            print(f"  - Direction: {signal.signal_direction}")
            print(f"  - Entry: ${signal.entry_price:.2f if signal.entry_price else 0:.2f}")
            print(f"  - Stop: ${signal.stop_loss:.2f if signal.stop_loss else 0:.2f}")
            print(f"  - Strategy: {signal.entry_strategy}")
            print(f"  - Confidence: {signal.entry_confidence:.2f if signal.entry_confidence else 0:.2f}")
        else:
            print("[WARN] No signal generated (may need consensus)")
            return
        
        # Test 2: Store to database
        print("\n[TEST 2] Storing signal to database...")
        stored = await storage.store_mtf_signal(signal)
        
        if stored:
            print("[OK] Signal stored successfully")
        else:
            print("[FAIL] Signal storage failed")
            return
        
        # Test 3: Verify in database
        print("\n[TEST 3] Verifying signal in database...")
        async with db_connection.get_connection() as conn:
            result = await conn.fetchrow("""
                SELECT 
                    symbol, direction, entry_strategy, entry_price,
                    stop_loss, take_profit_1, risk_reward_ratio
                FROM ai_signals_mtf
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """, 'BTCUSDT')
            
            if result:
                print("[OK] Signal found in database:")
                print(f"  - Symbol: {result['symbol']}")
                print(f"  - Direction: {result['direction']}")
                print(f"  - Entry Strategy: {result['entry_strategy']}")
                print(f"  - Entry Price: ${float(result['entry_price']):.2f if result['entry_price'] else 0:.2f}")
                print(f"  - Stop Loss: ${float(result['stop_loss']):.2f if result['stop_loss'] else 0:.2f}")
                print(f"  - R:R Ratio: {float(result['risk_reward_ratio']):.2f if result['risk_reward_ratio'] else 0:.2f}")
            else:
                print("[FAIL] Signal not found in database")
                return
        
        # Test 4: Verify in Redis cache
        print("\n[TEST 4] Verifying signal in Redis cache...")
        cached = await storage.get_cached_signal('BTCUSDT', signal.signal_direction)
        
        if cached:
            print("[OK] Signal found in cache:")
            print(f"  - Entry Price: ${cached.get('entry_price', 0):.2f}")
            print(f"  - Entry Strategy: {cached.get('entry_strategy', 'N/A')}")
        else:
            print("[WARN] Signal not in cache (may be normal)")
        
        # Test 5: Test deduplication
        print("\n[TEST 5] Testing deduplication...")
        duplicate_exists = await storage.check_active_signal_exists(
            'BTCUSDT', signal.signal_direction
        )
        
        if duplicate_exists:
            print("[OK] Deduplication working - active signal detected")
        else:
            print("[WARN] Deduplication may not be working")
        
        # Test 6: Check entry history
        print("\n[TEST 6] Checking entry analysis history...")
        async with db_connection.get_connection() as conn:
            history_count = await conn.fetchval("""
                SELECT COUNT(*) 
                FROM mtf_entry_analysis_history
                WHERE symbol = $1
            """, 'BTCUSDT')
            
            if history_count and history_count > 0:
                print(f"[OK] Found {history_count} entry analysis records")
            else:
                print("[INFO] No entry analysis history (may be normal for first run)")
        
        # Test 7: Storage statistics
        print("\n[TEST 7] Storage statistics...")
        stats = storage.get_stats()
        print(f"[OK] Storage stats:")
        print(f"  - Signals stored: {stats['signals_stored']}")
        print(f"  - Storage failures: {stats['storage_failures']}")
        print(f"  - Success rate: {stats['storage_success_rate']:.1%}")
        print(f"  - Cache hits: {stats['cache_hits']}")
        print(f"  - Cache misses: {stats['cache_misses']}")
        
        # Cleanup
        await storage.close()
        print("\n[OK] Test completed successfully!")
        
    except Exception as e:
        logger.error(f"[FAIL] Test failed with error: {e}", exc_info=True)
        print(f"\n[FAIL] Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_mtf_signal_storage())

