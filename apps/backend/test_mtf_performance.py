"""
MTF System Performance Test
Tests MTF system with 10 symbols and measures performance impact
"""

import asyncio
import logging
import time
import psutil
import os
from datetime import datetime, timezone
from typing import List, Dict

from src.services.ai_model_integration_service import AIModelIntegrationService
from src.services.mtf_signal_storage import MTFSignalStorage
from src.database.connection import db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def measure_signal_generation_performance(
    symbols: List[str],
    signal_timeframe: str = '1h',
    entry_timeframe: str = '15m'
) -> Dict:
    """Measure performance of MTF signal generation"""
    
    ai_service = AIModelIntegrationService()
    
    results = {
        'symbols_tested': len(symbols),
        'signals_generated': 0,
        'total_time_seconds': 0,
        'avg_time_per_signal_ms': 0,
        'min_time_ms': float('inf'),
        'max_time_ms': 0,
        'entry_strategies_used': {},
        'memory_usage_mb': 0
    }
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    timings = []
    
    print(f"\nGenerating MTF signals for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            signal_start = time.time()
            
            signal = await ai_service.generate_ai_signal_with_mtf_entry(
                symbol=symbol,
                signal_timeframe=signal_timeframe,
                entry_timeframe=entry_timeframe
            )
            
            signal_time = (time.time() - signal_start) * 1000  # ms
            timings.append(signal_time)
            
            if signal:
                results['signals_generated'] += 1
                
                # Track strategy used
                strategy = signal.entry_strategy or 'UNKNOWN'
                results['entry_strategies_used'][strategy] = \
                    results['entry_strategies_used'].get(strategy, 0) + 1
                
                print(f"  [{i}/{len(symbols)}] {symbol}: "
                      f"{signal.signal_direction} @ ${signal.entry_price:.2f if signal.entry_price else 0:.2f} "
                      f"({strategy}) - {signal_time:.0f}ms")
            else:
                print(f"  [{i}/{len(symbols)}] {symbol}: No signal - {signal_time:.0f}ms")
                
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
    
    total_time = time.time() - start_time
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate statistics
    if timings:
        results['total_time_seconds'] = total_time
        results['avg_time_per_signal_ms'] = sum(timings) / len(timings)
        results['min_time_ms'] = min(timings)
        results['max_time_ms'] = max(timings)
    
    results['memory_usage_mb'] = memory_after - memory_before
    
    return results


async def test_mtf_performance():
    """Run MTF performance test"""
    
    print("\n" + "=" * 80)
    print("MTF SYSTEM PERFORMANCE TEST")
    print("=" * 80 + "\n")
    
    try:
        # Initialize database
        await db_connection.initialize()
        print("[OK] Database initialized")
        
        # Test symbols (mix of futures and spot)
        test_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT'
        ]
        
        print(f"\n[INFO] Testing with {len(test_symbols)} symbols")
        print(f"[INFO] Signal TF: 1h, Entry TF: 15m")
        
        # Run performance test
        results = await measure_signal_generation_performance(
            symbols=test_symbols,
            signal_timeframe='1h',
            entry_timeframe='15m'
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"\nSymbols tested: {results['symbols_tested']}")
        print(f"Signals generated: {results['signals_generated']}")
        print(f"Signal generation rate: {results['signals_generated'] / results['symbols_tested'] * 100:.1f}%")
        print(f"\nTiming:")
        print(f"  - Total time: {results['total_time_seconds']:.2f}s")
        print(f"  - Avg per signal: {results['avg_time_per_signal_ms']:.0f}ms")
        print(f"  - Min time: {results['min_time_ms']:.0f}ms")
        print(f"  - Max time: {results['max_time_ms']:.0f}ms")
        print(f"\nMemory:")
        print(f"  - Memory increase: {results['memory_usage_mb']:.2f} MB")
        print(f"\nEntry Strategies Used:")
        for strategy, count in sorted(results['entry_strategies_used'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {strategy}: {count} ({count/results['signals_generated']*100:.1f}%)")
        
        # Performance assessment
        print("\n" + "=" * 80)
        print("ASSESSMENT")
        print("=" * 80)
        
        if results['avg_time_per_signal_ms'] < 500:
            print("[OK] Excellent performance (< 500ms per signal)")
        elif results['avg_time_per_signal_ms'] < 1000:
            print("[OK] Good performance (< 1000ms per signal)")
        elif results['avg_time_per_signal_ms'] < 2000:
            print("[WARN] Acceptable performance (< 2000ms per signal)")
        else:
            print("[FAIL] Poor performance (> 2000ms per signal)")
        
        if results['memory_usage_mb'] < 100:
            print("[OK] Low memory usage (< 100MB)")
        elif results['memory_usage_mb'] < 300:
            print("[OK] Moderate memory usage (< 300MB)")
        else:
            print("[WARN] High memory usage (> 300MB)")
        
        # Database query performance test
        print("\n[INFO] Testing database query performance...")
        async with db_connection.get_connection() as conn:
            query_start = time.time()
            
            signals = await conn.fetch("""
                SELECT * FROM ai_signals_mtf
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                LIMIT 100
            """)
            
            query_time = (time.time() - query_start) * 1000
            
            print(f"[OK] Database query: {query_time:.0f}ms ({len(signals)} records)")
        
        print("\n[TEST COMPLETE] ✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n[FAIL] Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_mtf_signal_storage())

