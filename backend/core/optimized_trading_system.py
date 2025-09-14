#!/usr/bin/env python3
"""
Optimized Trading System for AlphaPulse
Complete integration of all optimized components implementing the optimization playbook
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

# Import optimized components
from ..strategies.optimized_pattern_detector import OptimizedPatternDetector, OptimizedPatternSignal
from ..strategies.optimized_signal_generator import OptimizedSignalGenerator, OptimizedTradingSignal
from data.optimized_data_processor import OptimizedDataProcessor, OptimizedDataChunk

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTradingResult:
    """Complete trading result with all optimized components"""
    symbol: str
    timestamp: datetime
    data_chunk: OptimizedDataChunk
    patterns: List[OptimizedPatternSignal]
    signal: Optional[OptimizedTradingSignal]
    processing_time_ms: float
    cache_hits: int
    performance_metrics: Dict

class OptimizedTradingSystem:
    """
    Complete optimized trading system implementing the optimization playbook
    """
    
    def __init__(self, config: Dict = None, max_workers: int = 4):
        """Initialize optimized trading system"""
        self.config = config or {}
        self.max_workers = max_workers
        self.is_running = False
        
        # Initialize all optimized components
        logger.info("🚀 Initializing Optimized Trading System...")
        
        self.pattern_detector = OptimizedPatternDetector(max_workers=max_workers)
        self.signal_generator = OptimizedSignalGenerator(config, max_workers)
        self.data_processor = OptimizedDataProcessor(max_workers=max_workers)
        
        # Trading configuration
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h'])
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_strength = self.config.get('min_strength', 0.6)
        
        # Performance tracking
        self.stats = {
            'total_analyses': 0,
            'total_signals': 0,
            'total_patterns': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'successful_signals': 0,
            'failed_signals': 0
        }
        
        # Results storage
        self.results_history = deque(maxlen=10000)
        self.signals_history = deque(maxlen=1000)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"✅ Optimized Trading System initialized with {max_workers} workers")
        logger.info(f"📊 Monitoring {len(self.symbols)} symbols across {len(self.timeframes)} timeframes")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    async def start(self):
        """Start the optimized trading system"""
        if self.is_running:
            logger.warning("⚠️ Trading system is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Optimized Trading System...")
        
        try:
            # Start main processing loop
            await self._main_processing_loop()
        except Exception as e:
            logger.error(f"❌ Error in main processing loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the optimized trading system"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping Optimized Trading System...")
        self.is_running = False
        
        # Clear caches
        self.pattern_detector.clear_cache()
        self.signal_generator.clear_cache()
        self.data_processor.clear_cache()
        
        logger.info("✅ Optimized Trading System stopped")
    
    async def _main_processing_loop(self):
        """Main processing loop with optimized operations"""
        logger.info("🔄 Starting main processing loop...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # **5. PARALLELIZE ACROSS CONTRACTS & TIMEFRAMES**
                # Process all symbols and timeframes in parallel
                await self._process_all_markets_parallel()
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                self._update_system_stats(processing_time)
                
                # Log performance summary
                if self.stats['total_analyses'] % 100 == 0:
                    self._log_performance_summary()
                
                # Wait for next cycle
                await asyncio.sleep(1)  # 1 second cycle
                
            except Exception as e:
                logger.error(f"❌ Error in main processing loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_all_markets_parallel(self):
        """**5. PARALLELIZE ACROSS CONTRACTS & TIMEFRAMES**"""
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all market analysis tasks
            future_to_market = {}
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    market_key = f"{symbol}_{timeframe}"
                    future = executor.submit(
                        self._analyze_market_sync, symbol, timeframe
                    )
                    future_to_market[future] = market_key
            
            # Collect results as they complete
            for future in as_completed(future_to_market):
                market_key = future_to_market[future]
                try:
                    result = future.result()
                    if result:
                        self.results_history.append(result)
                        if result.signal:
                            self.signals_history.append(result.signal)
                            self.stats['total_signals'] += 1
                            self.stats['successful_signals'] += 1
                        else:
                            self.stats['failed_signals'] += 1
                        
                        self.stats['total_analyses'] += 1
                        self.stats['total_patterns'] += len(result.patterns)
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing market {market_key}: {e}")
                    self.stats['failed_signals'] += 1
    
    def _analyze_market_sync(self, symbol: str, timeframe: str) -> Optional[OptimizedTradingResult]:
        """Synchronous market analysis (for parallel processing)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._analyze_single_market(symbol, timeframe)
            )
        finally:
            loop.close()
    
    async def _analyze_single_market(self, symbol: str, timeframe: str) -> Optional[OptimizedTradingResult]:
        """Analyze a single market with optimized pipeline"""
        start_time = time.time()
        
        try:
            # **1. VECTORIZE PATTERN CALCULATIONS**
            # Generate sample data (in production, this would come from data feeds)
            data = self._generate_sample_data(symbol, timeframe)
            
            if data is None or len(data) < 50:
                return None
            
            # **2. CACHE REPETITIVE INDICATORS**
            # Process data with optimized processor
            data_chunk = await self.data_processor.process_data_chunk_optimized(
                symbol, data, timeframe
            )
            
            if data_chunk is None:
                return None
            
            # **3. FILTER FIRST, DETECT LATER**
            # Detect patterns using optimized detector
            patterns = self.pattern_detector.detect_patterns_vectorized(data_chunk.data)
            
            # **4. COMBINE RELATED PATTERNS INTO ONE PASS**
            # Generate signals using optimized generator
            signal = await self.signal_generator.generate_signal_optimized(
                symbol, data_chunk.data, timeframe
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate cache hits
            cache_hits = (
                (1 if data_chunk.cache_hit else 0) +
                (1 if signal and signal.cache_hit else 0)
            )
            
            # Create result
            result = OptimizedTradingResult(
                symbol=symbol,
                timestamp=datetime.now(),
                data_chunk=data_chunk,
                patterns=patterns,
                signal=signal,
                processing_time_ms=processing_time,
                cache_hits=cache_hits,
                performance_metrics={
                    'data_processing_time': data_chunk.processing_time_ms,
                    'pattern_detection_time': getattr(signal, 'processing_time_ms', 0) if signal else 0,
                    'total_patterns': len(patterns),
                    'signal_confidence': signal.confidence if signal else 0,
                    'signal_strength': signal.strength if signal else 0
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol} on {timeframe}: {e}")
            return None
    
    def _generate_sample_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate sample data for testing (replace with real data feeds)"""
        try:
            # Generate realistic sample data
            np.random.seed(hash(f"{symbol}_{timeframe}") % 1000)
            n = 1000
            
            # Base price with trend
            base_price = 100 if 'BTC' in symbol else 50 if 'ETH' in symbol else 1
            trend = np.cumsum(np.random.randn(n) * 0.01)
            closes = base_price * (1 + trend)
            
            # Generate OHLCV data
            highs = closes + np.random.rand(n) * base_price * 0.02
            lows = closes - np.random.rand(n) * base_price * 0.02
            opens = np.roll(closes, 1)
            opens[0] = closes[0]
            volumes = np.random.randint(1000, 10000, n)
            
            # Add some pattern-like data
            # Hammer pattern at 20% of data
            hammer_idx = int(n * 0.2)
            if hammer_idx < n:
                opens[hammer_idx] = closes[hammer_idx] - base_price * 0.01
                lows[hammer_idx] = closes[hammer_idx] - base_price * 0.03
                highs[hammer_idx] = closes[hammer_idx] + base_price * 0.005
            
            # Doji pattern at 40% of data
            doji_idx = int(n * 0.4)
            if doji_idx < n:
                opens[doji_idx] = closes[doji_idx] + base_price * 0.005
                highs[doji_idx] = closes[doji_idx] + base_price * 0.02
                lows[doji_idx] = closes[doji_idx] - base_price * 0.02
            
            # Engulfing pattern at 60% of data
            engulfing_idx = int(n * 0.6)
            if engulfing_idx < n:
                opens[engulfing_idx] = closes[engulfing_idx] - base_price * 0.02
                closes[engulfing_idx] = closes[engulfing_idx] + base_price * 0.03
                highs[engulfing_idx] = closes[engulfing_idx] + base_price * 0.01
                lows[engulfing_idx] = opens[engulfing_idx] - base_price * 0.01
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error generating sample data for {symbol}: {e}")
            return None
    
    def _update_system_stats(self, processing_time: float):
        """Update system performance statistics"""
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / max(self.stats['total_analyses'], 1)
        )
        
        # Calculate cache hit rate
        total_cache_opportunities = self.stats['total_analyses'] * 2  # data + signal
        if total_cache_opportunities > 0:
            self.stats['cache_hit_rate'] = (
                (self.pattern_detector.stats['cache_hits'] + 
                 self.signal_generator.stats['cache_hits'] + 
                 self.data_processor.stats['cache_hits']) / total_cache_opportunities
            )
    
    def _log_performance_summary(self):
        """Log performance summary"""
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Total Analyses: {self.stats['total_analyses']}")
        logger.info(f"   Total Signals: {self.stats['total_signals']}")
        logger.info(f"   Total Patterns: {self.stats['total_patterns']}")
        logger.info(f"   Avg Processing Time: {self.stats['avg_processing_time_ms']:.2f}ms")
        logger.info(f"   Cache Hit Rate: {self.stats['cache_hit_rate']:.2%}")
        logger.info(f"   Success Rate: {self.stats['successful_signals'] / max(self.stats['total_signals'], 1):.2%}")
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'system_stats': self.stats,
            'pattern_detector_stats': self.pattern_detector.get_performance_stats(),
            'signal_generator_stats': self.signal_generator.get_performance_stats(),
            'data_processor_stats': self.data_processor.get_performance_stats(),
            'recent_signals': self._get_recent_signals_summary(),
            'recent_patterns': self._get_recent_patterns_summary()
        }
    
    def _get_recent_signals_summary(self) -> List[Dict]:
        """Get summary of recent signals"""
        recent_signals = list(self.signals_history)[-10:]  # Last 10 signals
        
        return [
            {
                'symbol': signal.symbol,
                'type': signal.signal_type,
                'pattern': signal.pattern,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'timestamp': signal.timestamp.isoformat(),
                'processing_time_ms': signal.processing_time_ms
            }
            for signal in recent_signals
        ]
    
    def _get_recent_patterns_summary(self) -> Dict:
        """Get summary of recent patterns"""
        if not self.results_history:
            return {"message": "No patterns detected yet"}
        
        recent_results = list(self.results_history)[-50:]  # Last 50 results
        
        pattern_counts = {}
        symbol_patterns = {}
        
        for result in recent_results:
            for pattern in result.patterns:
                pattern_name = pattern.pattern
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                
                if result.symbol not in symbol_patterns:
                    symbol_patterns[result.symbol] = {}
                symbol_patterns[result.symbol][pattern_name] = symbol_patterns[result.symbol].get(pattern_name, 0) + 1
        
        return {
            'total_patterns': sum(pattern_counts.values()),
            'pattern_counts': pattern_counts,
            'symbol_patterns': symbol_patterns
        }
    
    async def run_performance_test(self, duration_seconds: int = 60):
        """Run performance test for specified duration"""
        logger.info(f"🧪 Starting performance test for {duration_seconds} seconds...")
        
        # Set running state for test
        self.is_running = True
        
        start_time = time.time()
        test_stats = {
            'start_time': start_time,
            'total_cycles': 0,
            'total_analyses': 0,
            'total_signals': 0,
            'total_patterns': 0,
            'processing_times': []
        }
        
        while time.time() - start_time < duration_seconds and self.is_running:
            cycle_start = time.time()
            
            # Process one cycle
            await self._process_all_markets_parallel()
            
            cycle_time = (time.time() - cycle_start) * 1000
            test_stats['processing_times'].append(cycle_time)
            test_stats['total_cycles'] += 1
            test_stats['total_analyses'] = self.stats['total_analyses']
            test_stats['total_signals'] = self.stats['total_signals']
            test_stats['total_patterns'] = self.stats['total_patterns']
            
            # Log progress every 10 cycles
            if test_stats['total_cycles'] % 10 == 0:
                logger.info(f"🧪 Test Progress: {test_stats['total_cycles']} cycles, "
                           f"{test_stats['total_analyses']} analyses, "
                           f"{test_stats['total_signals']} signals")
            
            await asyncio.sleep(0.1)  # 100ms cycles for testing
        
        # Calculate test results
        test_duration = time.time() - start_time
        
        if test_stats['processing_times']:
            avg_cycle_time = np.mean(test_stats['processing_times'])
            max_cycle_time = np.max(test_stats['processing_times'])
            min_cycle_time = np.min(test_stats['processing_times'])
        else:
            avg_cycle_time = max_cycle_time = min_cycle_time = 0.0
        
        logger.info(f"🧪 Performance Test Results:")
        logger.info(f"   Test Duration: {test_duration:.2f} seconds")
        logger.info(f"   Total Cycles: {test_stats['total_cycles']}")
        logger.info(f"   Total Analyses: {test_stats['total_analyses']}")
        logger.info(f"   Total Signals: {test_stats['total_signals']}")
        logger.info(f"   Total Patterns: {test_stats['total_patterns']}")
        logger.info(f"   Avg Cycle Time: {avg_cycle_time:.2f}ms")
        logger.info(f"   Max Cycle Time: {max_cycle_time:.2f}ms")
        logger.info(f"   Min Cycle Time: {min_cycle_time:.2f}ms")
        logger.info(f"   Analyses per Second: {test_stats['total_analyses'] / test_duration:.2f}")
        logger.info(f"   Signals per Second: {test_stats['total_signals'] / test_duration:.2f}")
        logger.info(f"   Patterns per Second: {test_stats['total_patterns'] / test_duration:.2f}")

# Example usage
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        'timeframes': ['1m', '5m', '15m'],
        'min_confidence': 0.7,
        'min_strength': 0.6
    }
    
    # Create and run optimized trading system
    trading_system = OptimizedTradingSystem(config, max_workers=4)
    
    async def main():
        try:
            # Run performance test
            await trading_system.run_performance_test(duration_seconds=30)
            
            # Get final stats
            stats = trading_system.get_system_stats()
            print(f"\n📊 Final System Statistics:")
            print(json.dumps(stats, indent=2, default=str))
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        finally:
            await trading_system.stop()
    
    # Run the system
    asyncio.run(main())
