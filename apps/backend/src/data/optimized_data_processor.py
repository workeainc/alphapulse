#!/usr/bin/env python3
"""
Optimized Data Processor for AlphaPulse
Implements the complete optimization playbook for maximum performance
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
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class OptimizedDataChunk:
    """Optimized data chunk with performance metrics"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    indicators: Dict[str, np.ndarray]
    patterns: List[Any]
    timestamp: datetime
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    data_hash: str = ""

class OptimizedDataProcessor:
    """
    Ultra-optimized data processor implementing the complete optimization playbook
    """
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 1000):
        """Initialize optimized data processor"""
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        # **2. CACHE REPETITIVE INDICATORS**
        self.data_cache = {}
        self.indicator_cache = {}
        self.pattern_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cache_cleanup = time.time()
        
        # Data buffers for real-time processing
        self.data_buffers = {}  # symbol -> deque of recent data
        self.processing_queue = deque(maxlen=10000)
        
        # Performance tracking
        self.stats = {
            'total_chunks_processed': 0,
            'cache_hits': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'data_points_processed': 0,
            'indicators_calculated': 0,
            'patterns_detected': 0
        }
        
        # Import optimized components
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from ..src.strategies.optimized_pattern_detector import OptimizedPatternDetector
            from ..src.strategies.indicators import TechnicalIndicators
            self.pattern_detector = OptimizedPatternDetector(max_workers=max_workers)
            self.indicators = TechnicalIndicators()
        except ImportError as e:
            logger.warning(f"⚠️ Could not import optimized components: {e}")
            self.pattern_detector = None
            self.indicators = None
        
        logger.info(f"Optimized Data Processor initialized with {max_workers} workers")
    
    async def process_data_chunk_optimized(self, 
                                         symbol: str, 
                                         data: pd.DataFrame, 
                                         timeframe: str,
                                         market_data: Dict = None) -> OptimizedDataChunk:
        """
        **1. VECTORIZE PATTERN CALCULATIONS**
        Process data chunk with optimized operations
        """
        start_time = time.time()
        
        # Validate inputs before processing
        if not self._validate_input_data(data, symbol, timeframe):
            logger.warning(f"Invalid input data for {symbol} {timeframe}")
            return None
        
        try:
            if len(data) < 10:  # Need minimum data for processing
                return None
            
            # **3. FILTER FIRST, DETECT LATER**
            # Apply fast preconditions to skip irrelevant processing
            if not self._apply_data_preconditions(data):
                return None
            
            # Create data hash for caching
            data_hash = self._create_data_hash(data, symbol, timeframe)
            
            # **2. CACHE REPETITIVE INDICATORS**
            # Check if we have cached results
            if data_hash in self.data_cache:
                self.stats['cache_hits'] += 1
                cached_chunk = self.data_cache[data_hash]
                cached_chunk.cache_hit = True
                return cached_chunk
            
            # **4. COMBINE RELATED PATTERNS INTO ONE PASS**
            # Process data in optimized pipeline
            processed_data = await self._process_data_pipeline(data, symbol, timeframe, market_data)
            
            if not processed_data:
                return None
            
            # Create optimized data chunk
            chunk = OptimizedDataChunk(
                symbol=symbol,
                timeframe=timeframe,
                data=processed_data['data'],
                indicators=processed_data['indicators'],
                patterns=processed_data['patterns'],
                timestamp=datetime.now(),
                data_hash=data_hash
            )
            
            # Cache the results
            self.data_cache[data_hash] = chunk
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            chunk.processing_time_ms = processing_time
            self._update_stats(processing_time, len(data), len(processed_data['patterns']))
            
            logger.info(f"⚡ Processed {symbol} data chunk: {len(data)} points, "
                       f"{len(processed_data['patterns'])} patterns in {processing_time:.2f}ms")
            
            return chunk
            
        except ValueError as e:
            logger.error(f"Data validation error for {symbol}: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, len(data), 0)
            return None
        except ImportError as e:
            logger.error(f"Module import error for {symbol}: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, len(data), 0)
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing data chunk for {symbol}: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, len(data), 0)
            return None
    
    def _validate_input_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Validate input data before processing"""
        try:
            if data is None or data.empty:
                return False
            
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Check for invalid values
            for col in required_columns:
                if data[col].isnull().any() or (data[col] <= 0).any():
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Data validation error: {e}", exc_info=True)
            return False
    
    def _apply_data_preconditions(self, data: pd.DataFrame) -> bool:
        """**3. FILTER FIRST, DETECT LATER** - Fast preconditions to skip irrelevant processing"""
        
        # **Fast preconditions using vectorized operations:**
        
        # Check if data has required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check if price data is valid (no NaN or infinite values)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if data[col].isna().any() or np.isinf(data[col]).any():
                return False
        
        # Check if price movements are reasonable
        price_changes = data['close'].pct_change().abs()
        if price_changes.max() > 0.5:  # More than 50% change in one period
            return False
        
        # Check if high/low relationships are valid
        if not ((data['high'] >= data['low']).all() and 
                (data['high'] >= data['open']).all() and 
                (data['high'] >= data['close']).all() and
                (data['low'] <= data['open']).all() and 
                (data['low'] <= data['close']).all()):
            return False
        
        return True
    
    def _create_data_hash(self, data: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Create hash for data caching"""
        # Create a hash based on data shape, last few values, and metadata
        hash_data = f"{symbol}_{timeframe}_{data.shape}_{data['close'].iloc[-1]:.6f}_{data['volume'].iloc[-1] if 'volume' in data.columns else 0}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    async def _process_data_pipeline(self, data: pd.DataFrame, symbol: str, timeframe: str, market_data: Dict = None) -> Optional[Dict]:
        """Process data through optimized pipeline with context managers"""
        try:
            # Use context manager for resource management
            with self._create_processing_context(symbol, timeframe) as context:
                # Step 1: Calculate technical indicators
                indicators = await self._calculate_indicators_vectorized(data, context)
                
                # Step 2: Detect patterns (moved outside loop for batch processing)
                patterns = await self._detect_patterns_batch(data, indicators, context)
                
                # Step 3: Apply market regime analysis
                market_regime = self._analyze_market_regime(data, indicators)
                
                return {
                    'data': data,
                    'indicators': indicators,
                    'patterns': patterns,
                    'market_regime': market_regime
                }
                
        except Exception as e:
            logger.error(f"Pipeline processing error for {symbol}: {e}", exc_info=True)
            return None
    
    def _create_processing_context(self, symbol: str, timeframe: str):
        """Create processing context manager for resource management"""
        class ProcessingContext:
            def __init__(self, processor, symbol, timeframe):
                self.processor = processor
                self.symbol = symbol
                self.timeframe = timeframe
                self.start_time = time.time()
            
            def __enter__(self):
                logger.debug(f"Starting processing context for {self.symbol} {self.timeframe}")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                processing_time = (time.time() - self.start_time) * 1000
                if exc_type is None:
                    logger.debug(f"Processing context completed for {self.symbol} in {processing_time:.2f}ms")
                else:
                    logger.error(f"Processing context failed for {self.symbol}: {exc_val}")
        
        return ProcessingContext(self, symbol, timeframe)
    
    async def _calculate_indicators_vectorized(self, data: pd.DataFrame, context) -> Dict[str, np.ndarray]:
        """Calculate technical indicators using vectorized operations"""
        try:
            indicators = {}
            
            # Vectorized calculations for better performance
            if len(data) >= 20:
                # Moving averages
                indicators['sma_20'] = data['close'].rolling(20).mean().values
                indicators['sma_50'] = data['close'].rolling(50).mean().values
                indicators['ema_12'] = data['close'].ewm(span=12).mean().values
                indicators['ema_26'] = data['close'].ewm(span=26).mean().values
                
                # RSI calculation
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).values
                
                # MACD
                indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
                indicators['macd_signal'] = pd.Series(indicators['macd']).ewm(span=9).mean().values
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}", exc_info=True)
            return {}
    
    async def _detect_patterns_batch(self, data: pd.DataFrame, indicators: Dict, context) -> List:
        """Detect patterns in batch to avoid per-item error handling"""
        try:
            if self.pattern_detector is None:
                return []
            
            # Batch pattern detection
            patterns = self.pattern_detector.detect_patterns_vectorized(data)
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}", exc_info=True)
            return []
    
    def _analyze_market_regime(self, data: pd.DataFrame, indicators: Dict) -> str:
        """Analyze market regime based on indicators"""
        try:
            if 'rsi' not in indicators or len(indicators['rsi']) == 0:
                return 'unknown'
            
            current_rsi = indicators['rsi'][-1]
            current_macd = indicators.get('macd', [0])[-1] if 'macd' in indicators else 0
            
            # Simple regime classification
            if current_rsi > 70 and current_macd < 0:
                return 'overbought'
            elif current_rsi < 30 and current_macd > 0:
                return 'oversold'
            elif current_rsi > 50 and current_macd > 0:
                return 'bullish'
            elif current_rsi < 50 and current_macd < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Market regime analysis error: {e}", exc_info=True)
            return 'unknown'
    
    async def process_multiple_chunks_parallel(self, chunks_data: Dict[str, Dict]) -> Dict[str, OptimizedDataChunk]:
        """
        **5. PARALLELIZE ACROSS CONTRACTS & TIMEFRAMES**
        Process multiple data chunks in parallel
        """
        logger.info(f"🔄 Starting parallel data processing for {len(chunks_data)} chunks")
        
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_key = {
                executor.submit(
                    self._process_chunk_sync, 
                    key, 
                    chunks_data[key]['data'], 
                    chunks_data[key]['timeframe'],
                    chunks_data[key].get('market_data')
                ): key
                for key in chunks_data.keys()
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    chunk = future.result()
                    results[key] = chunk
                    if chunk:
                        logger.info(f"✅ Processed chunk for {key}: {len(chunk.data)} points, {len(chunk.patterns)} patterns")
                    else:
                        logger.info(f"⏭️ No processing result for {key}")
                except Exception as e:
                    logger.error(f"❌ Error processing chunk for {key}: {e}")
                    results[key] = None
        
        return results
    
    def _process_chunk_sync(self, symbol: str, data: pd.DataFrame, timeframe: str, market_data: Dict = None):
        """Synchronous wrapper for chunk processing (for parallel processing)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_data_chunk_optimized(symbol, data, timeframe, market_data)
            )
        finally:
            loop.close()
    
    def add_to_buffer(self, symbol: str, data_point: Dict):
        """Add data point to buffer for real-time processing"""
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = deque(maxlen=self.chunk_size)
        
        self.data_buffers[symbol].append(data_point)
        
        # Check if buffer is ready for processing
        if len(self.data_buffers[symbol]) >= self.chunk_size:
            # Convert buffer to DataFrame and process
            df = pd.DataFrame(list(self.data_buffers[symbol]))
            self.processing_queue.append({
                'symbol': symbol,
                'data': df,
                'timestamp': datetime.now()
            })
    
    async def process_buffers(self) -> List[OptimizedDataChunk]:
        """Process all ready buffers"""
        results = []
        
        while self.processing_queue:
            chunk_info = self.processing_queue.popleft()
            
            try:
                chunk = await self.process_data_chunk_optimized(
                    chunk_info['symbol'],
                    chunk_info['data'],
                    '1m',  # Default timeframe for real-time data
                    {}
                )
                
                if chunk:
                    results.append(chunk)
                    
            except Exception as e:
                logger.error(f"❌ Error processing buffer for {chunk_info['symbol']}: {e}")
        
        return results
    
    def _update_stats(self, processing_time: float, data_points: int, patterns: int):
        """Update performance statistics"""
        self.stats['total_chunks_processed'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_chunks_processed']
        )
        self.stats['data_points_processed'] += data_points
        self.stats['patterns_detected'] += patterns
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self.data_cache),
            'indicator_cache_size': len(self.indicator_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_chunks_processed'], 1),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.data_buffers.items()},
            'queue_size': len(self.processing_queue)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.data_cache.clear()
        self.indicator_cache.clear()
        self.pattern_cache.clear()
        logger.info("🧹 Data processor cache cleared")
    
    def cleanup_old_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        
        # Remove old cache entries
        keys_to_remove = []
        for key, chunk in self.data_cache.items():
            if (current_time - chunk.timestamp.timestamp()) > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        if keys_to_remove:
            logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old cache entries")

# Example usage and performance comparison
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Test optimized data processor
    processor = OptimizedDataProcessor()
    
    # Performance test
    start_time = time.time()
    chunk = asyncio.run(processor.process_data_chunk_optimized('BTCUSDT', df, '1h'))
    optimized_time = (time.time() - start_time) * 1000
    
    print(f"🚀 Optimized Data Processing Results:")
    if chunk:
        print(f"   Data points processed: {len(chunk.data)}")
        print(f"   Indicators calculated: {len(chunk.indicators)}")
        print(f"   Patterns detected: {len(chunk.patterns)}")
        print(f"   Processing time: {optimized_time:.2f}ms")
    else:
        print(f"   No processing result")
        print(f"   Processing time: {optimized_time:.2f}ms")
    
    print(f"   Performance stats: {processor.get_performance_stats()}")
