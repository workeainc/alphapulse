#!/usr/bin/env python3
"""
AlphaPulse Optimized Trading Integration Service
Bridges optimized components with existing trading engine
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Import optimized components
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from ..src.strategies.optimized_pattern_detector import OptimizedPatternDetector
    from ..src.strategies.optimized_signal_generator import OptimizedSignalGenerator
    from src.data.optimized_data_processor import OptimizedDataProcessor
    from optimized_trading_system import OptimizedTradingSystem
    OPTIMIZED_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Optimized components not available: {e}")
    OPTIMIZED_AVAILABLE = False

# Import existing components
from src.app.core.config import settings
from src.app.services.market_data_service import MarketDataService
from src.app.services.risk_manager import RiskManager
from src.app.strategies.strategy_manager import StrategyManager
from src.app.database.models import SignalRecommendation, Strategy, MarketData
from src.app.database.connection import get_db

logger = logging.getLogger(__name__)

class OptimizedTradingIntegration:
    """
    Integration service that bridges optimized components with existing AlphaPulse system
    """
    
    def __init__(self, max_workers: int = 4):
        self.is_running = False
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Existing components
        self.market_data_service = MarketDataService()
        self.risk_manager = RiskManager()
        self.strategy_manager = StrategyManager()
        
        # Optimized components (if available)
        self.optimized_system = None
        self.optimized_pattern_detector = None
        self.optimized_signal_generator = None
        self.optimized_data_processor = None
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'optimized_signals': 0,
            'legacy_signals': 0,
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize optimized components
        self._initialize_optimized_components()
    
    def _initialize_optimized_components(self):
        """Initialize optimized components if available"""
        if not OPTIMIZED_AVAILABLE:
            logger.warning("⚠️ Optimized components not available - using legacy system")
            return
        
        try:
            logger.info("🚀 Initializing optimized trading components...")
            
            # Initialize optimized system
            self.optimized_system = OptimizedTradingSystem(max_workers=self.max_workers)
            
            # Initialize individual components
            self.optimized_pattern_detector = OptimizedPatternDetector(max_workers=self.max_workers)
            self.optimized_signal_generator = OptimizedSignalGenerator(
                pattern_detector=self.optimized_pattern_detector
            )
            self.optimized_data_processor = OptimizedDataProcessor(max_workers=self.max_workers)
            
            logger.info("✅ Optimized components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing optimized components: {e}")
            self.optimized_system = None
            self.optimized_pattern_detector = None
            self.optimized_signal_generator = None
            self.optimized_data_processor = None
    
    async def start(self):
        """Start the integration service"""
        if self.is_running:
            logger.warning("Integration service is already running")
            return
        
        logger.info("🚀 Starting Optimized Trading Integration...")
        self.is_running = True
        
        try:
            # Start existing services
            await self.market_data_service.start()
            await self.risk_manager.start()
            await self.strategy_manager.start()
            
            # Start optimized system if available
            if self.optimized_system:
                await self.optimized_system.start()
            
            logger.info("✅ Optimized Trading Integration started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting integration service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the integration service"""
        if not self.is_running:
            logger.warning("Integration service is not running")
            return
        
        logger.info("🛑 Stopping Optimized Trading Integration...")
        self.is_running = False
        
        try:
            # Stop existing services
            await self.market_data_service.stop()
            await self.risk_manager.stop()
            await self.strategy_manager.stop()
            
            # Stop optimized system if available
            if self.optimized_system:
                await self.optimized_system.stop()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Optimized Trading Integration stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping integration service: {e}")
    
    async def process_market_data(self, market_data: pd.DataFrame, 
                                symbols: List[str] = None) -> Dict[str, Any]:
        """
        Process market data using both optimized and legacy systems
        Returns combined results with performance metrics
        """
        if not self.is_running:
            raise RuntimeError("Integration service is not running")
        
        start_time = datetime.now()
        results = {
            'signals': [],
            'patterns': [],
            'performance': {},
            'optimization_used': False
        }
        
        try:
            # Use optimized system if available
            if self.optimized_system and not market_data.empty:
                logger.info("🔄 Processing with optimized system...")
                
                # Process with optimized system
                optimized_results = await self._process_with_optimized_system(
                    market_data, symbols
                )
                
                results.update(optimized_results)
                results['optimization_used'] = True
                self.performance_stats['optimized_signals'] += len(results['signals'])
                
            else:
                logger.info("🔄 Processing with legacy system...")
                
                # Fallback to legacy system
                legacy_results = await self._process_with_legacy_system(
                    market_data, symbols
                )
                
                results.update(legacy_results)
                self.performance_stats['legacy_signals'] += len(results['signals'])
            
            # Update performance stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats['processing_times'].append(processing_time)
            self.performance_stats['total_signals'] += len(results['signals'])
            
            results['performance'] = {
                'processing_time': processing_time,
                'signals_generated': len(results['signals']),
                'patterns_detected': len(results['patterns']),
                'optimization_used': results['optimization_used']
            }
            
            logger.info(f"✅ Processed {len(results['signals'])} signals in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _process_with_optimized_system(self, market_data: pd.DataFrame, 
                                           symbols: List[str] = None) -> Dict[str, Any]:
        """Process market data using optimized system"""
        try:
            # Process data with optimized data processor
            processed_data = await self._run_in_executor(
                self.optimized_data_processor.process_data, market_data
            )
            
            if processed_data is None or processed_data.empty:
                return {'signals': [], 'patterns': []}
            
            # Detect patterns with optimized pattern detector
            patterns = await self._run_in_executor(
                self.optimized_pattern_detector.detect_all_patterns, processed_data
            )
            
            # Generate signals with optimized signal generator
            signals = await self._run_in_executor(
                self.optimized_signal_generator.generate_signals, processed_data
            )
            
            # Update cache stats
            if hasattr(self.optimized_pattern_detector, 'stats'):
                self.performance_stats['cache_hits'] += self.optimized_pattern_detector.stats.get('cache_hits', 0)
                self.performance_stats['cache_misses'] += self.optimized_pattern_detector.stats.get('cache_misses', 0)
            
            return {
                'signals': signals,
                'patterns': patterns,
                'processed_data': processed_data
            }
            
        except Exception as e:
            logger.error(f"❌ Error in optimized processing: {e}")
            return {'signals': [], 'patterns': []}
    
    async def _process_with_legacy_system(self, market_data: pd.DataFrame, 
                                        symbols: List[str] = None) -> Dict[str, Any]:
        """Process market data using legacy system"""
        try:
            # Get signals from strategy manager
            strategy_signals = await self.strategy_manager.get_strategy_signals(market_data)
            
            # Get patterns from existing pattern detector
            patterns = await self._get_legacy_patterns(market_data)
            
            return {
                'signals': strategy_signals,
                'patterns': patterns
            }
            
        except Exception as e:
            logger.error(f"❌ Error in legacy processing: {e}")
            return {'signals': [], 'patterns': []}
    
    async def _get_legacy_patterns(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get patterns using legacy pattern detection"""
        try:
            # Import legacy pattern detector
            from ..src.strategies.pattern_detector import PatternDetector
            
            pattern_detector = PatternDetector()
            patterns = []
            
            for symbol in market_data.index.get_level_values(0).unique():
                symbol_data = market_data.loc[symbol]
                if not symbol_data.empty:
                    symbol_patterns = pattern_detector.detect_patterns(symbol_data)
                    for pattern in symbol_patterns:
                        pattern['symbol'] = symbol
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Error getting legacy patterns: {e}")
            return []
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """Run function in thread pool executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate additional metrics
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
            stats['min_processing_time'] = np.min(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['max_processing_time'] = 0.0
            stats['min_processing_time'] = 0.0
        
        # Calculate cache efficiency
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_efficiency'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_efficiency'] = 0.0
        
        # Add optimization status
        stats['optimized_components_available'] = OPTIMIZED_AVAILABLE
        stats['optimized_system_active'] = self.optimized_system is not None
        
        return stats
    
    async def clear_optimization_cache(self):
        """Clear optimization caches"""
        try:
            if self.optimized_pattern_detector:
                self.optimized_pattern_detector.clear_cache()
            
            if self.optimized_data_processor:
                self.optimized_data_processor.clear_cache()
            
            logger.info("🧹 Optimization caches cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing optimization cache: {e}")
    
    async def run_optimization_benchmark(self, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run performance benchmark comparing optimized vs legacy systems"""
        try:
            logger.info("🏁 Running optimization benchmark...")
            
            if test_data is None:
                # Generate test data
                test_data = self._generate_benchmark_data()
            
            # Test optimized system
            optimized_start = datetime.now()
            optimized_results = await self._process_with_optimized_system(test_data)
            optimized_time = (datetime.now() - optimized_start).total_seconds()
            
            # Test legacy system
            legacy_start = datetime.now()
            legacy_results = await self._process_with_legacy_system(test_data)
            legacy_time = (datetime.now() - legacy_start).total_seconds()
            
            # Calculate improvements
            time_improvement = ((legacy_time - optimized_time) / legacy_time * 100) if legacy_time > 0 else 0
            signal_improvement = len(optimized_results['signals']) - len(legacy_results['signals'])
            
            benchmark_results = {
                'optimized_time': optimized_time,
                'legacy_time': legacy_time,
                'time_improvement_percent': time_improvement,
                'optimized_signals': len(optimized_results['signals']),
                'legacy_signals': len(legacy_results['signals']),
                'signal_difference': signal_improvement,
                'optimized_patterns': len(optimized_results['patterns']),
                'legacy_patterns': len(legacy_results['patterns'])
            }
            
            logger.info(f"🏁 Benchmark completed: {time_improvement:.1f}% faster with optimization")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Error running benchmark: {e}")
            return {'error': str(e)}
    
    def _generate_benchmark_data(self) -> pd.DataFrame:
        """Generate test data for benchmarking"""
        # Generate 1000 rows of realistic market data
        dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        base_price = 50000
        returns = np.random.normal(0, 0.02, 1000)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLCV
            open_price = price
            close_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and capabilities"""
        return {
            'optimized_components_available': OPTIMIZED_AVAILABLE,
            'optimized_system_active': self.optimized_system is not None,
            'pattern_detector_optimized': self.optimized_pattern_detector is not None,
            'signal_generator_optimized': self.optimized_signal_generator is not None,
            'data_processor_optimized': self.optimized_data_processor is not None,
            'max_workers': self.max_workers,
            'is_running': self.is_running
        }
