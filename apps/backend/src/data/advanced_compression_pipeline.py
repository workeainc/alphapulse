#!/usr/bin/env python3
"""
Week 7.2 Phase 2: Advanced Compression Pipeline

This pipeline integrates multiple compression strategies and provides:
- Intelligent compression method selection
- Batch processing for high-volume data
- Compression analytics and monitoring
- Memory optimization and adaptive compression
- Integration with existing data pipelines

Author: AlphaPulse Team
Date: 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import json
import time

# Import our compression service
from .compression_service import CompressionService, CompressionConfig, CompressionResult, CompressionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for the compression pipeline"""
    total_data_processed: int = 0
    total_compression_time: float = 0.0
    average_compression_time: float = 0.0
    total_memory_saved: float = 0.0
    compression_efficiency: float = 0.0
    last_processed: Optional[datetime] = None


@dataclass
class BatchCompressionConfig:
    """Configuration for batch compression processing"""
    batch_size: int = 1000
    max_batch_delay: float = 1.0  # seconds
    enable_parallel_processing: bool = True
    max_workers: int = 4
    compression_threshold: float = 0.1  # Minimum compression ratio to apply
    memory_pressure_threshold: float = 0.8  # Memory usage threshold for aggressive compression


class AdvancedCompressionPipeline:
    """
    Advanced compression pipeline for market data
    
    This pipeline provides intelligent compression method selection,
    batch processing, and adaptive compression based on system resources.
    """
    
    def __init__(self, config: Optional[BatchCompressionConfig] = None):
        """Initialize the advanced compression pipeline"""
        self.config = config or BatchCompressionConfig()
        
        # Initialize compression service
        compression_config = CompressionConfig(
            enable_delta_encoding=True,
            enable_run_length=True,
            enable_quantization=True,
            adaptive_compression=True,
            delta_threshold=0.0001,
            quantization_precision=6
        )
        self.compression_service = CompressionService(compression_config)
        
        # Data buffers for batch processing
        self.data_buffers: Dict[str, Dict[str, deque]] = {}
        self.batch_timers: Dict[str, Dict[str, float]] = {}
        
        # Pipeline statistics and metrics
        self.metrics = PipelineMetrics()
        self.callbacks: List[Callable] = []
        
        # Performance monitoring
        self.processing_times: deque = deque(maxlen=1000)
        self.compression_ratios: deque = deque(maxlen=1000)
        
        # Memory pressure monitoring
        self.memory_pressure_level: float = 0.0
        self.last_memory_check: datetime = datetime.now()
        
        logger.info("✅ Advanced Compression Pipeline initialized")
    
    async def initialize(self):
        """Initialize the pipeline and all components"""
        logger.info("🚀 Initializing Advanced Compression Pipeline...")
        
        # Initialize compression service
        await self.compression_service.initialize()
        
        # Set up compression callbacks
        self.compression_service.add_callback(self._on_compression_completed)
        
        # Initialize memory monitoring
        asyncio.create_task(self._monitor_memory_pressure())
        
        logger.info("✅ Advanced Compression Pipeline ready")
    
    async def _monitor_memory_pressure(self):
        """Monitor memory pressure for adaptive compression"""
        while True:
            try:
                # Check memory usage
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_percent = memory_info.rss / (1024 * 1024 * 1024)  # GB
                
                # Calculate memory pressure (0.0 to 1.0)
                self.memory_pressure_level = min(memory_percent / 8.0, 1.0)  # Assume 8GB is max
                
                # Log memory pressure every minute
                if self.metrics.total_data_processed % 1000 == 0:
                    logger.info(f"Memory pressure: {self.memory_pressure_level:.2f} ({memory_percent:.2f} GB)")
                
                self.last_memory_check = datetime.now()
                await asyncio.sleep(60)  # Check every minute
                
            except ImportError:
                # psutil not available, use mock data
                self.memory_pressure_level = 0.3  # Mock 30% pressure
                await asyncio.sleep(60)
            except Exception as e:
                logger.warning(f"Memory pressure monitoring error: {e}")
                await asyncio.sleep(60)
    
    def add_callback(self, callback: Callable):
        """Add a callback for compression events"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            logger.info(f"✅ Added compression callback: {callback.__name__}")
    
    def _trigger_callbacks(self, result: CompressionResult):
        """Trigger all registered callbacks with compression result"""
        for callback in self.callbacks:
            try:
                asyncio.create_task(callback(result))
            except Exception as e:
                logger.error(f"❌ Error in compression callback {callback.__name__}: {e}")
    
    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a new symbol"""
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = {
                'price': deque(maxlen=self.config.batch_size * 2),
                'volume': deque(maxlen=self.config.batch_size * 2),
                'order_book': deque(maxlen=self.config.batch_size * 2),
                'timestamp': deque(maxlen=self.config.batch_size * 2)
            }
        
        if symbol not in self.batch_timers:
            self.batch_timers[symbol] = {
                'price': time.time(),
                'volume': time.time(),
                'order_book': time.time()
            }
    
    async def process_market_data(self, symbol: str, data_type: str, 
                                data: Any, timestamp: Optional[datetime] = None) -> Optional[CompressionResult]:
        """
        Process market data for compression
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('price', 'volume', 'order_book')
            data: Data value or list of values
            timestamp: Timestamp of the data
            
        Returns:
            CompressionResult if batch is processed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize buffers if needed
        self._initialize_symbol_buffers(symbol)
        
        # Add data to buffer
        if isinstance(data, (list, np.ndarray)):
            self.data_buffers[symbol][data_type].extend(data)
        else:
            self.data_buffers[symbol][data_type].append(data)
        
        # Add timestamp
        self.data_buffers[symbol]['timestamp'].append(timestamp)
        
        # Check if we should process the batch
        should_process = self._should_process_batch(symbol, data_type)
        
        if should_process:
            return await self._process_batch(symbol, data_type)
        
        return None
    
    def _should_process_batch(self, symbol: str, data_type: str) -> bool:
        """Determine if a batch should be processed"""
        buffer = self.data_buffers[symbol][data_type]
        timer = self.batch_timers[symbol][data_type]
        current_time = time.time()
        
        # Check if buffer is full
        if len(buffer) >= self.config.batch_size:
            return True
        
        # Check if enough time has passed
        if current_time - timer >= self.config.max_batch_delay:
            return True
        
        # Check memory pressure (process immediately if high pressure)
        if self.memory_pressure_level > self.config.memory_pressure_threshold:
            return True
        
        return False
    
    async def _process_batch(self, symbol: str, data_type: str) -> CompressionResult:
        """Process a batch of data for compression"""
        start_time = time.time()
        
        # Get data from buffer
        buffer = self.data_buffers[symbol][data_type]
        if not buffer:
            return CompressionResult(
                original_size=0,
                compressed_size=0,
                compression_ratio=0.0,
                compression_type=CompressionType.DELTA
            )
        
        # Convert to list and clear buffer
        data_list = list(buffer)
        buffer.clear()
        
        # Reset timer
        self.batch_timers[symbol][data_type] = time.time()
        
        # Apply compression
        try:
            result = await self.compression_service.compress_market_data(symbol, data_type, data_list)
            
            # Update metrics
            self._update_metrics(result, time.time() - start_time)
            
            logger.info(f"✅ Processed batch: {symbol} {data_type} - {len(data_list)} items, {result.compression_ratio:.2%} reduction")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing batch {symbol} {data_type}: {e}")
            
            # Return empty result on error
            return CompressionResult(
                original_size=len(data_list) * 8,
                compressed_size=len(data_list) * 8,
                compression_ratio=0.0,
                compression_type=CompressionType.DELTA
            )
    
    def _update_metrics(self, result: CompressionResult, processing_time: float):
        """Update pipeline metrics"""
        self.metrics.total_data_processed += 1
        self.metrics.total_compression_time += processing_time
        self.metrics.total_memory_saved += (result.original_size - result.compressed_size) / (1024 * 1024)  # MB
        self.metrics.last_processed = datetime.now()
        
        # Update average processing time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 0:
            self.metrics.average_compression_time = np.mean(self.processing_times)
        
        # Update compression efficiency
        self.compression_ratios.append(result.compression_ratio)
        if len(self.compression_ratios) > 0:
            self.metrics.compression_efficiency = np.mean(self.compression_ratios)
    
    def _on_compression_completed(self, result: CompressionResult):
        """Callback when compression is completed"""
        # Trigger pipeline callbacks
        self._trigger_callbacks(result)
    
    async def force_process_batch(self, symbol: str, data_type: str) -> Optional[CompressionResult]:
        """Force processing of a batch regardless of size or timing"""
        if symbol not in self.data_buffers or data_type not in self.data_buffers[symbol]:
            return None
        
        buffer = self.data_buffers[symbol][data_type]
        if not buffer:
            return None
        
        return await self._process_batch(symbol, data_type)
    
    async def process_all_pending_batches(self) -> List[CompressionResult]:
        """Process all pending batches across all symbols and data types"""
        results = []
        
        for symbol in self.data_buffers.keys():
            for data_type in ['price', 'volume', 'order_book']:
                if data_type in self.data_buffers[symbol]:
                    buffer = self.data_buffers[symbol][data_type]
                    if len(buffer) > 0:
                        result = await self._process_batch(symbol, data_type)
                        if result:
                            results.append(result)
        
        return results
    
    def get_buffer_status(self) -> Dict[str, Dict[str, int]]:
        """Get current buffer status for all symbols"""
        status = {}
        for symbol in self.data_buffers.keys():
            status[symbol] = {}
            for data_type in ['price', 'volume', 'order_book']:
                if data_type in self.data_buffers[symbol]:
                    status[symbol][data_type] = len(self.data_buffers[symbol][data_type])
                else:
                    status[symbol][data_type] = 0
        return status
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics"""
        return {
            'total_data_processed': self.metrics.total_data_processed,
            'total_compression_time': self.metrics.total_compression_time,
            'average_compression_time': self.metrics.average_compression_time,
            'total_memory_saved_mb': self.metrics.total_memory_saved,
            'compression_efficiency': self.metrics.compression_efficiency,
            'last_processed': self.metrics.last_processed,
            'memory_pressure_level': self.memory_pressure_level,
            'last_memory_check': self.last_memory_check,
            'buffer_status': self.get_buffer_status(),
            'processing_times_count': len(self.processing_times),
            'compression_ratios_count': len(self.compression_ratios)
        }
    
    def get_compression_service_statistics(self) -> Dict[str, Any]:
        """Get statistics from the underlying compression service"""
        return self.compression_service.get_service_statistics()
    
    async def optimize_compression_settings(self):
        """Optimize compression settings based on performance data"""
        if len(self.compression_ratios) < 10:
            return  # Need more data for optimization
        
        # Analyze recent compression performance
        recent_ratios = list(self.compression_ratios)[-50:]
        avg_ratio = np.mean(recent_ratios)
        
        # Adjust compression settings based on performance
        if avg_ratio < 0.1:  # Less than 10% compression
            # Increase compression aggressiveness
            self.compression_service.config.delta_threshold *= 0.9
            self.compression_service.config.quantization_precision = max(4, self.compression_service.config.quantization_precision - 1)
            logger.info(f"🔄 Optimizing compression settings: delta_threshold={self.compression_service.config.delta_threshold:.6f}, precision={self.compression_service.config.quantization_precision}")
        
        elif avg_ratio > 0.3:  # More than 30% compression
            # Decrease compression aggressiveness for better quality
            self.compression_service.config.delta_threshold *= 1.1
            self.compression_service.config.quantization_precision = min(8, self.compression_service.config.quantization_precision + 1)
            logger.info(f"🔄 Optimizing compression settings: delta_threshold={self.compression_service.config.delta_threshold:.6f}, precision={self.compression_service.config.quantization_precision}")
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Advanced Compression Pipeline...")
        
        # Process any remaining batches
        await self.process_all_pending_batches()
        
        # Close compression service
        await self.compression_service.close()
        
        # Clear buffers and data
        self.data_buffers.clear()
        self.batch_timers.clear()
        self.callbacks.clear()
        
        logger.info("✅ Advanced Compression Pipeline closed")


# Example usage and testing
async def main():
    """Example usage of the Advanced Compression Pipeline"""
    # Create pipeline with custom config
    config = BatchCompressionConfig(
        batch_size=500,
        max_batch_delay=0.5,
        enable_parallel_processing=True,
        max_workers=2,
        compression_threshold=0.05,
        memory_pressure_threshold=0.7
    )
    
    pipeline = AdvancedCompressionPipeline(config)
    await pipeline.initialize()
    
    # Add a simple callback
    def compression_callback(result: CompressionResult):
        print(f"📦 Pipeline compression: {result.compression_ratio:.2%} reduction")
    
    pipeline.add_callback(compression_callback)
    
    # Test with simulated market data
    print("🧪 Testing Advanced Compression Pipeline...")
    
    # Simulate continuous price updates
    print("\n📈 Testing continuous price updates...")
    for i in range(2000):
        price = 50000 + (i * 0.01) + (np.random.random() - 0.5) * 10
        result = await pipeline.process_market_data("BTCUSDT", "price", price)
        if result:
            print(f"  Batch processed: {result.compression_ratio:.2%} reduction")
    
    # Simulate volume updates
    print("\n📊 Testing volume updates...")
    for i in range(1000):
        volume = 100000 + (np.random.random() * 50000)
        result = await pipeline.process_market_data("BTCUSDT", "volume", volume)
        if result:
            print(f"  Volume batch: {result.compression_ratio:.2%} reduction")
    
    # Force process remaining batches
    print("\n🔄 Processing remaining batches...")
    results = await pipeline.process_all_pending_batches()
    print(f"  Processed {len(results)} remaining batches")
    
    # Print pipeline metrics
    print(f"\n📊 Pipeline Metrics:")
    metrics = pipeline.get_pipeline_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Print compression service statistics
    print(f"\n📊 Compression Service Statistics:")
    stats = pipeline.get_compression_service_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Optimize compression settings
    print(f"\n🔧 Optimizing compression settings...")
    await pipeline.optimize_compression_settings()
    
    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
