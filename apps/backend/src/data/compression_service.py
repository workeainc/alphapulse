#!/usr/bin/env python3
"""
Week 7.2 Phase 2: In-Memory Data Compression & Caching

This service provides advanced compression techniques for time-series market data:
- Delta encoding for price and volume data
- Multi-level compression strategies
- Memory optimization for high-frequency data
- Compression analytics and monitoring
- Adaptive compression based on data patterns

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
import zlib
import struct
import pickle
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Types of compression algorithms"""
    DELTA = "delta"
    RUN_LENGTH = "run_length"
    DICTIONARY = "dictionary"
    QUANTIZATION = "quantization"
    HYBRID = "hybrid"


@dataclass
class CompressionResult:
    """Result of compression operation"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_type: CompressionType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompressionConfig:
    """Configuration for compression service"""
    enable_delta_encoding: bool = True
    enable_run_length: bool = True
    enable_dictionary: bool = True
    enable_quantization: bool = True
    delta_threshold: float = 0.0001  # Minimum delta to encode
    quantization_precision: int = 6  # Decimal places for quantization
    run_length_threshold: int = 3  # Minimum consecutive values for RLE
    compression_level: int = 6  # Zlib compression level (0-9)
    adaptive_compression: bool = True  # Enable adaptive compression
    memory_threshold: float = 0.8  # Memory usage threshold for compression
    batch_size: int = 1000  # Batch size for processing


class DeltaEncoder:
    """Delta encoding for time-series data compression"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.last_values: Dict[str, Dict[str, float]] = {}
        self.delta_buffers: Dict[str, Dict[str, deque]] = {}
        self.stats = {
            'total_encodings': 0,
            'total_deltas': 0,
            'average_delta': 0.0,
            'compression_ratios': []
        }
    
    def encode_price_data(self, symbol: str, prices: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Encode price data using delta encoding"""
        if not prices:
            return [], {}
        
        # Initialize if first time
        if symbol not in self.last_values:
            self.last_values[symbol] = {'price': prices[0]}
            self.delta_buffers[symbol] = {'price': deque(maxlen=1000)}
        
        # Calculate deltas
        deltas = []
        last_price = self.last_values[symbol]['price']
        
        for price in prices:
            delta = price - last_price
            deltas.append(delta)
            last_price = price
        
        # Update last value
        self.last_values[symbol]['price'] = last_price
        
        # Store deltas for analysis
        self.delta_buffers[symbol]['price'].extend(deltas)
        
        # Calculate statistics
        self.stats['total_encodings'] += 1
        self.stats['total_deltas'] += len(deltas)
        
        # Calculate compression ratio (simplified)
        original_size = len(prices) * 8  # 8 bytes per float
        compressed_size = len(deltas) * 8  # Still 8 bytes, but deltas are smaller
        compression_ratio = 1.0 - (compressed_size / original_size)
        
        self.stats['compression_ratios'].append(compression_ratio)
        if len(self.stats['compression_ratios']) > 100:
            self.stats['compression_ratios'] = self.stats['compression_ratios'][-100:]
        
        metadata = {
            'first_price': prices[0],
            'last_price': prices[-1],
            'delta_count': len(deltas),
            'average_delta': np.mean(deltas),
            'delta_std': np.std(deltas),
            'compression_ratio': compression_ratio
        }
        
        return deltas, metadata
    
    def decode_price_data(self, symbol: str, first_price: float, deltas: List[float]) -> List[float]:
        """Decode price data from delta encoding"""
        if not deltas:
            return []
        
        prices = [first_price]
        current_price = first_price
        
        for delta in deltas:
            current_price += delta
            prices.append(current_price)
        
        return prices
    
    def get_encoding_statistics(self) -> Dict[str, Any]:
        """Get delta encoding statistics"""
        return {
            'total_encodings': self.stats['total_encodings'],
            'total_deltas': self.stats['total_deltas'],
            'average_compression_ratio': np.mean(self.stats['compression_ratios']) if self.stats['compression_ratios'] else 0.0,
            'compression_ratios': self.stats['compression_ratios'][-10:] if self.stats['compression_ratios'] else []
        }


class RunLengthEncoder:
    """Run-length encoding for repeated values"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.stats = {
            'total_encodings': 0,
            'total_runs': 0,
            'average_run_length': 0.0
        }
    
    def encode(self, data: List[Any]) -> Tuple[List[Tuple[Any, int]], Dict[str, Any]]:
        """Encode data using run-length encoding"""
        if not data:
            return [], {}
        
        encoded = []
        current_value = data[0]
        current_count = 1
        
        for value in data[1:]:
            if value == current_value:
                current_count += 1
            else:
                if current_count >= self.config.run_length_threshold:
                    encoded.append((current_value, current_count))
                else:
                    # Add individual values for short runs
                    for _ in range(current_count):
                        encoded.append((current_value, 1))
                
                current_value = value
                current_count = 1
        
        # Handle last run
        if current_count >= self.config.run_length_threshold:
            encoded.append((current_value, current_count))
        else:
            for _ in range(current_count):
                encoded.append((current_value, 1))
        
        # Calculate statistics
        self.stats['total_encodings'] += 1
        self.stats['total_runs'] += len(encoded)
        
        # Calculate compression ratio
        original_size = len(data)
        compressed_size = len(encoded)
        compression_ratio = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0
        
        metadata = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'run_count': len(encoded)
        }
        
        return encoded, metadata
    
    def decode(self, encoded_data: List[Tuple[Any, int]]) -> List[Any]:
        """Decode run-length encoded data"""
        decoded = []
        for value, count in encoded_data:
            decoded.extend([value] * count)
        return decoded


class QuantizationEncoder:
    """Quantization encoding for floating-point precision reduction"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.stats = {
            'total_quantizations': 0,
            'total_values': 0,
            'average_precision_loss': 0.0
        }
    
    def encode(self, data: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Encode data using quantization"""
        if not data:
            return [], {}
        
        # Quantize to specified precision
        precision_factor = 10 ** self.config.quantization_precision
        quantized = [round(value * precision_factor) / precision_factor for value in data]
        
        # Calculate precision loss
        precision_loss = np.mean([abs(orig - quant) for orig, quant in zip(data, quantized)])
        
        # Calculate compression ratio (simplified - in practice, this would reduce memory)
        original_size = len(data) * 8  # 8 bytes per float
        compressed_size = len(quantized) * 8  # Still 8 bytes, but with reduced precision
        
        # Calculate statistics
        self.stats['total_quantizations'] += 1
        self.stats['total_values'] += len(data)
        self.stats['average_precision_loss'] = precision_loss
        
        metadata = {
            'original_precision': 'full',
            'quantized_precision': self.config.quantization_precision,
            'precision_loss': precision_loss,
            'compression_ratio': 0.0  # No size reduction, only precision reduction
        }
        
        return quantized, metadata


class CompressionService:
    """
    Main compression service for market data
    
    This service provides multiple compression strategies and automatically
    selects the best method based on data characteristics and memory constraints.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize the compression service"""
        self.config = config or CompressionConfig()
        
        # Initialize compression components
        self.delta_encoder = DeltaEncoder(self.config)
        self.run_length_encoder = RunLengthEncoder(self.config)
        self.quantization_encoder = QuantizationEncoder(self.config)
        
        # Compression history and statistics
        self.compression_history: deque = deque(maxlen=1000)
        self.memory_usage: float = 0.0
        self.callbacks: List[Callable] = []
        
        # Service statistics
        self.stats = {
            'total_compressions': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'average_compression_ratio': 0.0,
            'compression_types_used': {},
            'memory_savings': 0.0,
            'last_compression': None
        }
        
        logger.info("✅ Compression Service initialized")
    
    async def initialize(self):
        """Initialize the service"""
        logger.info("🚀 Initializing Compression Service...")
        
        # Initialize memory monitoring
        asyncio.create_task(self._monitor_memory_usage())
        
        logger.info("✅ Compression Service ready")
    
    async def _monitor_memory_usage(self):
        """Monitor memory usage for adaptive compression"""
        while True:
            try:
                # Simple memory usage estimation
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # GB
                
                # Log memory usage every 5 minutes
                if self.stats['total_compressions'] % 100 == 0:
                    logger.info(f"Memory usage: {self.memory_usage:.2f} GB")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except ImportError:
                # psutil not available, use mock data
                self.memory_usage = 0.5  # Mock 500MB
                await asyncio.sleep(300)
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                await asyncio.sleep(300)
    
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
    
    async def compress_market_data(self, symbol: str, data_type: str, 
                                 data: List[Any]) -> CompressionResult:
        """
        Compress market data using the best available method
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('price', 'volume', 'order_book', etc.)
            data: List of data values to compress
            
        Returns:
            CompressionResult with compression details
        """
        if not data:
            return CompressionResult(
                original_size=0,
                compressed_size=0,
                compression_ratio=0.0,
                compression_type=CompressionType.DELTA
            )
        
        # Select compression method based on data type and characteristics
        compression_type = self._select_compression_method(data_type, data)
        
        # Apply compression
        if compression_type == CompressionType.DELTA:
            compressed_data, metadata = self._apply_delta_compression(symbol, data_type, data)
        elif compression_type == CompressionType.RUN_LENGTH:
            compressed_data, metadata = self._apply_run_length_compression(data)
        elif compression_type == CompressionType.QUANTIZATION:
            compressed_data, metadata = self._apply_quantization_compression(data)
        elif compression_type == CompressionType.HYBRID:
            compressed_data, metadata = await self._apply_hybrid_compression(symbol, data_type, data)
        else:
            # No compression
            compressed_data, metadata = data, {}
        
        # Calculate compression results
        original_size = len(data) * 8  # Approximate size in bytes
        compressed_size = self._estimate_compressed_size(compressed_data, compression_type)
        compression_ratio = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0
        
        # Create result
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_type=compression_type,
            metadata=metadata
        )
        
        # Update statistics
        self._update_statistics(result)
        
        # Store in history
        self.compression_history.append(result)
        
        # Trigger callbacks
        self._trigger_callbacks(result)
        
        logger.info(f"✅ Compressed {data_type} data for {symbol}: {compression_ratio:.2%} reduction")
        
        return result
    
    def _select_compression_method(self, data_type: str, data: List[Any]) -> CompressionType:
        """Select the best compression method based on data characteristics"""
        if not data:
            return CompressionType.DELTA
        
        # Analyze data characteristics
        data_array = np.array(data)
        
        # Check for repeated values (good for RLE)
        unique_ratio = len(np.unique(data_array)) / len(data_array)
        if unique_ratio < 0.3 and self.config.enable_run_length:
            return CompressionType.RUN_LENGTH
        
        # Check for small deltas (good for delta encoding)
        if len(data_array) > 1:
            deltas = np.diff(data_array)
            avg_delta = np.mean(np.abs(deltas))
            if avg_delta < self.config.delta_threshold and self.config.enable_delta_encoding:
                return CompressionType.DELTA
        
        # Check for high precision (good for quantization)
        if data_type in ['price', 'volume'] and self.config.enable_quantization:
            # Check if values have many decimal places
            decimal_places = self._count_decimal_places(data_array)
            if decimal_places > self.config.quantization_precision:
                return CompressionType.QUANTIZATION
        
        # Use hybrid if multiple methods are good
        if (self.config.enable_delta_encoding and self.config.enable_run_length and 
            self.config.adaptive_compression):
            return CompressionType.HYBRID
        
        # Default to delta encoding
        return CompressionType.DELTA
    
    def _count_decimal_places(self, data: np.ndarray) -> int:
        """Count the maximum number of decimal places in the data"""
        max_decimal_places = 0
        for value in data:
            if isinstance(value, (int, float)):
                str_value = str(float(value))
                if '.' in str_value:
                    decimal_part = str_value.split('.')[1]
                    max_decimal_places = max(max_decimal_places, len(decimal_part))
        return max_decimal_places
    
    def _apply_delta_compression(self, symbol: str, data_type: str, data: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Apply delta compression to data"""
        if data_type == 'price':
            return self.delta_encoder.encode_price_data(symbol, data)
        else:
            # For other numeric types, use simple delta encoding
            deltas = [data[0]] + [data[i] - data[i-1] for i in range(1, len(data))]
            metadata = {
                'first_value': data[0],
                'delta_count': len(deltas),
                'average_delta': np.mean(deltas[1:]) if len(deltas) > 1 else 0.0
            }
            return deltas, metadata
    
    def _apply_run_length_compression(self, data: List[Any]) -> Tuple[List[Tuple[Any, int]], Dict[str, Any]]:
        """Apply run-length encoding to data"""
        return self.run_length_encoder.encode(data)
    
    def _apply_quantization_compression(self, data: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Apply quantization compression to data"""
        return self.quantization_encoder.encode(data)
    
    async def _apply_hybrid_compression(self, symbol: str, data_type: str, data: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """Apply multiple compression methods in sequence"""
        # First apply delta encoding
        if data_type in ['price', 'volume']:
            delta_data, delta_metadata = self._apply_delta_compression(symbol, data_type, data)
        else:
            delta_data, delta_metadata = data, {}
        
        # Then apply run-length encoding if beneficial
        if len(delta_data) > 1:
            rle_data, rle_metadata = self._apply_run_length_compression(delta_data)
            
            # Choose the better compression
            if rle_metadata.get('compression_ratio', 0) > 0.1:  # 10% improvement
                metadata = {**delta_metadata, **rle_metadata, 'method': 'hybrid_delta_rle'}
                return rle_data, metadata
        
        metadata = {**delta_metadata, 'method': 'hybrid_delta_only'}
        return delta_data, metadata
    
    def _estimate_compressed_size(self, compressed_data: List[Any], compression_type: CompressionType) -> int:
        """Estimate the size of compressed data in bytes"""
        if not compressed_data:
            return 0
        
        if compression_type == CompressionType.RUN_LENGTH:
            # Each run is (value, count) pair
            return len(compressed_data) * 16  # Approximate size
        elif compression_type == CompressionType.DELTA:
            # Deltas are still floats
            return len(compressed_data) * 8
        elif compression_type == CompressionType.QUANTIZATION:
            # Quantized values are still floats
            return len(compressed_data) * 8
        else:
            # Default estimate
            return len(compressed_data) * 8
    
    def _update_statistics(self, result: CompressionResult):
        """Update service statistics"""
        self.stats['total_compressions'] += 1
        self.stats['total_original_size'] += result.original_size
        self.stats['total_compressed_size'] += result.compressed_size
        self.stats['last_compression'] = result.timestamp
        
        # Update compression type usage
        comp_type = result.compression_type.value
        if comp_type not in self.stats['compression_types_used']:
            self.stats['compression_types_used'][comp_type] = 0
        self.stats['compression_types_used'][comp_type] += 1
        
        # Calculate average compression ratio
        if self.stats['total_compressions'] > 0:
            self.stats['average_compression_ratio'] = (
                self.stats['total_compressed_size'] / self.stats['total_original_size']
            )
        
        # Calculate memory savings
        self.stats['memory_savings'] = (
            self.stats['total_original_size'] - self.stats['total_compressed_size']
        ) / (1024 * 1024)  # MB
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'total_compressions': self.stats['total_compressions'],
            'total_original_size_mb': self.stats['total_original_size'] / (1024 * 1024),
            'total_compressed_size_mb': self.stats['total_compressed_size'] / (1024 * 1024),
            'average_compression_ratio': self.stats['average_compression_ratio'],
            'compression_types_used': self.stats['compression_types_used'],
            'memory_savings_mb': self.stats['memory_savings'],
            'last_compression': self.stats['last_compression'],
            'memory_usage_gb': self.memory_usage,
            'compression_history_size': len(self.compression_history)
        }
    
    def get_compression_history(self, limit: int = 100) -> List[CompressionResult]:
        """Get recent compression history"""
        return list(self.compression_history)[-limit:]
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Compression Service...")
        
        # Clear history and statistics
        self.compression_history.clear()
        self.callbacks.clear()
        
        logger.info("✅ Compression Service closed")


# Example usage and testing
async def main():
    """Example usage of the Compression Service"""
    # Create service with custom config
    config = CompressionConfig(
        enable_delta_encoding=True,
        enable_run_length=True,
        enable_quantization=True,
        adaptive_compression=True,
        delta_threshold=0.001,
        quantization_precision=4
    )
    
    service = CompressionService(config)
    await service.initialize()
    
    # Add a simple callback
    def compression_callback(result: CompressionResult):
        print(f"📦 Compression completed: {result.compression_ratio:.2%} reduction")
    
    service.add_callback(compression_callback)
    
    # Test with different data types
    print("🧪 Testing Compression Service...")
    
    # Test price data (good for delta encoding)
    print("\n📈 Testing price data compression...")
    prices = [50000 + (i * 0.01) for i in range(1000)]
    result = await service.compress_market_data("BTCUSDT", "price", prices)
    print(f"  Price compression: {result.compression_ratio:.2%} reduction")
    print(f"  Method: {result.compression_type.value}")
    
    # Test volume data (good for run-length encoding)
    print("\n📊 Testing volume data compression...")
    volumes = [100000] * 500 + [200000] * 300 + [150000] * 200
    result = await service.compress_market_data("BTCUSDT", "volume", volumes)
    print(f"  Volume compression: {result.compression_ratio:.2%} reduction")
    print(f"  Method: {result.compression_type.value}")
    
    # Test order book data (mixed characteristics)
    print("\n📚 Testing order book data compression...")
    spreads = [0.001 + (i * 0.0001) for i in range(1000)]
    result = await service.compress_market_data("BTCUSDT", "spread", spreads)
    print(f"  Spread compression: {result.compression_ratio:.2%} reduction")
    print(f"  Method: {result.compression_type.value}")
    
    # Print statistics
    print(f"\n📊 Service Statistics:")
    stats = service.get_service_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print compression history
    print(f"\n📋 Recent Compression History:")
    history = service.get_compression_history(limit=5)
    for i, result in enumerate(history):
        print(f"  {i+1}. {result.compression_type.value}: {result.compression_ratio:.2%} reduction")
    
    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
