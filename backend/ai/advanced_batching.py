"""
Advanced Batching System for AlphaPulse
Dynamic batch size optimization and adaptive batching strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class AdvancedBatchingSystem:
    """
    Advanced batching system with dynamic batch size optimization,
    adaptive batching strategies, and real-time performance monitoring.
    """
    
    def __init__(self, initial_batch_size: int = 100,
                 max_batch_size: int = 1000,
                 min_batch_size: int = 10,
                 target_latency_ms: float = 50.0,
                 performance_window: int = 100):
        """
        Initialize advanced batching system.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            target_latency_ms: Target latency in milliseconds
            performance_window: Number of batches to consider for optimization
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.target_latency_ms = target_latency_ms
        self.performance_window = performance_window
        
        # Current batch size (dynamically adjusted)
        self.current_batch_size = initial_batch_size
        
        # Performance tracking
        self.batch_times: deque = deque(maxlen=performance_window)
        self.batch_sizes: deque = deque(maxlen=performance_window)
        self.throughput_history: deque = deque(maxlen=performance_window)
        
        # Adaptive batching state
        self.last_optimization_time = datetime.now()
        self.optimization_interval = timedelta(seconds=30)  # Optimize every 30 seconds
        
        # Batch queues by model type
        self.batch_queues: Dict[str, deque] = defaultdict(deque)
        self.batch_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Performance metrics
        self.total_batches_processed = 0
        self.total_predictions = 0
        self.avg_batch_time_ms = 0.0
        self.avg_throughput = 0.0
        
        # Optimization parameters
        self.learning_rate = 0.1  # How quickly to adjust batch size
        self.stability_threshold = 0.05  # Minimum change threshold
        
        logger.info(f"AdvancedBatchingSystem initialized with batch_size={initial_batch_size}")
    
    def add_to_batch(self, model_type: str, input_data: np.ndarray, 
                    metadata: Dict[str, Any] = None) -> str:
        """
        Add input data to a batch queue.
        
        Args:
            model_type: Type of model (e.g., 'pattern', 'regime', 'ensemble')
            input_data: Input data for prediction
            metadata: Additional metadata for the request
            
        Returns:
            Batch ID for tracking
        """
        batch_id = f"{model_type}_{int(time.time() * 1000000)}"
        
        batch_item = {
            'id': batch_id,
            'input_data': input_data,
            'metadata': metadata or {},
            'timestamp': datetime.now(),
            'future': asyncio.Future()
        }
        
        with self.batch_locks[model_type]:
            self.batch_queues[model_type].append(batch_item)
        
        logger.debug(f"Added item {batch_id} to {model_type} batch queue")
        return batch_id
    
    def should_process_batch(self, model_type: str) -> bool:
        """
        Determine if a batch should be processed based on current conditions.
        
        Args:
            model_type: Type of model
            
        Returns:
            True if batch should be processed
        """
        with self.batch_locks[model_type]:
            queue_size = len(self.batch_queues[model_type])
        
        # Process if queue is full or if we have enough items
        if queue_size >= self.current_batch_size:
            return True
        
        # Process if we have a minimum number of items and enough time has passed
        if queue_size >= self.min_batch_size:
            # Check if enough time has passed since last batch
            if self.total_batches_processed > 0:
                avg_batch_time = np.mean(list(self.batch_times)[-10:]) if self.batch_times else 0
                if avg_batch_time > 0:
                    # Process if we're approaching target latency
                    return True
        
        return False
    
    def get_batch(self, model_type: str) -> Tuple[List[Dict], int]:
        """
        Get items from batch queue for processing.
        
        Args:
            model_type: Type of model
            
        Returns:
            Tuple of (batch_items, actual_batch_size)
        """
        with self.batch_locks[model_type]:
            queue = self.batch_queues[model_type]
            batch_size = min(self.current_batch_size, len(queue))
            
            if batch_size == 0:
                return [], 0
            
            batch_items = []
            for _ in range(batch_size):
                if queue:
                    batch_items.append(queue.popleft())
            
            return batch_items, len(batch_items)
    
    def process_batch(self, model_type: str, inference_function, 
                     batch_items: List[Dict]) -> List[Dict]:
        """
        Process a batch of items using the provided inference function.
        
        Args:
            model_type: Type of model
            inference_function: Function to perform inference
            batch_items: List of batch items to process
            
        Returns:
            List of results with predictions
        """
        if not batch_items:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare batch input
            batch_inputs = [item['input_data'] for item in batch_items]
            batch_input_array = np.array(batch_inputs)
            
            # Perform inference
            predictions = inference_function(batch_input_array)
            
            # Calculate batch processing time
            batch_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(batch_time_ms, len(batch_items))
            
            # Prepare results
            results = []
            for i, item in enumerate(batch_items):
                result = {
                    'id': item['id'],
                    'prediction': predictions[i] if isinstance(predictions, np.ndarray) else predictions,
                    'metadata': item['metadata'],
                    'batch_time_ms': batch_time_ms,
                    'batch_size': len(batch_items)
                }
                results.append(result)
            
            logger.debug(f"Processed {model_type} batch: {len(batch_items)} items in {batch_time_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error processing {model_type} batch: {e}")
            # Return error results
            return [{'id': item['id'], 'error': str(e), 'metadata': item['metadata']} 
                   for item in batch_items]
    
    def _update_performance_metrics(self, batch_time_ms: float, batch_size: int):
        """Update performance tracking metrics"""
        self.batch_times.append(batch_time_ms)
        self.batch_sizes.append(batch_size)
        
        # Calculate throughput (predictions per second)
        throughput = (batch_size / batch_time_ms) * 1000 if batch_time_ms > 0 else 0
        self.throughput_history.append(throughput)
        
        # Update totals
        self.total_batches_processed += 1
        self.total_predictions += batch_size
        
        # Update averages
        self.avg_batch_time_ms = np.mean(list(self.batch_times))
        self.avg_throughput = np.mean(list(self.throughput_history))
    
    def optimize_batch_size(self):
        """Dynamically optimize batch size based on performance metrics"""
        current_time = datetime.now()
        
        # Only optimize periodically
        if current_time - self.last_optimization_time < self.optimization_interval:
            return
        
        if len(self.batch_times) < 10:  # Need enough data
            return
        
        # Calculate current performance
        current_avg_time = np.mean(list(self.batch_times)[-10:])
        current_avg_throughput = np.mean(list(self.throughput_history)[-10:])
        
        # Determine if we need to adjust batch size
        if current_avg_time > self.target_latency_ms * 1.1:  # Too slow
            # Reduce batch size
            new_batch_size = max(self.min_batch_size, 
                               int(self.current_batch_size * (1 - self.learning_rate)))
            reason = "reducing latency"
        elif current_avg_time < self.target_latency_ms * 0.9:  # Too fast
            # Increase batch size to improve throughput
            new_batch_size = min(self.max_batch_size, 
                               int(self.current_batch_size * (1 + self.learning_rate)))
            reason = "improving throughput"
        else:
            # Performance is good, no change needed
            return
        
        # Only change if the difference is significant
        if abs(new_batch_size - self.current_batch_size) / self.current_batch_size < self.stability_threshold:
            return
        
        old_batch_size = self.current_batch_size
        self.current_batch_size = new_batch_size
        self.last_optimization_time = current_time
        
        logger.info(f"🔄 Optimized batch size: {old_batch_size} → {new_batch_size} ({reason})")
        logger.info(f"   Current performance: {current_avg_time:.2f}ms avg, {current_avg_throughput:.1f} pred/s")
    
    def get_adaptive_batch_size(self, model_type: str, 
                              current_load: float = 1.0) -> int:
        """
        Get adaptive batch size based on current system load and model type.
        
        Args:
            model_type: Type of model
            current_load: Current system load (0.0 to 1.0)
            
        Returns:
            Recommended batch size
        """
        # Base batch size
        base_size = self.current_batch_size
        
        # Adjust based on model type
        model_adjustments = {
            'pattern': 1.0,      # Standard
            'regime': 1.2,       # Slightly larger (regime models are often faster)
            'ensemble': 0.8,     # Smaller (ensemble models are more complex)
        }
        
        adjustment = model_adjustments.get(model_type, 1.0)
        
        # Adjust based on system load
        if current_load > 0.8:  # High load
            load_adjustment = 0.7  # Reduce batch size
        elif current_load < 0.3:  # Low load
            load_adjustment = 1.3  # Increase batch size
        else:
            load_adjustment = 1.0
        
        # Calculate final batch size
        adaptive_size = int(base_size * adjustment * load_adjustment)
        
        # Ensure within bounds
        adaptive_size = max(self.min_batch_size, min(self.max_batch_size, adaptive_size))
        
        return adaptive_size
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'current_batch_size': self.current_batch_size,
            'total_batches_processed': self.total_batches_processed,
            'total_predictions': self.total_predictions,
            'avg_batch_time_ms': self.avg_batch_time_ms,
            'avg_throughput': self.avg_throughput,
            'queue_sizes': {},
            'recent_performance': {}
        }
        
        # Queue sizes
        for model_type in self.batch_queues:
            with self.batch_locks[model_type]:
                stats['queue_sizes'][model_type] = len(self.batch_queues[model_type])
        
        # Recent performance (last 10 batches)
        if self.batch_times:
            recent_times = list(self.batch_times)[-10:]
            recent_throughput = list(self.throughput_history)[-10:]
            
            stats['recent_performance'] = {
                'avg_batch_time_ms': np.mean(recent_times),
                'min_batch_time_ms': np.min(recent_times),
                'max_batch_time_ms': np.max(recent_times),
                'avg_throughput': np.mean(recent_throughput),
                'throughput_std': np.std(recent_throughput)
            }
        
        return stats
    
    def reset_performance_tracking(self):
        """Reset performance tracking data"""
        self.batch_times.clear()
        self.batch_sizes.clear()
        self.throughput_history.clear()
        self.total_batches_processed = 0
        self.total_predictions = 0
        self.avg_batch_time_ms = 0.0
        self.avg_throughput = 0.0
        
        logger.info("🔄 Performance tracking reset")
    
    def set_target_latency(self, target_latency_ms: float):
        """Update target latency"""
        self.target_latency_ms = target_latency_ms
        logger.info(f"🎯 Target latency updated: {target_latency_ms}ms")
    
    def set_batch_size_bounds(self, min_size: int, max_size: int):
        """Update batch size bounds"""
        self.min_batch_size = min_size
        self.max_batch_size = max_size
        
        # Adjust current batch size if needed
        if self.current_batch_size < min_size:
            self.current_batch_size = min_size
        elif self.current_batch_size > max_size:
            self.current_batch_size = max_size
        
        logger.info(f"📏 Batch size bounds updated: {min_size} - {max_size}")


# Global advanced batching system instance
advanced_batching_system = AdvancedBatchingSystem()
