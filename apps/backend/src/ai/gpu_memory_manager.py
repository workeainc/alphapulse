"""
GPU Memory Manager for AlphaPulse
Optimized GPU memory allocation, deallocation, and management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import time
import gc
import threading
from collections import defaultdict, deque
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. Install with: pip install pynvml")

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """
    GPU memory manager for optimized memory allocation, deallocation,
    and automatic garbage collection.
    """
    
    def __init__(self, enable_monitoring: bool = True,
                 memory_threshold: float = 0.8,  # 80% memory usage threshold
                 cleanup_interval: int = 60,  # Cleanup every 60 seconds
                 enable_auto_cleanup: bool = True):
        """
        Initialize GPU memory manager.
        
        Args:
            enable_monitoring: Enable GPU memory monitoring
            memory_threshold: Memory usage threshold for cleanup (0.0 to 1.0)
            cleanup_interval: Interval between cleanups in seconds
            enable_auto_cleanup: Enable automatic memory cleanup
        """
        self.enable_monitoring = enable_monitoring
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # Memory tracking
        self.memory_usage_history: deque = deque(maxlen=1000)
        self.memory_allocations: Dict[str, Dict] = {}
        self.memory_pool: Dict[str, Any] = {}
        
        # Performance metrics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.total_cleanups = 0
        self.last_cleanup_time = datetime.now()
        
        # Threading
        self.lock = threading.Lock()
        self.cleanup_thread = None
        self.running = False
        
        # Initialize NVML if available
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"✅ NVML initialized with {self.device_count} GPU(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
        
        # Start monitoring if enabled
        if self.enable_monitoring and self.enable_auto_cleanup:
            self.start_monitoring()
        
        logger.info(f"GPUMemoryManager initialized (monitoring: {enable_monitoring})")
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get GPU memory information.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory information
        """
        if not self.nvml_available:
            return {
                'total_memory_mb': 0,
                'used_memory_mb': 0,
                'free_memory_mb': 0,
                'memory_usage_percent': 0,
                'available': False
            }
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_mb = memory_info.total / (1024 * 1024)
            used_mb = memory_info.used / (1024 * 1024)
            free_mb = memory_info.free / (1024 * 1024)
            usage_percent = (memory_info.used / memory_info.total) * 100
            
            return {
                'total_memory_mb': total_mb,
                'used_memory_mb': used_mb,
                'free_memory_mb': free_mb,
                'memory_usage_percent': usage_percent,
                'available': True
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {
                'total_memory_mb': 0,
                'used_memory_mb': 0,
                'free_memory_mb': 0,
                'memory_usage_percent': 0,
                'available': False
            }
    
    def allocate_memory(self, allocation_id: str, size_bytes: int,
                       allocation_type: str = "tensor") -> bool:
        """
        Track memory allocation.
        
        Args:
            allocation_id: Unique identifier for the allocation
            size_bytes: Size of allocation in bytes
            allocation_type: Type of allocation (tensor, model, cache, etc.)
            
        Returns:
            True if allocation tracked successfully
        """
        with self.lock:
            self.memory_allocations[allocation_id] = {
                'size_bytes': size_bytes,
                'size_mb': size_bytes / (1024 * 1024),
                'type': allocation_type,
                'allocated_at': datetime.now(),
                'last_accessed': datetime.now()
            }
            
            self.total_allocations += 1
            
            logger.debug(f"Tracked memory allocation: {allocation_id} ({size_bytes / (1024*1024):.2f}MB)")
            
            # Check if cleanup is needed
            if self.enable_auto_cleanup:
                self._check_memory_threshold()
            
            return True
    
    def deallocate_memory(self, allocation_id: str) -> bool:
        """
        Track memory deallocation.
        
        Args:
            allocation_id: Unique identifier for the allocation
            
        Returns:
            True if deallocation tracked successfully
        """
        with self.lock:
            if allocation_id in self.memory_allocations:
                allocation = self.memory_allocations.pop(allocation_id)
                self.total_deallocations += 1
                
                logger.debug(f"Tracked memory deallocation: {allocation_id} ({allocation['size_mb']:.2f}MB)")
                return True
            
            return False
    
    def update_memory_access(self, allocation_id: str):
        """Update last access time for memory allocation"""
        with self.lock:
            if allocation_id in self.memory_allocations:
                self.memory_allocations[allocation_id]['last_accessed'] = datetime.now()
    
    def get_memory_pool(self, pool_name: str) -> Optional[Any]:
        """Get memory pool by name"""
        return self.memory_pool.get(pool_name)
    
    def set_memory_pool(self, pool_name: str, pool_data: Any):
        """Set memory pool data"""
        with self.lock:
            self.memory_pool[pool_name] = pool_data
    
    def _check_memory_threshold(self):
        """Check if memory usage exceeds threshold"""
        if not self.nvml_available:
            return
        
        memory_info = self.get_gpu_memory_info()
        if not memory_info['available']:
            return
        
        usage_percent = memory_info['memory_usage_percent'] / 100
        
        if usage_percent > self.memory_threshold:
            logger.warning(f"🚨 GPU memory usage high: {usage_percent:.1%}")
            self.cleanup_memory()
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Clean up GPU memory.
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_start = datetime.now()
        
        with self.lock:
            # Get current memory info
            memory_info = self.get_gpu_memory_info()
            
            # Calculate memory usage from tracked allocations
            total_tracked_mb = sum(alloc['size_mb'] for alloc in self.memory_allocations.values())
            
            # Find old allocations to clean up
            current_time = datetime.now()
            old_allocations = []
            
            for alloc_id, allocation in self.memory_allocations.items():
                age = current_time - allocation['last_accessed']
                
                # Clean up allocations older than 5 minutes
                if age > timedelta(minutes=5):
                    old_allocations.append(alloc_id)
            
            # Clean up old allocations
            cleaned_mb = 0
            for alloc_id in old_allocations:
                allocation = self.memory_allocations.pop(alloc_id)
                cleaned_mb += allocation['size_mb']
            
            # Force garbage collection
            gc.collect()
            
            # Clear memory pools if needed
            if force or len(old_allocations) > 0:
                self.memory_pool.clear()
            
            cleanup_time = (datetime.now() - cleanup_start).total_seconds()
            self.total_cleanups += 1
            self.last_cleanup_time = datetime.now()
            
            # Record memory usage
            self.memory_usage_history.append({
                'timestamp': current_time,
                'usage_percent': memory_info['memory_usage_percent'],
                'total_tracked_mb': total_tracked_mb,
                'cleaned_mb': cleaned_mb
            })
            
            result = {
                'cleaned_allocations': len(old_allocations),
                'cleaned_memory_mb': cleaned_mb,
                'cleanup_time_seconds': cleanup_time,
                'current_usage_percent': memory_info['memory_usage_percent'],
                'total_tracked_mb': total_tracked_mb
            }
            
            logger.info(f"🧹 Memory cleanup: {len(old_allocations)} allocations, {cleaned_mb:.2f}MB freed in {cleanup_time:.2f}s")
            
            return result
    
    def start_monitoring(self):
        """Start automatic memory monitoring"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("🔍 Started GPU memory monitoring")
    
    def stop_monitoring(self):
        """Stop automatic memory monitoring"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("🛑 Stopped GPU memory monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check memory threshold
                self._check_memory_threshold()
                
                # Periodic cleanup
                if self.enable_auto_cleanup:
                    time_since_cleanup = datetime.now() - self.last_cleanup_time
                    if time_since_cleanup.total_seconds() > self.cleanup_interval:
                        self.cleanup_memory()
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self.lock:
            # Current GPU memory info
            memory_info = self.get_gpu_memory_info()
            
            # Calculate tracked memory
            total_tracked_mb = sum(alloc['size_mb'] for alloc in self.memory_allocations.values())
            
            # Group allocations by type
            allocations_by_type = defaultdict(list)
            for alloc_id, allocation in self.memory_allocations.items():
                allocations_by_type[allocation['type']].append(allocation)
            
            # Calculate type-specific stats
            type_stats = {}
            for alloc_type, allocations in allocations_by_type.items():
                type_stats[alloc_type] = {
                    'count': len(allocations),
                    'total_mb': sum(alloc['size_mb'] for alloc in allocations),
                    'avg_size_mb': np.mean([alloc['size_mb'] for alloc in allocations])
                }
            
            # Recent memory usage
            recent_usage = []
            if self.memory_usage_history:
                recent_usage = list(self.memory_usage_history)[-10:]
            
            stats = {
                'gpu_memory': memory_info,
                'tracked_allocations': {
                    'total_count': len(self.memory_allocations),
                    'total_mb': total_tracked_mb,
                    'by_type': type_stats
                },
                'performance': {
                    'total_allocations': self.total_allocations,
                    'total_deallocations': self.total_deallocations,
                    'total_cleanups': self.total_cleanups,
                    'last_cleanup': self.last_cleanup_time.isoformat()
                },
                'recent_usage': recent_usage,
                'memory_pools': list(self.memory_pool.keys())
            }
            
            return stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage based on current patterns"""
        with self.lock:
            # Analyze memory usage patterns
            if not self.memory_usage_history:
                return {'message': 'No memory usage data available'}
            
            recent_usage = list(self.memory_usage_history)[-50:]
            
            # Calculate average usage
            avg_usage = np.mean([u['usage_percent'] for u in recent_usage])
            max_usage = np.max([u['usage_percent'] for u in recent_usage])
            
            # Generate optimization recommendations
            recommendations = []
            
            if avg_usage > 70:
                recommendations.append("Consider reducing batch sizes to lower memory usage")
            
            if max_usage > 90:
                recommendations.append("High peak memory usage detected - enable more aggressive cleanup")
            
            # Check for memory leaks
            if self.total_allocations > self.total_deallocations * 1.5:
                recommendations.append("Potential memory leak detected - check allocation/deallocation balance")
            
            # Optimize cleanup interval
            if len(recent_usage) > 10:
                usage_variance = np.var([u['usage_percent'] for u in recent_usage])
                if usage_variance > 100:  # High variance
                    recommendations.append("High memory usage variance - consider more frequent cleanup")
            
            result = {
                'avg_usage_percent': avg_usage,
                'max_usage_percent': max_usage,
                'recommendations': recommendations,
                'optimization_applied': False
            }
            
            # Apply automatic optimizations
            if avg_usage > 80:
                # Increase cleanup frequency
                self.cleanup_interval = max(30, self.cleanup_interval - 15)
                result['optimization_applied'] = True
                result['new_cleanup_interval'] = self.cleanup_interval
                logger.info(f"🔄 Optimized cleanup interval: {self.cleanup_interval}s")
            
            return result
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_monitoring()


# Global GPU memory manager instance
gpu_memory_manager = GPUMemoryManager()
