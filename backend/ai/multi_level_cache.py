"""
Multi-Level Cache System for AlphaPulse
L1 (recent signals) and L2 (frequent patterns) caching for optimal performance
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"  # Recent signals (fast access)
    L2 = "l2"  # Frequent patterns (medium access)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    ttl: Optional[timedelta] = None

class L1Cache:
    """L1 Cache for recent signals (fast access)"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
        logger.info(f"L1Cache initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() - entry.timestamp > entry.ttl:
                del self.cache[key]
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = datetime.now()
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in L1 cache"""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl=ttl or timedelta(seconds=self.ttl_seconds)
            )
            
            self.cache[key] = entry
            
            # Evict if needed
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self.cache:
            self.cache.popitem(last=False)
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count"""
        with self.lock:
            expired_keys = []
            now = datetime.now()
            
            for key, entry in self.cache.items():
                if entry.ttl and now - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need to track hits/misses
                'avg_access_count': sum(e.access_count for e in self.cache.values()) / len(self.cache) if self.cache else 0
            }

class L2Cache:
    """L2 Cache for frequent patterns (medium access)"""
    
    def __init__(self, max_size: int = 5000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
        logger.info(f"L2Cache initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L2 cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and datetime.now() - entry.timestamp > entry.ttl:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = datetime.now()
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set value in L2 cache"""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl=ttl or timedelta(seconds=self.ttl_seconds)
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            
            # Evict if needed
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count"""
        with self.lock:
            expired_keys = []
            now = datetime.now()
            
            for key, entry in self.cache.items():
                if entry.ttl and now - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need to track hits/misses
                'avg_access_count': sum(e.access_count for e in self.cache.values()) / len(self.cache) if self.cache else 0
            }

class MultiLevelCache:
    """Multi-level cache system with L1 and L2 caches"""
    
    def __init__(self, 
                l1_max_size: int = 1000,
                l2_max_size: int = 5000,
                l1_ttl_seconds: int = 60,
                l2_ttl_seconds: int = 3600,
                cleanup_interval: int = 300):
        
        # Initialize caches
        self.l1_cache = L1Cache(max_size=l1_max_size, ttl_seconds=l1_ttl_seconds)
        self.l2_cache = L2Cache(max_size=l2_max_size, ttl_seconds=l2_ttl_seconds)
        
        # Configuration
        self.cleanup_interval = cleanup_interval
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        
        # Cleanup task
        self.cleanup_task = None
        self.running = False
        
        logger.info("MultiLevelCache initialized")
    
    async def start(self):
        """Start the cache system"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("MultiLevelCache started")
    
    async def stop(self):
        """Stop the cache system"""
        if not self.running:
            return
        
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MultiLevelCache stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            self.l2_hits += 1
            # Promote to L1 cache
            self.l1_cache.set(key, value)
            return value
        
        # Cache miss
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.L1) -> None:
        """Set value in specified cache level"""
        if level == CacheLevel.L1:
            self.l1_cache.set(key, value)
        else:
            self.l2_cache.set(key, value)
    
    def set_both(self, key: str, value: Any) -> None:
        """Set value in both L1 and L2 caches"""
        self.l1_cache.set(key, value)
        self.l2_cache.set(key, value)
    
    def generate_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        # Create a deterministic string representation
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_signal_key(self, symbol: str, price: float, volume: float, direction: str) -> str:
        """Generate cache key for signal data"""
        data = {
            'symbol': symbol,
            'price': round(price, 6),
            'volume': round(volume, 2),
            'direction': direction,
            'timestamp': int(time.time() / 60)  # Round to minute
        }
        return self.generate_key(data)
    
    def get_pattern_key(self, pattern_type: str, timeframe: str, confidence: float) -> str:
        """Generate cache key for pattern data"""
        data = {
            'pattern_type': pattern_type,
            'timeframe': timeframe,
            'confidence': round(confidence, 3),
            'timestamp': int(time.time() / 300)  # Round to 5 minutes
        }
        return self.generate_key(data)
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while self.running:
            try:
                # Clear expired entries
                l1_expired = self.l1_cache.clear_expired()
                l2_expired = self.l2_cache.clear_expired()
                
                if l1_expired > 0 or l2_expired > 0:
                    logger.debug(f"Cleared {l1_expired} L1 and {l2_expired} L2 expired entries")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        l1_hit_rate = self.l1_hits / total_requests if total_requests > 0 else 0.0
        l2_hit_rate = self.l2_hits / total_requests if total_requests > 0 else 0.0
        overall_hit_rate = (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0.0
        
        return {
            'l1_cache': self.l1_cache.get_stats(),
            'l2_cache': self.l2_cache.get_stats(),
            'performance': {
                'l1_hits': self.l1_hits,
                'l2_hits': self.l2_hits,
                'misses': self.misses,
                'total_requests': total_requests,
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'overall_hit_rate': overall_hit_rate
            }
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.l1_cache.cache.clear()
        self.l2_cache.cache.clear()
        self.l2_cache.access_order.clear()
        logger.info("All caches cleared")

# Global multi-level cache instance
multi_level_cache = MultiLevelCache(
    l1_max_size=1000,
    l2_max_size=5000,
    l1_ttl_seconds=60,
    l2_ttl_seconds=3600,
    cleanup_interval=300
)
