#!/usr/bin/env python3
"""
Enhanced Cache Manager for AlphaPlus
Ultra-low latency caching layer with Redis integration
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis.asyncio as redis
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: datetime
    ttl: int
    access_count: int = 0
    last_accessed: datetime = None

class EnhancedCacheManager:
    """
    Enhanced cache manager with Redis integration for ultra-low latency
    Integrates seamlessly with existing TimescaleDB architecture
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 max_memory_cache_size: int = 10000,
                 enable_redis: bool = True):
        """
        Initialize enhanced cache manager
        
        Args:
            redis_url: Redis connection URL
            max_memory_cache_size: Maximum in-memory cache entries
            enable_redis: Whether to use Redis (fallback to memory-only)
        """
        self.redis_url = redis_url
        self.max_memory_cache_size = max_memory_cache_size
        self.enable_redis = enable_redis
        
        # In-memory cache for ultra-fast access
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_order = deque()  # LRU order
        
        # Redis client for persistent cache
        self.redis_client = None
        self._init_redis()
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'avg_response_time_ms': 0
        }
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        
        # Cache keys for different data types
        self.cache_keys = {
            'market_data': 'market:{symbol}:{timeframe}',
            'signals': 'signals:{symbol}:{timeframe}',
            'patterns': 'patterns:{symbol}:{timeframe}',
            'indicators': 'indicators:{symbol}:{timeframe}',
            'real_time': 'realtime:{symbol}:{timeframe}',
            'websocket': 'ws:{symbol}:{timeframe}'
        }
        
        logger.info("🚀 Enhanced Cache Manager initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not self.enable_redis:
            logger.info("⚠️ Redis disabled, using memory-only cache")
            return
            
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            asyncio.create_task(self._test_redis_connection())
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed, using memory-only cache: {e}")
            self.redis_client = None
            self.enable_redis = False
    
    async def _test_redis_connection(self):
        """Test Redis connection"""
        try:
            await self.redis_client.ping()
            logger.info("✅ Redis connection test successful")
        except Exception as e:
            logger.error(f"❌ Redis connection test failed: {e}")
            self.redis_client = None
            self.enable_redis = False
    
    def _get_cache_key(self, data_type: str, symbol: str, timeframe: str, **kwargs) -> str:
        """Generate cache key for data"""
        base_key = self.cache_keys.get(data_type, f"{data_type}:{{symbol}}:{{timeframe}}")
        key = base_key.format(symbol=symbol, timeframe=timeframe, **kwargs)
        return key
    
    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[List[Dict]]:
        """Get market data from cache with fallback to database"""
        start_time = time.time()
        cache_key = self._get_cache_key('market_data', symbol, timeframe)
        
        try:
            # Try memory cache first (fastest)
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not self._is_expired(entry):
                    self._update_memory_cache_stats(entry)
                    self.stats['memory_hits'] += 1
                    self._record_response_time(start_time)
                    return entry.data[:limit]
            
            # Try Redis cache
            if self.redis_client and self.enable_redis:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Store in memory cache for faster access
                    await self._store_in_memory_cache(cache_key, data, ttl=300)  # 5 minutes
                    self.stats['redis_hits'] += 1
                    self._record_response_time(start_time)
                    return data[:limit]
            
            # Cache miss
            self.stats['misses'] += 1
            self._record_response_time(start_time)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting market data from cache: {e}")
            self.stats['misses'] += 1
            return None
    
    async def store_market_data(self, symbol: str, timeframe: str, data: List[Dict], ttl: int = 300) -> bool:
        """Store market data in cache"""
        try:
            cache_key = self._get_cache_key('market_data', symbol, timeframe)
            
            # Store in memory cache
            await self._store_in_memory_cache(cache_key, data, ttl)
            
            # Store in Redis cache
            if self.redis_client and self.enable_redis:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(data, default=str)
                )
            
            logger.debug(f"📊 Stored market data in cache: {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing market data in cache: {e}")
            return False
    
    async def get_real_time_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get real-time data from cache (for WebSocket delivery)"""
        start_time = time.time()
        cache_key = self._get_cache_key('real_time', symbol, timeframe)
        
        try:
            # Memory cache for real-time data (ultra-fast)
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not self._is_expired(entry):
                    self._update_memory_cache_stats(entry)
                    self.stats['memory_hits'] += 1
                    self._record_response_time(start_time)
                    return entry.data
            
            # Redis cache for real-time data
            if self.redis_client and self.enable_redis:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    await self._store_in_memory_cache(cache_key, data, ttl=60)  # 1 minute
                    self.stats['redis_hits'] += 1
                    self._record_response_time(start_time)
                    return data
            
            self.stats['misses'] += 1
            self._record_response_time(start_time)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting real-time data from cache: {e}")
            return None
    
    async def store_real_time_data(self, symbol: str, timeframe: str, data: Dict, ttl: int = 60) -> bool:
        """Store real-time data in cache (for WebSocket delivery)"""
        try:
            cache_key = self._get_cache_key('real_time', symbol, timeframe)
            
            # Store in memory cache (priority for real-time)
            await self._store_in_memory_cache(cache_key, data, ttl)
            
            # Store in Redis cache
            if self.redis_client and self.enable_redis:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(data, default=str)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing real-time data in cache: {e}")
            return False
    
    async def get_signals(self, symbol: str, timeframe: str, limit: int = 50) -> Optional[List[Dict]]:
        """Get trading signals from cache"""
        start_time = time.time()
        cache_key = self._get_cache_key('signals', symbol, timeframe)
        
        try:
            # Memory cache
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not self._is_expired(entry):
                    self._update_memory_cache_stats(entry)
                    self.stats['memory_hits'] += 1
                    self._record_response_time(start_time)
                    return entry.data[:limit]
            
            # Redis cache
            if self.redis_client and self.enable_redis:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    await self._store_in_memory_cache(cache_key, data, ttl=180)  # 3 minutes
                    self.stats['redis_hits'] += 1
                    self._record_response_time(start_time)
                    return data[:limit]
            
            self.stats['misses'] += 1
            self._record_response_time(start_time)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting signals from cache: {e}")
            return None
    
    async def store_signals(self, symbol: str, timeframe: str, signals: List[Dict], ttl: int = 180) -> bool:
        """Store trading signals in cache"""
        try:
            cache_key = self._get_cache_key('signals', symbol, timeframe)
            
            # Store in memory cache
            await self._store_in_memory_cache(cache_key, signals, ttl)
            
            # Store in Redis cache
            if self.redis_client and self.enable_redis:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(signals, default=str)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing signals in cache: {e}")
            return False
    
    async def _store_in_memory_cache(self, key: str, data: Any, ttl: int) -> None:
        """Store data in memory cache with LRU eviction"""
        try:
            # Remove from order if exists
            if key in self.memory_cache_order:
                self.memory_cache_order.remove(key)
            
            # Add to front (most recently used)
            self.memory_cache_order.appendleft(key)
            
            # Store entry
            self.memory_cache[key] = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                ttl=ttl,
                access_count=0,
                last_accessed=datetime.now()
            )
            
            # Evict if cache is full
            if len(self.memory_cache) > self.max_memory_cache_size:
                await self._evict_lru_entry()
                
        except Exception as e:
            logger.error(f"❌ Error storing in memory cache: {e}")
    
    async def _evict_lru_entry(self) -> None:
        """Evict least recently used entry"""
        try:
            if self.memory_cache_order:
                # Remove oldest entry
                oldest_key = self.memory_cache_order.pop()
                if oldest_key in self.memory_cache:
                    del self.memory_cache[oldest_key]
                    self.stats['evictions'] += 1
                    logger.debug(f"🗑️ Evicted cache entry: {oldest_key}")
        except Exception as e:
            logger.error(f"❌ Error evicting LRU entry: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - entry.timestamp > timedelta(seconds=entry.ttl)
    
    def _update_memory_cache_stats(self, entry: CacheEntry) -> None:
        """Update memory cache entry statistics"""
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        # Move to front of LRU order
        if entry in self.memory_cache_order:
            self.memory_cache_order.remove(entry)
        self.memory_cache_order.appendleft(entry)
    
    def _record_response_time(self, start_time: float) -> None:
        """Record response time for statistics"""
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        self.response_times.append(response_time)
        
        # Update average response time
        if self.response_times:
            self.stats['avg_response_time_ms'] = np.mean(self.response_times)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats['memory_hits'] + self.stats['redis_hits'] + self.stats['misses']
        hit_rate = (self.stats['memory_hits'] + self.stats['redis_hits']) / total_requests if total_requests > 0 else 0
        
        return {
            'memory_hits': self.stats['memory_hits'],
            'redis_hits': self.stats['redis_hits'],
            'misses': self.stats['misses'],
            'total_requests': total_requests,
            'hit_rate': round(hit_rate * 100, 2),
            'memory_hit_rate': round(self.stats['memory_hits'] / total_requests * 100, 2) if total_requests > 0 else 0,
            'redis_hit_rate': round(self.stats['redis_hits'] / total_requests * 100, 2) if total_requests > 0 else 0,
            'evictions': self.stats['evictions'],
            'avg_response_time_ms': round(self.stats['avg_response_time_ms'], 2),
            'memory_cache_size': len(self.memory_cache),
            'max_memory_cache_size': self.max_memory_cache_size,
            'redis_enabled': self.enable_redis,
            'redis_connected': self.redis_client is not None
        }
    
    async def clear_cache(self, pattern: str = None) -> int:
        """Clear cache entries matching pattern"""
        try:
            cleared_count = 0
            
            if pattern:
                # Clear specific pattern
                keys_to_remove = [key for key in self.memory_cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.memory_cache_order:
                        self.memory_cache_order.remove(key)
                    cleared_count += 1
                
                # Clear Redis cache
                if self.redis_client and self.enable_redis:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                        cleared_count += len(keys)
            else:
                # Clear all cache
                cleared_count = len(self.memory_cache)
                self.memory_cache.clear()
                self.memory_cache_order.clear()
                
                if self.redis_client and self.enable_redis:
                    await self.redis_client.flushdb()
            
            logger.info(f"🗑️ Cleared {cleared_count} cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            return 0
    
    async def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries"""
        try:
            expired_keys = []
            current_time = datetime.now()
            
            for key, entry in self.memory_cache.items():
                if current_time - entry.timestamp > timedelta(seconds=entry.ttl):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                if key in self.memory_cache_order:
                    self.memory_cache_order.remove(key)
            
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up expired entries: {e}")
            return 0
    
    async def close(self):
        """Close cache manager and connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing cache manager: {e}")
