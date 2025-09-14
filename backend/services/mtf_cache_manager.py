import asyncio
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import redis
import json
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MTFCacheEntry:
    """Cache entry for MTF data"""
    data: pd.DataFrame
    timestamp: datetime
    ttl: datetime
    timeframe: str
    symbol: str
    is_completed: bool

class MTFCacheManager:
    """
    Multi-Timeframe Cache Manager for efficient MTF data caching
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", max_cache_size: int = 1000):
        self.redis_url = redis_url
        self.max_cache_size = max_cache_size
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, MTFCacheEntry] = {}
        
        # Redis client for persistent cache
        self.redis_client = None
        self._init_redis()
        
        # Timeframe hierarchy (higher to lower)
        self.timeframe_hierarchy = ["1d", "4h", "1h", "15m", "5m", "1m"]
        
        # Cache statistics
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        logger.info("🚀 MTF Cache Manager initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established for MTF cache")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed, using memory-only cache: {e}")
            self.redis_client = None
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key for symbol and timeframe"""
        return f"mtf:{symbol}:{timeframe}"
    
    def _calculate_ttl(self, timeframe: str) -> datetime:
        """Calculate TTL until next higher timeframe candle closes"""
        now = datetime.utcnow()
        
        # Timeframe intervals in minutes
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        
        if timeframe not in timeframe_minutes:
            return now + timedelta(hours=1)  # Default 1 hour TTL
        
        # Find next higher timeframe
        current_index = self.timeframe_hierarchy.index(timeframe)
        if current_index > 0:
            higher_timeframe = self.timeframe_hierarchy[current_index - 1]
            higher_minutes = timeframe_minutes[higher_timeframe]
        else:
            # Already highest timeframe, use 2x current timeframe
            higher_minutes = timeframe_minutes[timeframe] * 2
        
        # Calculate next candle close time
        minutes_since_epoch = int(now.timestamp() / 60)
        minutes_to_next_candle = higher_minutes - (minutes_since_epoch % higher_minutes)
        
        ttl = now + timedelta(minutes=minutes_to_next_candle)
        return ttl
    
    async def get_mtf_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get MTF data from cache
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1h", "4h")
            
        Returns:
            DataFrame with MTF data or None if not in cache
        """
        self.cache_stats['total_requests'] += 1
        cache_key = self._get_cache_key(symbol, timeframe)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if datetime.utcnow() < entry.ttl:
                self.cache_stats['hits'] += 1
                logger.debug(f"✅ MTF cache hit: {symbol} {timeframe}")
                return entry.data
            else:
                # Expired, remove from cache
                del self.memory_cache[cache_key]
                self.cache_stats['evictions'] += 1
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data_dict = json.loads(cached_data)
                    df = pd.DataFrame(data_dict['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Check if still valid
                    ttl = datetime.fromisoformat(data_dict['ttl'])
                    if datetime.utcnow() < ttl:
                        # Add to memory cache
                        entry = MTFCacheEntry(
                            data=df,
                            timestamp=datetime.fromisoformat(data_dict['timestamp']),
                            ttl=ttl,
                            timeframe=timeframe,
                            symbol=symbol,
                            is_completed=data_dict['is_completed']
                        )
                        self._add_to_memory_cache(cache_key, entry)
                        
                        self.cache_stats['hits'] += 1
                        logger.debug(f"✅ MTF Redis cache hit: {symbol} {timeframe}")
                        return df
            except Exception as e:
                logger.warning(f"Error reading from Redis cache: {e}")
        
        self.cache_stats['misses'] += 1
        logger.debug(f"❌ MTF cache miss: {symbol} {timeframe}")
        return None
    
    async def set_mtf_data(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                          is_completed: bool = True) -> bool:
        """
        Store MTF data in cache
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: DataFrame with MTF data
            is_completed: Whether this is completed candle data
            
        Returns:
            True if successfully cached
        """
        try:
            cache_key = self._get_cache_key(symbol, timeframe)
            ttl = self._calculate_ttl(timeframe)
            
            # Create cache entry
            entry = MTFCacheEntry(
                data=data,
                timestamp=datetime.utcnow(),
                ttl=ttl,
                timeframe=timeframe,
                symbol=symbol,
                is_completed=is_completed
            )
            
            # Add to memory cache
            self._add_to_memory_cache(cache_key, entry)
            
            # Add to Redis cache
            if self.redis_client:
                cache_data = {
                    'data': data.to_dict('records'),
                    'timestamp': entry.timestamp.isoformat(),
                    'ttl': ttl.isoformat(),
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'is_completed': is_completed
                }
                
                # Calculate Redis TTL in seconds
                redis_ttl = int((ttl - datetime.utcnow()).total_seconds())
                if redis_ttl > 0:
                    self.redis_client.setex(
                        cache_key, 
                        redis_ttl, 
                        json.dumps(cache_data)
                    )
            
            logger.debug(f"💾 MTF data cached: {symbol} {timeframe} (TTL: {ttl})")
            return True
            
        except Exception as e:
            logger.error(f"Error caching MTF data: {e}")
            return False
    
    def _add_to_memory_cache(self, cache_key: str, entry: MTFCacheEntry):
        """Add entry to memory cache with size management"""
        # Remove expired entries first
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if current_time >= entry.ttl
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.cache_stats['evictions'] += 1
        
        # Add new entry
        self.memory_cache[cache_key] = entry
        
        # Evict oldest entries if cache is full
        if len(self.memory_cache) > self.max_cache_size:
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].timestamp
            )
            del self.memory_cache[oldest_key]
            self.cache_stats['evictions'] += 1
    
    async def get_higher_timeframe_context(self, symbol: str, current_timeframe: str) -> Optional[Dict]:
        """
        Get context from higher timeframes for market bias
        
        Args:
            symbol: Trading symbol
            current_timeframe: Current timeframe being analyzed
            
        Returns:
            Dictionary with higher timeframe context
        """
        try:
            current_index = self.timeframe_hierarchy.index(current_timeframe)
            context = {}
            
            # Get data from higher timeframes
            for i in range(current_index):
                higher_timeframe = self.timeframe_hierarchy[i]
                data = await self.get_mtf_data(symbol, higher_timeframe)
                
                if data is not None and len(data) > 0:
                    context[higher_timeframe] = self._analyze_timeframe_context(data)
            
            return context if context else None
            
        except Exception as e:
            logger.error(f"Error getting higher timeframe context: {e}")
            return None
    
    def _analyze_timeframe_context(self, data: pd.DataFrame) -> Dict:
        """Analyze timeframe data to extract context"""
        if len(data) < 20:
            return {'trend': 'neutral', 'strength': 0.0, 'confidence': 0.0}
        
        # Calculate EMAs
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
        current_price = data['close'].iloc[-1]
        ema_20 = data['ema_20'].iloc[-1]
        ema_50 = data['ema_50'].iloc[-1]
        
        # Determine trend
        if current_price > ema_20 > ema_50:
            trend = 'bullish'
            strength = min((current_price - ema_50) / ema_50, 1.0)
        elif current_price < ema_20 < ema_50:
            trend = 'bearish'
            strength = min((ema_50 - current_price) / ema_50, 1.0)
        else:
            trend = 'neutral'
            strength = 0.0
        
        # Calculate confidence based on data quality
        confidence = min(len(data) / 100, 1.0)  # More data = higher confidence
        
        return {
            'trend': trend,
            'strength': strength,
            'confidence': confidence,
            'current_price': current_price,
            'ema_20': ema_20,
            'ema_50': ema_50
        }
    
    async def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """Clear cache entries"""
        if symbol and timeframe:
            cache_key = self._get_cache_key(symbol, timeframe)
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            if self.redis_client:
                self.redis_client.delete(cache_key)
        else:
            # Clear all cache
            self.memory_cache.clear()
            if self.redis_client:
                # Clear all MTF cache keys
                keys = self.redis_client.keys("mtf:*")
                if keys:
                    self.redis_client.delete(*keys)
        
        logger.info(f"🧹 MTF cache cleared for {symbol or 'all'} {timeframe or 'timeframes'}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['total_requests']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_cache_size': len(self.memory_cache),
            'redis_available': self.redis_client is not None
        }
    
    async def cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if current_time >= entry.ttl
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.cache_stats['evictions'] += 1
        
        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired MTF cache entries")
