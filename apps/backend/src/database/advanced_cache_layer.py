"""
Advanced Hybrid Cache Layer for AlphaPlus
Implements RAM + Disk storage with ultra-low latency
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: datetime
    ttl: datetime
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    priority: int = 1  # 1=low, 5=high

class AdvancedCacheLayer:
    """Hybrid storage layer with Redis + TimescaleDB"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db_session_factory = None,
                 max_memory_mb: int = 1024,
                 hot_data_hours: int = 24,
                 batch_size: int = 1000,
                 flush_interval: int = 30):
        
        self.redis_url = redis_url
        self.db_session_factory = db_session_factory
        self.max_memory_mb = max_memory_mb
        self.hot_data_hours = hot_data_hours
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Redis client
        self.redis_client = None
        
        # Memory cache for ultra-fast access
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_size = 0
        
        # Write buffers for async flushing
        self.write_buffers = {
            'candlestick_data': [],
            'signals': [],
            'trades': [],
            'patterns': []
        }
        
        # Background tasks
        self.flush_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # Performance metrics
        self.stats = {
            'redis_hits': 0,
            'memory_hits': 0,
            'cache_misses': 0,
            'writes_buffered': 0,
            'writes_flushed': 0,
            'avg_response_time_ms': 0.0
        }
        
        logger.info("🚀 Advanced Cache Layer initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start background tasks"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Start background tasks
            self.is_running = True
            self.flush_task = asyncio.create_task(self._flush_buffer_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("✅ Background tasks started")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache layer: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown cache layer gracefully"""
        self.is_running = False
        
        # Flush remaining data
        await self._flush_all_buffers()
        
        # Cancel background tasks
        if self.flush_task:
            self.flush_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Cache layer shutdown complete")
    
    async def get_candlestick_data(self, symbol: str, timeframe: str, 
                                  start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get candlestick data with hybrid caching"""
        start = datetime.now()
        
        try:
            # Check memory cache first (ultra-fast)
            memory_key = f"mem:{symbol}:{timeframe}:{start_time.strftime('%Y%m%d')}"
            if memory_key in self.memory_cache:
                entry = self.memory_cache[memory_key]
                if entry.ttl > datetime.now():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self.stats['memory_hits'] += 1
                    self.stats['avg_response_time_ms'] = (datetime.now() - start).total_seconds() * 1000
                    return entry.data
            
            # Check Redis cache
            redis_key = f"candle:{symbol}:{timeframe}:{start_time.strftime('%Y%m%d')}"
            if self.redis_client:
                cached_data = await self.redis_client.get(redis_key)
                if cached_data:
                    data_dict = json.loads(cached_data)
                    df = pd.DataFrame(data_dict['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Add to memory cache for faster access
                    self._add_to_memory_cache(memory_key, df, data_dict['ttl'])
                    
                    self.stats['redis_hits'] += 1
                    self.stats['avg_response_time_ms'] = (datetime.now() - start).total_seconds() * 1000
                    return df
            
            # Cache miss - fetch from database
            self.stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting candlestick data: {e}")
            return None
    
    async def set_candlestick_data(self, symbol: str, timeframe: str, 
                                  data: pd.DataFrame, ttl_hours: int = 24) -> bool:
        """Set candlestick data with hybrid storage"""
        try:
            # Store in Redis (hot data)
            redis_key = f"candle:{symbol}:{timeframe}:{datetime.now().strftime('%Y%m%d')}"
            memory_key = f"mem:{symbol}:{timeframe}:{datetime.now().strftime('%Y%m%d')}"
            
            # Prepare data for storage
            data_dict = {
                'data': data.to_dict('records'),
                'timestamp': datetime.now().isoformat(),
                'ttl': (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            }
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    redis_key,
                    ttl_hours * 3600,
                    json.dumps(data_dict)
                )
            
            # Add to memory cache
            ttl = datetime.now() + timedelta(hours=ttl_hours)
            self._add_to_memory_cache(memory_key, data, ttl)
            
            # Buffer for TimescaleDB (cold storage)
            self.write_buffers['candlestick_data'].extend(data.to_dict('records'))
            self.stats['writes_buffered'] += len(data)
            
            # Flush if buffer is full
            if len(self.write_buffers['candlestick_data']) >= self.batch_size:
                await self._flush_candlestick_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting candlestick data: {e}")
            return False
    
    async def get_latest_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get latest signals with ultra-fast access"""
        try:
            # Always check memory first for latest data
            memory_key = f"mem:signals:latest:{symbol or 'all'}"
            if memory_key in self.memory_cache:
                entry = self.memory_cache[memory_key]
                if entry.ttl > datetime.now():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self.stats['memory_hits'] += 1
                    return entry.data[:limit]
            
            # Check Redis for latest signals
            redis_key = f"signals:latest:{symbol or 'all'}"
            if self.redis_client:
                cached_data = await self.redis_client.get(redis_key)
                if cached_data:
                    signals = json.loads(cached_data)
                    
                    # Add to memory cache
                    ttl = datetime.now() + timedelta(minutes=5)  # Short TTL for latest data
                    self._add_to_memory_cache(memory_key, signals, ttl)
                    
                    self.stats['redis_hits'] += 1
                    return signals[:limit]
            
            self.stats['cache_misses'] += 1
            return []
            
        except Exception as e:
            logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def set_signal(self, signal: Dict) -> bool:
        """Set signal with immediate cache update"""
        try:
            # Add to memory cache immediately
            memory_key = f"mem:signal:{signal['id']}"
            ttl = datetime.now() + timedelta(hours=1)
            self._add_to_memory_cache(memory_key, signal, ttl)
            
            # Update latest signals cache
            latest_key = f"mem:signals:latest:{signal.get('symbol', 'all')}"
            if latest_key in self.memory_cache:
                latest_signals = self.memory_cache[latest_key].data
                latest_signals.insert(0, signal)
                latest_signals = latest_signals[:100]  # Keep only latest 100
                self.memory_cache[latest_key].data = latest_signals
            
            # Buffer for TimescaleDB
            self.write_buffers['signals'].append(signal)
            self.stats['writes_buffered'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting signal: {e}")
            return False
    
    def _add_to_memory_cache(self, key: str, data: Any, ttl: datetime, priority: int = 1):
        """Add data to memory cache with LRU eviction"""
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl=ttl,
            access_count=1,
            last_accessed=datetime.now(),
            size_bytes=len(str(data)),
            priority=priority
        )
        
        # Check memory limit
        if self.memory_size + entry.size_bytes > self.max_memory_mb * 1024 * 1024:
            self._evict_lru_entries(entry.size_bytes)
        
        self.memory_cache[key] = entry
        self.memory_size += entry.size_bytes
    
    def _evict_lru_entries(self, needed_bytes: int):
        """Evict least recently used entries"""
        # Sort by priority, then by last accessed time
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: (x[1].priority, x[1].last_accessed or x[1].timestamp)
        )
        
        freed_bytes = 0
        for key, entry in sorted_entries:
            if freed_bytes >= needed_bytes:
                break
            
            del self.memory_cache[key]
            freed_bytes += entry.size_bytes
            self.memory_size -= entry.size_bytes
    
    async def _flush_buffer_loop(self):
        """Background task to flush buffers to TimescaleDB"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_all_buffers()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    async def _flush_all_buffers(self):
        """Flush all write buffers to TimescaleDB"""
        if not self.db_session_factory:
            return
        
        try:
            async with self.db_session_factory() as session:
                # Flush candlestick data
                if self.write_buffers['candlestick_data']:
                    await self._flush_candlestick_buffer(session)
                
                # Flush signals
                if self.write_buffers['signals']:
                    await self._flush_signals_buffer(session)
                
                # Flush trades
                if self.write_buffers['trades']:
                    await self._flush_trades_buffer(session)
                
                # Flush patterns
                if self.write_buffers['patterns']:
                    await self._flush_patterns_buffer(session)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
    
    async def _flush_candlestick_buffer(self, session: AsyncSession = None):
        """Flush candlestick data buffer"""
        if not self.write_buffers['candlestick_data']:
            return
        
        try:
            if not session:
                async with self.db_session_factory() as session:
                    await self._flush_candlestick_buffer(session)
                    await session.commit()
                return
            
            # Batch insert
            query = text("""
                INSERT INTO candlestick_data 
                (symbol, timestamp, open, high, low, close, volume, timeframe, indicators, patterns)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :timeframe, :indicators, :patterns)
                ON CONFLICT (timestamp, id) DO NOTHING
            """)
            
            await session.execute(query, self.write_buffers['candlestick_data'])
            
            flushed_count = len(self.write_buffers['candlestick_data'])
            self.write_buffers['candlestick_data'].clear()
            self.stats['writes_flushed'] += flushed_count
            
            logger.debug(f"✅ Flushed {flushed_count} candlestick records")
            
        except Exception as e:
            logger.error(f"Error flushing candlestick buffer: {e}")
    
    async def _flush_signals_buffer(self, session: AsyncSession):
        """Flush signals buffer"""
        if not self.write_buffers['signals']:
            return
        
        try:
            query = text("""
                INSERT INTO signals 
                (id, symbol, side, strategy, confidence, strength, timestamp, price, stop_loss, take_profit, metadata, status)
                VALUES (:id, :symbol, :side, :strategy, :confidence, :strength, :timestamp, :price, :stop_loss, :take_profit, :metadata, :status)
                ON CONFLICT (id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    strength = EXCLUDED.strength,
                    price = EXCLUDED.price,
                    metadata = EXCLUDED.metadata,
                    status = EXCLUDED.status
            """)
            
            await session.execute(query, self.write_buffers['signals'])
            
            flushed_count = len(self.write_buffers['signals'])
            self.write_buffers['signals'].clear()
            self.stats['writes_flushed'] += flushed_count
            
            logger.debug(f"✅ Flushed {flushed_count} signal records")
            
        except Exception as e:
            logger.error(f"Error flushing signals buffer: {e}")
    
    async def _flush_trades_buffer(self, session: AsyncSession):
        """Flush trades buffer"""
        if not self.write_buffers['trades']:
            return
        
        try:
            query = text("""
                INSERT INTO trades 
                (signal_id, symbol, side, entry_price, quantity, timestamp, strategy, confidence, status, exit_price, exit_timestamp, pnl)
                VALUES (:signal_id, :symbol, :side, :entry_price, :quantity, :timestamp, :strategy, :confidence, :status, :exit_price, :exit_timestamp, :pnl)
                ON CONFLICT (signal_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    exit_price = EXCLUDED.exit_price,
                    exit_timestamp = EXCLUDED.exit_timestamp,
                    pnl = EXCLUDED.pnl
            """)
            
            await session.execute(query, self.write_buffers['trades'])
            
            flushed_count = len(self.write_buffers['trades'])
            self.write_buffers['trades'].clear()
            self.stats['writes_flushed'] += flushed_count
            
            logger.debug(f"✅ Flushed {flushed_count} trade records")
            
        except Exception as e:
            logger.error(f"Error flushing trades buffer: {e}")
    
    async def _flush_patterns_buffer(self, session: AsyncSession):
        """Flush patterns buffer"""
        if not self.write_buffers['patterns']:
            return
        
        try:
            query = text("""
                INSERT INTO candlestick_patterns 
                (symbol, timestamp, pattern_name, confidence, volume_confirmation, trend_alignment, metadata)
                VALUES (:symbol, :timestamp, :pattern_name, :confidence, :volume_confirmation, :trend_alignment, :metadata)
                ON CONFLICT (symbol, timestamp, pattern_name) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    volume_confirmation = EXCLUDED.volume_confirmation,
                    trend_alignment = EXCLUDED.trend_alignment,
                    metadata = EXCLUDED.metadata
            """)
            
            await session.execute(query, self.write_buffers['patterns'])
            
            flushed_count = len(self.write_buffers['patterns'])
            self.write_buffers['patterns'].clear()
            self.stats['writes_flushed'] += flushed_count
            
            logger.debug(f"✅ Flushed {flushed_count} pattern records")
            
        except Exception as e:
            logger.error(f"Error flushing patterns buffer: {e}")
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired cache entries"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Cleanup expired memory cache entries
                expired_keys = []
                for key, entry in self.memory_cache.items():
                    if entry.ttl < datetime.now():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.memory_cache[key]
                    self.memory_size -= entry.size_bytes
                    del self.memory_cache[key]
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            **self.stats,
            'memory_cache_size': len(self.memory_cache),
            'memory_usage_mb': self.memory_size / (1024 * 1024),
            'buffer_sizes': {k: len(v) for k, v in self.write_buffers.items()}
        }
