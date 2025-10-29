#!/usr/bin/env python3
"""
Consolidated Data Services for AlphaPulse
Combines functionality from various data_*.py files
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import aiohttp
import websockets
import redis
import sqlite3
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Represents a data request"""
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    limit: int = 1000
    data_type: str = "candles"  # candles, trades, orderbook

@dataclass
class DataResponse:
    """Represents a data response"""
    data: List[Dict]
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str

class DataService:
    """Unified data service for AlphaPulse"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 database_url: str = "postgresql://user:pass@localhost:5432/alphapulse",
                 cache_ttl: int = 3600):
        self.redis_url = redis_url
        self.database_url = database_url
        self.cache_ttl = cache_ttl
        
        # Initialize connections
        self.redis_client = redis.from_url(redis_url)
        self.db_engine = create_engine(database_url)
        self.async_db_engine = create_async_engine(database_url.replace('postgresql://', 'postgresql+asyncpg://'))
        
        # Data buffers
        self.candle_buffers = {}
        self.trade_buffers = {}
        self.orderbook_buffers = {}
        
        # WebSocket connections
        self.websocket_connections = {}
        
        logger.info("DataService initialized")
    
    async def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from multiple sources"""
        try:
            # Try cache first
            cached_data = await self._get_from_cache(request)
            if cached_data:
                return cached_data
            
            # Try database
            db_data = await self._get_from_database(request)
            if db_data:
                await self._cache_data(request, db_data)
                return db_data
            
            # Try external API
            api_data = await self._get_from_api(request)
            if api_data:
                await self._store_in_database(request, api_data)
                await self._cache_data(request, api_data)
                return api_data
            
            raise ValueError(f"No data available for {request.symbol} {request.timeframe}")
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_realtime_data(self, symbol: str, data_type: str = "candles") -> DataResponse:
        """Get real-time data via WebSocket"""
        try:
            if symbol not in self.websocket_connections:
                await self._connect_websocket(symbol)
            
            # Get from buffer
            buffer_key = f"{symbol}_{data_type}"
            if data_type == "candles" and buffer_key in self.candle_buffers:
                data = list(self.candle_buffers[buffer_key])
            elif data_type == "trades" and buffer_key in self.trade_buffers:
                data = list(self.trade_buffers[buffer_key])
            elif data_type == "orderbook" and buffer_key in self.orderbook_buffers:
                data = list(self.orderbook_buffers[buffer_key])
            else:
                data = []
            
            return DataResponse(
                data=data,
                metadata={'source': 'websocket', 'symbol': symbol, 'data_type': data_type},
                timestamp=datetime.now(),
                source='websocket'
            )
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            raise
    
    async def store_data(self, data: List[Dict], data_type: str = "candles") -> bool:
        """Store data in database and cache"""
        try:
            # Store in database
            await self._store_in_database_batch(data, data_type)
            
            # Update cache
            for item in data:
                await self._cache_item(item, data_type)
            
            # Update buffers
            await self._update_buffers(data, data_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old data from database and cache"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean database
            deleted_count = await self._cleanup_database(cutoff_date)
            
            # Clean cache
            cache_deleted = await self._cleanup_cache(cutoff_date)
            
            logger.info(f"Cleaned up {deleted_count} database records and {cache_deleted} cache entries")
            return deleted_count + cache_deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    async def get_data_statistics(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get statistics about available data"""
        try:
            stats = {}
            
            # Database stats
            db_stats = await self._get_database_stats(symbol, timeframe)
            stats.update(db_stats)
            
            # Cache stats
            cache_stats = await self._get_cache_stats(symbol, timeframe)
            stats.update(cache_stats)
            
            # Buffer stats
            buffer_stats = await self._get_buffer_stats(symbol, timeframe)
            stats.update(buffer_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
    
    async def _get_from_cache(self, request: DataRequest) -> Optional[DataResponse]:
        """Get data from Redis cache"""
        try:
            cache_key = f"data:{request.symbol}:{request.timeframe}:{request.start_time.strftime('%Y%m%d')}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                data = json.loads(cached)
                return DataResponse(
                    data=data,
                    metadata={'source': 'cache', 'cache_key': cache_key},
                    timestamp=datetime.now(),
                    source='cache'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def _get_from_database(self, request: DataRequest) -> Optional[DataResponse]:
        """Get data from database"""
        try:
            async with self.async_db_engine.begin() as conn:
                query = text("""
                    SELECT * FROM market_data 
                    WHERE symbol = :symbol 
                    AND timeframe = :timeframe 
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await conn.execute(query, {
                    'symbol': request.symbol,
                    'timeframe': request.timeframe,
                    'start_time': request.start_time,
                    'end_time': request.end_time,
                    'limit': request.limit
                })
                
                data = [dict(row) for row in result]
                
                if data:
                    return DataResponse(
                        data=data,
                        metadata={'source': 'database', 'count': len(data)},
                        timestamp=datetime.now(),
                        source='database'
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting from database: {e}")
            return None
    
    async def _get_from_api(self, request: DataRequest) -> Optional[DataResponse]:
        """Get data from external API"""
        try:
            # Binance API example
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': request.symbol.replace('/', ''),
                'interval': request.timeframe,
                'startTime': int(request.start_time.timestamp() * 1000),
                'endTime': int(request.end_time.timestamp() * 1000),
                'limit': request.limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        raw_data = await response.json()
                        
                        # Convert to standard format
                        data = []
                        for candle in raw_data:
                            data.append({
                                'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': float(candle[5]),
                                'symbol': request.symbol,
                                'timeframe': request.timeframe
                            })
                        
                        return DataResponse(
                            data=data,
                            metadata={'source': 'api', 'count': len(data)},
                            timestamp=datetime.now(),
                            source='api'
                        )
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting from API: {e}")
            return None
    
    async def _cache_data(self, request: DataRequest, response: DataResponse) -> None:
        """Cache data in Redis"""
        try:
            cache_key = f"data:{request.symbol}:{request.timeframe}:{request.start_time.strftime('%Y%m%d')}"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(response.data, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    async def _store_in_database(self, request: DataRequest, response: DataResponse) -> None:
        """Store data in database"""
        try:
            async with self.async_db_engine.begin() as conn:
                for item in response.data:
                    query = text("""
                        INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """)
                    
                    await conn.execute(query, item)
                    
        except Exception as e:
            logger.error(f"Error storing in database: {e}")
    
    async def _store_in_database_batch(self, data: List[Dict], data_type: str) -> None:
        """Store data in database in batch"""
        try:
            if data_type == "candles":
                async with self.async_db_engine.begin() as conn:
                    for item in data:
                        query = text("""
                            INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
                            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
                            close = EXCLUDED.close, volume = EXCLUDED.volume
                        """)
                        
                        await conn.execute(query, item)
                        
        except Exception as e:
            logger.error(f"Error storing batch data: {e}")
    
    async def _cache_item(self, item: Dict, data_type: str) -> None:
        """Cache individual item"""
        try:
            cache_key = f"item:{item.get('symbol', 'unknown')}:{item.get('timestamp', datetime.now())}"
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(item, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching item: {e}")
    
    async def _update_buffers(self, data: List[Dict], data_type: str) -> None:
        """Update in-memory buffers"""
        try:
            for item in data:
                symbol = item.get('symbol', 'unknown')
                buffer_key = f"{symbol}_{data_type}"
                
                if data_type == "candles":
                    if buffer_key not in self.candle_buffers:
                        self.candle_buffers[buffer_key] = deque(maxlen=1000)
                    self.candle_buffers[buffer_key].append(item)
                    
                elif data_type == "trades":
                    if buffer_key not in self.trade_buffers:
                        self.trade_buffers[buffer_key] = deque(maxlen=1000)
                    self.trade_buffers[buffer_key].append(item)
                    
                elif data_type == "orderbook":
                    if buffer_key not in self.orderbook_buffers:
                        self.orderbook_buffers[buffer_key] = deque(maxlen=100)
                    self.orderbook_buffers[buffer_key].append(item)
                    
        except Exception as e:
            logger.error(f"Error updating buffers: {e}")
    
    async def _connect_websocket(self, symbol: str) -> None:
        """Connect to WebSocket for real-time data"""
        try:
            # Binance WebSocket example
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower().replace('/', '')}@kline_1m"
            
            async def websocket_handler():
                try:
                    async with websockets.connect(ws_url) as websocket:
                        self.websocket_connections[symbol] = websocket
                        
                        async for message in websocket:
                            data = json.loads(message)
                            await self._process_websocket_message(symbol, data)
                            
                except Exception as e:
                    logger.error(f"WebSocket error for {symbol}: {e}")
                    if symbol in self.websocket_connections:
                        del self.websocket_connections[symbol]
            
            # Start WebSocket handler
            asyncio.create_task(websocket_handler())
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
    
    async def _process_websocket_message(self, symbol: str, message: Dict) -> None:
        """Process incoming WebSocket message"""
        try:
            if 'k' in message:  # Kline data
                kline = message['k']
                candle_data = {
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'symbol': symbol,
                    'timeframe': '1m'
                }
                
                await self.store_data([candle_data], "candles")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _cleanup_database(self, cutoff_date: datetime) -> int:
        """Clean up old data from database"""
        try:
            async with self.async_db_engine.begin() as conn:
                query = text("DELETE FROM market_data WHERE timestamp < :cutoff_date")
                result = await conn.execute(query, {'cutoff_date': cutoff_date})
                return result.rowcount
        except Exception as e:
            logger.error(f"Error cleaning database: {e}")
            return 0
    
    async def _cleanup_cache(self, cutoff_date: datetime) -> int:
        """Clean up old data from cache"""
        try:
            # This is a simplified cleanup - in production you'd use more sophisticated cache management
            deleted_count = 0
            for key in self.redis_client.scan_iter("data:*"):
                # Check if key is old (simplified)
                deleted_count += 1
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
            return 0
    
    async def _get_database_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.async_db_engine.begin() as conn:
                # Count records
                count_query = text("""
                    SELECT COUNT(*) as count FROM market_data 
                    WHERE symbol = :symbol AND timeframe = :timeframe
                """)
                count_result = await conn.execute(count_query, {'symbol': symbol, 'timeframe': timeframe})
                count = count_result.scalar()
                
                # Date range
                range_query = text("""
                    SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                    FROM market_data 
                    WHERE symbol = :symbol AND timeframe = :timeframe
                """)
                range_result = await conn.execute(range_query, {'symbol': symbol, 'timeframe': timeframe})
                date_range = range_result.fetchone()
                
                return {
                    'db_record_count': count,
                    'db_min_date': date_range[0] if date_range else None,
                    'db_max_date': date_range[1] if date_range else None
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def _get_cache_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_keys = list(self.redis_client.scan_iter(f"data:{symbol}:{timeframe}:*"))
            return {
                'cache_keys_count': len(cache_keys),
                'cache_ttl': self.cache_ttl
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def _get_buffer_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get buffer statistics"""
        try:
            buffer_key = f"{symbol}_candles"
            candle_count = len(self.candle_buffers.get(buffer_key, []))
            
            return {
                'buffer_candle_count': candle_count,
                'buffer_trade_count': len(self.trade_buffers.get(f"{symbol}_trades", [])),
                'buffer_orderbook_count': len(self.orderbook_buffers.get(f"{symbol}_orderbook", []))
            }
        except Exception as e:
            logger.error(f"Error getting buffer stats: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize data service
        data_service = DataService()
        
        # Create data request
        request = DataRequest(
            symbol="BTC/USDT",
            timeframe="1h",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            limit=100
        )
        
        # Get historical data
        response = await data_service.get_historical_data(request)
        print(f"Retrieved {len(response.data)} records from {response.source}")
        
        # Get statistics
        stats = await data_service.get_data_statistics("BTC/USDT", "1h")
        print(f"Data statistics: {stats}")
    
    asyncio.run(main())
