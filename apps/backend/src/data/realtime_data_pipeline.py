#!/usr/bin/env python3
"""
Real-time Data Pipeline for AlphaPlus TimescaleDB Integration
Handles WebSocket data ingestion, Redis caching, and TimescaleDB storage
"""

import asyncio
import logging
import json
import redis
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class OHLCVData:
    """OHLCV data structure"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    source: str = "websocket"

class RealTimeDataPipeline:
    """Real-time data pipeline for WebSocket to TimescaleDB"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 db_url: str = None):
        """
        Initialize the data pipeline
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            db_url: TimescaleDB connection URL
        """
        # Note: Actual Redis is on port 56379, but connecting via localhost:6379 (update if needed)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.db_pool = None
        
        # Data buffers (increased capacity for 100 symbols)
        self.ohlcv_buffer = {}  # Supports unlimited symbols
        self.orderbook_buffer = {}
        self.max_buffer_size_per_symbol = 200  # Keep last 200 candles per symbol
        
        # Performance tracking
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'db_inserts': 0,
            'redis_writes': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Database connection pool closed")
    
    async def process_websocket_message(self, message: Dict[str, Any]) -> bool:
        """
        Process incoming WebSocket message and store in Redis/TimescaleDB
        
        Args:
            message: WebSocket message data
            
        Returns:
            bool: True if processed successfully
        """
        try:
            self.stats['messages_received'] += 1
            
            # Process different message types
            if message.get('type') == 'kline':
                await self._process_kline_data(message)
            elif message.get('type') == 'orderbook':
                await self._process_orderbook_data(message)
            elif message.get('type') == 'trade':
                await self._process_trade_data(message)
            else:
                logger.debug(f"Unknown message type: {message.get('type')}")
                return False
            
            self.stats['messages_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing WebSocket message: {e}")
            self.stats['errors'] += 1
            return False
    
    async def _process_kline_data(self, message: Dict[str, Any]):
        """Process kline/candlestick data"""
        try:
            # Extract kline data
            symbol = message.get('symbol', '').upper()
            timeframe = message.get('timeframe', '1m')
            timestamp_str = message.get('timestamp')
            
            if not all([symbol, timeframe, timestamp_str]):
                logger.warning("Incomplete kline data received")
                return
            
            # Convert timestamp to timezone-aware datetime
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str).tz_localize('UTC')
            else:
                timestamp = timestamp_str
            
            # Robust volume parsing with error handling
            def safe_float(value, default=0.0):
                """Safely convert to float with error handling"""
                try:
                    if isinstance(value, str):
                        return float(value) if value else default
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse volume value: {value}, using default: {default}")
                    return default
            
            def safe_int(value, default=0):
                """Safely convert to int with error handling"""
                try:
                    if isinstance(value, str):
                        return int(float(value)) if value else default
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse int value: {value}, using default: {default}")
                    return default
            
            # Create OHLCV data object with safe parsing
            ohlcv = OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=safe_float(message.get('open', 0)),
                high=safe_float(message.get('high', 0)),
                low=safe_float(message.get('low', 0)),
                close=safe_float(message.get('close', 0)),
                volume=safe_float(message.get('volume', 0)),
                quote_volume=safe_float(message.get('quote_volume', 0)) if message.get('quote_volume') else None,
                trades_count=safe_int(message.get('trades', 0)) if message.get('trades') else None,
                source="websocket"
            )
            
            # Store in Redis buffer
            await self._store_ohlcv_redis(ohlcv)
            
            # Store in TimescaleDB (batch insert)
            await self._store_ohlcv_timescaledb(ohlcv)
            
        except Exception as e:
            logger.error(f"❌ Error processing kline data: {e}")
            raise
    
    async def _process_orderbook_data(self, message: Dict[str, Any]):
        """Process order book data"""
        try:
            symbol = message.get('symbol', '').upper()
            timestamp_str = message.get('timestamp')
            
            # Convert timestamp to timezone-aware datetime
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str).tz_localize('UTC')
            else:
                timestamp = timestamp_str
            bids = message.get('bids', [])
            asks = message.get('asks', [])
            
            if not all([symbol, timestamp, bids, asks]):
                logger.warning("Incomplete orderbook data received")
                return
            
            # Calculate best bid/ask and spread
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0
            mid_price = (best_bid + best_ask) / 2 if best_ask > 0 and best_bid > 0 else 0
            
            # Store in Redis
            orderbook_key = f"orderbook:{symbol}:{timestamp}"
            orderbook_data = {
                'symbol': symbol,
                'timestamp': timestamp.isoformat(),
                'bids': bids,
                'asks': asks,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'mid_price': mid_price,
                'source': 'websocket'
            }
            
            self.redis_client.setex(orderbook_key, 3600, json.dumps(orderbook_data))  # 1 hour TTL
            self.stats['redis_writes'] += 1
            
            # Store in TimescaleDB (batch insert)
            await self._store_orderbook_timescaledb(orderbook_data)
            
        except Exception as e:
            logger.error(f"❌ Error processing orderbook data: {e}")
            raise
    
    async def _process_trade_data(self, message: Dict[str, Any]):
        """Process trade data"""
        try:
            symbol = message.get('symbol', '').upper()
            timestamp_str = message.get('timestamp')
            
            # Convert timestamp to timezone-aware datetime
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str).tz_localize('UTC')
            else:
                timestamp = timestamp_str
            price = float(message.get('price', 0))
            quantity = float(message.get('quantity', 0))
            side = message.get('side', 'unknown')
            
            if not all([symbol, timestamp, price, quantity]):
                logger.warning("Incomplete trade data received")
                return
            
            # Store in Redis for real-time analysis
            trade_key = f"trade:{symbol}:{timestamp}"
            trade_data = {
                'symbol': symbol,
                'timestamp': timestamp.isoformat(),
                'price': price,
                'quantity': quantity,
                'side': side,
                'source': 'websocket'
            }
            
            self.redis_client.setex(trade_key, 300, json.dumps(trade_data))  # 5 minutes TTL
            self.stats['redis_writes'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing trade data: {e}")
            raise
    
    async def _store_ohlcv_redis(self, ohlcv: OHLCVData):
        """Store OHLCV data in Redis buffer"""
        try:
            buffer_key = f"ohlcv_buffer:{ohlcv.symbol}:{ohlcv.timeframe}"
            
            # Add to Redis list (FIFO)
            ohlcv_data = {
                'symbol': ohlcv.symbol,
                'timeframe': ohlcv.timeframe,
                'timestamp': ohlcv.timestamp.isoformat(),
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume,
                'quote_volume': ohlcv.quote_volume,
                'trades_count': ohlcv.trades_count,
                'source': ohlcv.source
            }
            
            self.redis_client.lpush(buffer_key, json.dumps(ohlcv_data))
            
            # Keep only last 1000 items per symbol/timeframe
            self.redis_client.ltrim(buffer_key, 0, 999)
            
            self.stats['redis_writes'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error storing OHLCV in Redis: {e}")
            raise
    
    async def _store_ohlcv_timescaledb(self, ohlcv: OHLCVData):
        """Store OHLCV data in TimescaleDB"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ohlcv_data (
                        symbol, timeframe, timestamp, open, high, low, close, volume,
                        quote_volume, trades_count, source, data_quality_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT DO NOTHING
                """, 
                ohlcv.symbol, ohlcv.timeframe, ohlcv.timestamp, ohlcv.open, ohlcv.high, 
                ohlcv.low, ohlcv.close, ohlcv.volume, ohlcv.quote_volume, 
                ohlcv.trades_count, ohlcv.source, 1.0)
                
                self.stats['db_inserts'] += 1
                
        except Exception as e:
            logger.error(f"❌ Error storing OHLCV in TimescaleDB: {e}")
            raise
    
    async def _store_orderbook_timescaledb(self, orderbook_data: Dict[str, Any]):
        """Store order book data in TimescaleDB"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO order_book_data (
                        symbol, timestamp, bids, asks, best_bid, best_ask, spread, mid_price, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT DO NOTHING
                """,
                orderbook_data['symbol'], 
                datetime.fromisoformat(orderbook_data['timestamp'].replace('Z', '+00:00')),
                json.dumps(orderbook_data['bids']),
                json.dumps(orderbook_data['asks']),
                orderbook_data['best_bid'],
                orderbook_data['best_ask'],
                orderbook_data['spread'],
                orderbook_data['mid_price'],
                orderbook_data['source'])
                
                self.stats['db_inserts'] += 1
                
        except Exception as e:
            logger.error(f"❌ Error storing orderbook in TimescaleDB: {e}")
            raise
    
    async def batch_flush_redis_to_timescaledb(self, batch_size: int = 1000):
        """Flush Redis buffer to TimescaleDB in batches"""
        try:
            # Get all buffer keys
            buffer_keys = self.redis_client.keys("ohlcv_buffer:*")
            
            for buffer_key in buffer_keys:
                # Get batch of data from Redis
                data_items = self.redis_client.lrange(buffer_key, 0, batch_size - 1)
                
                if not data_items:
                    continue
                
                # Parse and prepare batch data
                batch_data = []
                for item in data_items:
                    try:
                        data = json.loads(item)
                        batch_data.append(data)
                    except json.JSONDecodeError:
                        continue
                
                if not batch_data:
                    continue
                
                # Batch insert to TimescaleDB
                async with self.db_pool.acquire() as conn:
                    await conn.executemany("""
                        INSERT INTO ohlcv_data (
                            symbol, timeframe, timestamp, open, high, low, close, volume,
                            quote_volume, trades_count, source, data_quality_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """, [
                        (
                            item['symbol'], item['timeframe'], 
                            datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                            item['open'], item['high'], item['low'], item['close'], item['volume'],
                            item.get('quote_volume'), item.get('trades_count'), item['source'], 1.0
                        )
                        for item in batch_data
                    ])
                
                # Remove processed items from Redis
                self.redis_client.ltrim(buffer_key, batch_size, -1)
                
                logger.info(f"✅ Flushed {len(batch_data)} items from {buffer_key}")
                
        except Exception as e:
            logger.error(f"❌ Error in batch flush: {e}")
            raise
    
    async def get_latest_ohlcv_data(self, symbol: str, timeframe: str, periods: int = 100) -> List[Dict[str, Any]]:
        """Get latest OHLCV data from TimescaleDB"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT timestamp, open, high, low, close, volume, quote_volume, trades_count
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """, symbol, timeframe, periods)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting latest OHLCV data: {e}")
            return []
    
    async def calculate_technical_indicators(self, symbol: str, timeframe: str):
        """Calculate and store technical indicators"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get latest OHLCV data
                ohlcv_data = await self.get_latest_ohlcv_data(symbol, timeframe, 100)
                
                if len(ohlcv_data) < 20:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    return
                
                # Convert to DataFrame for calculations
                df = pd.DataFrame(ohlcv_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Calculate indicators
                indicators = {}
                
                # SMA 20
                if len(df) >= 20:
                    indicators['SMA_20'] = df['close'].rolling(window=20).mean().iloc[-1]
                
                # SMA 50
                if len(df) >= 50:
                    indicators['SMA_50'] = df['close'].rolling(window=50).mean().iloc[-1]
                
                # RSI 14
                if len(df) >= 14:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    indicators['RSI_14'] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # ATR 14
                if len(df) >= 14:
                    try:
                        high_low = df['high'] - df['low']
                        high_close = np.abs(df['high'] - df['close'].shift())
                        low_close = np.abs(df['low'] - df['close'].shift())
                        
                        # Convert to numpy arrays to avoid decimal issues
                        high_low_vals = high_low.values
                        high_close_vals = high_close.values
                        low_close_vals = low_close.values
                        
                        true_range_vals = np.maximum(high_low_vals, np.maximum(high_close_vals, low_close_vals))
                        true_range = pd.Series(true_range_vals, index=df.index)
                        
                        indicators['ATR_14'] = true_range.rolling(window=14).mean().iloc[-1]
                    except Exception as e:
                        logger.warning(f"Failed to calculate ATR: {e}")
                        indicators['ATR_14'] = None
                
                # Store indicators
                current_time = datetime.now(timezone.utc)
                for indicator_name, value in indicators.items():
                    try:
                        # Validate and convert value
                        if pd.notna(value) and not pd.isnull(value):
                            # Convert to float and check for valid numeric value
                            float_value = float(value)
                            if not (np.isnan(float_value) or np.isinf(float_value)):
                                await conn.execute("""
                                    INSERT INTO technical_indicators (
                                        symbol, timeframe, timestamp, indicator_name, indicator_value, calculation_method
                                    ) VALUES ($1, $2, $3, $4, $5, $6)
                                    ON CONFLICT DO NOTHING
                                """, symbol, timeframe, current_time, indicator_name, float_value, 'python_calculation')
                            else:
                                logger.debug(f"Skipping invalid indicator {indicator_name}: {value}")
                        else:
                            logger.debug(f"Skipping null indicator {indicator_name}: {value}")
                    except (ValueError, TypeError, OverflowError) as e:
                        logger.warning(f"Failed to convert indicator {indicator_name} value {value}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to store indicator {indicator_name}: {e}")
                        continue
                
                logger.info(f"✅ Calculated indicators for {symbol} {timeframe}: {list(indicators.keys())}")
                
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't raise the error, just log it and continue
            logger.warning("Continuing without indicator calculation...")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        return {
            'messages_received': self.stats['messages_received'],
            'messages_processed': self.stats['messages_processed'],
            'db_inserts': self.stats['db_inserts'],
            'redis_writes': self.stats['redis_writes'],
            'errors': self.stats['errors'],
            'success_rate': (self.stats['messages_processed'] / max(self.stats['messages_received'], 1)) * 100
        }

# Example usage
async def main():
    """Example usage of the real-time data pipeline"""
    pipeline = RealTimeDataPipeline()
    
    try:
        await pipeline.initialize()
        
        # Simulate WebSocket message processing
        sample_message = {
            'type': 'kline',
            'symbol': 'BTCUSDT',
            'timeframe': '1m',
            'timestamp': datetime.now(timezone.utc),
            'open': 45000.0,
            'high': 45100.0,
            'low': 44900.0,
            'close': 45050.0,
            'volume': 100.5,
            'quote_volume': 4525000.0,
            'trades': 150
        }
        
        await pipeline.process_websocket_message(sample_message)
        
        # Calculate indicators
        await pipeline.calculate_technical_indicators('BTCUSDT', '1m')
        
        # Get stats
        stats = pipeline.get_pipeline_stats()
        print(f"Pipeline stats: {stats}")
        
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
