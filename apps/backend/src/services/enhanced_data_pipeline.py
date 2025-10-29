#!/usr/bin/env python3
"""
Enhanced Data Pipeline for AlphaPlus
Ultra-low latency data processing with Redis cache integration
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# Import existing components
from .enhanced_cache_manager import EnhancedCacheManager
from src.data.candlestick_collector import CandlestickCollector
from src.core.websocket_binance import BinanceWebSocketClient
from src.database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_processing_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    errors_count: int = 0
    last_update: datetime = None

class EnhancedDataPipeline:
    """
    Enhanced data pipeline with Redis cache integration
    Provides ultra-low latency data processing while maintaining TimescaleDB compatibility
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 database_config: Dict = None,
                 symbols: List[str] = None,
                 timeframes: List[str] = None,
                 enable_cache: bool = True,
                 enable_websocket: bool = True):
        """
        Initialize enhanced data pipeline
        
        Args:
            redis_url: Redis connection URL
            database_config: TimescaleDB configuration
            symbols: Trading symbols to process
            timeframes: Timeframes to process
            enable_cache: Enable Redis caching
            enable_websocket: Enable WebSocket real-time data
        """
        self.redis_url = redis_url
        self.database_config = database_config or {
            'host': 'postgres',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h']
        self.enable_cache = enable_cache
        self.enable_websocket = enable_websocket
        
        # Initialize components
        self.cache_manager = EnhancedCacheManager(
            redis_url=redis_url,
            enable_redis=enable_cache
        )
        self.db_connection = TimescaleDBConnection(self.database_config)
        self.candlestick_collector = CandlestickCollector()
        self.websocket_client = BinanceWebSocketClient(
            symbols=[s.replace('/', '') for s in self.symbols],
            timeframes=self.timeframes
        ) if enable_websocket else None
        
        # Data buffers
        self.data_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.signal_buffers = defaultdict(lambda: deque(maxlen=100))
        self.pattern_buffers = defaultdict(lambda: deque(maxlen=100))
        
        # Pipeline state
        self.is_running = False
        self.processing_tasks = []
        
        # Performance metrics
        self.metrics = PipelineMetrics()
        self.processing_times = deque(maxlen=1000)
        self.latency_times = deque(maxlen=1000)
        
        # Callbacks for real-time updates
        self.data_callbacks = []
        self.signal_callbacks = []
        self.pattern_callbacks = []
        
        logger.info("🚀 Enhanced Data Pipeline initialized")
    
    async def initialize(self):
        """Initialize pipeline components"""
        try:
            # Initialize database connection
            await self.db_connection.initialize()
            logger.info("✅ Database connection initialized")
            
            # Initialize cache manager
            await self.cache_manager._test_redis_connection()
            logger.info("✅ Cache manager initialized")
            
            # Initialize WebSocket client
            if self.websocket_client:
                await self.websocket_client.connect()
                logger.info("✅ WebSocket client initialized")
            
            logger.info("🎉 Enhanced Data Pipeline fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}")
            raise
    
    async def start(self):
        """Start the enhanced data pipeline"""
        try:
            self.is_running = True
            logger.info("🚀 Starting Enhanced Data Pipeline...")
            
            # Start data collection tasks
            tasks = []
            
            # Start WebSocket data collection
            if self.websocket_client:
                tasks.append(asyncio.create_task(self._websocket_data_collection()))
            
            # Start REST API data collection (fallback)
            tasks.append(asyncio.create_task(self._rest_data_collection()))
            
            # Start cache management
            tasks.append(asyncio.create_task(self._cache_management()))
            
            # Start performance monitoring
            tasks.append(asyncio.create_task(self._performance_monitoring()))
            
            # Start data processing
            tasks.append(asyncio.create_task(self._data_processing_loop()))
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"❌ Error in data pipeline: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the enhanced data pipeline"""
        try:
            self.is_running = False
            logger.info("🛑 Stopping Enhanced Data Pipeline...")
            
            # Cancel all tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Close connections
            if self.websocket_client:
                await self.websocket_client.disconnect()
            
            await self.cache_manager.close()
            await self.db_connection.close()
            
            logger.info("✅ Enhanced Data Pipeline stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping pipeline: {e}")
    
    async def _websocket_data_collection(self):
        """Collect real-time data via WebSocket"""
        try:
            logger.info("📡 Starting WebSocket data collection...")
            
            async for data in self.websocket_client.listen():
                if not self.is_running:
                    break
                
                start_time = time.time()
                
                try:
                    # Process incoming data
                    processed_data = await self._process_incoming_data(data)
                    
                    if processed_data:
                        # Store in cache for ultra-fast access
                        await self._store_data_in_cache(processed_data)
                        
                        # Store in database for persistence
                        await self._store_data_in_database(processed_data)
                        
                        # Update buffers
                        await self._update_buffers(processed_data)
                        
                        # Notify callbacks
                        await self._notify_data_callbacks(processed_data)
                        
                        # Record metrics
                        self._record_processing_metrics(start_time)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing WebSocket data: {e}")
                    self.metrics.errors_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error in WebSocket data collection: {e}")
    
    async def _rest_data_collection(self):
        """Collect data via REST API (fallback)"""
        try:
            logger.info("🌐 Starting REST API data collection...")
            
            while self.is_running:
                try:
                    for symbol in self.symbols:
                        for timeframe in self.timeframes:
                            # Check cache first
                            cached_data = await self.cache_manager.get_market_data(symbol, timeframe)
                            
                            if cached_data:
                                self.metrics.cache_hits += 1
                                continue
                            
                            # Fetch from exchange if not in cache
                            start_time = time.time()
                            exchange_data = await self._fetch_from_exchange(symbol, timeframe)
                            
                            if exchange_data:
                                # Process and store data
                                processed_data = await self._process_exchange_data(exchange_data, symbol, timeframe)
                                
                                if processed_data:
                                    await self._store_data_in_cache(processed_data)
                                    await self._store_data_in_database(processed_data)
                                    await self._update_buffers(processed_data)
                                    
                                    self._record_processing_metrics(start_time)
                            
                            self.metrics.cache_misses += 1
                    
                    # Wait before next collection cycle
                    await asyncio.sleep(30)  # 30-second intervals
                    
                except Exception as e:
                    logger.error(f"❌ Error in REST data collection: {e}")
                    self.metrics.errors_count += 1
                    await asyncio.sleep(60)  # Wait longer on error
            
        except Exception as e:
            logger.error(f"❌ Error in REST data collection: {e}")
    
    async def _process_incoming_data(self, data: Dict) -> Optional[Dict]:
        """Process incoming WebSocket data"""
        try:
            if 'e' not in data or data['e'] != 'kline':
                return None
            
            kline = data['k']
            symbol = data['s']
            timeframe = self._convert_timeframe(kline['i'])
            
            processed_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_complete': kline['x'],
                'source': 'websocket'
            }
            
            # Calculate technical indicators if candlestick is complete
            if processed_data['is_complete']:
                indicators = await self._calculate_indicators(processed_data)
                processed_data['indicators'] = indicators
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing incoming data: {e}")
            return None
    
    async def _process_exchange_data(self, data: List, symbol: str, timeframe: str) -> Optional[Dict]:
        """Process exchange data from REST API"""
        try:
            if not data or len(data) == 0:
                return None
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Get latest data point
            latest = df.iloc[-1]
            
            processed_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': latest['timestamp'],
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
                'volume': float(latest['volume']),
                'is_complete': True,
                'source': 'rest_api'
            }
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(processed_data)
            processed_data['indicators'] = indicators
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing exchange data: {e}")
            return None
    
    async def _calculate_indicators(self, data: Dict) -> Dict:
        """Calculate technical indicators for data point"""
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            
            # Get historical data for calculations
            historical_data = await self._get_historical_data(symbol, timeframe, limit=100)
            
            if not historical_data or len(historical_data) < 20:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Calculate indicators
            indicators = {}
            
            # RSI
            if len(df) >= 14:
                indicators['rsi'] = self._calculate_rsi(df['close'].values, period=14)
            
            # MACD
            if len(df) >= 26:
                macd_data = self._calculate_macd(df['close'].values)
                indicators['macd'] = macd_data['macd']
                indicators['macd_signal'] = macd_data['signal']
                indicators['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            if len(df) >= 20:
                bb_data = self._calculate_bollinger_bands(df['close'].values, period=20)
                indicators['bb_upper'] = bb_data['upper']
                indicators['bb_middle'] = bb_data['middle']
                indicators['bb_lower'] = bb_data['lower']
            
            # ATR
            if len(df) >= 14:
                indicators['atr'] = self._calculate_atr(df, period=14)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get historical data from cache or database"""
        try:
            # Try cache first
            cached_data = await self.cache_manager.get_market_data(symbol, timeframe, limit)
            if cached_data:
                return cached_data
            
            # Fallback to database
            async with self.db_connection.get_async_session() as session:
                query = f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM enhanced_market_data
                    WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                """
                result = await session.execute(query)
                data = [dict(row) for row in result]
                return data
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return []
    
    async def _store_data_in_cache(self, data: Dict):
        """Store data in cache for fast access"""
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            
            # Store market data
            await self.cache_manager.store_market_data(symbol, timeframe, [data], ttl=300)
            
            # Store real-time data for WebSocket delivery
            real_time_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': data['timestamp'].isoformat(),
                'price': data['close'],
                'volume': data['volume'],
                'indicators': data.get('indicators', {}),
                'source': data['source']
            }
            await self.cache_manager.store_real_time_data(symbol, timeframe, real_time_data, ttl=60)
            
        except Exception as e:
            logger.error(f"❌ Error storing data in cache: {e}")
    
    async def _store_data_in_database(self, data: Dict):
        """Store data in TimescaleDB for persistence"""
        try:
            async with self.db_connection.get_async_session() as session:
                query = """
                    INSERT INTO enhanced_market_data (
                        symbol, timeframe, timestamp, open, high, low, close, volume,
                        rsi, macd, macd_signal, bollinger_upper, bollinger_lower, bollinger_middle, atr,
                        data_quality_score, created_at
                    ) VALUES (
                        :symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume,
                        :rsi, :macd, :macd_signal, :bb_upper, :bb_lower, :bb_middle, :atr,
                        :quality_score, NOW()
                    )
                """
                
                indicators = data.get('indicators', {})
                
                await session.execute(query, {
                    'symbol': data['symbol'],
                    'timeframe': data['timeframe'],
                    'timestamp': data['timestamp'],
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume'],
                    'rsi': indicators.get('rsi'),
                    'macd': indicators.get('macd'),
                    'macd_signal': indicators.get('macd_signal'),
                    'bb_upper': indicators.get('bb_upper'),
                    'bb_lower': indicators.get('bb_lower'),
                    'bb_middle': indicators.get('bb_middle'),
                    'atr': indicators.get('atr'),
                    'quality_score': 0.95  # High quality for real-time data
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing data in database: {e}")
    
    async def _update_buffers(self, data: Dict):
        """Update in-memory buffers"""
        try:
            symbol = data['symbol']
            timeframe = data['timeframe']
            
            # Update data buffer
            self.data_buffers[symbol][timeframe].append(data)
            
            # Update signal buffer if new signal generated
            if 'signal' in data:
                self.signal_buffers[symbol].append(data['signal'])
            
            # Update pattern buffer if new pattern detected
            if 'pattern' in data:
                self.pattern_buffers[symbol].append(data['pattern'])
                
        except Exception as e:
            logger.error(f"❌ Error updating buffers: {e}")
    
    async def _notify_data_callbacks(self, data: Dict):
        """Notify registered callbacks of new data"""
        try:
            for callback in self.data_callbacks:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Error in data callback: {e}")
        except Exception as e:
            logger.error(f"❌ Error notifying callbacks: {e}")
    
    async def _cache_management(self):
        """Manage cache operations"""
        try:
            while self.is_running:
                try:
                    # Clean up expired entries
                    await self.cache_manager.cleanup_expired_entries()
                    
                    # Update metrics
                    self.metrics.last_update = datetime.now()
                    
                    # Wait before next cleanup
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Error in cache management: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in cache management: {e}")
    
    async def _performance_monitoring(self):
        """Monitor pipeline performance"""
        try:
            while self.is_running:
                try:
                    # Get cache statistics
                    cache_stats = await self.cache_manager.get_cache_stats()
                    
                    # Update pipeline metrics
                    self.metrics.cache_hits = cache_stats['memory_hits'] + cache_stats['redis_hits']
                    self.metrics.cache_misses = cache_stats['misses']
                    self.metrics.avg_latency_ms = cache_stats['avg_response_time_ms']
                    
                    # Calculate average processing time
                    if self.processing_times:
                        self.metrics.avg_processing_time_ms = np.mean(self.processing_times)
                    
                    # Log performance metrics
                    logger.info(f"📊 Pipeline Performance - "
                              f"Cache Hit Rate: {cache_stats['hit_rate']}%, "
                              f"Avg Latency: {self.metrics.avg_latency_ms:.2f}ms, "
                              f"Errors: {self.metrics.errors_count}")
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(60)  # 1 minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in performance monitoring: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in performance monitoring: {e}")
    
    async def _data_processing_loop(self):
        """Main data processing loop"""
        try:
            while self.is_running:
                try:
                    # Process any pending data
                    await self._process_pending_data()
                    
                    # Wait before next processing cycle
                    await asyncio.sleep(1)  # 1 second
                    
                except Exception as e:
                    logger.error(f"❌ Error in data processing loop: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Error in data processing loop: {e}")
    
    async def _process_pending_data(self):
        """Process any pending data in buffers"""
        try:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    if symbol in self.data_buffers and timeframe in self.data_buffers[symbol]:
                        buffer = self.data_buffers[symbol][timeframe]
                        
                        if len(buffer) > 0:
                            # Process latest data point
                            latest_data = buffer[-1]
                            
                            # Generate signals if conditions are met
                            if latest_data.get('is_complete', False):
                                signal = await self._generate_signal(latest_data)
                                if signal:
                                    await self._store_signal(signal)
                            
                            # Detect patterns if conditions are met
                            pattern = await self._detect_pattern(latest_data)
                            if pattern:
                                await self._store_pattern(pattern)
                
        except Exception as e:
            logger.error(f"❌ Error processing pending data: {e}")
    
    async def _generate_signal(self, data: Dict) -> Optional[Dict]:
        """Generate trading signal from data"""
        try:
            # Implement signal generation logic here
            # This is a placeholder - integrate with your existing signal generation
            return None
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            return None
    
    async def _detect_pattern(self, data: Dict) -> Optional[Dict]:
        """Detect patterns from data"""
        try:
            # Implement pattern detection logic here
            # This is a placeholder - integrate with your existing pattern detection
            return None
        except Exception as e:
            logger.error(f"❌ Error detecting pattern: {e}")
            return None
    
    async def _store_signal(self, signal: Dict):
        """Store signal in cache and database"""
        try:
            symbol = signal['symbol']
            timeframe = signal['timeframe']
            
            # Store in cache
            await self.cache_manager.store_signals(symbol, timeframe, [signal])
            
            # Store in database
            async with self.db_connection.get_async_session() as session:
                query = """
                    INSERT INTO signal_history (
                        signal_id, symbol, timeframe, direction, signal_type,
                        entry_price, stop_loss, take_profit, confidence,
                        pattern_type, signal_generated_at, status, created_at
                    ) VALUES (
                        :signal_id, :symbol, :timeframe, :direction, :signal_type,
                        :entry_price, :stop_loss, :take_profit, :confidence,
                        :pattern_type, :generated_at, 'generated', NOW()
                    )
                """
                
                await session.execute(query, signal)
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing signal: {e}")
    
    async def _store_pattern(self, pattern: Dict):
        """Store pattern in database"""
        try:
            async with self.db_connection.get_async_session() as session:
                query = """
                    INSERT INTO pattern_detections (
                        pattern_id, symbol, timeframe, pattern_type, pattern_category,
                        direction, confidence, strength, entry_price, stop_loss, take_profit,
                        risk_reward_ratio, pattern_start_time, pattern_end_time,
                        volume_confirmation, technical_indicators, data_points_used,
                        data_quality_score, status, created_at
                    ) VALUES (
                        :pattern_id, :symbol, :timeframe, :pattern_type, :pattern_category,
                        :direction, :confidence, :strength, :entry_price, :stop_loss, :take_profit,
                        :risk_reward_ratio, :start_time, :end_time,
                        :volume_confirmation, :indicators, :data_points, :quality_score,
                        'active', NOW()
                    )
                """
                
                await session.execute(query, pattern)
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing pattern: {e}")
    
    async def _fetch_from_exchange(self, symbol: str, timeframe: str) -> Optional[List]:
        """Fetch data from exchange via REST API"""
        try:
            # This is a placeholder - integrate with your existing exchange connector
            # For now, return None to indicate no data available
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching from exchange: {e}")
            return None
    
    def _convert_timeframe(self, interval: str) -> str:
        """Convert Binance interval to standard timeframe"""
        timeframe_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        return timeframe_map.get(interval, interval)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        try:
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self._calculate_ema(np.array([macd_line]), signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line),
                'signal': float(signal_line),
                'histogram': float(histogram)
            }
        except Exception:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2 / (period + 1)
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            return ema
        except Exception:
            return prices[-1] if len(prices) > 0 else 0.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}
            
            recent_prices = prices[-period:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            return {
                'upper': float(middle + (std_dev * std)),
                'middle': float(middle),
                'lower': float(middle - (std_dev * std))
            }
        except Exception:
            return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr[-period:])
            
            return float(atr)
        except Exception:
            return 0.0
    
    def _record_processing_metrics(self, start_time: float):
        """Record processing time metrics"""
        try:
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            self.metrics.total_processed += 1
        except Exception as e:
            logger.error(f"❌ Error recording metrics: {e}")
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        try:
            cache_stats = await self.cache_manager.get_cache_stats()
            
            return {
                'pipeline_running': self.is_running,
                'total_processed': self.metrics.total_processed,
                'cache_hit_rate': cache_stats['hit_rate'],
                'avg_latency_ms': self.metrics.avg_latency_ms,
                'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
                'errors_count': self.metrics.errors_count,
                'last_update': self.metrics.last_update.isoformat() if self.metrics.last_update else None,
                'symbols_count': len(self.symbols),
                'timeframes_count': len(self.timeframes),
                'websocket_enabled': self.enable_websocket,
                'cache_enabled': self.enable_cache,
                'cache_stats': cache_stats
            }
        except Exception as e:
            logger.error(f"❌ Error getting pipeline stats: {e}")
            return {}
    
    def add_data_callback(self, callback: Callable):
        """Add callback for data updates"""
        self.data_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable):
        """Add callback for signal updates"""
        self.signal_callbacks.append(callback)
    
    def add_pattern_callback(self, callback: Callable):
        """Add callback for pattern updates"""
        self.pattern_callbacks.append(callback)
