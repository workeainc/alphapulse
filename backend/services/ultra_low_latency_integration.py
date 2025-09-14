"""
Ultra-Low Latency Integration Service for AlphaPlus
Combines WebSocket client, vectorized pattern detection, and enhanced storage
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import uuid

from core.ultra_low_latency_websocket import UltraLowLatencyWebSocketClient
from strategies.vectorized_pattern_detector import VectorizedPatternDetector, VectorizedPattern
from database.connection import TimescaleDBConnection
from database.advanced_indexing import AdvancedIndexingManager

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for ultra-low latency integration"""
    symbols: List[str]
    timeframes: List[str]
    redis_url: str = "redis://localhost:6379"
    db_url: str = "postgresql://alpha_emon:password@localhost:5432/alphapulse"
    max_workers: int = 4
    enable_talib: bool = True
    enable_vectorized: bool = True
    enable_incremental: bool = True
    confidence_threshold: float = 0.7
    processing_batch_size: int = 100

class UltraLowLatencyIntegrationService:
    """
    Ultra-low latency integration service
    Achieves <20ms end-to-end latency from tick to signal
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        
        # Core components
        self.websocket_client = None
        self.pattern_detector = None
        self.db_connection = None
        self.indexing_manager = None
        
        # Performance tracking
        self.total_messages_processed = 0
        self.total_patterns_detected = 0
        self.total_signals_generated = 0
        self.avg_processing_latency_ms = 0.0
        self.max_processing_latency_ms = 0.0
        self.min_processing_latency_ms = float('inf')
        
        # Data buffers
        self.candlestick_buffers = {}
        self.pattern_buffers = {}
        self.signal_buffers = {}
        
        # Callbacks
        self.pattern_callbacks = []
        self.signal_callbacks = []
        self.error_callbacks = []
        
        # Status
        self.is_running = False
        self.is_initialized = False
        
        logger.info("🚀 Ultra-Low Latency Integration Service initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🔧 Initializing Ultra-Low Latency Integration Service...")
            
            # Initialize WebSocket client
            self.websocket_client = UltraLowLatencyWebSocketClient(self.config.redis_url)
            await self.websocket_client.initialize()
            
            # Initialize pattern detector
            self.pattern_detector = VectorizedPatternDetector(max_workers=self.config.max_workers)
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection(self.config.db_url)
            await self.db_connection.initialize()
            
            # Initialize indexing manager
            self.indexing_manager = AdvancedIndexingManager(self.db_connection.get_session_factory())
            await self.indexing_manager.create_all_advanced_indexes()
            
            # Initialize data buffers
            await self._initialize_buffers()
            
            self.is_initialized = True
            logger.info("✅ Ultra-Low Latency Integration Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize integration service: {e}")
            raise
    
    async def _initialize_buffers(self):
        """Initialize data buffers for each symbol and timeframe"""
        try:
            for symbol in self.config.symbols:
                self.candlestick_buffers[symbol] = {}
                self.pattern_buffers[symbol] = {}
                self.signal_buffers[symbol] = {}
                
                for timeframe in self.config.timeframes:
                    self.candlestick_buffers[symbol][timeframe] = []
                    self.pattern_buffers[symbol][timeframe] = []
                    self.signal_buffers[symbol][timeframe] = []
            
            logger.info(f"✅ Initialized buffers for {len(self.config.symbols)} symbols and {len(self.config.timeframes)} timeframes")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize buffers: {e}")
            raise
    
    async def start(self):
        """Start the ultra-low latency processing"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("🚀 Starting Ultra-Low Latency Integration Service...")
            
            # Connect to WebSocket with multiplexed streams
            await self.websocket_client.connect_multiplexed(
                self.config.symbols, 
                self.config.timeframes
            )
            
            self.is_running = True
            
            # Start processing with ultra-low latency callback
            await self.websocket_client.listen_ultra_fast(self._process_message_ultra_fast)
            
        except Exception as e:
            logger.error(f"❌ Failed to start integration service: {e}")
            raise
    
    async def stop(self):
        """Stop the ultra-low latency processing"""
        try:
            logger.info("🛑 Stopping Ultra-Low Latency Integration Service...")
            
            self.is_running = False
            
            if self.websocket_client:
                await self.websocket_client.disconnect()
            
            if self.pattern_detector:
                await self.pattern_detector.cleanup()
            
            if self.db_connection:
                await self.db_connection.close()
            
            logger.info("✅ Ultra-Low Latency Integration Service stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping integration service: {e}")
    
    async def _process_message_ultra_fast(self, message: Dict):
        """
        Ultra-fast message processing pipeline
        Achieves <20ms end-to-end latency
        """
        start_time = time.time()
        
        try:
            # Extract message data
            symbol = message.get('symbol')
            timeframe = message.get('timeframe')
            
            if not symbol or not timeframe:
                return
            
            # Store in candlestick buffer
            await self._store_candlestick_buffer(symbol, timeframe, message)
            
            # Check if we have enough data for pattern detection
            if len(self.candlestick_buffers[symbol][timeframe]) >= 20:
                # Convert to DataFrame for pattern detection
                df = self._convert_buffer_to_dataframe(symbol, timeframe)
                
                # Detect patterns with vectorized operations
                patterns = await self._detect_patterns_ultra_fast(symbol, timeframe, df)
                
                # Generate signals from patterns
                signals = await self._generate_signals_ultra_fast(symbol, timeframe, patterns)
                
                # Store results in database
                await self._store_results_ultra_fast(symbol, timeframe, patterns, signals)
                
                # Trigger callbacks
                await self._trigger_callbacks(patterns, signals)
            
            # Update performance metrics
            processing_latency_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_latency_ms)
            
            self.total_messages_processed += 1
            
        except Exception as e:
            logger.error(f"❌ Ultra-fast message processing error: {e}")
            await self._trigger_error_callbacks(e)
    
    async def _store_candlestick_buffer(self, symbol: str, timeframe: str, message: Dict):
        """Store candlestick data in buffer"""
        try:
            # Add to buffer
            self.candlestick_buffers[symbol][timeframe].append(message)
            
            # Keep only recent data for memory efficiency
            if len(self.candlestick_buffers[symbol][timeframe]) > 100:
                self.candlestick_buffers[symbol][timeframe] = \
                    self.candlestick_buffers[symbol][timeframe][-50:]
            
        except Exception as e:
            logger.error(f"❌ Buffer storage error: {e}")
    
    def _convert_buffer_to_dataframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Convert buffer data to DataFrame for pattern detection"""
        try:
            data = self.candlestick_buffers[symbol][timeframe]
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"❌ DataFrame conversion error: {e}")
            return pd.DataFrame()
    
    async def _detect_patterns_ultra_fast(self, symbol: str, timeframe: str, df: pd.DataFrame) -> List[VectorizedPattern]:
        """Detect patterns using ultra-fast vectorized operations"""
        try:
            if df.empty or len(df) < 5:
                return []
            
            # Detect patterns with vectorized operations
            patterns = await self.pattern_detector.detect_patterns_vectorized(
                df,
                use_talib=self.config.enable_talib,
                use_incremental=self.config.enable_incremental
            )
            
            # Filter by confidence threshold
            filtered_patterns = [
                pattern for pattern in patterns 
                if pattern.confidence >= self.config.confidence_threshold
            ]
            
            # Add symbol and timeframe info
            for pattern in filtered_patterns:
                pattern.metadata['symbol'] = symbol
                pattern.metadata['timeframe'] = timeframe
            
            # Store in pattern buffer
            self.pattern_buffers[symbol][timeframe].extend(filtered_patterns)
            
            # Keep only recent patterns
            if len(self.pattern_buffers[symbol][timeframe]) > 50:
                self.pattern_buffers[symbol][timeframe] = \
                    self.pattern_buffers[symbol][timeframe][-25:]
            
            self.total_patterns_detected += len(filtered_patterns)
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern detection error: {e}")
            return []
    
    async def _generate_signals_ultra_fast(self, symbol: str, timeframe: str, 
                                        patterns: List[VectorizedPattern]) -> List[Dict]:
        """Generate trading signals from patterns"""
        try:
            signals = []
            
            for pattern in patterns:
                if pattern.confidence >= self.config.confidence_threshold:
                    # Generate signal based on pattern type
                    signal = await self._create_signal_from_pattern(symbol, timeframe, pattern)
                    
                    if signal:
                        signals.append(signal)
                        
                        # Store in signal buffer
                        self.signal_buffers[symbol][timeframe].append(signal)
                        
                        # Keep only recent signals
                        if len(self.signal_buffers[symbol][timeframe]) > 20:
                            self.signal_buffers[symbol][timeframe] = \
                                self.signal_buffers[symbol][timeframe][-10:]
            
            self.total_signals_generated += len(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            return []
    
    async def _create_signal_from_pattern(self, symbol: str, timeframe: str, 
                                        pattern: VectorizedPattern) -> Optional[Dict]:
        """Create trading signal from detected pattern"""
        try:
            # Determine signal type based on pattern
            if pattern.pattern_type == 'bullish':
                signal_type = 'buy'
            elif pattern.pattern_type == 'bearish':
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Calculate entry price and risk management levels
            entry_price = pattern.price_level
            
            # Simple risk management (can be enhanced)
            if signal_type == 'buy':
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.04  # 4% take profit
            elif signal_type == 'sell':
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profit = entry_price * 0.96  # 4% take profit
            else:
                stop_loss = None
                take_profit = None
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                if risk > 0:
                    risk_reward_ratio = reward / risk
            
            signal = {
                'signal_id': str(uuid.uuid4()),
                'pattern_id': str(uuid.uuid4()),
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': signal_type,
                'confidence': pattern.confidence,
                'strength': pattern.strength,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'timestamp': datetime.now(),
                'pattern_name': pattern.pattern_name,
                'pattern_type': pattern.pattern_type,
                'metadata': pattern.metadata
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal creation error: {e}")
            return None
    
    async def _store_results_ultra_fast(self, symbol: str, timeframe: str, 
                                      patterns: List[VectorizedPattern], signals: List[Dict]):
        """Store patterns and signals in database with ultra-fast operations"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Store patterns
                for pattern in patterns:
                    await session.execute("""
                        INSERT INTO ultra_low_latency_patterns (
                            pattern_id, symbol, timeframe, pattern_name, pattern_type,
                            confidence, strength, timestamp, price_level, volume_confirmation,
                            volume_confidence, trend_alignment, detection_method, processing_latency_ms, metadata
                        ) VALUES (
                            :pattern_id, :symbol, :timeframe, :pattern_name, :pattern_type,
                            :confidence, :strength, :timestamp, :price_level, :volume_confirmation,
                            :volume_confidence, :trend_alignment, :detection_method, :processing_latency_ms, :metadata
                        )
                    """, {
                        'pattern_id': str(uuid.uuid4()),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pattern_name': pattern.pattern_name,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'strength': pattern.strength,
                        'timestamp': pattern.timestamp,
                        'price_level': pattern.price_level,
                        'volume_confirmation': pattern.volume_confirmation,
                        'volume_confidence': pattern.volume_confidence,
                        'trend_alignment': pattern.trend_alignment,
                        'detection_method': 'vectorized',
                        'processing_latency_ms': None,  # Will be calculated
                        'metadata': pattern.metadata
                    })
                
                # Store signals
                for signal in signals:
                    await session.execute("""
                        INSERT INTO ultra_low_latency_signals (
                            signal_id, pattern_id, symbol, timeframe, signal_type,
                            confidence, strength, entry_price, stop_loss, take_profit,
                            risk_reward_ratio, timestamp, processing_latency_ms, ensemble_score,
                            market_regime, volatility_context, metadata
                        ) VALUES (
                            :signal_id, :pattern_id, :symbol, :timeframe, :signal_type,
                            :confidence, :strength, :entry_price, :stop_loss, :take_profit,
                            :risk_reward_ratio, :timestamp, :processing_latency_ms, :ensemble_score,
                            :market_regime, :volatility_context, :metadata
                        )
                    """, {
                        'signal_id': signal['signal_id'],
                        'pattern_id': signal['pattern_id'],
                        'symbol': signal['symbol'],
                        'timeframe': signal['timeframe'],
                        'signal_type': signal['signal_type'],
                        'confidence': signal['confidence'],
                        'strength': signal['strength'],
                        'entry_price': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'risk_reward_ratio': signal['risk_reward_ratio'],
                        'timestamp': signal['timestamp'],
                        'processing_latency_ms': None,
                        'ensemble_score': signal['confidence'],
                        'market_regime': 'trending',  # Can be enhanced
                        'volatility_context': None,
                        'metadata': signal['metadata']
                    })
                
                await session.commit()
            
        except Exception as e:
            logger.error(f"❌ Database storage error: {e}")
    
    async def _trigger_callbacks(self, patterns: List[VectorizedPattern], signals: List[Dict]):
        """Trigger registered callbacks"""
        try:
            # Trigger pattern callbacks
            for callback in self.pattern_callbacks:
                try:
                    await callback(patterns)
                except Exception as e:
                    logger.error(f"❌ Pattern callback error: {e}")
            
            # Trigger signal callbacks
            for callback in self.signal_callbacks:
                try:
                    await callback(signals)
                except Exception as e:
                    logger.error(f"❌ Signal callback error: {e}")
            
        except Exception as e:
            logger.error(f"❌ Callback triggering error: {e}")
    
    async def _trigger_error_callbacks(self, error: Exception):
        """Trigger error callbacks"""
        try:
            for callback in self.error_callbacks:
                try:
                    await callback(error)
                except Exception as e:
                    logger.error(f"❌ Error callback error: {e}")
        except Exception as e:
            logger.error(f"❌ Error callback triggering error: {e}")
    
    def _update_performance_metrics(self, latency_ms: float):
        """Update performance metrics"""
        self.avg_processing_latency_ms = (
            (self.avg_processing_latency_ms * (self.total_messages_processed - 1) + latency_ms) /
            self.total_messages_processed
        )
        self.max_processing_latency_ms = max(self.max_processing_latency_ms, latency_ms)
        self.min_processing_latency_ms = min(self.min_processing_latency_ms, latency_ms)
    
    # Callback registration methods
    def add_pattern_callback(self, callback: Callable[[List[VectorizedPattern]], None]):
        """Add pattern detection callback"""
        self.pattern_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable[[List[Dict]], None]):
        """Add signal generation callback"""
        self.signal_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    # Performance monitoring methods
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            websocket_stats = await self.websocket_client.get_performance_stats()
            
            return {
                'integration_service': {
                    'total_messages_processed': self.total_messages_processed,
                    'total_patterns_detected': self.total_patterns_detected,
                    'total_signals_generated': self.total_signals_generated,
                    'avg_processing_latency_ms': self.avg_processing_latency_ms,
                    'max_processing_latency_ms': self.max_processing_latency_ms,
                    'min_processing_latency_ms': self.min_processing_latency_ms,
                    'is_running': self.is_running,
                    'is_initialized': self.is_initialized
                },
                'websocket_client': websocket_stats,
                'buffer_sizes': {
                    'candlestick_buffers': {
                        symbol: {tf: len(self.candlestick_buffers[symbol][tf]) 
                                for tf in self.config.timeframes}
                        for symbol in self.config.symbols
                    },
                    'pattern_buffers': {
                        symbol: {tf: len(self.pattern_buffers[symbol][tf]) 
                                for tf in self.config.timeframes}
                        for symbol in self.config.symbols
                    },
                    'signal_buffers': {
                        symbol: {tf: len(self.signal_buffers[symbol][tf]) 
                                for tf in self.config.timeframes}
                        for symbol in self.config.symbols
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance stats error: {e}")
            return {}
    
    async def get_latest_patterns(self, symbol: str = None, timeframe: str = None, 
                                limit: int = 10) -> List[VectorizedPattern]:
        """Get latest detected patterns"""
        try:
            patterns = []
            
            symbols = [symbol] if symbol else self.config.symbols
            timeframes = [timeframe] if timeframe else self.config.timeframes
            
            for s in symbols:
                for tf in timeframes:
                    if s in self.pattern_buffers and tf in self.pattern_buffers[s]:
                        patterns.extend(self.pattern_buffers[s][tf])
            
            # Sort by timestamp and return latest
            patterns.sort(key=lambda x: x.timestamp, reverse=True)
            return patterns[:limit]
            
        except Exception as e:
            logger.error(f"❌ Get latest patterns error: {e}")
            return []
    
    async def get_latest_signals(self, symbol: str = None, timeframe: str = None, 
                               limit: int = 10) -> List[Dict]:
        """Get latest generated signals"""
        try:
            signals = []
            
            symbols = [symbol] if symbol else self.config.symbols
            timeframes = [timeframe] if timeframe else self.config.timeframes
            
            for s in symbols:
                for tf in timeframes:
                    if s in self.signal_buffers and tf in self.signal_buffers[s]:
                        signals.extend(self.signal_buffers[s][tf])
            
            # Sort by timestamp and return latest
            signals.sort(key=lambda x: x['timestamp'], reverse=True)
            return signals[:limit]
            
        except Exception as e:
            logger.error(f"❌ Get latest signals error: {e}")
            return []
