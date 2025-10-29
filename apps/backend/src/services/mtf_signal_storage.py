"""
MTF Signal Storage Service
Stores MTF-enhanced signals to database and Redis cache
"""

import logging
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import asdict

from src.services.ai_model_integration_service import AIModelSignal
from src.database.connection import TimescaleDBConnection
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MTFSignalStorage:
    """
    Storage service for MTF-enhanced AI signals
    Handles database persistence, Redis caching, and entry history tracking
    """
    
    def __init__(
        self,
        db_connection: TimescaleDBConnection,
        redis_url: str = "redis://localhost:56379"
    ):
        self.db_connection = db_connection
        self.redis_url = redis_url
        self.redis_client = None
        self.logger = logger
        
        # Statistics
        self.stats = {
            'signals_stored': 0,
            'storage_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'entry_history_stored': 0
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("✅ MTF Signal Storage initialized (Redis connected)")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def store_mtf_signal(self, signal: AIModelSignal) -> bool:
        """
        Store MTF signal to database and cache
        
        Args:
            signal: AIModelSignal with MTF entry data
        
        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Generate signal_id if not present
            signal_id = uuid.uuid4()
            
            # Prepare data for database
            signal_data = {
                'signal_id': signal_id,
                'symbol': signal.symbol.upper(),
                'direction': signal.signal_direction,
                'timestamp': signal.timestamp or datetime.now(timezone.utc),
                
                # Timeframes
                'signal_timeframe': signal.timeframe,
                'entry_timeframe': signal.entry_timeframe or signal.timeframe,
                
                # Prices
                'signal_price': float(signal.probability) if signal.probability else None,  # Store probability as signal price context
                'entry_price': float(signal.entry_price) if signal.entry_price else None,
                'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                'take_profit_1': float(signal.take_profit_levels[0]) if signal.take_profit_levels and len(signal.take_profit_levels) > 0 else None,
                'take_profit_2': float(signal.take_profit_levels[1]) if signal.take_profit_levels and len(signal.take_profit_levels) > 1 else None,
                'take_profit_3': float(signal.take_profit_levels[2]) if signal.take_profit_levels and len(signal.take_profit_levels) > 2 else None,
                
                # Entry Analysis
                'entry_strategy': signal.entry_strategy or 'UNKNOWN',
                'entry_pattern': signal.entry_pattern or 'UNKNOWN',
                'entry_confidence': float(signal.entry_confidence) if signal.entry_confidence else None,
                'fibonacci_level': float(signal.fibonacci_level) if signal.fibonacci_level else None,
                
                # Signal Quality
                'signal_confidence': float(signal.confidence_score),
                'signal_probability': float(signal.probability) if signal.probability else None,
                'consensus_achieved': signal.consensus_achieved,
                'consensus_score': float(signal.consensus_score) if signal.consensus_score else None,
                'agreeing_heads_count': len(signal.agreeing_heads) if signal.agreeing_heads else 0,
                
                # Entry Timing Indicators
                'atr_entry_tf': float(signal.atr_entry_tf) if signal.atr_entry_tf else None,
                'volume_confirmation': signal.metadata.get('mtf_analysis', {}).get('volume_confirmed', False) if signal.metadata else False,
                'ema_alignment': True if signal.entry_strategy and 'EMA' in signal.entry_strategy else False,
                
                # Risk Management
                'risk_reward_ratio': float(signal.risk_reward_ratio) if signal.risk_reward_ratio else None,
                
                # Model Analysis
                'model_heads_analysis': json.dumps({
                    'agreeing_heads': signal.agreeing_heads or [],
                    'model_reasoning': signal.model_reasoning or {},
                    'mtf_metadata': signal.metadata or {}
                }),
                
                # Data Quality
                'data_quality_score': float(signal.data_quality),
                
                # Status
                'is_active': True,
                'status': 'OPEN'
            }
            
            # Insert into database
            async with self.db_connection.get_connection() as conn:
                query = """
                    INSERT INTO ai_signals_mtf (
                        signal_id, symbol, direction, timestamp,
                        signal_timeframe, entry_timeframe,
                        signal_price, entry_price, stop_loss,
                        take_profit_1, take_profit_2, take_profit_3,
                        entry_strategy, entry_pattern, entry_confidence, fibonacci_level,
                        signal_confidence, signal_probability, consensus_achieved,
                        consensus_score, agreeing_heads_count,
                        atr_entry_tf, volume_confirmation, ema_alignment,
                        risk_reward_ratio, model_heads_analysis,
                        data_quality_score, is_active, status
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29
                    )
                """
                
                await conn.execute(
                    query,
                    signal_data['signal_id'],
                    signal_data['symbol'],
                    signal_data['direction'],
                    signal_data['timestamp'],
                    signal_data['signal_timeframe'],
                    signal_data['entry_timeframe'],
                    signal_data['signal_price'],
                    signal_data['entry_price'],
                    signal_data['stop_loss'],
                    signal_data['take_profit_1'],
                    signal_data['take_profit_2'],
                    signal_data['take_profit_3'],
                    signal_data['entry_strategy'],
                    signal_data['entry_pattern'],
                    signal_data['entry_confidence'],
                    signal_data['fibonacci_level'],
                    signal_data['signal_confidence'],
                    signal_data['signal_probability'],
                    signal_data['consensus_achieved'],
                    signal_data['consensus_score'],
                    signal_data['agreeing_heads_count'],
                    signal_data['atr_entry_tf'],
                    signal_data['volume_confirmation'],
                    signal_data['ema_alignment'],
                    signal_data['risk_reward_ratio'],
                    signal_data['model_heads_analysis'],
                    signal_data['data_quality_score'],
                    signal_data['is_active'],
                    signal_data['status']
                )
            
            self.logger.info(
                f"✅ Stored MTF signal: {signal.symbol} {signal.signal_direction} "
                f"@ ${signal.entry_price:.2f if signal.entry_price else 0:.2f} "
                f"({signal.entry_strategy})"
            )
            
            # Store entry analysis history if MTF metadata exists
            if signal.metadata and 'mtf_analysis' in signal.metadata:
                await self.store_entry_analysis_history(signal_id, signal)
            
            # Cache the signal
            await self.cache_signal(signal, signal_id)
            
            self.stats['signals_stored'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing MTF signal for {signal.symbol}: {e}")
            self.stats['storage_failures'] += 1
            return False
    
    async def store_entry_analysis_history(
        self, 
        signal_id: uuid.UUID, 
        signal: AIModelSignal
    ) -> bool:
        """
        Store entry analysis history for performance tracking
        
        Args:
            signal_id: UUID of the stored signal
            signal: AIModelSignal with MTF metadata
        
        Returns:
            bool: True if stored successfully
        """
        try:
            mtf_data = signal.metadata.get('mtf_analysis', {})
            ema_levels = mtf_data.get('ema_levels', {})
            
            async with self.db_connection.get_connection() as conn:
                query = """
                    INSERT INTO mtf_entry_analysis_history (
                        signal_id, symbol, entry_timeframe, timestamp,
                        ema_9_level, ema_21_level, ema_50_level,
                        selected_entry_price, selected_entry_reason,
                        candlestick_pattern, pattern_confidence
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    )
                """
                
                await conn.execute(
                    query,
                    signal_id,
                    signal.symbol.upper(),
                    signal.entry_timeframe or signal.timeframe,
                    signal.timestamp or datetime.now(timezone.utc),
                    ema_levels.get('ema_9'),
                    ema_levels.get('ema_21'),
                    ema_levels.get('ema_50'),
                    float(signal.entry_price) if signal.entry_price else None,
                    signal.entry_strategy or 'UNKNOWN',
                    signal.entry_pattern or 'UNKNOWN',
                    float(signal.entry_confidence) if signal.entry_confidence else None
                )
            
            self.stats['entry_history_stored'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing entry analysis history: {e}")
            return False
    
    async def cache_signal(
        self, 
        signal: AIModelSignal, 
        signal_id: uuid.UUID
    ) -> bool:
        """
        Cache signal in Redis for fast retrieval
        
        Args:
            signal: AIModelSignal to cache
            signal_id: UUID of the signal
        
        Returns:
            bool: True if cached successfully
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"mtf_signal:{signal.symbol}:{signal.signal_direction}"
            cache_data = {
                'signal_id': str(signal_id),
                'symbol': signal.symbol,
                'direction': signal.signal_direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit_levels': signal.take_profit_levels,
                'entry_strategy': signal.entry_strategy,
                'entry_confidence': signal.entry_confidence,
                'signal_confidence': signal.confidence_score,
                'timestamp': signal.timestamp.isoformat() if signal.timestamp else datetime.now(timezone.utc).isoformat()
            }
            
            # Store with 1-hour expiration
            await self.redis_client.setex(
                cache_key,
                3600,
                json.dumps(cache_data, default=str)
            )
            
            # Also set active signal marker for deduplication
            active_key = f"active_signal:{signal.symbol}:{signal.signal_direction}"
            await self.redis_client.setex(active_key, 3600, 'exists')
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error caching signal: {e}")
            return False
    
    async def get_cached_signal(
        self, 
        symbol: str, 
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached signal from Redis
        
        Args:
            symbol: Trading symbol
            direction: Signal direction (LONG/SHORT)
        
        Returns:
            Dict with signal data or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"mtf_signal:{symbol.upper()}:{direction}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.stats['cache_hits'] += 1
                return json.loads(cached_data)
            
            self.stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error retrieving cached signal: {e}")
            return None
    
    async def check_active_signal_exists(
        self, 
        symbol: str, 
        direction: str
    ) -> bool:
        """
        Check if an active signal already exists for symbol+direction
        
        Args:
            symbol: Trading symbol
            direction: Signal direction (LONG/SHORT)
        
        Returns:
            bool: True if active signal exists
        """
        try:
            # Check Redis cache first (fast)
            if self.redis_client:
                cache_key = f"active_signal:{symbol.upper()}:{direction}"
                cached = await self.redis_client.get(cache_key)
                if cached:
                    self.logger.debug(f"Active signal exists in cache: {symbol} {direction}")
                    return True
            
            # Check database
            async with self.db_connection.get_connection() as conn:
                query = """
                    SELECT signal_id 
                    FROM ai_signals_mtf 
                    WHERE symbol = $1 
                      AND direction = $2 
                      AND is_active = true 
                      AND status = 'OPEN'
                      AND timestamp > NOW() - INTERVAL '24 hours'
                    LIMIT 1
                """
                result = await conn.fetchrow(query, symbol.upper(), direction)
                
                if result:
                    # Cache the result
                    if self.redis_client:
                        cache_key = f"active_signal:{symbol.upper()}:{direction}"
                        await self.redis_client.setex(cache_key, 3600, 'exists')
                    
                    self.logger.debug(f"Active signal exists in database: {symbol} {direction}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error checking active signal: {e}")
            return False  # Assume no duplicate on error (safer to store than skip)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            'signals_stored': self.stats['signals_stored'],
            'storage_failures': self.stats['storage_failures'],
            'storage_success_rate': (
                self.stats['signals_stored'] / 
                (self.stats['signals_stored'] + self.stats['storage_failures'])
                if (self.stats['signals_stored'] + self.stats['storage_failures']) > 0
                else 0
            ),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
                else 0
            ),
            'entry_history_stored': self.stats['entry_history_stored']
        }

