#!/usr/bin/env python3
"""
Ultra-Optimized Pattern Detection Service
Provides seamless integration with existing AlphaPlus applications
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.strategies.ultra_optimized_pattern_detector import UltraOptimizedPatternDetector, UltraOptimizedPatternSignal
from backend.database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

class UltraOptimizedPatternService:
    """
    Ultra-optimized pattern detection service with seamless integration
    """
    
    def __init__(self, db_connection: TimescaleDBConnection, max_workers: int = 8):
        """Initialize ultra-optimized pattern service"""
        self.db_connection = db_connection
        self.detector = UltraOptimizedPatternDetector(max_workers=max_workers)
        self.session_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0.0,
            'total_response_time_ms': 0.0
        }
        
        logger.info(f"🚀 Ultra-Optimized Pattern Service initialized with {max_workers} workers")
    
    async def initialize(self):
        """Initialize the service and ensure database tables exist"""
        try:
            # Ensure database connection
            if not self.db_connection.connected:
                await self.db_connection.initialize()
            
            # Verify tables exist
            await self._verify_tables_exist()
            
            logger.info("✅ Ultra-Optimized Pattern Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ultra-Optimized Pattern Service: {e}")
            raise
    
    async def _verify_tables_exist(self):
        """Verify that required tables exist"""
        try:
            async with self.db_connection.async_session() as session:
                # Check if ultra_optimized_patterns table exists
                result = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'ultra_optimized_patterns'
                    );
                """))
                
                if not result.scalar():
                    logger.warning("⚠️ ultra_optimized_patterns table not found. Please run migration 011.")
                    # Create a basic table structure for compatibility
                    await self._create_basic_tables(session)
                
        except Exception as e:
            logger.error(f"Error verifying tables: {e}")
            raise
    
    async def _create_basic_tables(self, session: AsyncSession):
        """Create basic tables for compatibility if migration not run"""
        try:
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS ultra_optimized_patterns (
                    id BIGSERIAL PRIMARY KEY,
                    pattern_id VARCHAR(50) UNIQUE NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    pattern_name VARCHAR(100) NOT NULL,
                    pattern_category VARCHAR(20) NOT NULL,
                    pattern_type VARCHAR(20) NOT NULL,
                    direction VARCHAR(10) NOT NULL,
                    confidence DECIMAL(4,3) NOT NULL,
                    strength VARCHAR(20) NOT NULL,
                    price_level DECIMAL(20,8) NOT NULL,
                    volume_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
                    volume_confidence DECIMAL(4,3) NOT NULL DEFAULT 0.0,
                    trend_alignment VARCHAR(20) NOT NULL DEFAULT 'neutral',
                    multi_timeframe_boost DECIMAL(4,3) NOT NULL DEFAULT 0.0,
                    processing_time_ms DECIMAL(8,2),
                    vectorized_operations BOOLEAN NOT NULL DEFAULT TRUE,
                    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                    technical_indicators JSONB,
                    market_conditions JSONB,
                    pattern_metadata JSONB,
                    performance_metrics JSONB,
                    data_points_used INTEGER NOT NULL,
                    data_quality_score DECIMAL(4,3) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    timestamp TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))
            
            await session.commit()
            logger.info("✅ Created basic ultra_optimized_patterns table for compatibility")
            
        except Exception as e:
            logger.error(f"Error creating basic tables: {e}")
            raise
    
    async def detect_patterns(self, 
                            symbol: str, 
                            timeframe: str, 
                            data: pd.DataFrame,
                            use_cache: bool = True,
                            parallel_processing: bool = True) -> List[Dict[str, Any]]:
        """
        **SEAMLESS INTEGRATION METHOD**
        Detect patterns with ultra-optimized performance while maintaining compatibility
        """
        start_time = datetime.now()
        self.session_stats['total_requests'] += 1
        
        try:
            # Validate input data
            if not self._validate_input_data(data, symbol, timeframe):
                logger.warning(f"Invalid input data for {symbol} {timeframe}")
                return []
            
            # Check cache first if enabled
            if use_cache:
                cached_patterns = await self._get_cached_patterns(symbol, timeframe, data)
                if cached_patterns:
                    self.session_stats['cache_hits'] += 1
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_session_stats(response_time)
                    logger.info(f"⚡ Cache hit for {symbol} {timeframe}: {len(cached_patterns)} patterns")
                    return cached_patterns
            
            # Perform ultra-optimized pattern detection
            patterns = self.detector.detect_patterns_ultra_optimized(data, symbol, timeframe)
            
            # Convert to compatible format
            compatible_patterns = await self._convert_to_compatible_format(patterns, symbol, timeframe, data)
            
            # Store patterns in database
            await self._store_patterns(compatible_patterns)
            
            # Cache results if enabled
            if use_cache:
                await self._cache_patterns(symbol, timeframe, data, compatible_patterns)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_session_stats(response_time)
            
            logger.info(f"✅ Ultra-optimized detection for {symbol} {timeframe}: {len(compatible_patterns)} patterns in {response_time:.2f}ms")
            
            return compatible_patterns
            
        except Exception as e:
            logger.error(f"❌ Error in pattern detection for {symbol} {timeframe}: {e}")
            return []
    
    async def detect_patterns_multi_timeframe(self, 
                                            symbol: str, 
                                            data_dict: Dict[str, pd.DataFrame],
                                            use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        **MULTI-TIMEFRAME DETECTION**
        Detect patterns across multiple timeframes in parallel
        """
        start_time = datetime.now()
        self.session_stats['total_requests'] += 1
        
        try:
            # Validate all input data
            valid_data = {}
            for timeframe, data in data_dict.items():
                if self._validate_input_data(data, symbol, timeframe):
                    valid_data[timeframe] = data
                else:
                    logger.warning(f"Invalid input data for {symbol} {timeframe}")
            
            if not valid_data:
                return {}
            
            # Perform parallel pattern detection
            pattern_results = await self.detector.detect_patterns_parallel(valid_data)
            
            # Convert and store results
            all_results = {}
            for timeframe, patterns in pattern_results.items():
                data = valid_data[timeframe]
                compatible_patterns = await self._convert_to_compatible_format(patterns, symbol, timeframe, data)
                await self._store_patterns(compatible_patterns)
                all_results[timeframe] = compatible_patterns
                
                # Cache results if enabled
                if use_cache:
                    await self._cache_patterns(symbol, timeframe, data, compatible_patterns)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_session_stats(response_time)
            
            total_patterns = sum(len(patterns) for patterns in all_results.values())
            logger.info(f"✅ Multi-timeframe detection for {symbol}: {total_patterns} patterns across {len(all_results)} timeframes in {response_time:.2f}ms")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in multi-timeframe detection for {symbol}: {e}")
            return {}
    
    async def get_patterns(self, 
                          symbol: str, 
                          timeframe: str = None,
                          pattern_name: str = None,
                          min_confidence: float = 0.0,
                          limit: int = 100,
                          hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        **COMPATIBLE PATTERN RETRIEVAL**
        Get patterns from database with filtering options
        """
        try:
            async with self.db_connection.async_session() as session:
                # Build query
                query = """
                    SELECT 
                        pattern_id, symbol, timeframe, pattern_name, pattern_category,
                        pattern_type, direction, confidence, strength, price_level,
                        volume_confirmation, volume_confidence, trend_alignment,
                        multi_timeframe_boost, processing_time_ms, vectorized_operations,
                        cache_hit, technical_indicators, market_conditions,
                        pattern_metadata, performance_metrics, data_points_used,
                        data_quality_score, status, timestamp, created_at
                    FROM ultra_optimized_patterns
                    WHERE symbol = :symbol
                        AND timestamp >= NOW() - INTERVAL ':hours_back hours'
                """
                
                params = {
                    'symbol': symbol,
                    'hours_back': hours_back
                }
                
                if timeframe:
                    query += " AND timeframe = :timeframe"
                    params['timeframe'] = timeframe
                
                if pattern_name:
                    query += " AND pattern_name = :pattern_name"
                    params['pattern_name'] = pattern_name
                
                if min_confidence > 0:
                    query += " AND confidence >= :min_confidence"
                    params['min_confidence'] = min_confidence
                
                query += " ORDER BY timestamp DESC LIMIT :limit"
                params['limit'] = limit
                
                result = await session.execute(text(query), params)
                patterns = [dict(row) for row in result]
                
                logger.info(f"📊 Retrieved {len(patterns)} patterns for {symbol}")
                return patterns
                
        except Exception as e:
            logger.error(f"❌ Error retrieving patterns for {symbol}: {e}")
            return []
    
    async def get_performance_stats(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """
        **PERFORMANCE ANALYTICS**
        Get comprehensive performance statistics
        """
        try:
            # Get detector stats
            detector_stats = self.detector.get_performance_stats()
            
            # Get session stats
            session_stats = self.session_stats.copy()
            
            # Get database stats
            db_stats = await self._get_database_stats(symbol, timeframe)
            
            return {
                'detector_stats': detector_stats,
                'session_stats': session_stats,
                'database_stats': db_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}
    
    async def _validate_input_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Validate input data for pattern detection"""
        try:
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol} {timeframe}")
                return False
            
            # Check minimum data length
            if len(data) < 5:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(data)} rows")
                return False
            
            # Check for NaN values
            if data[required_columns].isnull().any().any():
                logger.warning(f"NaN values found in data for {symbol} {timeframe}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol} {timeframe}: {e}")
            return False
    
    async def _convert_to_compatible_format(self, 
                                          patterns: List[UltraOptimizedPatternSignal],
                                          symbol: str,
                                          timeframe: str,
                                          data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert ultra-optimized patterns to compatible format"""
        compatible_patterns = []
        
        for pattern in patterns:
            if pattern.index < len(data):
                candle = data.iloc[pattern.index]
                
                # Determine pattern category
                pattern_category = self._get_pattern_category(pattern.pattern)
                
                # Determine pattern type
                pattern_type = self._get_pattern_type(pattern.pattern)
                
                # Create compatible pattern dict
                compatible_pattern = {
                    'pattern_id': f"{symbol}_{timeframe}_{pattern.pattern}_{pattern.index}_{int(datetime.now().timestamp())}",
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pattern_name': pattern.pattern,
                    'pattern_category': pattern_category,
                    'pattern_type': pattern_type,
                    'direction': pattern.type,
                    'confidence': float(pattern.confidence),
                    'strength': self._get_strength_label(pattern.strength),
                    'price_level': float(candle['close']),
                    'volume_confirmation': pattern.volume_confirmation,
                    'volume_confidence': float(pattern.confidence) if pattern.volume_confirmation else 0.0,
                    'volume_pattern_type': 'volume_spike' if pattern.volume_confirmation else None,
                    'volume_strength': self._get_strength_label(pattern.confidence) if pattern.volume_confirmation else None,
                    'volume_context': pattern.metadata.get('volume_ratio') if pattern.metadata else None,
                    'trend_alignment': pattern.trend_alignment,
                    'multi_timeframe_boost': float(pattern.multi_timeframe_boost),
                    'processing_time_ms': float(pattern.processing_time_ms),
                    'vectorized_operations': True,
                    'cache_hit': False,  # Will be updated if cache hit
                    'technical_indicators': self._extract_technical_indicators(data, pattern.index),
                    'market_conditions': self._extract_market_conditions(data, pattern.index),
                    'pattern_metadata': pattern.metadata,
                    'performance_metrics': {
                        'detection_time_ms': pattern.processing_time_ms,
                        'vectorized': True,
                        'cache_hit': False
                    },
                    'data_points_used': len(data),
                    'data_quality_score': self._calculate_data_quality(data),
                    'status': 'active',
                    'timestamp': candle.get('timestamp', datetime.now()),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }
                
                compatible_patterns.append(compatible_pattern)
        
        return compatible_patterns
    
    async def _store_patterns(self, patterns: List[Dict[str, Any]]):
        """Store patterns in database"""
        if not patterns:
            return
        
        try:
            async with self.db_connection.async_session() as session:
                # Prepare bulk insert
                insert_query = """
                    INSERT INTO ultra_optimized_patterns (
                        pattern_id, symbol, timeframe, pattern_name, pattern_category,
                        pattern_type, direction, confidence, strength, price_level,
                        volume_confirmation, volume_confidence, volume_pattern_type,
                        volume_strength, volume_context, trend_alignment,
                        multi_timeframe_boost, processing_time_ms, vectorized_operations,
                        cache_hit, technical_indicators, market_conditions,
                        pattern_metadata, performance_metrics, data_points_used,
                        data_quality_score, status, timestamp, created_at, updated_at
                    ) VALUES (
                        :pattern_id, :symbol, :timeframe, :pattern_name, :pattern_category,
                        :pattern_type, :direction, :confidence, :strength, :price_level,
                        :volume_confirmation, :volume_confidence, :volume_pattern_type,
                        :volume_strength, :volume_context, :trend_alignment,
                        :multi_timeframe_boost, :processing_time_ms, :vectorized_operations,
                        :cache_hit, :technical_indicators, :market_conditions,
                        :pattern_metadata, :performance_metrics, :data_points_used,
                        :data_quality_score, :status, :timestamp, :created_at, :updated_at
                    )
                    ON CONFLICT (pattern_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        strength = EXCLUDED.strength,
                        volume_confirmation = EXCLUDED.volume_confirmation,
                        trend_alignment = EXCLUDED.trend_alignment,
                        updated_at = NOW()
                """
                
                await session.execute(text(insert_query), patterns)
                await session.commit()
                
                logger.info(f"💾 Stored {len(patterns)} patterns in database")
                
        except Exception as e:
            logger.error(f"❌ Error storing patterns: {e}")
    
    async def _get_cached_patterns(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
        """Get cached patterns if available"""
        try:
            # Simple cache check - in production, use Redis or similar
            data_hash = self.detector._create_data_hash(data)
            
            async with self.db_connection.async_session() as session:
                result = await session.execute(text("""
                    SELECT patterns_data FROM pattern_detection_cache
                    WHERE cache_key = :cache_key AND expires_at > NOW()
                """), {'cache_key': data_hash})
                
                row = result.fetchone()
                if row:
                    return row[0]
                
                return None
                
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _cache_patterns(self, symbol: str, timeframe: str, data: pd.DataFrame, patterns: List[Dict[str, Any]]):
        """Cache pattern results"""
        try:
            data_hash = self.detector._create_data_hash(data)
            
            async with self.db_connection.async_session() as session:
                await session.execute(text("""
                    INSERT INTO pattern_detection_cache (
                        cache_key, symbol, timeframe, data_hash, patterns_data,
                        cache_size_bytes, hit_count, last_accessed, expires_at, created_at
                    ) VALUES (
                        :cache_key, :symbol, :timeframe, :data_hash, :patterns_data,
                        :cache_size_bytes, 1, NOW(), NOW() + INTERVAL '5 minutes', NOW()
                    )
                    ON CONFLICT (cache_key) DO UPDATE SET
                        hit_count = pattern_detection_cache.hit_count + 1,
                        last_accessed = NOW(),
                        expires_at = NOW() + INTERVAL '5 minutes'
                """), {
                    'cache_key': data_hash,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_hash': data_hash,
                    'patterns_data': patterns,
                    'cache_size_bytes': len(str(patterns))
                })
                
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def _get_database_stats(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get database performance statistics"""
        try:
            async with self.db_connection.async_session() as session:
                # Get pattern counts
                count_query = "SELECT COUNT(*) FROM ultra_optimized_patterns"
                if symbol:
                    count_query += " WHERE symbol = :symbol"
                    if timeframe:
                        count_query += " AND timeframe = :timeframe"
                
                params = {}
                if symbol:
                    params['symbol'] = symbol
                if timeframe:
                    params['timeframe'] = timeframe
                
                result = await session.execute(text(count_query), params)
                total_patterns = result.scalar()
                
                # Get cache stats
                cache_result = await session.execute(text("""
                    SELECT COUNT(*) as total_cache,
                           COUNT(*) FILTER (WHERE expires_at > NOW()) as active_cache,
                           AVG(hit_count) as avg_hits
                    FROM pattern_detection_cache
                """))
                cache_stats = dict(cache_result.fetchone())
                
                return {
                    'total_patterns': total_patterns,
                    'cache_stats': cache_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def _get_pattern_category(self, pattern_name: str) -> str:
        """Get pattern category"""
        single_patterns = {'hammer', 'hanging_man', 'shooting_star', 'doji', 'spinning_top', 'marubozu'}
        double_patterns = {'engulfing', 'harami', 'piercing', 'dark_cloud_cover'}
        triple_patterns = {'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'}
        
        if pattern_name in single_patterns:
            return 'single'
        elif pattern_name in double_patterns:
            return 'double'
        elif pattern_name in triple_patterns:
            return 'triple'
        else:
            return 'complex'
    
    def _get_pattern_type(self, pattern_name: str) -> str:
        """Get pattern type"""
        reversal_patterns = {'hammer', 'hanging_man', 'shooting_star', 'engulfing', 'morning_star', 'evening_star'}
        continuation_patterns = {'three_white_soldiers', 'three_black_crows', 'marubozu'}
        indecision_patterns = {'doji', 'spinning_top'}
        
        if pattern_name in reversal_patterns:
            return 'reversal'
        elif pattern_name in continuation_patterns:
            return 'continuation'
        elif pattern_name in indecision_patterns:
            return 'indecision'
        else:
            return 'unknown'
    
    def _get_strength_label(self, strength: float) -> str:
        """Convert strength value to label"""
        if strength >= 0.8:
            return 'strong'
        elif strength >= 0.5:
            return 'moderate'
        else:
            return 'weak'
    
    def _extract_technical_indicators(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Extract technical indicators for pattern context"""
        try:
            if index >= len(data):
                return {}
            
            # Simple technical indicators
            closes = data['close'].values
            volumes = data['volume'].values if 'volume' in data.columns else None
            
            indicators = {}
            
            # SMA calculations
            if index >= 9:
                indicators['sma_10'] = float(np.mean(closes[index-9:index+1]))
            if index >= 19:
                indicators['sma_20'] = float(np.mean(closes[index-19:index+1]))
            
            # RSI calculation (simplified)
            if index >= 14:
                price_changes = np.diff(closes[index-14:index+1])
                gains = np.sum(price_changes[price_changes > 0])
                losses = -np.sum(price_changes[price_changes < 0])
                if losses != 0:
                    rs = gains / losses
                    indicators['rsi'] = float(100 - (100 / (1 + rs)))
            
            # Volume indicators
            if volumes is not None and index >= 19:
                indicators['volume_sma_20'] = float(np.mean(volumes[index-19:index+1]))
                if index < len(volumes):
                    indicators['current_volume'] = float(volumes[index])
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Error extracting technical indicators: {e}")
            return {}
    
    def _extract_market_conditions(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Extract market conditions for pattern context"""
        try:
            if index >= len(data):
                return {}
            
            closes = data['close'].values
            
            conditions = {}
            
            # Volatility calculation
            if index >= 19:
                returns = np.diff(np.log(closes[index-19:index+1]))
                conditions['volatility'] = float(np.std(returns) * np.sqrt(252))  # Annualized
            
            # Trend calculation
            if index >= 19:
                sma_20 = np.mean(closes[index-19:index+1])
                current_price = closes[index]
                conditions['trend'] = 'bullish' if current_price > sma_20 else 'bearish'
                conditions['trend_strength'] = float(abs(current_price - sma_20) / sma_20)
            
            return conditions
            
        except Exception as e:
            logger.warning(f"Error extracting market conditions: {e}")
            return {}
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            # Check for missing values
            missing_ratio = data[['open', 'high', 'low', 'close']].isnull().sum().sum() / (len(data) * 4)
            
            # Check for zero values
            zero_ratio = (data[['open', 'high', 'low', 'close']] == 0).sum().sum() / (len(data) * 4)
            
            # Check for price consistency
            price_consistency = 1.0
            if len(data) > 1:
                price_changes = np.abs(np.diff(data['close']))
                if np.max(price_changes) > 0:
                    price_consistency = 1.0 - (np.std(price_changes) / np.mean(price_changes))
            
            # Calculate overall quality score
            quality_score = (1.0 - missing_ratio) * (1.0 - zero_ratio) * price_consistency
            
            return float(max(0.0, min(1.0, quality_score)))
            
        except Exception as e:
            logger.warning(f"Error calculating data quality: {e}")
            return 0.5
    
    def _update_session_stats(self, response_time: float):
        """Update session statistics"""
        self.session_stats['total_response_time_ms'] += response_time
        self.session_stats['avg_response_time_ms'] = (
            self.session_stats['total_response_time_ms'] / self.session_stats['total_requests']
        )
    
    async def cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        try:
            # Clean up detector cache
            self.detector.cleanup_old_data()
            
            # Clean up database cache
            async with self.db_connection.async_session() as session:
                await session.execute(text("""
                    DELETE FROM pattern_detection_cache 
                    WHERE expires_at < NOW() - INTERVAL '1 hour'
                """))
                await session.commit()
            
            logger.info("🧹 Cleaned up old data")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def shutdown(self):
        """Shutdown the service gracefully"""
        try:
            # Clean up old data
            await self.cleanup_old_data()
            
            # Clear detector cache
            self.detector.clear_cache()
            
            logger.info("🛑 Ultra-Optimized Pattern Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_service():
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        await db_connection.initialize()
        
        # Initialize service
        service = UltraOptimizedPatternService(db_connection)
        await service.initialize()
        
        # Create test data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n = 1000
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.random.rand(n) * 2
        lows = closes - np.random.rand(n) * 2
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(1000, 10000, n)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
        })
        
        # Test pattern detection
        patterns = await service.detect_patterns("BTCUSDT", "1m", df)
        print(f"Detected {len(patterns)} patterns")
        
        # Test performance stats
        stats = await service.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        # Shutdown
        await service.shutdown()
    
    asyncio.run(test_service())
