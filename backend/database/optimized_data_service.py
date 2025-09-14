"""
Optimized Data Service for AlphaPlus
Integrates hybrid storage, materialized views, and advanced indexing for ultra-fast data access
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .advanced_cache_layer import AdvancedCacheLayer
from .materialized_views import MaterializedViewsManager
from .advanced_indexing import AdvancedIndexingManager

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    execution_time_ms: float
    cache_hit: bool
    rows_returned: int
    data_source: str  # 'memory', 'redis', 'database', 'materialized_view'

class OptimizedDataService:
    """Unified optimized data service with hybrid storage and advanced indexing"""
    
    def __init__(self, 
                 db_session_factory,
                 redis_url: str = "redis://localhost:6379",
                 max_memory_mb: int = 1024):
        
        self.db_session_factory = db_session_factory
        self.redis_url = redis_url
        self.max_memory_mb = max_memory_mb
        
        # Initialize components
        self.cache_layer = AdvancedCacheLayer(
            redis_url=redis_url,
            db_session_factory=db_session_factory,
            max_memory_mb=max_memory_mb
        )
        
        self.views_manager = MaterializedViewsManager(db_session_factory)
        self.indexing_manager = AdvancedIndexingManager(db_session_factory)
        
        # Performance tracking
        self.query_metrics: List[QueryMetrics] = []
        self.is_initialized = False
        
        logger.info("🚀 Optimized Data Service initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🔄 Initializing Optimized Data Service...")
            
            # Initialize cache layer
            await self.cache_layer.initialize()
            
            # Create materialized views
            await self.views_manager.create_all_views()
            
            # Create advanced indexes
            await self.indexing_manager.create_all_advanced_indexes()
            
            self.is_initialized = True
            logger.info("✅ Optimized Data Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Optimized Data Service: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all components gracefully"""
        try:
            if self.cache_layer:
                await self.cache_layer.shutdown()
            
            self.is_initialized = False
            logger.info("✅ Optimized Data Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Optimized Data Service: {e}")
    
    async def get_market_data(self, symbol: str, timeframe: str, 
                             hours: int = 24, use_cache: bool = True) -> pd.DataFrame:
        """Get market data with optimized caching and materialized views"""
        start_time = datetime.now()
        
        try:
            # Try cache first if enabled
            if use_cache:
                cached_data = await self.cache_layer.get_candlestick_data(
                    symbol, timeframe, 
                    start_time - timedelta(hours=hours), 
                    start_time
                )
                if cached_data is not None:
                    self._record_metrics("market_data", 
                                       (datetime.now() - start_time).total_seconds() * 1000,
                                       True, len(cached_data), "memory")
                    return cached_data
            
            # Try materialized view for recent data
            if hours <= 7:  # Use materialized view for recent data
                view_data = await self.views_manager.get_market_data_with_indicators(symbol, hours)
                if view_data:
                    df = pd.DataFrame(view_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Cache the result
                    if use_cache:
                        await self.cache_layer.set_candlestick_data(symbol, timeframe, df)
                    
                    self._record_metrics("market_data", 
                                       (datetime.now() - start_time).total_seconds() * 1000,
                                       False, len(df), "materialized_view")
                    return df
            
            # Fallback to direct database query with optimized function
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT * FROM get_optimized_market_data(:symbol, :timeframe, :hours)
                """)
                
                result = await session.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "hours": hours
                })
                
                data = [dict(row) for row in result]
                df = pd.DataFrame(data)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Cache the result
                    if use_cache:
                        await self.cache_layer.set_candlestick_data(symbol, timeframe, df)
                
                self._record_metrics("market_data", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(df), "database")
                return df
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    async def get_signals(self, symbol: str = None, hours: int = 24, 
                         min_confidence: float = 0.0, use_cache: bool = True) -> List[Dict]:
        """Get signals with optimized caching and materialized views"""
        start_time = datetime.now()
        
        try:
            # Try cache first for latest signals
            if use_cache and hours <= 1:
                cached_signals = await self.cache_layer.get_latest_signals(symbol)
                if cached_signals:
                    filtered_signals = [
                        s for s in cached_signals 
                        if s.get('confidence', 0) >= min_confidence
                    ]
                    
                    self._record_metrics("signals", 
                                       (datetime.now() - start_time).total_seconds() * 1000,
                                       True, len(filtered_signals), "memory")
                    return filtered_signals
            
            # Try materialized view for recent data
            if hours <= 30:  # Use materialized view for recent data
                view_signals = await self.views_manager.get_signals_with_context(symbol, hours)
                if view_signals:
                    filtered_signals = [
                        s for s in view_signals 
                        if s.get('confidence', 0) >= min_confidence
                    ]
                    
                    self._record_metrics("signals", 
                                       (datetime.now() - start_time).total_seconds() * 1000,
                                       False, len(filtered_signals), "materialized_view")
                    return filtered_signals
            
            # Fallback to optimized database function
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT * FROM get_optimized_signals(:symbol, :hours, :min_confidence)
                """)
                
                result = await session.execute(query, {
                    "symbol": symbol,
                    "hours": hours,
                    "min_confidence": min_confidence
                })
                
                signals = [dict(row) for row in result]
                
                self._record_metrics("signals", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(signals), "database")
                return signals
                
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []
    
    async def get_patterns(self, symbol: str = None, hours: int = 24, 
                          min_confidence: float = 0.0) -> List[Dict]:
        """Get patterns with optimized materialized views"""
        start_time = datetime.now()
        
        try:
            # Use materialized view for patterns
            view_patterns = await self.views_manager.get_patterns_with_context(symbol, hours)
            if view_patterns:
                filtered_patterns = [
                    p for p in view_patterns 
                    if p.get('confidence', 0) >= min_confidence
                ]
                
                self._record_metrics("patterns", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(filtered_patterns), "materialized_view")
                return filtered_patterns
            
            # Fallback to direct query with covering index
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT * FROM candlestick_patterns 
                    WHERE (:symbol IS NULL OR symbol = :symbol)
                        AND timestamp >= NOW() - INTERVAL ':hours hours'
                        AND confidence >= :min_confidence
                    ORDER BY timestamp DESC
                """)
                
                result = await session.execute(query, {
                    "symbol": symbol,
                    "hours": hours,
                    "min_confidence": min_confidence
                })
                
                patterns = [dict(row) for row in result]
                
                self._record_metrics("patterns", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(patterns), "database")
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return []
    
    async def get_performance_summary(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get performance summary with optimized materialized views"""
        start_time = datetime.now()
        
        try:
            # Use materialized view for performance data
            view_performance = await self.views_manager.get_performance_summary(symbol, hours)
            if view_performance:
                self._record_metrics("performance", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(view_performance), "materialized_view")
                return view_performance
            
            # Fallback to optimized database function
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT * FROM get_performance_summary(:symbol, :hours)
                """)
                
                result = await session.execute(query, {
                    "symbol": symbol,
                    "hours": hours
                })
                
                performance = [dict(row) for row in result]
                
                self._record_metrics("performance", 
                                   (datetime.now() - start_time).total_seconds() * 1000,
                                   False, len(performance), "database")
                return performance
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return []
    
    async def get_ui_data(self) -> Dict:
        """Get pre-computed UI data from cache view"""
        start_time = datetime.now()
        
        try:
            # Get from materialized view
            ui_data = await self.views_manager.get_ui_cache_data()
            
            self._record_metrics("ui_data", 
                               (datetime.now() - start_time).total_seconds() * 1000,
                               False, 1, "materialized_view")
            return ui_data
            
        except Exception as e:
            logger.error(f"Error getting UI data: {e}")
            return {}
    
    async def set_signal(self, signal: Dict) -> bool:
        """Set signal with immediate cache update"""
        start_time = datetime.now()
        
        try:
            # Update cache immediately
            success = await self.cache_layer.set_signal(signal)
            
            # Also store in database (async)
            async with self.db_session_factory() as session:
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
                
                await session.execute(query, signal)
                await session.commit()
            
            self._record_metrics("set_signal", 
                               (datetime.now() - start_time).total_seconds() * 1000,
                               False, 1, "cache_and_db")
            return success
            
        except Exception as e:
            logger.error(f"Error setting signal: {e}")
            return False
    
    async def set_candlestick_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Set candlestick data with hybrid storage"""
        start_time = datetime.now()
        
        try:
            # Store in cache and buffer for database
            success = await self.cache_layer.set_candlestick_data(symbol, timeframe, data)
            
            self._record_metrics("set_candlestick", 
                               (datetime.now() - start_time).total_seconds() * 1000,
                               False, len(data), "cache_and_buffer")
            return success
            
        except Exception as e:
            logger.error(f"Error setting candlestick data: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            cache_stats = self.cache_layer.get_stats()
            
            # Add query metrics
            query_stats = self._get_query_statistics()
            
            return {
                **cache_stats,
                'query_metrics': query_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def optimize_performance(self):
        """Run performance optimization tasks"""
        try:
            logger.info("🔄 Running performance optimization...")
            
            # Optimize indexes
            await self.indexing_manager.optimize_indexes()
            
            # Refresh materialized views
            views = [
                'market_data_with_indicators',
                'signals_with_context',
                'performance_summary',
                'ui_cache'
            ]
            
            for view in views:
                await self.views_manager.refresh_view(view)
            
            logger.info("✅ Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            # Get cache statistics
            cache_stats = await self.get_cache_stats()
            
            # Get index usage statistics
            index_stats = await self.indexing_manager.analyze_index_usage()
            
            # Get materialized view statistics
            view_stats = await self.views_manager.get_view_stats()
            
            # Get query recommendations
            index_recommendations = await self.indexing_manager.get_index_recommendations()
            
            return {
                'cache_performance': cache_stats,
                'index_usage': index_stats,
                'materialized_views': view_stats,
                'recommendations': index_recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance report: {e}")
            return {}
    
    def _record_metrics(self, query_type: str, execution_time_ms: float, 
                       cache_hit: bool, rows_returned: int, data_source: str):
        """Record query performance metrics"""
        metric = QueryMetrics(
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            cache_hit=cache_hit,
            rows_returned=rows_returned,
            data_source=data_source
        )
        
        self.query_metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.query_metrics) > 1000:
            self.query_metrics = self.query_metrics[-1000:]
    
    def _get_query_statistics(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        if not self.query_metrics:
            return {}
        
        # Group by query type
        query_types = {}
        for metric in self.query_metrics:
            if metric.query_type not in query_types:
                query_types[metric.query_type] = {
                    'count': 0,
                    'total_time_ms': 0.0,
                    'avg_time_ms': 0.0,
                    'cache_hits': 0,
                    'total_rows': 0,
                    'data_sources': {}
                }
            
            stats = query_types[metric.query_type]
            stats['count'] += 1
            stats['total_time_ms'] += metric.execution_time_ms
            stats['total_rows'] += metric.rows_returned
            
            if metric.cache_hit:
                stats['cache_hits'] += 1
            
            if metric.data_source not in stats['data_sources']:
                stats['data_sources'][metric.data_source] = 0
            stats['data_sources'][metric.data_source] += 1
        
        # Calculate averages
        for query_type, stats in query_types.items():
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['count']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['count'] if stats['count'] > 0 else 0
        
        return query_types
    
    async def warm_cache(self, symbols: List[str] = None):
        """Warm up cache with frequently accessed data"""
        try:
            logger.info("🔄 Warming up cache...")
            
            if not symbols:
                symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            
            # Warm up market data cache
            for symbol in symbols:
                for timeframe in ['1m', '5m', '15m', '1h']:
                    await self.get_market_data(symbol, timeframe, hours=1, use_cache=True)
            
            # Warm up signals cache
            await self.get_signals(hours=1, use_cache=True)
            
            # Warm up patterns cache
            await self.get_patterns(hours=1)
            
            logger.info("✅ Cache warming completed")
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
    
    async def cleanup_old_data(self):
        """Clean up old data and optimize storage"""
        try:
            logger.info("🔄 Cleaning up old data...")
            
            # This would typically be handled by TimescaleDB retention policies
            # but we can also manually clean up if needed
            
            async with self.db_session_factory() as session:
                # Clean up old query metrics (keep last 7 days)
                await session.execute(text("""
                    DELETE FROM query_performance_log 
                    WHERE timestamp < NOW() - INTERVAL '7 days'
                """))
                
                # Clean up old cache performance logs (keep last 30 days)
                await session.execute(text("""
                    DELETE FROM cache_performance_log 
                    WHERE timestamp < NOW() - INTERVAL '30 days'
                """))
                
                await session.commit()
            
            logger.info("✅ Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
