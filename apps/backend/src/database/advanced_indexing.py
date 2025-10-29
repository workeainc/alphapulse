"""
Advanced Indexing Strategy for AlphaPlus
Implements BRIN, partial, and covering indexes for ultra-fast queries
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class AdvancedIndexingManager:
    """Manages advanced indexing strategies for ultra-fast queries"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.index_stats = {}
        
        logger.info("🚀 Advanced Indexing Manager initialized")
    
    async def create_all_advanced_indexes(self):
        """Create all advanced indexes for optimal performance"""
        try:
            async with self.db_session_factory() as session:
                # Create BRIN indexes for time-series data
                await self._create_brin_indexes(session)
                
                # Create partial indexes for filtered queries
                await self._create_partial_indexes(session)
                
                # Create covering indexes for common queries
                await self._create_covering_indexes(session)
                
                # Create GIN indexes for JSONB fields
                await self._create_gin_indexes(session)
                
                # Create composite indexes for multi-column queries
                await self._create_composite_indexes(session)
                
                await session.commit()
                logger.info("✅ All advanced indexes created successfully")
                
        except Exception as e:
            logger.error(f"❌ Error creating advanced indexes: {e}")
            raise
    
    async def _create_brin_indexes(self, session: AsyncSession):
        """Create BRIN indexes for time-series data (low space, ultra-fast)"""
        try:
            # BRIN index for candlestick_data timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_data_timestamp_brin 
                ON candlestick_data USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            # BRIN index for signals timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_timestamp_brin 
                ON signals USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            # BRIN index for trades timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_timestamp_brin 
                ON trades USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            # BRIN index for patterns timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_timestamp_brin 
                ON candlestick_patterns USING BRIN (timestamp) 
                WITH (pages_per_range = 128);
            """))
            
            logger.info("✅ Created BRIN indexes for time-series data")
            
        except Exception as e:
            logger.error(f"Error creating BRIN indexes: {e}")
            raise
    
    async def _create_partial_indexes(self, session: AsyncSession):
        """Create partial indexes for filtered queries"""
        try:
            # Partial index for high-confidence signals only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_high_confidence 
                ON signals (symbol, timestamp DESC, confidence DESC) 
                WHERE confidence >= 0.8;
            """))
            
            # Partial index for active signals only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_active 
                ON signals (symbol, timestamp DESC) 
                WHERE status = 'active';
            """))
            
            # Partial index for winning trades only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_winners 
                ON trades (symbol, timestamp DESC, pnl DESC) 
                WHERE pnl > 0;
            """))
            
            # Partial index for losing trades only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_losers 
                ON trades (symbol, timestamp DESC, pnl ASC) 
                WHERE pnl < 0;
            """))
            
            # Partial index for high-confidence patterns only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_high_confidence 
                ON candlestick_patterns (symbol, pattern_name, timestamp DESC) 
                WHERE confidence >= 0.7;
            """))
            
            # Partial index for volume-confirmed patterns only
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_volume_confirmed 
                ON candlestick_patterns (symbol, pattern_name, timestamp DESC) 
                WHERE volume_confirmation = true;
            """))
            
            logger.info("✅ Created partial indexes for filtered queries")
            
        except Exception as e:
            logger.error(f"Error creating partial indexes: {e}")
            raise
    
    async def _create_covering_indexes(self, session: AsyncSession):
        """Create covering indexes (INCLUDE) for common queries"""
        try:
            # Covering index for candlestick data with indicators
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_covering 
                ON candlestick_data (symbol, timestamp DESC) 
                INCLUDE (open, high, low, close, volume, indicators, patterns);
            """))
            
            # Covering index for signals with metadata
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_covering 
                ON signals (symbol, timestamp DESC) 
                INCLUDE (side, strategy, confidence, strength, price, stop_loss, take_profit, status, metadata);
            """))
            
            # Covering index for trades with performance data
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_covering 
                ON trades (symbol, timestamp DESC) 
                INCLUDE (side, strategy, entry_price, quantity, exit_price, pnl, status);
            """))
            
            # Covering index for patterns with context
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_covering 
                ON candlestick_patterns (symbol, timestamp DESC) 
                INCLUDE (pattern_name, confidence, volume_confirmation, trend_alignment, metadata);
            """))
            
            logger.info("✅ Created covering indexes for common queries")
            
        except Exception as e:
            logger.error(f"Error creating covering indexes: {e}")
            raise
    
    async def _create_gin_indexes(self, session: AsyncSession):
        """Create GIN indexes for JSONB fields"""
        try:
            # GIN index for candlestick indicators
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_indicators_gin 
                ON candlestick_data USING GIN (indicators);
            """))
            
            # GIN index for signal metadata
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_metadata_gin 
                ON signals USING GIN (metadata);
            """))
            
            # GIN index for pattern metadata
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_metadata_gin 
                ON candlestick_patterns USING GIN (metadata);
            """))
            
            # GIN index for trade metadata
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_metadata_gin 
                ON trades USING GIN (metadata);
            """))
            
            # GIN index for strategy configs metadata
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_configs_metadata_gin 
                ON strategy_configs USING GIN (metadata);
            """))
            
            logger.info("✅ Created GIN indexes for JSONB fields")
            
        except Exception as e:
            logger.error(f"Error creating GIN indexes: {e}")
            raise
    
    async def _create_composite_indexes(self, session: AsyncSession):
        """Create composite indexes for multi-column queries"""
        try:
            # Composite index for symbol + timeframe + timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_symbol_tf_time 
                ON candlestick_data (symbol, timeframe, timestamp DESC);
            """))
            
            # Composite index for symbol + strategy + timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_strategy_time 
                ON signals (symbol, strategy, timestamp DESC);
            """))
            
            # Composite index for symbol + side + timestamp
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_side_time 
                ON trades (symbol, side, timestamp DESC);
            """))
            
            # Composite index for symbol + pattern + confidence
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_symbol_pattern_conf 
                ON candlestick_patterns (symbol, pattern_name, confidence DESC, timestamp DESC);
            """))
            
            # Composite index for strategy + performance metrics
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_strategy_performance 
                ON trades (strategy, pnl DESC, timestamp DESC);
            """))
            
            # Composite index for risk parameters
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_configs_risk_params 
                ON strategy_configs (max_loss_pct, stop_loss_pct, take_profit_pct, timestamp DESC);
            """))
            
            logger.info("✅ Created composite indexes for multi-column queries")
            
        except Exception as e:
            logger.error(f"Error creating composite indexes: {e}")
            raise
    
    async def create_functional_indexes(self, session: AsyncSession):
        """Create functional indexes for computed values"""
        try:
            # Functional index for signal strength * confidence
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_strength_score 
                ON signals ((confidence * CASE 
                    WHEN strength = 'strong' THEN 1.0
                    WHEN strength = 'medium' THEN 0.7
                    WHEN strength = 'weak' THEN 0.4
                    ELSE 0.1
                END)) DESC;
            """))
            
            # Functional index for trade profit percentage
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_profit_pct 
                ON trades (((exit_price - entry_price) / entry_price * 100)) DESC;
            """))
            
            # Functional index for pattern frequency
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patterns_frequency 
                ON candlestick_patterns (symbol, pattern_name, 
                    COUNT(*) OVER (PARTITION BY symbol, pattern_name ORDER BY timestamp)) DESC;
            """))
            
            logger.info("✅ Created functional indexes for computed values")
            
        except Exception as e:
            logger.error(f"Error creating functional indexes: {e}")
            raise
    
    async def create_partitioned_indexes(self, session: AsyncSession):
        """Create partitioned indexes for large tables"""
        try:
            # Partitioned index for candlestick data by symbol
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_partitioned 
                ON candlestick_data (timestamp DESC, symbol) 
                WHERE timestamp >= NOW() - INTERVAL '30 days';
            """))
            
            # Partitioned index for signals by symbol
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_partitioned 
                ON signals (timestamp DESC, symbol) 
                WHERE timestamp >= NOW() - INTERVAL '30 days';
            """))
            
            # Partitioned index for trades by symbol
            await session.execute(text("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_partitioned 
                ON trades (timestamp DESC, symbol) 
                WHERE timestamp >= NOW() - INTERVAL '30 days';
            """))
            
            logger.info("✅ Created partitioned indexes for large tables")
            
        except Exception as e:
            logger.error(f"Error creating partitioned indexes: {e}")
            raise
    
    async def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage and performance"""
        try:
            async with self.db_session_factory() as session:
                # Get index usage statistics
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes 
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC;
                """)
                
                result = await session.execute(query)
                index_usage = [dict(row) for row in result]
                
                # Get table statistics
                table_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        pg_size_pretty(pg_total_relation_size(relid)) as table_size
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                    ORDER BY seq_scan DESC;
                """)
                
                table_result = await session.execute(table_query)
                table_stats = [dict(row) for row in table_result]
                
                # Get slow queries
                slow_query = text("""
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        rows
                    FROM pg_stat_statements 
                    WHERE mean_time > 100  -- Queries taking more than 100ms
                    ORDER BY mean_time DESC
                    LIMIT 10;
                """)
                
                slow_result = await session.execute(slow_query)
                slow_queries = [dict(row) for row in slow_result]
                
                return {
                    'index_usage': index_usage,
                    'table_stats': table_stats,
                    'slow_queries': slow_queries
                }
                
        except Exception as e:
            logger.error(f"Error analyzing index usage: {e}")
            return {}
    
    async def optimize_indexes(self):
        """Optimize indexes based on usage patterns"""
        try:
            async with self.db_session_factory() as session:
                # Update table statistics
                await session.execute(text("ANALYZE;"))
                
                # Vacuum tables to reclaim space
                await session.execute(text("VACUUM ANALYZE candlestick_data;"))
                await session.execute(text("VACUUM ANALYZE signals;"))
                await session.execute(text("VACUUM ANALYZE trades;"))
                await session.execute(text("VACUUM ANALYZE candlestick_patterns;"))
                
                # Reindex if needed
                await session.execute(text("REINDEX INDEX CONCURRENTLY idx_candlestick_data_timestamp_brin;"))
                await session.execute(text("REINDEX INDEX CONCURRENTLY idx_signals_timestamp_brin;"))
                await session.execute(text("REINDEX INDEX CONCURRENTLY idx_trades_timestamp_brin;"))
                
                await session.commit()
                logger.info("✅ Index optimization completed")
                
        except Exception as e:
            logger.error(f"Error optimizing indexes: {e}")
            raise
    
    async def get_index_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for index improvements"""
        try:
            async with self.db_session_factory() as session:
                # Find tables without proper indexes
                query = text("""
                    SELECT 
                        t.tablename,
                        COUNT(i.indexname) as index_count,
                        pg_size_pretty(pg_total_relation_size(t.tablename::regclass)) as table_size
                    FROM pg_tables t
                    LEFT JOIN pg_indexes i ON t.tablename = i.tablename
                    WHERE t.schemaname = 'public'
                    GROUP BY t.tablename
                    HAVING COUNT(i.indexname) < 3  -- Tables with less than 3 indexes
                    ORDER BY pg_total_relation_size(t.tablename::regclass) DESC;
                """)
                
                result = await session.execute(query)
                recommendations = []
                
                for row in result:
                    recommendations.append({
                        'table': row[0],
                        'current_indexes': row[1],
                        'table_size': row[2],
                        'recommendation': f"Add indexes for {row[0]} table"
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Error getting index recommendations: {e}")
            return []
    
    async def create_index_for_query(self, query: str, table_name: str) -> bool:
        """Create a specific index for a given query pattern"""
        try:
            async with self.db_session_factory() as session:
                # This is a simplified version - in practice, you'd use query analysis
                # to determine the optimal index columns
                
                if 'candlestick_data' in query and 'symbol' in query and 'timestamp' in query:
                    await session.execute(text("""
                        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_candlestick_custom 
                        ON candlestick_data (symbol, timestamp DESC);
                    """))
                
                elif 'signals' in query and 'confidence' in query:
                    await session.execute(text("""
                        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_custom 
                        ON signals (confidence DESC, timestamp DESC);
                    """))
                
                elif 'trades' in query and 'pnl' in query:
                    await session.execute(text("""
                        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_custom 
                        ON trades (pnl DESC, timestamp DESC);
                    """))
                
                await session.commit()
                logger.info(f"✅ Created custom index for {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating custom index: {e}")
            return False
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        try:
            async with self.db_session_factory() as session:
                # Get index hit ratio
                hit_ratio_query = text("""
                    SELECT 
                        sum(idx_scan) as index_scans,
                        sum(seq_scan) as seq_scans,
                        CASE 
                            WHEN sum(idx_scan + seq_scan) > 0 
                            THEN round(sum(idx_scan)::numeric / (sum(idx_scan) + sum(seq_scan)) * 100, 2)
                            ELSE 0 
                        END as hit_ratio_percent
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public';
                """)
                
                hit_result = await session.execute(hit_ratio_query)
                hit_stats = hit_result.fetchone()
                
                # Get total indexes count and size
                index_query = text("""
                    SELECT 
                        count(*) as total_indexes,
                        sum(pg_relation_size(indexrelid)) as total_size_bytes
                    FROM pg_stat_user_indexes 
                    WHERE schemaname = 'public';
                """)
                
                index_result = await session.execute(index_query)
                index_stats = index_result.fetchone()
                
                # Get index usage details
                usage_query = text("""
                    SELECT 
                        indexname,
                        idx_scan as scans,
                        idx_tup_read as tuples_read,
                        pg_size_pretty(pg_relation_size(indexrelid)) as size
                    FROM pg_stat_user_indexes 
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                    LIMIT 10;
                """)
                
                usage_result = await session.execute(usage_query)
                usage_stats = [dict(row) for row in usage_result]
                
                return {
                    'index_hit_ratio': hit_stats.hit_ratio_percent if hit_stats else 0,
                    'total_indexes': index_stats.total_indexes if index_stats else 0,
                    'index_size_mb': (index_stats.total_size_bytes / (1024 * 1024)) if index_stats and index_stats.total_size_bytes else 0,
                    'index_scans': hit_stats.index_scans if hit_stats else 0,
                    'seq_scans': hit_stats.seq_scans if hit_stats else 0,
                    'top_indexes': usage_stats,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            return {
                'index_hit_ratio': 0,
                'total_indexes': 0,
                'index_size_mb': 0,
                'index_scans': 0,
                'seq_scans': 0,
                'top_indexes': [],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }