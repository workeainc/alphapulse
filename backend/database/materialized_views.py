"""
Pre-Joined Materialized Views for AlphaPlus
Creates optimized views that combine market data, indicators, and patterns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class MaterializedViewsManager:
    """Manages pre-joined materialized views for ultra-fast queries"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.refresh_tasks = {}
        self.is_running = False
        
        logger.info("🚀 Materialized Views Manager initialized")
    
    async def create_all_views(self):
        """Create all pre-joined materialized views"""
        try:
            async with self.db_session_factory() as session:
                # Create market data with indicators view
                await self._create_market_data_with_indicators_view(session)
                
                # Create signals with context view
                await self._create_signals_with_context_view(session)
                
                # Create patterns with market context view
                await self._create_patterns_with_context_view(session)
                
                # Create performance summary view
                await self._create_performance_summary_view(session)
                
                # Create UI cache view for frontend
                await self._create_ui_cache_view(session)
                
                await session.commit()
                logger.info("✅ All materialized views created successfully")
                
        except Exception as e:
            logger.error(f"❌ Error creating materialized views: {e}")
            raise
    
    async def _create_market_data_with_indicators_view(self, session: AsyncSession):
        """Create view combining market data with technical indicators"""
        try:
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_with_indicators
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', cd.timestamp) AS bucket,
                    cd.symbol,
                    cd.timeframe,
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.indicators->>'rsi_14' AS rsi_14,
                    cd.indicators->>'macd' AS macd,
                    cd.indicators->>'macd_signal' AS macd_signal,
                    cd.indicators->>'macd_histogram' AS macd_histogram,
                    cd.indicators->>'ema_9' AS ema_9,
                    cd.indicators->>'ema_21' AS ema_21,
                    cd.indicators->>'ema_50' AS ema_50,
                    cd.indicators->>'bb_upper' AS bb_upper,
                    cd.indicators->>'bb_middle' AS bb_middle,
                    cd.indicators->>'bb_lower' AS bb_lower,
                    cd.indicators->>'atr_14' AS atr_14,
                    cd.indicators->>'stoch_k' AS stoch_k,
                    cd.indicators->>'stoch_d' AS stoch_d,
                    cd.patterns,
                    COUNT(*) OVER (PARTITION BY cd.symbol ORDER BY cd.timestamp) AS candle_count
                FROM candlestick_data cd
                WHERE cd.timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY bucket, cd.symbol, cd.timeframe, cd.open, cd.high, cd.low, cd.close, cd.volume, 
                         cd.indicators, cd.patterns
                ORDER BY bucket DESC, cd.symbol;
            """))
            
            # Add refresh policy
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('market_data_with_indicators',
                    start_offset => INTERVAL '1 hour',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '1 minute'
                );
            """))
            
            logger.info("✅ Created market_data_with_indicators view")
            
        except Exception as e:
            logger.error(f"Error creating market_data_with_indicators view: {e}")
            raise
    
    async def _create_signals_with_context_view(self, session: AsyncSession):
        """Create view combining signals with market context"""
        try:
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS signals_with_context
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', s.timestamp) AS bucket,
                    s.id,
                    s.symbol,
                    s.side,
                    s.strategy,
                    s.confidence,
                    s.strength,
                    s.price,
                    s.stop_loss,
                    s.take_profit,
                    s.status,
                    s.metadata,
                    -- Market context at signal time
                    cd.open AS market_open,
                    cd.high AS market_high,
                    cd.low AS market_low,
                    cd.close AS market_close,
                    cd.volume AS market_volume,
                    cd.indicators->>'rsi_14' AS market_rsi,
                    cd.indicators->>'macd' AS market_macd,
                    cd.indicators->>'ema_9' AS market_ema_9,
                    cd.indicators->>'ema_21' AS market_ema_21,
                    -- Pattern context
                    cp.pattern_name,
                    cp.confidence AS pattern_confidence,
                    cp.volume_confirmation,
                    cp.trend_alignment,
                    -- Performance metrics
                    t.pnl,
                    t.exit_price,
                    t.exit_timestamp,
                    CASE 
                        WHEN t.pnl > 0 THEN 'win'
                        WHEN t.pnl < 0 THEN 'loss'
                        ELSE 'pending'
                    END AS trade_result
                FROM signals s
                LEFT JOIN candlestick_data cd ON s.symbol = cd.symbol 
                    AND s.timestamp = cd.timestamp
                LEFT JOIN candlestick_patterns cp ON s.symbol = cp.symbol 
                    AND s.timestamp = cp.timestamp
                LEFT JOIN trades t ON s.id = t.signal_id
                WHERE s.timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY bucket, s.id, s.symbol, s.side, s.strategy, s.confidence, s.strength,
                         s.price, s.stop_loss, s.take_profit, s.status, s.metadata,
                         cd.open, cd.high, cd.low, cd.close, cd.volume, cd.indicators,
                         cp.pattern_name, cp.confidence, cp.volume_confirmation, cp.trend_alignment,
                         t.pnl, t.exit_price, t.exit_timestamp
                ORDER BY bucket DESC, s.symbol;
            """))
            
            # Add refresh policy
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('signals_with_context',
                    start_offset => INTERVAL '2 hours',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '5 minutes'
                );
            """))
            
            logger.info("✅ Created signals_with_context view")
            
        except Exception as e:
            logger.error(f"Error creating signals_with_context view: {e}")
            raise
    
    async def _create_patterns_with_context_view(self, session: AsyncSession):
        """Create view combining patterns with market context"""
        try:
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS patterns_with_context
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', cp.timestamp) AS bucket,
                    cp.symbol,
                    cp.pattern_name,
                    cp.confidence,
                    cp.volume_confirmation,
                    cp.trend_alignment,
                    cp.metadata,
                    -- Market context
                    cd.open,
                    cd.high,
                    cd.low,
                    cd.close,
                    cd.volume,
                    cd.indicators->>'rsi_14' AS rsi_14,
                    cd.indicators->>'macd' AS macd,
                    cd.indicators->>'ema_9' AS ema_9,
                    cd.indicators->>'ema_21' AS ema_21,
                    cd.indicators->>'bb_upper' AS bb_upper,
                    cd.indicators->>'bb_lower' AS bb_lower,
                    -- Signal outcomes
                    COUNT(s.id) AS signal_count,
                    COUNT(CASE WHEN s.confidence >= 0.8 THEN 1 END) AS high_confidence_signals,
                    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) AS winning_signals,
                    COUNT(CASE WHEN t.pnl < 0 THEN 1 END) AS losing_signals,
                    AVG(t.pnl) AS avg_pnl,
                    -- Pattern frequency
                    COUNT(*) OVER (PARTITION BY cp.symbol, cp.pattern_name ORDER BY cp.timestamp) AS pattern_frequency
                FROM candlestick_patterns cp
                LEFT JOIN candlestick_data cd ON cp.symbol = cd.symbol 
                    AND cp.timestamp = cd.timestamp
                LEFT JOIN signals s ON cp.symbol = s.symbol 
                    AND cp.timestamp = s.timestamp
                LEFT JOIN trades t ON s.id = t.signal_id
                WHERE cp.timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY bucket, cp.symbol, cp.pattern_name, cp.confidence, cp.volume_confirmation,
                         cp.trend_alignment, cp.metadata, cd.open, cd.high, cd.low, cd.close, cd.volume, cd.indicators
                ORDER BY bucket DESC, cp.symbol, cp.pattern_name;
            """))
            
            # Add refresh policy
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('patterns_with_context',
                    start_offset => INTERVAL '2 hours',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '5 minutes'
                );
            """))
            
            logger.info("✅ Created patterns_with_context view")
            
        except Exception as e:
            logger.error(f"Error creating patterns_with_context view: {e}")
            raise
    
    async def _create_performance_summary_view(self, session: AsyncSession):
        """Create view for performance analytics"""
        try:
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS performance_summary
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', t.timestamp) AS bucket,
                    t.symbol,
                    t.strategy,
                    -- Trade metrics
                    COUNT(*) AS total_trades,
                    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) AS winning_trades,
                    COUNT(CASE WHEN t.pnl < 0 THEN 1 END) AS losing_trades,
                    COUNT(CASE WHEN t.status = 'open' THEN 1 END) AS open_trades,
                    -- Performance metrics
                    AVG(t.pnl) AS avg_pnl,
                    SUM(t.pnl) AS total_pnl,
                    MAX(t.pnl) AS max_win,
                    MIN(t.pnl) AS max_loss,
                    STDDEV(t.pnl) AS pnl_stddev,
                    -- Win rate and ratios
                    CASE 
                        WHEN COUNT(*) > 0 THEN 
                            COUNT(CASE WHEN t.pnl > 0 THEN 1 END)::FLOAT / COUNT(*)
                        ELSE 0 
                    END AS win_rate,
                    CASE 
                        WHEN AVG(CASE WHEN t.pnl < 0 THEN ABS(t.pnl) END) > 0 THEN
                            AVG(CASE WHEN t.pnl > 0 THEN t.pnl END) / AVG(CASE WHEN t.pnl < 0 THEN ABS(t.pnl) END)
                        ELSE 0 
                    END AS profit_factor,
                    -- Risk metrics
                    MAX(t.entry_price) AS max_entry_price,
                    MIN(t.entry_price) AS min_entry_price,
                    AVG(t.entry_price) AS avg_entry_price,
                    -- Signal quality
                    AVG(s.confidence) AS avg_signal_confidence,
                    COUNT(CASE WHEN s.confidence >= 0.8 THEN 1 END) AS high_confidence_signals
                FROM trades t
                LEFT JOIN signals s ON t.signal_id = s.id
                WHERE t.timestamp >= NOW() - INTERVAL '90 days'
                GROUP BY bucket, t.symbol, t.strategy
                ORDER BY bucket DESC, t.symbol, t.strategy;
            """))
            
            # Add refresh policy
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('performance_summary',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour'
                );
            """))
            
            logger.info("✅ Created performance_summary view")
            
        except Exception as e:
            logger.error(f"Error creating performance_summary view: {e}")
            raise
    
    async def _create_ui_cache_view(self, session: AsyncSession):
        """Create view optimized for frontend UI queries"""
        try:
            await session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ui_cache
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('5 minutes', NOW()) AS bucket,
                    -- Latest market data snapshot
                    jsonb_build_object(
                        'symbols', (
                            SELECT jsonb_object_agg(symbol, jsonb_build_object(
                                'price', close,
                                'change_24h', (close - LAG(close, 1440) OVER (PARTITION BY symbol ORDER BY timestamp)) / LAG(close, 1440) OVER (PARTITION BY symbol ORDER BY timestamp) * 100,
                                'volume_24h', SUM(volume) OVER (PARTITION BY symbol ORDER BY timestamp ROWS BETWEEN 1440 PRECEDING AND CURRENT ROW),
                                'rsi', indicators->>'rsi_14',
                                'macd', indicators->>'macd',
                                'trend', CASE 
                                    WHEN indicators->>'ema_9' > indicators->>'ema_21' THEN 'bullish'
                                    ELSE 'bearish'
                                END
                            ))
                            FROM (
                                SELECT DISTINCT ON (symbol) 
                                    symbol, close, volume, indicators, timestamp
                                FROM candlestick_data 
                                WHERE timestamp >= NOW() - INTERVAL '1 day'
                                ORDER BY symbol, timestamp DESC
                            ) latest_data
                        ),
                        'signals', (
                            SELECT jsonb_agg(jsonb_build_object(
                                'id', id,
                                'symbol', symbol,
                                'side', side,
                                'confidence', confidence,
                                'price', price,
                                'timestamp', timestamp
                            ))
                            FROM (
                                SELECT id, symbol, side, confidence, price, timestamp
                                FROM signals 
                                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                                ORDER BY timestamp DESC
                                LIMIT 50
                            ) recent_signals
                        ),
                        'patterns', (
                            SELECT jsonb_agg(jsonb_build_object(
                                'symbol', symbol,
                                'pattern', pattern_name,
                                'confidence', confidence,
                                'timestamp', timestamp
                            ))
                            FROM (
                                SELECT symbol, pattern_name, confidence, timestamp
                                FROM candlestick_patterns 
                                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                                ORDER BY timestamp DESC
                                LIMIT 50
                            ) recent_patterns
                        ),
                        'performance', (
                            SELECT jsonb_build_object(
                                'total_trades', COUNT(*),
                                'win_rate', COUNT(CASE WHEN pnl > 0 THEN 1 END)::FLOAT / COUNT(*),
                                'total_pnl', SUM(pnl),
                                'avg_pnl', AVG(pnl)
                            )
                            FROM trades 
                            WHERE timestamp >= NOW() - INTERVAL '24 hours'
                        )
                    ) AS ui_data
                FROM candlestick_data
                WHERE timestamp >= NOW() - INTERVAL '1 minute'
                GROUP BY bucket;
            """))
            
            # Add refresh policy
            await session.execute(text("""
                SELECT add_continuous_aggregate_policy('ui_cache',
                    start_offset => INTERVAL '10 minutes',
                    end_offset => INTERVAL '5 minutes',
                    schedule_interval => INTERVAL '5 minutes'
                );
            """))
            
            logger.info("✅ Created ui_cache view")
            
        except Exception as e:
            logger.error(f"Error creating ui_cache view: {e}")
            raise
    
    async def get_market_data_with_indicators(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get market data with indicators from materialized view"""
        try:
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT *
                    FROM market_data_with_indicators
                    WHERE symbol = :symbol 
                        AND bucket >= NOW() - INTERVAL ':hours hours'
                    ORDER BY bucket DESC
                """)
                
                result = await session.execute(query, {
                    "symbol": symbol,
                    "hours": hours
                })
                
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting market data with indicators: {e}")
            return []
    
    async def get_signals_with_context(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get signals with market context from materialized view"""
        try:
            async with self.db_session_factory() as session:
                if symbol:
                    query = text("""
                        SELECT *
                        FROM signals_with_context
                        WHERE symbol = :symbol 
                            AND bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                    """)
                    params = {"symbol": symbol, "hours": hours}
                else:
                    query = text("""
                        SELECT *
                        FROM signals_with_context
                        WHERE bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                        LIMIT 100
                    """)
                    params = {"hours": hours}
                
                result = await session.execute(query, params)
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting signals with context: {e}")
            return []
    
    async def get_patterns_with_context(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get patterns with market context from materialized view"""
        try:
            async with self.db_session_factory() as session:
                if symbol:
                    query = text("""
                        SELECT *
                        FROM patterns_with_context
                        WHERE symbol = :symbol 
                            AND bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                    """)
                    params = {"symbol": symbol, "hours": hours}
                else:
                    query = text("""
                        SELECT *
                        FROM patterns_with_context
                        WHERE bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                        LIMIT 100
                    """)
                    params = {"hours": hours}
                
                result = await session.execute(query, params)
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting patterns with context: {e}")
            return []
    
    async def get_performance_summary(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get performance summary from materialized view"""
        try:
            async with self.db_session_factory() as session:
                if symbol:
                    query = text("""
                        SELECT *
                        FROM performance_summary
                        WHERE symbol = :symbol 
                            AND bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                    """)
                    params = {"symbol": symbol, "hours": hours}
                else:
                    query = text("""
                        SELECT *
                        FROM performance_summary
                        WHERE bucket >= NOW() - INTERVAL ':hours hours'
                        ORDER BY bucket DESC
                    """)
                    params = {"hours": hours}
                
                result = await session.execute(query, params)
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return []
    
    async def get_ui_cache_data(self) -> Dict:
        """Get pre-computed UI data from cache view"""
        try:
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT ui_data
                    FROM ui_cache
                    WHERE bucket >= NOW() - INTERVAL '10 minutes'
                    ORDER BY bucket DESC
                    LIMIT 1
                """)
                
                result = await session.execute(query)
                row = result.fetchone()
                
                if row:
                    return row[0]
                else:
                    return {}
                
        except Exception as e:
            logger.error(f"Error getting UI cache data: {e}")
            return {}
    
    async def refresh_view(self, view_name: str):
        """Manually refresh a specific materialized view"""
        try:
            async with self.db_session_factory() as session:
                await session.execute(text(f"REFRESH MATERIALIZED VIEW {view_name}"))
                await session.commit()
                logger.info(f"✅ Refreshed {view_name} view")
                
        except Exception as e:
            logger.error(f"Error refreshing {view_name} view: {e}")
            raise
    
    async def get_view_stats(self) -> Dict[str, Any]:
        """Get statistics about materialized views"""
        try:
            async with self.db_session_factory() as session:
                query = text("""
                    SELECT 
                        schemaname,
                        matviewname,
                        matviewowner,
                        definition
                    FROM pg_matviews 
                    WHERE schemaname = 'public'
                    ORDER BY matviewname;
                """)
                
                result = await session.execute(query)
                views = [dict(row) for row in result]
                
                # Get row counts for each view
                stats = {}
                for view in views:
                    view_name = view['matviewname']
                    count_query = text(f"SELECT COUNT(*) FROM {view_name}")
                    count_result = await session.execute(count_query)
                    count = count_result.scalar()
                    stats[view_name] = {
                        'row_count': count,
                        'owner': view['matviewowner']
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting view stats: {e}")
            return {}
