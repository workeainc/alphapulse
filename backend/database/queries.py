import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import text, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class TimescaleQueries:
    """TimescaleDB-specific queries for efficient time-series operations"""

    @staticmethod
    async def get_market_data_timeframe(
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """Get market data for a specific timeframe using TimescaleDB time_bucket"""
        query = text("""
            SELECT 
                time_bucket(:timeframe_interval, timestamp) AS bucket,
                symbol,
                first(open, timestamp) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close, timestamp) AS close,
                sum(volume) AS volume,
                count(*) AS candle_count
            FROM market_data
            WHERE symbol = :symbol 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
            GROUP BY bucket, symbol
            ORDER BY bucket DESC
            LIMIT :limit
        """)
        
        # Convert timeframe to interval
        timeframe_map = {
            "1m": "1 minute",
            "5m": "5 minutes", 
            "15m": "15 minutes",
            "30m": "30 minutes",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day"
        }
        
        result = await session.execute(query, {
            "symbol": symbol,
            "timeframe_interval": timeframe_map.get(timeframe, "1 hour"),
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit
        })
        
        return [dict(row) for row in result]

    @staticmethod
    async def get_latest_market_data(
        session: AsyncSession,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get latest market data for a symbol"""
        query = text("""
            SELECT 
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        result = await session.execute(query, {
            "symbol": symbol,
            "limit": limit
        })
        
        return [dict(row) for row in result]

    @staticmethod
    async def get_sentiment_summary(
        session: AsyncSession,
        symbol: str,
        hours: int = 24
    ) -> Optional[Dict]:
        """Get sentiment summary for the last N hours"""
        query = text("""
            SELECT 
                avg(sentiment_score) AS avg_sentiment,
                count(*) AS sentiment_count,
                min(sentiment_score) AS min_sentiment,
                max(sentiment_score) AS max_sentiment,
                stddev(sentiment_score) AS sentiment_volatility
            FROM sentiment_data
            WHERE symbol = :symbol 
                AND timestamp >= NOW() - INTERVAL ':hours hours'
        """)
        
        result = await session.execute(query, {
            "symbol": symbol,
            "hours": hours
        })
        
        row = result.fetchone()
        return dict(row) if row else None

    @staticmethod
    async def get_trade_performance_summary(
        session: AsyncSession,
        days: int = 30
    ) -> Dict:
        """Get trading performance summary for the last N days"""
        query = text("""
            SELECT 
                count(*) AS total_trades,
                count(CASE WHEN pnl > 0 THEN 1 END) AS winning_trades,
                count(CASE WHEN pnl < 0 THEN 1 END) AS losing_trades,
                sum(pnl) AS total_pnl,
                avg(pnl) AS avg_pnl,
                avg(CASE WHEN pnl > 0 THEN pnl END) AS avg_win,
                avg(CASE WHEN pnl < 0 THEN pnl END) AS avg_loss,
                max(pnl) AS max_win,
                min(pnl) AS max_loss
            FROM trades
            WHERE entry_time >= NOW() - INTERVAL ':days days'
                AND status = 'closed'
        """)
        
        result = await session.execute(query, {"days": days})
        row = result.fetchone()
        
        if row:
            data = dict(row)
            if data['total_trades'] > 0:
                data['win_rate'] = data['winning_trades'] / data['total_trades']
            else:
                data['win_rate'] = 0
            return data
        
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_win": 0,
            "max_loss": 0,
            "win_rate": 0
        }

    @staticmethod
    async def get_volatility_analysis(
        session: AsyncSession,
        symbol: str,
        days: int = 30
    ) -> Dict:
        """Get volatility analysis for a symbol"""
        query = text("""
            SELECT 
                avg(high - low) as avg_range,
                avg(abs(close - lag(close) OVER (ORDER BY timestamp))) as avg_price_change,
                max(high - low) as max_range,
                min(high - low) as min_range,
                stddev(close) as price_volatility
            FROM market_data
            WHERE symbol = :symbol 
                AND timestamp >= NOW() - INTERVAL ':days days'
        """)
        
        result = await session.execute(query, {"symbol": symbol, "days": days})
        row = result.fetchone()
        
        return dict(row) if row else {}

    @staticmethod
    async def get_latency_metrics_summary(
        session: AsyncSession,
        hours: int = 24
    ) -> Dict:
        """Get latency metrics summary for the last N hours"""
        query = text("""
            SELECT 
                operation_type,
                model_id,
                count(*) as total_operations,
                count(CASE WHEN success THEN 1 END) as successful_operations,
                avg(total_latency_ms) as avg_latency_ms,
                min(total_latency_ms) as min_latency_ms,
                max(total_latency_ms) as max_latency_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY total_latency_ms) as p99_latency_ms,
                avg(fetch_time_ms) as avg_fetch_ms,
                avg(preprocess_time_ms) as avg_preprocess_ms,
                avg(inference_time_ms) as avg_inference_ms,
                avg(postprocess_time_ms) as avg_postprocess_ms
            FROM latency_metrics
            WHERE timestamp >= NOW() - INTERVAL ':hours hours'
            GROUP BY operation_type, model_id
            ORDER BY avg_latency_ms DESC
        """)
        
        result = await session.execute(query, {"hours": hours})
        rows = result.fetchall()
        
        return [dict(row) for row in rows]

    @staticmethod
    async def get_latency_metrics_by_symbol(
        session: AsyncSession,
        symbol: str,
        hours: int = 24
    ) -> Dict:
        """Get latency metrics for a specific symbol"""
        query = text("""
            SELECT 
                time_bucket('1 hour', timestamp) AS hour_bucket,
                count(*) as operations_count,
                avg(total_latency_ms) as avg_latency_ms,
                max(total_latency_ms) as max_latency_ms,
                count(CASE WHEN success THEN 1 END) as successful_operations,
                avg(fetch_time_ms) as avg_fetch_ms,
                avg(preprocess_time_ms) as avg_preprocess_ms,
                avg(inference_time_ms) as avg_inference_ms,
                avg(postprocess_time_ms) as avg_postprocess_ms
            FROM latency_metrics
            WHERE symbol = :symbol 
                AND timestamp >= NOW() - INTERVAL ':hours hours'
            GROUP BY hour_bucket
            ORDER BY hour_bucket DESC
        """)
        
        result = await session.execute(query, {"symbol": symbol, "hours": hours})
        rows = result.fetchall()
        
        return [dict(row) for row in rows]

    @staticmethod
    async def get_latency_metrics_by_strategy(
        session: AsyncSession,
        strategy_name: str,
        hours: int = 24
    ) -> Dict:
        """Get latency metrics for a specific strategy"""
        query = text("""
            SELECT 
                time_bucket('1 hour', timestamp) AS hour_bucket,
                count(*) as operations_count,
                avg(total_latency_ms) as avg_latency_ms,
                max(total_latency_ms) as max_latency_ms,
                count(CASE WHEN success THEN 1 END) as successful_operations,
                avg(fetch_time_ms) as avg_fetch_ms,
                avg(preprocess_time_ms) as avg_preprocess_ms,
                avg(inference_time_ms) as avg_inference_ms,
                avg(postprocess_time_ms) as avg_postprocess_ms
            FROM latency_metrics
            WHERE strategy_name = :strategy_name 
                AND timestamp >= NOW() - INTERVAL ':hours hours'
            GROUP BY hour_bucket
            ORDER BY hour_bucket DESC
        """)
        
        result = await session.execute(query, {"strategy_name": strategy_name, "hours": hours})
        rows = result.fetchall()
        
        return [dict(row) for row in rows]

    @staticmethod
    async def get_latency_trends(
        session: AsyncSession,
        operation_type: str = None,
        model_id: str = None,
        hours: int = 24
    ) -> Dict:
        """Get latency trends over time"""
        where_clause = "WHERE timestamp >= NOW() - INTERVAL ':hours hours'"
        params = {"hours": hours}
        
        if operation_type:
            where_clause += " AND operation_type = :operation_type"
            params["operation_type"] = operation_type
        
        if model_id:
            where_clause += " AND model_id = :model_id"
            params["model_id"] = model_id
        
        query = text(f"""
            SELECT 
                time_bucket('15 minutes', timestamp) AS time_bucket,
                count(*) as operations_count,
                avg(total_latency_ms) as avg_latency_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
                count(CASE WHEN success THEN 1 END) as successful_operations
            FROM latency_metrics
            {where_clause}
            GROUP BY time_bucket
            ORDER BY time_bucket DESC
        """)
        
        result = await session.execute(query, params)
        rows = result.fetchall()
        
        return [dict(row) for row in rows]

    @staticmethod
    async def get_high_latency_operations(
        session: AsyncSession,
        threshold_ms: float = 1000,
        hours: int = 24
    ) -> Dict:
        """Get operations that exceeded latency threshold"""
        query = text("""
            SELECT 
                timestamp,
                operation_type,
                model_id,
                symbol,
                strategy_name,
                total_latency_ms,
                fetch_time_ms,
                preprocess_time_ms,
                inference_time_ms,
                postprocess_time_ms,
                success,
                error_message
            FROM latency_metrics
            WHERE total_latency_ms > :threshold_ms
                AND timestamp >= NOW() - INTERVAL ':hours hours'
            ORDER BY total_latency_ms DESC
            LIMIT 100
        """)
        
        result = await session.execute(query, {"threshold_ms": threshold_ms, "hours": hours})
        rows = result.fetchall()
        
        return [dict(row) for row in rows]

    @staticmethod
    async def get_latency_performance_comparison(
        session: AsyncSession,
        model_id_1: str,
        model_id_2: str,
        hours: int = 24
    ) -> Dict:
        """Compare latency performance between two models"""
        query = text("""
            SELECT 
                model_id,
                count(*) as total_operations,
                avg(total_latency_ms) as avg_latency_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY total_latency_ms) as p99_latency_ms,
                count(CASE WHEN success THEN 1 END) as successful_operations,
                avg(fetch_time_ms) as avg_fetch_ms,
                avg(preprocess_time_ms) as avg_preprocess_ms,
                avg(inference_time_ms) as avg_inference_ms,
                avg(postprocess_time_ms) as avg_postprocess_ms
            FROM latency_metrics
            WHERE model_id IN (:model_id_1, :model_id_2)
                AND timestamp >= NOW() - INTERVAL ':hours hours'
            GROUP BY model_id
        """)
        
        result = await session.execute(query, {
            "model_id_1": model_id_1, 
            "model_id_2": model_id_2, 
            "hours": hours
        })
        rows = result.fetchall()
        
        models_data = {row.model_id: dict(row) for row in rows}
        
        # Calculate improvement percentages
        comparison = {
            "model_1": models_data.get(model_id_1, {}),
            "model_2": models_data.get(model_id_2, {}),
            "improvements": {}
        }
        
        if model_id_1 in models_data and model_id_2 in models_data:
            m1 = models_data[model_id_1]
            m2 = models_data[model_id_2]
            
            if m1['avg_latency_ms'] > 0 and m2['avg_latency_ms'] > 0:
                comparison["improvements"] = {
                    "avg_latency_improvement_pct": ((m1['avg_latency_ms'] - m2['avg_latency_ms']) / m1['avg_latency_ms']) * 100,
                    "p95_latency_improvement_pct": ((m1['p95_latency_ms'] - m2['p95_latency_ms']) / m1['p95_latency_ms']) * 100,
                    "p99_latency_improvement_pct": ((m1['p99_latency_ms'] - m2['p99_latency_ms']) / m1['p99_latency_ms']) * 100,
                    "success_rate_improvement_pct": ((m2['successful_operations'] / m2['total_operations']) - (m1['successful_operations'] / m1['total_operations'])) * 100
                }
        
        return comparison

    @staticmethod
    async def get_support_resistance_levels(
        session: AsyncSession,
        symbol: str,
        days: int = 30
    ) -> Dict:
        """Get support and resistance levels using TimescaleDB"""
        query = text("""
            WITH price_levels AS (
                SELECT 
                    close,
                    count(*) as frequency
                FROM market_data
                WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL ':days days'
                GROUP BY close
                HAVING count(*) >= 3
            ),
            resistance AS (
                SELECT close, frequency
                FROM price_levels
                WHERE close > (
                    SELECT avg(close) 
                    FROM market_data 
                    WHERE symbol = :symbol 
                        AND timestamp >= NOW() - INTERVAL ':days days'
                )
                ORDER BY frequency DESC, close DESC
                LIMIT 5
            ),
            support AS (
                SELECT close, frequency
                FROM price_levels
                WHERE close < (
                    SELECT avg(close) 
                    FROM market_data 
                    WHERE symbol = :symbol 
                        AND timestamp >= NOW() - INTERVAL ':days days'
                )
                ORDER BY frequency DESC, close ASC
                LIMIT 5
            )
            SELECT 
                'resistance' as level_type,
                close as price,
                frequency
            FROM resistance
            UNION ALL
            SELECT 
                'support' as level_type,
                close as price,
                frequency
            FROM support
            ORDER BY level_type DESC, frequency DESC
        """)
        
        result = await session.execute(query, {
            "symbol": symbol,
            "days": days
        })
        
        levels = {"support": [], "resistance": []}
        for row in result:
            data = dict(row)
            if data['level_type'] == 'support':
                levels['support'].append({
                    'price': float(data['price']),
                    'frequency': data['frequency']
                })
            else:
                levels['resistance'].append({
                    'price': float(data['price']),
                    'frequency': data['frequency']
                })
        
        return levels

    @staticmethod
    async def get_market_regime_analysis(
        session: AsyncSession,
        symbol: str,
        days: int = 14
    ) -> Dict:
        """Analyze market regime using ATR and price action"""
        query = text("""
            WITH daily_stats AS (
                SELECT 
                    time_bucket('1 day', timestamp) AS day,
                    symbol,
                    (max(high) - min(low)) / avg(close) AS daily_range,
                    avg(close) AS avg_price,
                    sum(volume) AS total_volume
                FROM market_data
                WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL ':days days'
                GROUP BY day, symbol
            ),
            regime_stats AS (
                SELECT 
                    avg(daily_range) AS avg_volatility,
                    stddev(daily_range) AS volatility_std,
                    avg(total_volume) AS avg_volume
                FROM daily_stats
            )
            SELECT 
                CASE 
                    WHEN daily_range > (SELECT avg_volatility + volatility_std FROM regime_stats) 
                    THEN 'volatile'
                    WHEN daily_range < (SELECT avg_volatility - volatility_std FROM regime_stats) 
                    THEN 'ranging'
                    ELSE 'trending'
                END as regime,
                count(*) as regime_count
            FROM daily_stats, regime_stats
            GROUP BY regime
        """)
        
        result = await session.execute(query, {
            "symbol": symbol,
            "days": days
        })
        
        regimes = {}
        total_days = 0
        for row in result:
            data = dict(row)
            regimes[data['regime']] = data['regime_count']
            total_days += data['regime_count']
        
        # Calculate percentages
        for regime in regimes:
            regimes[regime] = {
                'days': regimes[regime],
                'percentage': (regimes[regime] / total_days * 100) if total_days > 0 else 0
            }
        
        return regimes

    @staticmethod
    async def get_continuous_aggregate_data(
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        hours: int = 24
    ) -> List[Dict]:
        """Get data from continuous aggregates for efficient querying"""
        view_map = {
            "1h": "market_data_1h",
            "4h": "market_data_4h"
        }
        
        if timeframe not in view_map:
            return []
        
        query = text(f"""
            SELECT 
                bucket,
                symbol,
                open,
                high,
                low,
                close,
                volume
            FROM {view_map[timeframe]}
            WHERE symbol = :symbol 
                AND bucket >= NOW() - INTERVAL ':hours hours'
            ORDER BY bucket DESC
        """)
        
        result = await session.execute(query, {
            "symbol": symbol,
            "hours": hours
        })
        
        return [dict(row) for row in result]

    @staticmethod
    async def cleanup_old_data(
        session: AsyncSession,
        days: int = 90
    ) -> int:
        """Clean up old data to maintain performance"""
        queries = [
            text("DELETE FROM market_data WHERE timestamp < NOW() - INTERVAL ':days days'"),
            text("DELETE FROM sentiment_data WHERE timestamp < NOW() - INTERVAL ':days days'"),
            text("DELETE FROM trades WHERE entry_time < NOW() - INTERVAL ':days days'")
        ]
        
        total_deleted = 0
        for query in queries:
            result = await session.execute(query, {"days": days})
            total_deleted += result.rowcount
        
        await session.commit()
        return total_deleted

    # Accuracy Benchmarking Queries
    @staticmethod
    async def get_model_accuracy_summary(
        session: AsyncSession,
        model_id: str = None,
        symbol: str = None,
        days: int = 30
    ) -> List[Dict]:
        """Get accuracy metrics summary for models"""
        base_query = f"""
            SELECT 
                model_id,
                symbol,
                strategy_name,
                avg(f1_score) as avg_f1_score,
                avg(precision) as avg_precision,
                avg(recall) as avg_recall,
                avg(accuracy) as avg_accuracy,
                avg(roc_auc) as avg_roc_auc,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                count(*) as evaluation_count,
                max(evaluation_date) as latest_evaluation
            FROM model_accuracy_benchmarks
            WHERE evaluation_date >= NOW() - INTERVAL '{days} days'
        """
        
        if model_id:
            base_query += f" AND model_id = '{model_id}'"
        
        if symbol:
            base_query += f" AND symbol = '{symbol}'"
        
        base_query += """
            GROUP BY model_id, symbol, strategy_name
            ORDER BY avg_f1_score DESC, avg_win_rate DESC
        """
        
        result = await session.execute(text(base_query))
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_model_comparison_data(
        session: AsyncSession,
        baseline_model_id: str,
        optimized_model_id: str,
        symbol: str,
        days: int = 90
    ) -> List[Dict]:
        """Get comparison data between two models"""
        query = text("""
            SELECT 
                model_id,
                evaluation_date,
                f1_score,
                precision,
                recall,
                accuracy,
                roc_auc,
                win_rate,
                profit_factor,
                total_return,
                sharpe_ratio,
                max_drawdown,
                total_trades,
                winning_trades,
                losing_trades
            FROM model_accuracy_benchmarks
            WHERE model_id IN (:baseline_model_id, :optimized_model_id)
                AND symbol = :symbol
                AND evaluation_date >= NOW() - INTERVAL ':days days'
            ORDER BY model_id, evaluation_date
        """)
        
        result = await session.execute(query, {
            "baseline_model_id": baseline_model_id,
            "optimized_model_id": optimized_model_id,
            "symbol": symbol,
            "days": days
        })
        
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_best_performing_models(
        session: AsyncSession,
        symbol: str = None,
        strategy_name: str = None,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict]:
        """Get best performing models by F1 score and win rate"""
        base_query = """
            SELECT 
                model_id,
                symbol,
                strategy_name,
                f1_score,
                precision,
                recall,
                accuracy,
                roc_auc,
                win_rate,
                profit_factor,
                total_return,
                sharpe_ratio,
                total_trades,
                evaluation_date
            FROM model_accuracy_benchmarks
            WHERE evaluation_date >= NOW() - INTERVAL ':days days'
        """
        
        params = {"days": days}
        
        if symbol:
            base_query += " AND symbol = :symbol"
            params["symbol"] = symbol
        
        if strategy_name:
            base_query += " AND strategy_name = :strategy_name"
            params["strategy_name"] = strategy_name
        
        base_query += """
            ORDER BY f1_score DESC, win_rate DESC
            LIMIT :limit
        """
        params["limit"] = limit
        
        result = await session.execute(text(base_query), params)
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_accuracy_trends(
        session: AsyncSession,
        model_id: str,
        symbol: str,
        days: int = 90
    ) -> List[Dict]:
        """Get accuracy trends over time for a specific model"""
        query = text("""
            SELECT 
                time_bucket('1 day', evaluation_date) as day_bucket,
                avg(f1_score) as avg_f1_score,
                avg(precision) as avg_precision,
                avg(recall) as avg_recall,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                count(*) as evaluation_count
            FROM model_accuracy_benchmarks
            WHERE model_id = :model_id
                AND symbol = :symbol
                AND evaluation_date >= NOW() - INTERVAL ':days days'
            GROUP BY day_bucket
            ORDER BY day_bucket DESC
        """)
        
        result = await session.execute(query, {
            "model_id": model_id,
            "symbol": symbol,
            "days": days
        })
        
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_model_performance_ranking(
        session: AsyncSession,
        symbol: str = None,
        days: int = 30
    ) -> List[Dict]:
        """Get model performance ranking across all models"""
        base_query = """
            SELECT 
                model_id,
                symbol,
                strategy_name,
                avg(f1_score) as avg_f1_score,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(total_return) as avg_total_return,
                count(*) as evaluation_count,
                rank() OVER (ORDER BY avg(f1_score) DESC) as f1_rank,
                rank() OVER (ORDER BY avg(win_rate) DESC) as win_rate_rank,
                rank() OVER (ORDER BY avg(profit_factor) DESC) as profit_factor_rank
            FROM model_accuracy_benchmarks
            WHERE evaluation_date >= NOW() - INTERVAL ':days days'
        """
        
        params = {"days": days}
        
        if symbol:
            base_query += " AND symbol = :symbol"
            params["symbol"] = symbol
        
        base_query += """
            GROUP BY model_id, symbol, strategy_name
            HAVING count(*) >= 3
            ORDER BY avg_f1_score DESC, avg_win_rate DESC
        """
        
        result = await session.execute(text(base_query), params)
        return [dict(row) for row in result.fetchall()]

    # Model Comparison Queries
    @staticmethod
    async def get_model_comparison_summary(
        session: AsyncSession,
        baseline_model_id: str,
        optimized_model_id: str,
        days: int = 90
    ) -> Dict:
        """Get comparison summary between two models"""
        query = text(f"""
            SELECT 
                'baseline' as model_type,
                model_id,
                avg(f1_score) as avg_f1_score,
                avg(precision) as avg_precision,
                avg(recall) as avg_recall,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(max_drawdown) as avg_max_drawdown,
                count(*) as evaluation_count
            FROM model_accuracy_benchmarks
            WHERE model_id = '{baseline_model_id}'
                AND evaluation_date >= NOW() - INTERVAL '{days} days'
            GROUP BY model_id
            
            UNION ALL
            
            SELECT 
                'optimized' as model_type,
                model_id,
                avg(f1_score) as avg_f1_score,
                avg(precision) as avg_precision,
                avg(recall) as avg_recall,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(max_drawdown) as avg_max_drawdown,
                count(*) as evaluation_count
            FROM model_accuracy_benchmarks
            WHERE model_id = '{optimized_model_id}'
                AND evaluation_date >= NOW() - INTERVAL '{days} days'
            GROUP BY model_id
        """)
        
        result = await session.execute(query)
        
        rows = [dict(row) for row in result.fetchall()]
        
        # Organize results
        summary = {
            'baseline': None,
            'optimized': None,
            'improvements': {}
        }
        
        for row in rows:
            if row['model_type'] == 'baseline':
                summary['baseline'] = row
            elif row['model_type'] == 'optimized':
                summary['optimized'] = row
        
        # Calculate improvements if both models exist
        if summary['baseline'] and summary['optimized']:
            baseline = summary['baseline']
            optimized = summary['optimized']
            
            summary['improvements'] = {
                'f1_improvement': optimized['avg_f1_score'] - baseline['avg_f1_score'],
                'precision_improvement': optimized['avg_precision'] - baseline['avg_precision'],
                'recall_improvement': optimized['avg_recall'] - baseline['avg_recall'],
                'win_rate_improvement': optimized['avg_win_rate'] - baseline['avg_win_rate'],
                'profit_factor_improvement': optimized['avg_profit_factor'] - baseline['avg_profit_factor'],
                'total_return_improvement': optimized['avg_total_return'] - baseline['avg_total_return'],
                'sharpe_ratio_improvement': optimized['avg_sharpe_ratio'] - baseline['avg_sharpe_ratio'],
                'max_drawdown_improvement': baseline['avg_max_drawdown'] - optimized['avg_max_drawdown']  # Lower is better
            }
        
        return summary

    @staticmethod
    async def get_model_evolution_trends(
        session: AsyncSession,
        model_id: str,
        days: int = 180
    ) -> List[Dict]:
        """Get model performance evolution over time"""
        query = text(f"""
            SELECT 
                time_bucket('7 days', evaluation_date) as week_bucket,
                avg(f1_score) as avg_f1_score,
                avg(precision) as avg_precision,
                avg(recall) as avg_recall,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(max_drawdown) as avg_max_drawdown,
                count(*) as evaluation_count
            FROM model_accuracy_benchmarks
            WHERE model_id = '{model_id}'
                AND evaluation_date >= NOW() - INTERVAL '{days} days'
            GROUP BY week_bucket
            ORDER BY week_bucket DESC
        """)
        
        result = await session.execute(query)
        
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_promotion_candidates(
        session: AsyncSession,
        symbol: str = None,
        strategy_name: str = None,
        days: int = 30
    ) -> List[Dict]:
        """Get models that meet promotion criteria"""
        base_query = f"""
            WITH model_performance AS (
                SELECT 
                    model_id,
                    symbol,
                    strategy_name,
                    avg(f1_score) as avg_f1_score,
                    avg(win_rate) as avg_win_rate,
                    avg(profit_factor) as avg_profit_factor,
                    avg(total_return) as avg_total_return,
                    avg(sharpe_ratio) as avg_sharpe_ratio,
                    count(*) as evaluation_count,
                    max(evaluation_date) as latest_evaluation
                FROM model_accuracy_benchmarks
                WHERE evaluation_date >= NOW() - INTERVAL '{days} days'
        """
        
        if symbol:
            base_query += f" AND symbol = '{symbol}'"
        
        if strategy_name:
            base_query += f" AND strategy_name = '{strategy_name}'"
        
        base_query += """
                GROUP BY model_id, symbol, strategy_name
                HAVING count(*) >= 3
            )
            SELECT 
                *,
                CASE 
                    WHEN avg_f1_score >= 0.7 AND avg_win_rate >= 0.6 AND avg_profit_factor >= 1.5 
                    THEN 'high_potential'
                    WHEN avg_f1_score >= 0.65 AND avg_win_rate >= 0.55 AND avg_profit_factor >= 1.3 
                    THEN 'promising'
                    ELSE 'needs_improvement'
                END as promotion_status
            FROM model_performance
            ORDER BY avg_f1_score DESC, avg_win_rate DESC
        """
        
        result = await session.execute(text(base_query))
        return [dict(row) for row in result.fetchall()]

    @staticmethod
    async def get_model_rollback_analysis(
        session: AsyncSession,
        model_id: str,
        days: int = 30
    ) -> Dict:
        """Analyze if a model should be rolled back"""
        query = text(f"""
            SELECT 
                avg(f1_score) as avg_f1_score,
                avg(win_rate) as avg_win_rate,
                avg(profit_factor) as avg_profit_factor,
                avg(total_return) as avg_total_return,
                avg(sharpe_ratio) as avg_sharpe_ratio,
                avg(max_drawdown) as avg_max_drawdown,
                count(*) as evaluation_count,
                min(evaluation_date) as first_evaluation,
                max(evaluation_date) as latest_evaluation
            FROM model_accuracy_benchmarks
            WHERE model_id = '{model_id}'
                AND evaluation_date >= NOW() - INTERVAL '{days} days'
        """)
        
        result = await session.execute(query)
        
        row = result.fetchone()
        if not row:
            return {
                'should_rollback': False,
                'reason': 'No recent evaluations',
                'metrics': {}
            }
        
        metrics = dict(row)
        
        # Determine if rollback is needed
        should_rollback = False
        reasons = []
        
        if metrics['avg_f1_score'] < 0.6:
            should_rollback = True
            reasons.append(f"Low F1 score: {metrics['avg_f1_score']:.3f}")
        
        if metrics['avg_win_rate'] < 0.5:
            should_rollback = True
            reasons.append(f"Low win rate: {metrics['avg_win_rate']:.3f}")
        
        if metrics['avg_profit_factor'] < 1.2:
            should_rollback = True
            reasons.append(f"Low profit factor: {metrics['avg_profit_factor']:.3f}")
        
        if metrics['avg_total_return'] < 0:
            should_rollback = True
            reasons.append(f"Negative total return: {metrics['avg_total_return']:.2f}")
        
        return {
            'should_rollback': should_rollback,
            'reason': '; '.join(reasons) if reasons else 'Performance acceptable',
            'metrics': metrics
        }
