#!/usr/bin/env python3
"""
Retrieval Performance Service for AlphaPulse Trading Bot
Tests and optimizes TimescaleDB retrieval performance for large datasets (10M+ rows)
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import random

from sqlalchemy import text, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..src.database.connection import TimescaleDBConnection
from src.app.services.pattern_storage_service import PatternStorageService, PatternData

logger = logging.getLogger(__name__)

@dataclass
class QueryPerformanceResult:
    """Results from query performance testing"""
    query_name: str
    execution_time_ms: float
    rows_scanned: int
    rows_returned: int
    plan_type: str  # 'Index Scan', 'Seq Scan', 'Index Only Scan', etc.
    index_used: Optional[str]
    total_cost: float
    actual_time_ms: float
    planning_time_ms: float
    buffer_hits: int
    buffer_reads: int
    shared_hits: int
    shared_reads: int

@dataclass
class PerformanceBenchmark:
    """Complete benchmark results"""
    total_patterns: int
    test_queries: List[QueryPerformanceResult]
    index_recommendations: List[str]
    optimization_actions: List[str]
    overall_score: float
    timestamp: datetime

class RetrievalPerformanceService:
    """Service for testing and optimizing TimescaleDB retrieval performance"""
    
    def __init__(self):
        self.db_connection = TimescaleDBConnection()
        self.pattern_storage = PatternStorageService()
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_execution_time_ms': 100.0,  # 100ms max for queries
            'max_rows_scanned_ratio': 10.0,  # Max 10x rows scanned vs returned
            'min_index_scan_ratio': 0.8,     # 80% of queries should use indexes
        }
        
        # Common query patterns to test
        self.test_queries = [
            {
                'name': 'latest_patterns_by_symbol',
                'description': 'Get latest patterns for a specific symbol',
                'query': """
                    SELECT * FROM candlestick_patterns 
                    WHERE symbol = :symbol 
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """,
                'params': {'symbol': 'BTCUSDT', 'limit': 100}
            },
            {
                'name': 'patterns_by_name_and_date_range',
                'description': 'Get patterns by name within date range',
                'query': """
                    SELECT * FROM candlestick_patterns 
                    WHERE pattern_name = :pattern_name 
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp DESC
                """,
                'params': {
                    'pattern_name': 'hammer',
                    'start_time': datetime.now(timezone.utc) - timedelta(days=30),
                    'end_time': datetime.now(timezone.utc)
                }
            },
            {
                'name': 'high_confidence_volume_confirmed',
                'description': 'Get high confidence patterns with volume confirmation',
                'query': """
                    SELECT * FROM candlestick_patterns 
                    WHERE confidence >= :min_confidence 
                    AND volume_confirmation = true
                    AND timestamp >= :start_time
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT :limit
                """,
                'params': {
                    'min_confidence': 0.8,
                    'start_time': datetime.now(timezone.utc) - timedelta(days=7),
                    'limit': 50
                }
            },
            {
                'name': 'multi_timeframe_patterns',
                'description': 'Get patterns across multiple timeframes for a symbol',
                'query': """
                    SELECT * FROM candlestick_patterns 
                    WHERE symbol = :symbol 
                    AND timeframe IN (:timeframe1, :timeframe2, :timeframe3)
                    AND timestamp >= :start_time
                    ORDER BY timeframe, timestamp DESC
                """,
                'params': {
                    'symbol': 'ETHUSDT',
                    'timeframe1': '1h',
                    'timeframe2': '4h', 
                    'timeframe3': '1d',
                    'start_time': datetime.now(timezone.utc) - timedelta(days=14)
                }
            },
            {
                'name': 'trend_alignment_analysis',
                'description': 'Analyze patterns by trend alignment and strength',
                'query': """
                    SELECT 
                        trend_alignment,
                        strength,
                        COUNT(*) as pattern_count,
                        AVG(confidence) as avg_confidence
                    FROM candlestick_patterns 
                    WHERE timestamp >= :start_time
                    GROUP BY trend_alignment, strength
                    ORDER BY pattern_count DESC
                """,
                'params': {
                    'start_time': datetime.now(timezone.utc) - timedelta(days=30)
                }
            }
        ]
    
    async def initialize(self):
        """Initialize the retrieval performance service"""
        if self._initialized:
            return
            
        try:
            self.db_connection.initialize()
            await self.pattern_storage.initialize()
            
            # Apply advanced optimizations
            await self._create_indexes()
            await self._setup_continuous_aggregates()
            await self._optimize_chunk_settings()
            
            self._initialized = True
            self.logger.info("✅ Retrieval Performance Service initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Retrieval Performance Service: {e}")
            raise
    
    async def generate_synthetic_data(self, target_rows: int = 10_000_000) -> int:
        """Generate synthetic pattern data for performance testing"""
        try:
            if not self._initialized:
                await self.initialize()
            
            self.logger.info(f"🚀 Starting synthetic data generation for {target_rows:,} rows...")
            
            # Check current row count
            current_count = await self._get_current_row_count()
            if current_count >= target_rows:
                self.logger.info(f"✅ Database already has {current_count:,} rows (target: {target_rows:,})")
                return current_count
            
            rows_needed = target_rows - current_count
            batch_size = 100_000  # Process in 100K batches
            total_batches = (rows_needed + batch_size - 1) // batch_size
            
            self.logger.info(f"📊 Need to generate {rows_needed:,} rows in {total_batches} batches")
            
            # Generate data using PostgreSQL generate_series for maximum performance
            await self._generate_data_with_generate_series(rows_needed, batch_size)
            
            # Verify final count
            final_count = await self._get_current_row_count()
            self.logger.info(f"✅ Synthetic data generation completed: {final_count:,} total rows")
            
            return final_count
            
        except Exception as e:
            self.logger.error(f"❌ Error generating synthetic data: {e}")
            raise
    
    async def _generate_data_with_generate_series(self, rows_needed: int, batch_size: int):
        """Use PostgreSQL generate_series for ultra-fast data generation with memory-efficient batching"""
        try:
            # Use smaller batches to avoid memory issues
            actual_batch_size = min(batch_size, 100_000)  # Max 100K per batch
            total_batches = (rows_needed + actual_batch_size - 1) // actual_batch_size
            
            self.logger.info(f"📊 Generating {rows_needed:,} rows in {total_batches} batches of {actual_batch_size:,}")
            
            for batch_num in range(total_batches):
                start_row = batch_num * actual_batch_size + 1
                end_row = min((batch_num + 1) * actual_batch_size, rows_needed)
                current_batch_size = end_row - start_row + 1
                
                self.logger.info(f"🚀 Batch {batch_num + 1}/{total_batches}: rows {start_row:,} to {end_row:,}")
                
                async with self.db_connection.get_async_session() as session:
                    # Use generate_series with range to create synthetic data efficiently
                    insert_query = text(f"""
                        INSERT INTO candlestick_patterns (
                            symbol, timeframe, pattern_name, timestamp, confidence,
                            strength, price_level, volume_confirmation, volume_confidence,
                            volume_pattern_type, volume_strength, volume_context, trend_alignment,
                            metadata, created_at, updated_at
                        )
                        SELECT 
                            (ARRAY['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 
                                    'LTCUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'TRXUSDT'])[(generate_series % 10) + 1] as symbol,
                            (ARRAY['1m', '5m', '15m', '1h', '4h', '1d'])[(generate_series % 6) + 1] as timeframe,
                            (ARRAY['hammer', 'shooting_star', 'doji', 'engulfing', 'morning_star',
                                    'evening_star', 'three_white_soldiers', 'three_black_crows',
                                    'tweezer_top', 'tweezer_bottom'])[(generate_series % 10) + 1] as pattern_name,
                            NOW() - (generate_series * INTERVAL '1 minute') as timestamp,
                            0.5 + (random() * 0.5) as confidence,
                            (ARRAY['weak', 'moderate', 'strong'])[(generate_series % 3) + 1] as strength,
                            100 + (random() * 900) as price_level,
                            (generate_series % 2)::boolean as volume_confirmation,
                            0.0 + (random() * 1.0) as volume_confidence,
                            (ARRAY['spike', 'divergence', 'climax', 'dry-up'])[(generate_series % 4) + 1] as volume_pattern_type,
                            (ARRAY['weak', 'moderate', 'strong'])[(generate_series % 3) + 1] as volume_strength,
                            '{{"context": "synthetic"}}'::jsonb as volume_context,
                            (ARRAY['bullish', 'bearish', 'neutral'])[(generate_series % 3) + 1] as trend_alignment,
                            '{{"source": "synthetic", "batch": "generate_series"}}'::jsonb as metadata,
                            NOW() - (generate_series * INTERVAL '1 minute') as created_at,
                            NOW() - (generate_series * INTERVAL '1 minute') as updated_at
                        FROM generate_series({start_row}, {end_row});
                    """)
                    
                    await session.execute(insert_query)
                    await session.commit()
                    
                    self.logger.info(f"✅ Batch {batch_num + 1} completed: {current_batch_size:,} rows")
                
            self.logger.info(f"✅ Generated {rows_needed:,} synthetic rows using generate_series with batching")
                
        except Exception as e:
            self.logger.error(f"❌ Error in generate_series data generation: {e}")
            raise
    
    async def _get_current_row_count(self) -> int:
        """Get current row count in candlestick_patterns table"""
        try:
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM candlestick_patterns"))
                return result.scalar()
        except Exception as e:
            self.logger.error(f"❌ Error getting row count: {e}")
            return 0
    
    async def run_performance_benchmark(self) -> PerformanceBenchmark:
        """Run comprehensive performance benchmark on all test queries"""
        try:
            if not self._initialized:
                await self.initialize()
            
            self.logger.info("🚀 Starting comprehensive performance benchmark...")
            
            # Get current row count
            total_patterns = await self._get_current_row_count()
            self.logger.info(f"📊 Benchmarking against {total_patterns:,} patterns")
            
            # Test each query with EXPLAIN ANALYZE
            test_results = []
            for query_info in self.test_queries:
                try:
                    result = await self._test_query_performance(query_info)
                    test_results.append(result)
                    self.logger.info(f"✅ {query_info['name']}: {result.execution_time_ms:.2f}ms")
                except Exception as e:
                    self.logger.error(f"❌ Error testing {query_info['name']}: {e}")
            
            # Analyze results and generate recommendations
            index_recommendations = await self._analyze_index_recommendations(test_results)
            optimization_actions = await self._generate_optimization_actions(test_results)
            overall_score = self._calculate_overall_performance_score(test_results)
            
            benchmark = PerformanceBenchmark(
                total_patterns=total_patterns,
                test_queries=test_results,
                index_recommendations=index_recommendations,
                optimization_actions=optimization_actions,
                overall_score=overall_score,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.logger.info(f"🏆 Benchmark completed. Overall score: {overall_score:.2f}/10")
            return benchmark
            
        except Exception as e:
            self.logger.error(f"❌ Error running performance benchmark: {e}")
            raise
    
    async def _test_query_performance(self, query_info: Dict) -> QueryPerformanceResult:
        """Test a single query with EXPLAIN ANALYZE"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Run EXPLAIN ANALYZE
                explain_query = text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_info['query']}")
                result = await session.execute(explain_query, query_info['params'])
                explain_result = result.fetchone()
                
                if not explain_result or not explain_result[0]:
                    raise Exception("No EXPLAIN ANALYZE result")
                
                # Parse the JSON result
                plan_data = explain_result[0][0]  # First plan in the array
                
                # Extract performance metrics
                execution_time_ms = plan_data.get('Execution Time', 0)
                planning_time_ms = plan_data.get('Planning Time', 0)
                total_cost = plan_data.get('Total Cost', 0)
                
                # Analyze the plan structure
                plan_type, index_used, rows_scanned = self._analyze_query_plan(plan_data)
                
                # Get actual query results for row count
                actual_result = await session.execute(text(query_info['query']), query_info['params'])
                rows_returned = len(actual_result.fetchall())
                
                # Get buffer statistics
                buffer_hits = plan_data.get('Shared Hit Blocks', 0)
                buffer_reads = plan_data.get('Shared Read Blocks', 0)
                shared_hits = plan_data.get('Shared Hit Blocks', 0)
                shared_reads = plan_data.get('Shared Read Blocks', 0)
                
                return QueryPerformanceResult(
                    query_name=query_info['name'],
                    execution_time_ms=execution_time_ms,
                    rows_scanned=rows_scanned,
                    rows_returned=rows_returned,
                    plan_type=plan_type,
                    index_used=index_used,
                    total_cost=total_cost,
                    actual_time_ms=execution_time_ms,
                    planning_time_ms=planning_time_ms,
                    buffer_hits=buffer_hits,
                    buffer_reads=buffer_reads,
                    shared_hits=shared_hits,
                    shared_reads=shared_reads
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error testing query {query_info['name']}: {e}")
            raise
    
    def _analyze_query_plan(self, plan_data: Dict) -> Tuple[str, Optional[str], int]:
        """Analyze query plan to determine scan type and index usage"""
        try:
            # Recursively analyze the plan tree
            def analyze_node(node):
                node_type = node.get('Node Type', 'Unknown')
                index_name = node.get('Index Name')
                rows_scanned = node.get('Actual Rows', 0)
                
                # Check for index usage
                if 'Index' in node_type:
                    return node_type, index_name, rows_scanned
                elif 'Seq Scan' in node_type:
                    return 'Seq Scan', None, rows_scanned
                elif 'Bitmap' in node_type:
                    return 'Bitmap Scan', index_name, rows_scanned
                
                # Recursively check children
                children = node.get('Plans', [])
                for child in children:
                    child_type, child_index, child_rows = analyze_node(child)
                    if child_type != 'Unknown':
                        return child_type, child_index, child_rows
                
                return node_type, None, rows_scanned
            
            return analyze_node(plan_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing query plan: {e}")
            return 'Unknown', None, 0
    
    async def _analyze_index_recommendations(self, test_results: List[QueryPerformanceResult]) -> List[str]:
        """Analyze test results and recommend index improvements"""
        recommendations = []
        
        try:
            # Check for Seq Scans that could benefit from indexes
            seq_scans = [r for r in test_results if 'Seq Scan' in r.plan_type]
            if seq_scans:
                recommendations.append(f"⚠️ Found {len(seq_scans)} queries using Seq Scan - consider adding indexes")
                
                for result in seq_scans:
                    recommendations.append(f"  - {result.query_name}: {result.execution_time_ms:.2f}ms, {result.rows_scanned:,} rows scanned")
            
            # Check for slow queries
            slow_queries = [r for r in test_results if r.execution_time_ms > self.performance_thresholds['max_execution_time_ms']]
            if slow_queries:
                recommendations.append(f"⚠️ Found {len(slow_queries)} slow queries (>100ms)")
                
                for result in slow_queries:
                    recommendations.append(f"  - {result.query_name}: {result.execution_time_ms:.2f}ms")
            
            # Check row scan efficiency
            inefficient_queries = []
            for result in test_results:
                if result.rows_scanned > 0:
                    ratio = result.rows_scanned / max(result.rows_returned, 1)
                    if ratio > self.performance_thresholds['max_rows_scanned_ratio']:
                        inefficient_queries.append((result, ratio))
            
            if inefficient_queries:
                recommendations.append(f"⚠️ Found {len(inefficient_queries)} queries scanning too many rows")
                
                for result, ratio in inefficient_queries:
                    recommendations.append(f"  - {result.query_name}: scans {ratio:.1f}x more rows than returned")
            
            # Check index usage ratio
            index_scans = len([r for r in test_results if 'Index' in r.plan_type])
            total_queries = len(test_results)
            index_ratio = index_scans / total_queries if total_queries > 0 else 0
            
            if index_ratio < self.performance_thresholds['min_index_scan_ratio']:
                recommendations.append(f"⚠️ Only {index_ratio:.1%} of queries use indexes (target: {self.performance_thresholds['min_index_scan_ratio']:.1%})")
            
            if not recommendations:
                recommendations.append("✅ All queries are performing well within thresholds")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing index recommendations: {e}")
            return ["❌ Error analyzing recommendations"]
    
    async def _generate_optimization_actions(self, test_results: List[QueryPerformanceResult]) -> List[str]:
        """Generate specific optimization actions based on test results"""
        actions = []
        
        try:
            # Check for specific optimization opportunities
            for result in test_results:
                if result.execution_time_ms > 100:
                    actions.append(f"🔧 Optimize {result.query_name}: currently {result.execution_time_ms:.2f}ms")
                
                if result.rows_scanned > result.rows_returned * 10:
                    actions.append(f"🔧 Add index for {result.query_name}: scans {result.rows_scanned:,} vs returns {result.rows_returned:,}")
            
            # General optimization suggestions
            actions.append("🔧 Consider adding composite indexes for common query patterns")
            actions.append("🔧 Review chunk_time_interval for optimal TimescaleDB performance")
            actions.append("🔧 Consider continuous aggregates for frequently accessed aggregations")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"❌ Error generating optimization actions: {e}")
            return ["❌ Error generating actions"]
    
    def _calculate_overall_performance_score(self, test_results: List[QueryPerformanceResult]) -> float:
        """Calculate overall performance score (0-10)"""
        try:
            if not test_results:
                return 0.0
            
            total_score = 0.0
            max_score = len(test_results) * 10.0
            
            for result in test_results:
                score = 10.0
                
                # Deduct points for slow execution
                if result.execution_time_ms > 100:
                    score -= min(5.0, (result.execution_time_ms - 100) / 20)
                
                # Deduct points for inefficient scans
                if result.rows_scanned > 0:
                    ratio = result.rows_scanned / max(result.rows_returned, 1)
                    if ratio > 10:
                        score -= min(3.0, (ratio - 10) / 5)
                
                # Deduct points for Seq Scans
                if 'Seq Scan' in result.plan_type:
                    score -= 2.0
                
                # Ensure score doesn't go below 0
                score = max(0.0, score)
                total_score += score
            
            return (total_score / max_score) * 10.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating performance score: {e}")
            return 0.0
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Run benchmark
            benchmark = await self.run_performance_benchmark()
            
            # Format results for reporting
            report = {
                'timestamp': benchmark.timestamp.isoformat(),
                'total_patterns': benchmark.total_patterns,
                'overall_score': benchmark.overall_score,
                'query_performance': [],
                'index_recommendations': benchmark.index_recommendations,
                'optimization_actions': benchmark.optimization_actions,
                'summary': {
                    'total_queries': len(benchmark.test_queries),
                    'fast_queries': len([q for q in benchmark.test_queries if q.execution_time_ms <= 50]),
                    'medium_queries': len([q for q in benchmark.test_queries if 50 < q.execution_time_ms <= 100]),
                    'slow_queries': len([q for q in benchmark.test_queries if q.execution_time_ms > 100]),
                    'index_scans': len([q for q in benchmark.test_queries if 'Index' in q.plan_type]),
                    'seq_scans': len([q for q in benchmark.test_queries if 'Seq Scan' in q.plan_type])
                }
            }
            
            # Add detailed query performance
            for result in benchmark.test_queries:
                report['query_performance'].append({
                    'name': result.query_name,
                    'execution_time_ms': result.execution_time_ms,
                    'rows_scanned': result.rows_scanned,
                    'rows_returned': result.rows_returned,
                    'plan_type': result.plan_type,
                    'index_used': result.index_used,
                    'efficiency_ratio': result.rows_scanned / max(result.rows_returned, 1) if result.rows_returned > 0 else 0
                })
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance report: {e}")
            return {'error': str(e)}

    async def _create_indexes(self):
        """Create optimized indexes for pattern queries"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Composite index for symbol + timeframe + timestamp (most common query)
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timeframe_timestamp 
                    ON candlestick_patterns (symbol, timeframe, timestamp DESC);
                """))
                
                # Index for pattern name queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_name_timestamp 
                    ON candlestick_patterns (pattern_name, timestamp DESC);
                """))
                
                # Index for confidence-based queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_confidence_timestamp 
                    ON candlestick_patterns (confidence DESC, timestamp DESC);
                """))
                
                # Index for trend alignment queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_trend_timestamp 
                    ON candlestick_patterns (trend_alignment, timestamp DESC);
                """))
                
                # Index for volume pattern type queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_volume_type_timestamp 
                    ON candlestick_patterns (volume_pattern_type, timestamp DESC);
                """))
                
                # Index for volume strength queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_volume_strength_timestamp 
                    ON candlestick_patterns (volume_strength, timestamp DESC);
                """))
                
                # GIN index for JSON metadata queries
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_metadata_gin 
                    ON candlestick_patterns USING GIN (metadata);
                """))
                
                # Advanced composite indexes for common query patterns
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_symbol_confidence_timestamp 
                    ON candlestick_patterns (symbol, confidence DESC, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_symbol_volume_conf_timestamp 
                    ON candlestick_patterns (symbol, volume_confirmation, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_timeframe_confidence_timestamp 
                    ON candlestick_patterns (timeframe, confidence DESC, timestamp DESC);
                """))
                
                # Partial indexes for high-confidence patterns (most valuable)
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_high_confidence 
                    ON candlestick_patterns (symbol, timeframe, timestamp DESC) 
                    WHERE confidence >= 0.8;
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_volume_confirmed 
                    ON candlestick_patterns (symbol, timeframe, timestamp DESC) 
                    WHERE volume_confirmation = true;
                """))
                
                await session.commit()
                self.logger.info("✅ Created advanced optimized indexes for pattern queries")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating indexes: {e}")
            raise
    
    async def _setup_continuous_aggregates(self):
        """Setup TimescaleDB continuous aggregates for common aggregations"""
        try:
            # Continuous aggregates must be created outside of transactions
            async with self.db_connection.get_async_session() as session:
                # Use raw connection for DDL operations
                conn = await session.connection()
                # Daily pattern counts by symbol and pattern type
                await conn.execute(text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS patterns_daily_summary
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 day', timestamp) AS day,
                        symbol,
                        pattern_name,
                        COUNT(*) as pattern_count,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN volume_confirmation = true THEN 1 END) as volume_confirmed_count
                    FROM candlestick_patterns
                    GROUP BY day, symbol, pattern_name;
                """))
                
                # Hourly pattern counts by symbol
                await conn.execute(text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS patterns_hourly_summary
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 hour', timestamp) AS hour,
                        symbol,
                        COUNT(*) as pattern_count,
                        AVG(confidence) as avg_confidence,
                        COUNT(CASE WHEN volume_confirmation = true THEN 1 END) as volume_confirmed_count
                    FROM candlestick_patterns
                    GROUP BY hour, symbol;
                """))
                
                # Trend alignment summary
                await conn.execute(text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS patterns_trend_summary
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 day', timestamp) AS day,
                        trend_alignment,
                        strength,
                        COUNT(*) as pattern_count,
                        AVG(confidence) as avg_confidence
                    FROM candlestick_patterns
                    GROUP BY day, trend_alignment, strength;
                """))
                
                # Add refresh policies
                await conn.execute(text("""
                    SELECT add_continuous_aggregate_policy('patterns_daily_summary',
                        start_offset => INTERVAL '3 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour');
                """))
                
                await conn.execute(text("""
                    SELECT add_continuous_aggregate_policy('patterns_hourly_summary',
                        start_offset => INTERVAL '1 day',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour');
                """))
                
                await conn.execute(text("""
                    SELECT add_continuous_aggregate_policy('patterns_trend_summary',
                        start_offset => INTERVAL '3 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour');
                """))
                
                self.logger.info("✅ Setup continuous aggregates for pattern analysis")
                
        except Exception as e:
            self.logger.error(f"⚠️ Could not setup continuous aggregates: {e}")
            # Non-critical, continue without continuous aggregates
    
    async def _optimize_chunk_settings(self):
        """Optimize TimescaleDB chunk settings for better performance"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Use raw connection for DDL operations
                conn = await session.connection()
                
                # Check if TimescaleDB is available and get current chunk settings
                try:
                    result = await conn.execute(text("""
                        SELECT chunk_time_interval 
                        FROM timescaledb_information.hypertables 
                        WHERE hypertable_name = 'candlestick_patterns';
                    """))
                    
                    current_interval = result.scalar()
                    self.logger.info(f"📊 Current chunk interval: {current_interval}")
                    
                    # Optimize chunk size based on data volume
                    # For 1M+ rows, smaller chunks (1 hour) provide better query performance
                    await conn.execute(text("""
                        SELECT set_chunk_time_interval('candlestick_patterns', INTERVAL '1 hour');
                    """))
                    
                    # Enable parallel workers for better performance
                    await conn.execute(text("""
                        ALTER TABLE candlestick_patterns SET (
                            parallel_workers = 4
                        );
                    """))
                    
                    self.logger.info("✅ Optimized chunk settings for better performance")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ TimescaleDB chunk optimization not available: {e}")
                    # Continue without chunk optimization
                
        except Exception as e:
            self.logger.error(f"⚠️ Could not optimize chunk settings: {e}")
            # Non-critical, continue with default settings
