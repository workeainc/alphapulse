#!/usr/bin/env python3
"""
Query Optimization Script for AlphaPlus Database
Analyzes query performance and provides optimization recommendations
"""

import asyncio
import logging
import time
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import sys
import os

# Add backend to path
sys.path.append('backend')

from src.data.mock_database import MockDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Query analysis result"""
    query_name: str
    execution_time_ms: float
    rows_returned: int
    query_plan: str
    optimization_score: float
    recommendations: List[str]
    is_optimized: bool

class QueryOptimizer:
    """Analyze and optimize database queries"""
    
    def __init__(self, db_path: str = "test_database.db"):
        self.db_path = db_path
        self.db = MockDatabase(db_path)
        
        # Define test queries
        self.test_queries = {
            'ohlcv_simple_select': """
                SELECT * FROM ohlcv_data 
                WHERE symbol = 'BTCUSDT' 
                ORDER BY timestamp DESC 
                LIMIT 100
            """,
            'ohlcv_time_range': """
                SELECT symbol, AVG(close) as avg_price, MAX(volume) as max_volume
                FROM ohlcv_data
                WHERE timestamp >= datetime('now', '-1 hour')
                GROUP BY symbol
            """,
            'ohlcv_volume_analysis': """
                SELECT 
                    symbol,
                    timeframe,
                    AVG(volume) as avg_volume,
                    MAX(volume) as max_volume,
                    MIN(volume) as min_volume,
                    COUNT(*) as candle_count
                FROM ohlcv_data
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY symbol, timeframe
                HAVING COUNT(*) > 10
                ORDER BY avg_volume DESC
            """,
            'signals_performance': """
                SELECT 
                    s.symbol,
                    s.direction,
                    AVG(s.confidence) as avg_confidence,
                    COUNT(*) as signal_count,
                    COUNT(CASE WHEN s.outcome = 'success' THEN 1 END) as success_count
                FROM signals s
                WHERE s.timestamp >= datetime('now', '-7 days')
                GROUP BY s.symbol, s.direction
                HAVING COUNT(*) > 5
                ORDER BY avg_confidence DESC
            """,
            'indicators_calculation': """
                SELECT 
                    ti.symbol,
                    ti.timeframe,
                    ti.indicator_name,
                    AVG(ti.indicator_value) as avg_value,
                    MAX(ti.indicator_value) as max_value,
                    MIN(ti.indicator_value) as min_value
                FROM technical_indicators ti
                WHERE ti.timestamp >= datetime('now', '-1 hour')
                GROUP BY ti.symbol, ti.timeframe, ti.indicator_name
                ORDER BY ti.symbol, ti.timeframe
            """,
            'complex_join_analysis': """
                SELECT 
                    o.symbol,
                    o.timeframe,
                    AVG(o.close) as avg_price,
                    AVG(o.volume) as avg_volume,
                    COUNT(s.signal_id) as signal_count,
                    AVG(s.confidence) as avg_signal_confidence
                FROM ohlcv_data o
                LEFT JOIN signals s ON o.symbol = s.symbol AND o.timeframe = s.timeframe
                WHERE o.timestamp >= datetime('now', '-1 hour')
                GROUP BY o.symbol, o.timeframe
                HAVING COUNT(s.signal_id) > 0
                ORDER BY avg_signal_confidence DESC
            """
        }
    
    async def initialize(self):
        """Initialize the optimizer"""
        await self.db.initialize()
        logger.info("✅ Query optimizer initialized")
    
    async def close(self):
        """Close the optimizer"""
        await self.db.close()
    
    async def analyze_query(self, query_name: str, query_sql: str) -> QueryAnalysis:
        """Analyze a single query"""
        try:
            logger.info(f"🔍 Analyzing query: {query_name}")
            
            # Execute query and measure time
            start_time = time.time()
            result = await self.db.fetch(query_sql)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Get query plan (SQLite equivalent of EXPLAIN ANALYZE)
            plan_result = await self.db.fetch(f"EXPLAIN QUERY PLAN {query_sql}")
            query_plan = self._format_query_plan(plan_result)
            
            # Analyze query performance
            optimization_score = self._calculate_optimization_score(execution_time_ms, len(result), query_plan)
            recommendations = self._generate_recommendations(query_name, query_sql, execution_time_ms, query_plan)
            is_optimized = optimization_score >= 80
            
            return QueryAnalysis(
                query_name=query_name,
                execution_time_ms=execution_time_ms,
                rows_returned=len(result),
                query_plan=query_plan,
                optimization_score=optimization_score,
                recommendations=recommendations,
                is_optimized=is_optimized
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing query {query_name}: {e}")
            return QueryAnalysis(
                query_name=query_name,
                execution_time_ms=0,
                rows_returned=0,
                query_plan="Error in query execution",
                optimization_score=0,
                recommendations=[f"Fix query error: {str(e)}"],
                is_optimized=False
            )
    
    def _format_query_plan(self, plan_result: List[Dict]) -> str:
        """Format query plan for readability"""
        try:
            if not plan_result:
                return "No query plan available"
            
            plan_lines = []
            for row in plan_result:
                plan_lines.append(f"  {row.get('detail', 'Unknown step')}")
            
            return "\n".join(plan_lines)
            
        except Exception as e:
            return f"Error formatting query plan: {e}"
    
    def _calculate_optimization_score(self, execution_time_ms: float, rows_returned: int, query_plan: str) -> float:
        """Calculate optimization score (0-100)"""
        try:
            score = 100.0
            
            # Deduct points for slow execution
            if execution_time_ms > 1000:  # > 1 second
                score -= 30
            elif execution_time_ms > 500:  # > 500ms
                score -= 20
            elif execution_time_ms > 100:  # > 100ms
                score -= 10
            
            # Deduct points for inefficient operations
            if "SCAN" in query_plan.upper():
                score -= 15
            if "TEMPORARY" in query_plan.upper():
                score -= 10
            if "SORT" in query_plan.upper() and rows_returned > 1000:
                score -= 10
            
            # Bonus points for efficient operations
            if "INDEX" in query_plan.upper():
                score += 5
            if execution_time_ms < 10:  # Very fast
                score += 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimization score: {e}")
            return 0
    
    def _generate_recommendations(self, query_name: str, query_sql: str, execution_time_ms: float, query_plan: str) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Time-based recommendations
        if execution_time_ms > 1000:
            recommendations.append("Query is very slow (>1s). Consider adding indexes or rewriting the query.")
        elif execution_time_ms > 500:
            recommendations.append("Query is slow (>500ms). Review for optimization opportunities.")
        
        # Plan-based recommendations
        if "SCAN" in query_plan.upper():
            recommendations.append("Full table scan detected. Add appropriate indexes.")
        
        if "TEMPORARY" in query_plan.upper():
            recommendations.append("Temporary table created. Consider optimizing GROUP BY or ORDER BY clauses.")
        
        if "SORT" in query_plan.upper():
            recommendations.append("Sorting operation detected. Consider adding indexes on ORDER BY columns.")
        
        # Query-specific recommendations
        if "GROUP BY" in query_sql.upper() and "HAVING" in query_sql.upper():
            recommendations.append("Complex GROUP BY with HAVING. Consider filtering in WHERE clause first.")
        
        if "LEFT JOIN" in query_sql.upper() or "RIGHT JOIN" in query_sql.upper():
            recommendations.append("Outer join detected. Ensure join columns are indexed.")
        
        if "COUNT(*)" in query_sql.upper():
            recommendations.append("COUNT(*) operation. Consider using COUNT(column) for better performance.")
        
        # Index recommendations
        if "WHERE" in query_sql.upper():
            recommendations.append("Ensure WHERE clause columns are indexed.")
        
        if "ORDER BY" in query_sql.upper():
            recommendations.append("Consider adding composite indexes for ORDER BY columns.")
        
        if not recommendations:
            recommendations.append("Query appears to be well-optimized.")
        
        return recommendations
    
    async def analyze_all_queries(self) -> List[QueryAnalysis]:
        """Analyze all test queries"""
        logger.info("🔍 Analyzing all test queries...")
        
        analyses = []
        for query_name, query_sql in self.test_queries.items():
            analysis = await self.analyze_query(query_name, query_sql)
            analyses.append(analysis)
        
        return analyses
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            logger.info("📊 Generating optimization report...")
            
            # Analyze all queries
            analyses = await self.analyze_all_queries()
            
            # Calculate overall statistics
            total_queries = len(analyses)
            optimized_queries = sum(1 for a in analyses if a.is_optimized)
            avg_execution_time = sum(a.execution_time_ms for a in analyses) / total_queries
            avg_optimization_score = sum(a.optimization_score for a in analyses) / total_queries
            
            # Find slowest queries
            slowest_queries = sorted(analyses, key=lambda x: x.execution_time_ms, reverse=True)[:3]
            
            # Find queries needing optimization
            needs_optimization = [a for a in analyses if not a.is_optimized]
            
            # Generate index recommendations
            index_recommendations = self._generate_index_recommendations(analyses)
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'total_queries': total_queries,
                    'optimized_queries': optimized_queries,
                    'optimization_rate': (optimized_queries / total_queries) * 100,
                    'avg_execution_time_ms': avg_execution_time,
                    'avg_optimization_score': avg_optimization_score
                },
                'query_analyses': [
                    {
                        'name': a.query_name,
                        'execution_time_ms': a.execution_time_ms,
                        'rows_returned': a.rows_returned,
                        'optimization_score': a.optimization_score,
                        'is_optimized': a.is_optimized,
                        'recommendations': a.recommendations,
                        'query_plan': a.query_plan
                    }
                    for a in analyses
                ],
                'slowest_queries': [
                    {
                        'name': q.query_name,
                        'execution_time_ms': q.execution_time_ms,
                        'optimization_score': q.optimization_score
                    }
                    for q in slowest_queries
                ],
                'needs_optimization': [
                    {
                        'name': q.query_name,
                        'execution_time_ms': q.execution_time_ms,
                        'optimization_score': q.optimization_score,
                        'recommendations': q.recommendations
                    }
                    for q in needs_optimization
                ],
                'index_recommendations': index_recommendations
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating optimization report: {e}")
            return {'error': str(e)}
    
    def _generate_index_recommendations(self, analyses: List[QueryAnalysis]) -> List[Dict[str, str]]:
        """Generate index recommendations based on query analysis"""
        recommendations = []
        
        # Common index patterns
        index_patterns = {
            'ohlcv_symbol_timeframe': "CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe)",
            'ohlcv_timestamp': "CREATE INDEX idx_ohlcv_timestamp ON ohlcv_data (timestamp)",
            'ohlcv_symbol_timestamp': "CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv_data (symbol, timestamp)",
            'signals_symbol_direction': "CREATE INDEX idx_signals_symbol_direction ON signals (symbol, direction)",
            'signals_timestamp': "CREATE INDEX idx_signals_timestamp ON signals (timestamp)",
            'indicators_symbol_name': "CREATE INDEX idx_indicators_symbol_name ON technical_indicators (symbol, indicator_name)",
            'indicators_timestamp': "CREATE INDEX idx_indicators_timestamp ON technical_indicators (timestamp)"
        }
        
        for pattern_name, index_sql in index_patterns.items():
            recommendations.append({
                'index_name': pattern_name,
                'sql': index_sql,
                'reason': f"Recommended for {pattern_name} query patterns"
            })
        
        return recommendations
    
    def save_optimization_report(self, report: Dict[str, Any], filename: str = None):
        """Save optimization report to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Optimization report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving optimization report: {e}")
    
    async def create_recommended_indexes(self):
        """Create recommended indexes"""
        try:
            logger.info("🔧 Creating recommended indexes...")
            
            index_sqls = [
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data (symbol, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_direction ON signals (symbol, direction)",
                "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_name ON technical_indicators (symbol, indicator_name)",
                "CREATE INDEX IF NOT EXISTS idx_indicators_timestamp ON technical_indicators (timestamp)"
            ]
            
            for index_sql in index_sqls:
                try:
                    await self.db.execute(index_sql)
                    logger.info(f"✅ Created index: {index_sql.split()[-1]}")
                except Exception as e:
                    logger.warning(f"⚠️ Index creation failed: {e}")
            
            logger.info("✅ Index creation completed")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")

async def main():
    """Main function for query optimization"""
    optimizer = QueryOptimizer()
    
    try:
        await optimizer.initialize()
        
        # Create recommended indexes first
        await optimizer.create_recommended_indexes()
        
        # Generate optimization report
        logger.info("📊 Generating optimization report...")
        report = await optimizer.generate_optimization_report()
        
        # Display report
        logger.info("=" * 60)
        logger.info("📊 QUERY OPTIMIZATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Queries Analyzed: {report['summary']['total_queries']}")
        logger.info(f"Optimized Queries: {report['summary']['optimized_queries']}")
        logger.info(f"Optimization Rate: {report['summary']['optimization_rate']:.1f}%")
        logger.info(f"Average Execution Time: {report['summary']['avg_execution_time_ms']:.2f} ms")
        logger.info(f"Average Optimization Score: {report['summary']['avg_optimization_score']:.1f}/100")
        
        logger.info("\nSlowest Queries:")
        for i, query in enumerate(report['slowest_queries'], 1):
            logger.info(f"  {i}. {query['name']}: {query['execution_time_ms']:.2f} ms (Score: {query['optimization_score']:.1f})")
        
        if report['needs_optimization']:
            logger.info("\nQueries Needing Optimization:")
            for i, query in enumerate(report['needs_optimization'], 1):
                logger.info(f"  {i}. {query['name']}: {query['execution_time_ms']:.2f} ms (Score: {query['optimization_score']:.1f})")
                for rec in query['recommendations'][:2]:  # Show first 2 recommendations
                    logger.info(f"     - {rec}")
        
        logger.info("\nIndex Recommendations:")
        for i, index in enumerate(report['index_recommendations'], 1):
            logger.info(f"  {i}. {index['index_name']}")
            logger.info(f"     SQL: {index['sql']}")
        
        # Save report
        optimizer.save_optimization_report(report)
        
    finally:
        await optimizer.close()

if __name__ == "__main__":
    asyncio.run(main())
