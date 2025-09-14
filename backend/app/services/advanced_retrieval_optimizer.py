#!/usr/bin/env python3
"""
Advanced Retrieval Optimizer for AlphaPulse Trading Bot
Production-ready TimescaleDB retrieval optimization with auto-tuning capabilities
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import json
import statistics
from collections import defaultdict, deque

from sqlalchemy import text, func, inspect
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import TimescaleDBConnection
from app.services.retrieval_performance_service import RetrievalPerformanceService

logger = logging.getLogger(__name__)

@dataclass
class QueryPlanAnalysis:
    """Detailed analysis of a query execution plan"""
    query_hash: str
    query_text: str
    execution_count: int
    total_execution_time: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_deviation: float
    plan_type: str
    index_usage: Dict[str, int]
    scan_efficiency: float
    last_executed: datetime
    optimization_score: float

@dataclass
class IndexRecommendation:
    """Index optimization recommendation"""
    table_name: str
    columns: List[str]
    index_type: str  # 'btree', 'gin', 'partial', 'composite'
    priority: int  # 1-10, higher = more important
    expected_improvement: float
    creation_cost: str
    reasoning: str

@dataclass
class PerformanceAlert:
    """Performance degradation alert"""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    query_name: str
    current_performance: float
    threshold: float
    degradation_percent: float
    recommendation: str
    timestamp: datetime

class AdvancedRetrievalOptimizer:
    """Advanced retrieval optimization service with auto-tuning capabilities"""
    
    def __init__(self):
        self.db_connection = TimescaleDBConnection()
        self.retrieval_service = RetrievalPerformanceService()
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.query_performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.performance_thresholds = {
            'execution_time_ms': 100.0,
            'scan_efficiency_ratio': 10.0,
            'index_usage_ratio': 0.8,
            'query_timeout_ms': 5000.0
        }
        
        # Auto-optimization settings
        self.auto_optimization_enabled = True
        self.optimization_interval_hours = 24
        self.last_optimization = None
        
        # Monitoring state
        self.monitoring_active = False
        self.performance_alerts = []
        self.optimization_history = []
        
    async def initialize(self):
        """Initialize the advanced retrieval optimizer"""
        if self._initialized:
            return
            
        try:
            await self.retrieval_service.initialize()
            self.db_connection.initialize()
            
            # Setup monitoring tables
            await self._setup_monitoring_tables()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            self._initialized = True
            self.logger.info("✅ Advanced Retrieval Optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Advanced Retrieval Optimizer: {e}")
            raise
    
    async def _setup_monitoring_tables(self):
        """Setup tables for performance monitoring and optimization tracking"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Query performance history table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS query_performance_history (
                        id SERIAL PRIMARY KEY,
                        query_hash TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        rows_scanned INTEGER NOT NULL,
                        rows_returned INTEGER NOT NULL,
                        plan_type TEXT NOT NULL,
                        index_used TEXT,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Index recommendations table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS index_recommendations (
                        id SERIAL PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        columns TEXT[] NOT NULL,
                        index_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        expected_improvement REAL NOT NULL,
                        creation_cost TEXT NOT NULL,
                        reasoning TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        applied_at TIMESTAMPTZ
                    );
                """))
                
                # Performance alerts table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id SERIAL PRIMARY KEY,
                        alert_id TEXT UNIQUE NOT NULL,
                        severity TEXT NOT NULL,
                        query_name TEXT NOT NULL,
                        current_performance REAL NOT NULL,
                        threshold REAL NOT NULL,
                        degradation_percent REAL NOT NULL,
                        recommendation TEXT NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create indexes on monitoring tables
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_query_perf_hash_timestamp 
                    ON query_performance_history (query_hash, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_index_rec_status_priority 
                    ON index_recommendations (status, priority DESC);
                """))
                
                await session.commit()
                self.logger.info("✅ Monitoring tables setup completed")
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up monitoring tables: {e}")
            raise
    
    async def _start_background_monitoring(self):
        """Start background performance monitoring"""
        if self.monitoring_active:
            return
            
        try:
            self.monitoring_active = True
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._auto_optimization_loop())
            self.logger.info("✅ Background monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting background monitoring: {e}")
            self.monitoring_active = False
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring"""
        while self.monitoring_active:
            try:
                await self._check_performance_degradation()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _auto_optimization_loop(self):
        """Background loop for automatic optimization"""
        while self.monitoring_active:
            try:
                if self._should_run_optimization():
                    await self._run_auto_optimization()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"❌ Error in auto-optimization loop: {e}")
                await asyncio.sleep(300)
    
    def _should_run_optimization(self) -> bool:
        """Check if optimization should run"""
        if not self.auto_optimization_enabled:
            return False
            
        if self.last_optimization is None:
            return True
            
        hours_since_last = (datetime.now(timezone.utc) - self.last_optimization).total_seconds() / 3600
        return hours_since_last >= self.optimization_interval_hours
    
    async def _run_auto_optimization(self):
        """Run automatic optimization based on performance analysis"""
        try:
            self.logger.info("🚀 Starting automatic optimization...")
            
            # Analyze current performance
            performance_report = await self.retrieval_service.get_performance_report()
            
            # Generate index recommendations
            recommendations = await self._generate_index_recommendations(performance_report)
            
            # Apply high-priority recommendations
            applied_count = 0
            for rec in recommendations:
                if rec.priority >= 8:  # High priority
                    try:
                        await self._apply_index_recommendation(rec)
                        applied_count += 1
                        self.logger.info(f"✅ Applied high-priority index: {rec.columns}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to apply index {rec.columns}: {e}")
            
            # Update optimization history
            self.optimization_history.append({
                'timestamp': datetime.now(timezone.utc),
                'recommendations_generated': len(recommendations),
                'recommendations_applied': applied_count,
                'performance_score': performance_report.get('overall_score', 0)
            })
            
            self.last_optimization = datetime.now(timezone.utc)
            self.logger.info(f"✅ Auto-optimization completed: {applied_count} indexes applied")
            
        except Exception as e:
            self.logger.error(f"❌ Error in auto-optimization: {e}")
    
    async def _generate_index_recommendations(self, performance_report: Dict) -> List[IndexRecommendation]:
        """Generate intelligent index recommendations based on performance data"""
        recommendations = []
        
        try:
            query_performance = performance_report.get('query_performance', [])
            
            for query in query_performance:
                if query['execution_time_ms'] > self.performance_thresholds['execution_time_ms']:
                    # Analyze slow queries for index opportunities
                    rec = await self._analyze_query_for_index_optimization(query)
                    if rec:
                        recommendations.append(rec)
                
                if query['efficiency_ratio'] > self.performance_thresholds['scan_efficiency_ratio']:
                    # Analyze inefficient scans
                    rec = await self._analyze_scan_efficiency(query)
                    if rec:
                        recommendations.append(rec)
            
            # Sort by priority (higher first)
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            # Store recommendations
            await self._store_index_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating index recommendations: {e}")
            return []
    
    async def _analyze_query_for_index_optimization(self, query: Dict) -> Optional[IndexRecommendation]:
        """Analyze a specific query for index optimization opportunities"""
        try:
            query_name = query['name']
            
            # Define optimization strategies based on query type
            if 'symbol' in query_name.lower():
                return IndexRecommendation(
                    table_name='candlestick_patterns',
                    columns=['symbol', 'timestamp'],
                    index_type='btree',
                    priority=9,
                    expected_improvement=0.7,
                    creation_cost='low',
                    reasoning=f"Query {query_name} frequently filters by symbol and orders by timestamp"
                )
            
            elif 'pattern_name' in query_name.lower():
                return IndexRecommendation(
                    table_name='candlestick_patterns',
                    columns=['pattern_name', 'timestamp'],
                    index_type='btree',
                    priority=8,
                    expected_improvement=0.6,
                    creation_cost='low',
                    reasoning=f"Query {query_name} filters by pattern_name and orders by timestamp"
                )
            
            elif 'confidence' in query_name.lower():
                return IndexRecommendation(
                    table_name='candlestick_patterns',
                    columns=['confidence', 'timestamp'],
                    index_type='btree',
                    priority=7,
                    expected_improvement=0.5,
                    creation_cost='low',
                    reasoning=f"Query {query_name} filters by confidence threshold"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing query {query.get('name', 'unknown')}: {e}")
            return None
    
    async def _analyze_scan_efficiency(self, query: Dict) -> Optional[IndexRecommendation]:
        """Analyze scan efficiency for index optimization"""
        try:
            efficiency_ratio = query['efficiency_ratio']
            
            if efficiency_ratio > 50:  # Very inefficient
                return IndexRecommendation(
                    table_name='candlestick_patterns',
                    columns=['timestamp', 'symbol', 'timeframe'],
                    index_type='btree',
                    priority=10,
                    expected_improvement=0.8,
                    creation_cost='medium',
                    reasoning=f"Query scans {efficiency_ratio:.1f}x more rows than returned - needs composite index"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing scan efficiency: {e}")
            return None
    
    async def _apply_index_recommendation(self, recommendation: IndexRecommendation):
        """Apply an index recommendation"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Create the index
                columns_str = ', '.join(recommendation.columns)
                index_name = f"idx_auto_{recommendation.table_name}_{'_'.join(recommendation.columns)}"
                
                if recommendation.index_type == 'gin':
                    create_sql = f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {recommendation.table_name} USING GIN ({columns_str});
                    """
                else:
                    create_sql = f"""
                        CREATE INDEX IF NOT EXISTS {index_name} 
                        ON {recommendation.table_name} ({columns_str});
                    """
                
                await session.execute(text(create_sql))
                await session.commit()
                
                # Update recommendation status
                await session.execute(text("""
                    UPDATE index_recommendations 
                    SET status = 'applied', applied_at = NOW() 
                    WHERE columns = :columns AND table_name = :table_name
                """), {
                    'columns': recommendation.columns,
                    'table_name': recommendation.table_name
                })
                await session.commit()
                
                self.logger.info(f"✅ Applied index recommendation: {index_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Error applying index recommendation: {e}")
            raise
    
    async def _store_index_recommendations(self, recommendations: List[IndexRecommendation]):
        """Store index recommendations in database"""
        try:
            async with self.db_connection.get_async_session() as session:
                for rec in recommendations:
                    await session.execute(text("""
                        INSERT INTO index_recommendations 
                        (table_name, columns, index_type, priority, expected_improvement, creation_cost, reasoning)
                        VALUES (:table_name, :columns, :index_type, :priority, :expected_improvement, :creation_cost, :reasoning)
                        ON CONFLICT DO NOTHING
                    """), {
                        'table_name': rec.table_name,
                        'columns': rec.columns,
                        'index_type': rec.index_type,
                        'priority': rec.priority,
                        'expected_improvement': rec.expected_improvement,
                        'creation_cost': rec.creation_cost,
                        'reasoning': rec.reasoning
                    })
                
                await session.commit()
                self.logger.info(f"✅ Stored {len(recommendations)} index recommendations")
                
        except Exception as e:
            self.logger.error(f"❌ Error storing index recommendations: {e}")
    
    async def _check_performance_degradation(self):
        """Check for performance degradation and create alerts"""
        try:
            # Get recent performance data
            performance_report = await self.retrieval_service.get_performance_report()
            
            for query in performance_report.get('query_performance', []):
                if query['execution_time_ms'] > self.performance_thresholds['execution_time_ms']:
                    # Calculate degradation
                    degradation = (query['execution_time_ms'] - self.performance_thresholds['execution_time_ms']) / self.performance_thresholds['execution_time_ms']
                    
                    if degradation > 0.5:  # 50% degradation
                        alert = PerformanceAlert(
                            alert_id=f"perf_{query['name']}_{int(time.time())}",
                            severity='high' if degradation > 1.0 else 'medium',
                            query_name=query['name'],
                            current_performance=query['execution_time_ms'],
                            threshold=self.performance_thresholds['execution_time_ms'],
                            degradation_percent=degradation * 100,
                            recommendation=f"Query {query['name']} is {degradation:.1%} slower than threshold. Consider adding indexes or optimizing query.",
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        await self._create_performance_alert(alert)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking performance degradation: {e}")
    
    async def _create_performance_alert(self, alert: PerformanceAlert):
        """Create and store a performance alert"""
        try:
            async with self.db_connection.get_async_session() as session:
                await session.execute(text("""
                    INSERT INTO performance_alerts 
                    (alert_id, severity, query_name, current_performance, threshold, degradation_percent, recommendation)
                    VALUES (:alert_id, :severity, :query_name, :current_performance, :threshold, :degradation_percent, :recommendation)
                    ON CONFLICT (alert_id) DO UPDATE SET
                        severity = EXCLUDED.severity,
                        current_performance = EXCLUDED.current_performance,
                        degradation_percent = EXCLUDED.degradation_percent,
                        recommendation = EXCLUDED.recommendation,
                        timestamp = NOW()
                """), {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'query_name': alert.query_name,
                    'current_performance': alert.current_performance,
                    'threshold': alert.threshold,
                    'degradation_percent': alert.degradation_percent,
                    'recommendation': alert.recommendation
                })
                
                await session.commit()
                
                # Add to local alerts list
                self.performance_alerts.append(alert)
                
                self.logger.warning(f"⚠️ Performance alert created: {alert.query_name} - {alert.degradation_percent:.1f}% degradation")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating performance alert: {e}")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and recommendations"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get pending index recommendations
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(text("""
                    SELECT * FROM index_recommendations 
                    WHERE status = 'pending' 
                    ORDER BY priority DESC, created_at ASC
                """))
                
                pending_recommendations = []
                for row in result.fetchall():
                    pending_recommendations.append({
                        'table_name': row[1],
                        'columns': row[2],
                        'index_type': row[3],
                        'priority': row[4],
                        'expected_improvement': row[5],
                        'creation_cost': row[6],
                        'reasoning': row[7]
                    })
                
                # Get recent performance alerts
                result = await session.execute(text("""
                    SELECT * FROM performance_alerts 
                    WHERE acknowledged = FALSE 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """))
                
                active_alerts = []
                for row in result.fetchall():
                    active_alerts.append({
                        'severity': row[2],
                        'query_name': row[3],
                        'current_performance': row[4],
                        'threshold': row[5],
                        'degradation_percent': row[6],
                        'recommendation': row[7],
                        'timestamp': row[9].isoformat() if row[9] else None
                    })
            
            return {
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'next_optimization': (self.last_optimization + timedelta(hours=self.optimization_interval_hours)).isoformat() if self.last_optimization else None,
                'pending_recommendations': pending_recommendations,
                'active_alerts': active_alerts,
                'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
                'monitoring_active': self.monitoring_active
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting optimization status: {e}")
            return {'error': str(e)}
    
    async def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive optimization analysis and recommendations"""
        try:
            if not self._initialized:
                await self.initialize()
            
            self.logger.info("🚀 Starting comprehensive optimization analysis...")
            
            # Run performance benchmark with error handling
            try:
                benchmark = await self.retrieval_service.run_performance_benchmark()
            except Exception as e:
                if "out of shared memory" in str(e).lower() or "max_locks_per_transaction" in str(e).lower():
                    self.logger.warning("⚠️ Database memory limit reached - using fallback analysis")
                    return {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'benchmark_results': {
                            'overall_score': 0.0,
                            'total_patterns': 'unknown (memory limit)',
                            'query_count': 0
                        },
                        'recommendations': [
                            {
                                'table_name': 'candlestick_patterns',
                                'columns': ['timestamp', 'symbol'],
                                'index_type': 'btree',
                                'priority': 10,
                                'expected_improvement': 0.8,
                                'creation_cost': 'low',
                                'reasoning': 'Database memory limit reached - recommend basic index optimization'
                            }
                        ],
                        'index_analysis': {'total_indexes': 0, 'error': 'Memory limit'},
                        'optimization_plan': {
                            'immediate_actions': ['Increase PostgreSQL shared_buffers and max_locks_per_transaction'],
                            'short_term_actions': ['Review and optimize existing indexes'],
                            'long_term_actions': ['Consider data partitioning strategies'],
                            'estimated_effort': 'high',
                            'expected_improvement': 0.5
                        },
                        'estimated_improvement': 0.5,
                        'warning': 'Database memory limit reached - optimization limited'
                    }
                else:
                    raise e
            
            # Generate comprehensive recommendations
            recommendations = await self._generate_index_recommendations({
                'query_performance': [
                    {
                        'name': result.query_name,
                        'execution_time_ms': result.execution_time_ms,
                        'efficiency_ratio': result.rows_scanned / max(result.rows_returned, 1) if result.rows_returned > 0 else 0
                    }
                    for result in benchmark.test_queries
                ]
            })
            
            # Analyze current index usage with error handling
            try:
                index_analysis = await self._analyze_current_indexes()
            except Exception as e:
                self.logger.warning(f"⚠️ Index analysis failed: {e}")
                index_analysis = {
                    'total_indexes': 0,
                    'indexes': [],
                    'usage_statistics': {},
                    'unused_indexes': [],
                    'error': str(e)
                }
            
            # Generate optimization plan
            optimization_plan = await self._generate_optimization_plan(benchmark, recommendations, index_analysis)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'benchmark_results': {
                    'overall_score': benchmark.overall_score,
                    'total_patterns': benchmark.total_patterns,
                    'query_count': len(benchmark.test_queries)
                },
                'recommendations': [
                    {
                        'table_name': rec.table_name,
                        'columns': rec.columns,
                        'index_type': rec.index_type,
                        'priority': rec.priority,
                        'expected_improvement': rec.expected_improvement,
                        'creation_cost': rec.creation_cost,
                        'reasoning': rec.reasoning
                    }
                    for rec in recommendations
                ],
                'index_analysis': index_analysis,
                'optimization_plan': optimization_plan,
                'estimated_improvement': sum(rec.expected_improvement for rec in recommendations[:5]) / 5 if recommendations else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive optimization: {e}")
            return {'error': str(e)}
    
    async def _analyze_current_indexes(self) -> Dict[str, Any]:
        """Analyze current database indexes for optimization opportunities"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Get index information
                result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes 
                    WHERE tablename = 'candlestick_patterns'
                    ORDER BY indexname;
                """))
                
                indexes = []
                for row in result.fetchall():
                    indexes.append({
                        'name': row[2],
                        'definition': row[3],
                        'table': row[1]
                    })
                
                # Get index usage statistics - use correct column names
                result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        relname as tablename,
                        indexrelname as indexname,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE relname = 'candlestick_patterns'
                    ORDER BY idx_tup_read DESC;
                """))
                
                usage_stats = {}
                for row in result.fetchall():
                    usage_stats[row[2]] = {
                        'tuples_read': row[3] or 0,
                        'tuples_fetched': row[4] or 0
                    }
                
                return {
                    'total_indexes': len(indexes),
                    'indexes': indexes,
                    'usage_statistics': usage_stats,
                    'unused_indexes': [name for name, stats in usage_stats.items() if stats['tuples_read'] == 0]
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error analyzing current indexes: {e}")
            # Return basic info if detailed analysis fails
            return {
                'total_indexes': 0,
                'indexes': [],
                'usage_statistics': {},
                'unused_indexes': [],
                'error': str(e)
            }
    
    async def _generate_optimization_plan(self, benchmark, recommendations, index_analysis) -> Dict[str, Any]:
        """Generate a comprehensive optimization plan"""
        try:
            plan = {
                'immediate_actions': [],
                'short_term_actions': [],
                'long_term_actions': [],
                'estimated_effort': 'medium',
                'expected_improvement': 0.0
            }
            
            # Immediate actions (high priority, low cost)
            high_priority = [r for r in recommendations if r.priority >= 8]
            if high_priority:
                plan['immediate_actions'].extend([
                    f"Create {r.index_type} index on {r.table_name}({', '.join(r.columns)}) - Priority: {r.priority}"
                    for r in high_priority[:3]  # Top 3
                ])
            
            # Short term actions (medium priority)
            medium_priority = [r for r in recommendations if 5 <= r.priority < 8]
            if medium_priority:
                plan['short_term_actions'].extend([
                    f"Evaluate {r.index_type} index on {r.table_name}({', '.join(r.columns)}) - Expected improvement: {r.expected_improvement:.1%}"
                    for r in medium_priority[:5]  # Top 5
                ])
            
            # Long term actions
            if index_analysis.get('unused_indexes'):
                plan['long_term_actions'].append(
                    f"Review and potentially drop {len(index_analysis['unused_indexes'])} unused indexes"
                )
            
            # Calculate estimated improvement
            if recommendations:
                plan['expected_improvement'] = sum(r.expected_improvement for r in recommendations[:5]) / 5
            
            # Estimate effort
            total_recommendations = len(recommendations)
            if total_recommendations <= 3:
                plan['estimated_effort'] = 'low'
            elif total_recommendations <= 8:
                plan['estimated_effort'] = 'medium'
            else:
                plan['estimated_effort'] = 'high'
            
            return plan
            
        except Exception as e:
            self.logger.error(f"❌ Error generating optimization plan: {e}")
            return {'error': str(e)}
    
    async def stop(self):
        """Stop the advanced retrieval optimizer"""
        try:
            self.monitoring_active = False
            self.logger.info("✅ Advanced Retrieval Optimizer stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Advanced Retrieval Optimizer: {e}")
