#!/usr/bin/env python3
"""
Database Performance Monitoring Script for AlphaPlus
Monitors TimescaleDB performance metrics, query execution times, and system health
"""

import asyncio
import logging
import time
import psutil
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
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
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    query_execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    active_connections: int
    cache_hit_ratio: float
    slow_queries_count: int
    total_queries: int

class DatabasePerformanceMonitor:
    """Monitor database performance and system metrics"""
    
    def __init__(self, db_path: str = "test_database.db"):
        self.db_path = db_path
        self.db = MockDatabase(db_path)
        self.metrics_history = []
        self.start_time = time.time()
        
        # Performance thresholds
        self.thresholds = {
            'slow_query_ms': 1000,  # Queries slower than 1 second
            'high_memory_mb': 1000,  # Memory usage above 1GB
            'high_cpu_percent': 80,  # CPU usage above 80%
            'low_cache_hit_ratio': 0.8  # Cache hit ratio below 80%
        }
    
    async def initialize(self):
        """Initialize the monitor"""
        await self.db.initialize()
        logger.info("✅ Performance monitor initialized")
    
    async def close(self):
        """Close the monitor"""
        await self.db.close()
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'memory_percent': memory.percent
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
            return {
                'cpu_percent': 0,
                'memory_mb': 0,
                'disk_read_mb': 0,
                'disk_write_mb': 0,
                'memory_percent': 0
            }
    
    async def collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database-specific metrics"""
        try:
            # Get table sizes
            table_sizes = await self.db.fetch("""
                SELECT 
                    name as table_name,
                    (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as row_count
                FROM sqlite_master m
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            
            # Get total records
            total_records = 0
            for table in table_sizes:
                count = await self.db.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
                total_records += count
            
            # Get recent query performance (simulated)
            recent_queries = await self.db.fetch("""
                SELECT COUNT(*) as query_count
                FROM ohlcv_data
                WHERE created_at >= datetime('now', '-1 hour')
            """)
            
            query_count = recent_queries[0]['query_count'] if recent_queries else 0
            
            return {
                'total_records': total_records,
                'table_count': len(table_sizes),
                'recent_queries': query_count,
                'active_connections': 1,  # Mock database has 1 connection
                'cache_hit_ratio': 0.95,  # Simulated high cache hit ratio
                'slow_queries': 0  # No slow queries in mock
            }
            
        except Exception as e:
            logger.error(f"❌ Error collecting database metrics: {e}")
            return {
                'total_records': 0,
                'table_count': 0,
                'recent_queries': 0,
                'active_connections': 0,
                'cache_hit_ratio': 0,
                'slow_queries': 0
            }
    
    async def test_query_performance(self) -> Dict[str, float]:
        """Test query performance with various operations"""
        try:
            query_times = {}
            
            # Test 1: Simple SELECT query
            start_time = time.time()
            await self.db.fetch("SELECT COUNT(*) FROM ohlcv_data")
            query_times['simple_select'] = (time.time() - start_time) * 1000  # Convert to ms
            
            # Test 2: Complex JOIN query
            start_time = time.time()
            await self.db.fetch("""
                SELECT s.symbol, s.direction, s.confidence, COUNT(*) as signal_count
                FROM signals s
                GROUP BY s.symbol, s.direction
            """)
            query_times['complex_join'] = (time.time() - start_time) * 1000
            
            # Test 3: Time-based query
            start_time = time.time()
            await self.db.fetch("""
                SELECT symbol, AVG(close) as avg_price, MAX(volume) as max_volume
                FROM ohlcv_data
                WHERE timestamp >= datetime('now', '-1 hour')
                GROUP BY symbol
            """)
            query_times['time_based_query'] = (time.time() - start_time) * 1000
            
            # Test 4: Index usage test
            start_time = time.time()
            await self.db.fetch("""
                SELECT * FROM ohlcv_data
                WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            query_times['indexed_query'] = (time.time() - start_time) * 1000
            
            return query_times
            
        except Exception as e:
            logger.error(f"❌ Error testing query performance: {e}")
            return {}
    
    async def analyze_slow_queries(self, query_times: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze slow queries and provide recommendations"""
        slow_queries = []
        
        for query_name, execution_time in query_times.items():
            if execution_time > self.thresholds['slow_query_ms']:
                slow_queries.append({
                    'query_name': query_name,
                    'execution_time_ms': execution_time,
                    'recommendation': self._get_query_recommendation(query_name, execution_time)
                })
        
        return slow_queries
    
    def _get_query_recommendation(self, query_name: str, execution_time: float) -> str:
        """Get optimization recommendation for slow queries"""
        recommendations = {
            'simple_select': "Consider adding indexes on frequently queried columns",
            'complex_join': "Optimize JOIN conditions and consider denormalization",
            'time_based_query': "Use time-based partitioning or materialized views",
            'indexed_query': "Check index usage and consider composite indexes"
        }
        
        return recommendations.get(query_name, "Review query structure and add appropriate indexes")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            logger.info("📊 Generating performance report...")
            
            # Collect metrics
            system_metrics = await self.collect_system_metrics()
            db_metrics = await self.collect_database_metrics()
            query_times = await self.test_query_performance()
            
            # Analyze slow queries
            slow_queries = await self.analyze_slow_queries(query_times)
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                system_metrics, db_metrics, query_times
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                system_metrics, db_metrics, slow_queries
            )
            
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'performance_score': performance_score,
                'system_metrics': system_metrics,
                'database_metrics': db_metrics,
                'query_performance': query_times,
                'slow_queries': slow_queries,
                'recommendations': recommendations,
                'uptime_hours': (time.time() - self.start_time) / 3600
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_score(self, system_metrics: Dict, db_metrics: Dict, query_times: Dict) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100.0
            
            # Deduct points for high CPU usage
            if system_metrics['cpu_percent'] > self.thresholds['high_cpu_percent']:
                score -= 20
            
            # Deduct points for high memory usage
            if system_metrics['memory_mb'] > self.thresholds['high_memory_mb']:
                score -= 15
            
            # Deduct points for slow queries
            slow_query_penalty = sum(1 for time_ms in query_times.values() 
                                   if time_ms > self.thresholds['slow_query_ms']) * 10
            score -= slow_query_penalty
            
            # Deduct points for low cache hit ratio
            if db_metrics['cache_hit_ratio'] < self.thresholds['low_cache_hit_ratio']:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance score: {e}")
            return 0
    
    def _generate_recommendations(self, system_metrics: Dict, db_metrics: Dict, slow_queries: List) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # System recommendations
        if system_metrics['cpu_percent'] > self.thresholds['high_cpu_percent']:
            recommendations.append("High CPU usage detected. Consider optimizing queries or scaling resources.")
        
        if system_metrics['memory_mb'] > self.thresholds['high_memory_mb']:
            recommendations.append("High memory usage detected. Consider increasing memory or optimizing data structures.")
        
        # Database recommendations
        if db_metrics['cache_hit_ratio'] < self.thresholds['low_cache_hit_ratio']:
            recommendations.append("Low cache hit ratio. Consider increasing cache size or optimizing queries.")
        
        if slow_queries:
            recommendations.append(f"Found {len(slow_queries)} slow queries. Review and optimize these queries.")
        
        # General recommendations
        if db_metrics['total_records'] > 1000000:
            recommendations.append("Large dataset detected. Consider implementing data partitioning or archiving.")
        
        if not recommendations:
            recommendations.append("System performance is optimal. Continue monitoring for any changes.")
        
        return recommendations
    
    async def run_continuous_monitoring(self, duration_minutes: int = 5):
        """Run continuous performance monitoring"""
        logger.info(f"🔄 Starting continuous monitoring for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                report = await self.generate_performance_report()
                
                # Log key metrics
                logger.info(f"📊 Performance Score: {report['performance_score']:.1f}/100")
                logger.info(f"💻 CPU: {report['system_metrics']['cpu_percent']:.1f}%")
                logger.info(f"🧠 Memory: {report['system_metrics']['memory_mb']:.1f} MB")
                logger.info(f"📊 Total Records: {report['database_metrics']['total_records']}")
                
                # Check for alerts
                if report['performance_score'] < 70:
                    logger.warning(f"⚠️ Performance alert: Score {report['performance_score']:.1f}")
                
                # Store metrics
                self.metrics_history.append(report)
                
                # Wait before next measurement
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)
        
        logger.info("✅ Continuous monitoring completed")
    
    def save_performance_report(self, report: Dict[str, Any], filename: str = None):
        """Save performance report to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Performance report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error saving performance report: {e}")

async def main():
    """Main function for performance monitoring"""
    monitor = DatabasePerformanceMonitor()
    
    try:
        await monitor.initialize()
        
        # Generate single performance report
        logger.info("📊 Generating performance report...")
        report = await monitor.generate_performance_report()
        
        # Display report
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info(f"Performance Score: {report['performance_score']:.1f}/100")
        logger.info(f"System Metrics:")
        logger.info(f"  CPU Usage: {report['system_metrics']['cpu_percent']:.1f}%")
        logger.info(f"  Memory Usage: {report['system_metrics']['memory_mb']:.1f} MB")
        logger.info(f"  Memory Percent: {report['system_metrics']['memory_percent']:.1f}%")
        logger.info(f"Database Metrics:")
        logger.info(f"  Total Records: {report['database_metrics']['total_records']}")
        logger.info(f"  Table Count: {report['database_metrics']['table_count']}")
        logger.info(f"  Cache Hit Ratio: {report['database_metrics']['cache_hit_ratio']:.3f}")
        logger.info(f"Query Performance:")
        for query_name, time_ms in report['query_performance'].items():
            logger.info(f"  {query_name}: {time_ms:.2f} ms")
        
        if report['slow_queries']:
            logger.info("Slow Queries:")
            for slow_query in report['slow_queries']:
                logger.info(f"  {slow_query['query_name']}: {slow_query['execution_time_ms']:.2f} ms")
                logger.info(f"    Recommendation: {slow_query['recommendation']}")
        
        logger.info("Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        # Save report
        monitor.save_performance_report(report)
        
        # Run continuous monitoring for 2 minutes
        await monitor.run_continuous_monitoring(duration_minutes=2)
        
    finally:
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())
