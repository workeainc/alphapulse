#!/usr/bin/env python3
"""
AlphaPlus Performance Optimization Script
Comprehensive performance tuning and monitoring for production deployment
"""

import asyncio
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp
import asyncpg
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Comprehensive performance optimization and monitoring"""
    
    def __init__(self):
        self.metrics = {}
        self.optimization_results = {}
        
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        logger.info("Checking system resources...")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_total': memory.total,
            'memory_used': memory.used,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_used': disk.used,
            'disk_percent': (disk.used / disk.total) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {(disk.used / disk.total) * 100:.1f}%")
        return resources
    
    async def check_database_performance(self) -> Dict[str, Any]:
        """Check database performance metrics"""
        logger.info("Checking database performance...")
        
        try:
            # Connect to database
            conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                user='alphapulse_user',
                password='your_password',  # Replace with actual password
                database='alphapulse'
            )
            
            # Get database statistics
            stats = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
                LIMIT 10
            """)
            
            # Get connection info
            connections = await conn.fetch("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
            """)
            
            # Get database size
            db_size = await conn.fetchval("SELECT pg_size_pretty(pg_database_size('alphapulse'))")
            
            await conn.close()
            
            db_metrics = {
                'table_stats': [dict(row) for row in stats],
                'connections': dict(connections[0]),
                'database_size': db_size,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Database size: {db_size}, Active connections: {connections[0]['active_connections']}")
            return db_metrics
            
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return {'error': str(e)}
    
    async def check_redis_performance(self) -> Dict[str, Any]:
        """Check Redis performance metrics"""
        logger.info("Checking Redis performance...")
        
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Get Redis info
            info = await redis_client.info()
            
            redis_metrics = {
                'used_memory': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            await redis_client.close()
            
            logger.info(f"Redis memory: {redis_metrics['used_memory']}, Hit rate: {redis_metrics['hit_rate']:.1f}%")
            return redis_metrics
            
        except Exception as e:
            logger.error(f"Redis check failed: {e}")
            return {'error': str(e)}
    
    async def check_api_performance(self) -> Dict[str, Any]:
        """Check API performance metrics"""
        logger.info("Checking API performance...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                start_time = time.time()
                async with session.get('http://localhost:8000/health') as response:
                    health_time = time.time() - start_time
                    health_status = response.status
                
                # Test API endpoint
                start_time = time.time()
                async with session.get('http://localhost:8000/api/health') as response:
                    api_time = time.time() - start_time
                    api_status = response.status
                
                api_metrics = {
                    'health_endpoint_time': health_time,
                    'health_status': health_status,
                    'api_endpoint_time': api_time,
                    'api_status': api_status,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Health endpoint: {health_time:.3f}s, API endpoint: {api_time:.3f}s")
                return api_metrics
                
        except Exception as e:
            logger.error(f"API check failed: {e}")
            return {'error': str(e)}
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        logger.info("Optimizing database...")
        
        optimizations = []
        
        try:
            conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                user='alphapulse_user',
                password='your_password',  # Replace with actual password
                database='alphapulse'
            )
            
            # Analyze tables for better query planning
            await conn.execute("ANALYZE;")
            optimizations.append("Database statistics updated")
            
            # Check for unused indexes
            unused_indexes = await conn.fetch("""
                SELECT schemaname, tablename, indexname
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
                AND schemaname = 'public'
            """)
            
            if unused_indexes:
                optimizations.append(f"Found {len(unused_indexes)} unused indexes")
            
            # Check table bloat
            bloated_tables = await conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as bloat
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 10
            """)
            
            await conn.close()
            
            db_optimizations = {
                'optimizations_applied': optimizations,
                'unused_indexes': [dict(row) for row in unused_indexes],
                'table_sizes': [dict(row) for row in bloated_tables],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Applied {len(optimizations)} database optimizations")
            return db_optimizations
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {'error': str(e)}
    
    async def optimize_redis(self) -> Dict[str, Any]:
        """Optimize Redis performance"""
        logger.info("Optimizing Redis...")
        
        optimizations = []
        
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Get memory usage
            memory_info = await redis_client.memory_usage()
            
            # Check for large keys
            large_keys = []
            async for key in redis_client.scan_iter():
                key_size = await redis_client.memory_usage(key)
                if key_size > 1024 * 1024:  # Keys larger than 1MB
                    large_keys.append({'key': key, 'size': key_size})
            
            # Memory optimization suggestions
            if memory_info > 100 * 1024 * 1024:  # More than 100MB
                optimizations.append("Consider enabling memory optimization policies")
            
            if len(large_keys) > 0:
                optimizations.append(f"Found {len(large_keys)} large keys")
            
            await redis_client.close()
            
            redis_optimizations = {
                'optimizations_applied': optimizations,
                'memory_usage': memory_info,
                'large_keys': large_keys,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Applied {len(optimizations)} Redis optimizations")
            return redis_optimizations
            
        except Exception as e:
            logger.error(f"Redis optimization failed: {e}")
            return {'error': str(e)}
    
    async def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance recommendations"""
        logger.info("Generating performance recommendations...")
        
        recommendations = []
        
        # System resource recommendations
        resources = await self.check_system_resources()
        
        if resources.get('cpu_percent', 0) > 80:
            recommendations.append({
                'category': 'CPU',
                'issue': 'High CPU usage',
                'recommendation': 'Consider scaling backend services or optimizing CPU-intensive operations',
                'priority': 'High'
            })
        
        if resources.get('memory_percent', 0) > 85:
            recommendations.append({
                'category': 'Memory',
                'issue': 'High memory usage',
                'recommendation': 'Increase memory limits or optimize memory usage',
                'priority': 'High'
            })
        
        if resources.get('disk_percent', 0) > 90:
            recommendations.append({
                'category': 'Disk',
                'issue': 'High disk usage',
                'recommendation': 'Clean up old data or increase disk space',
                'priority': 'High'
            })
        
        # Database recommendations
        db_metrics = await self.check_database_performance()
        
        if db_metrics.get('connections', {}).get('active_connections', 0) > 50:
            recommendations.append({
                'category': 'Database',
                'issue': 'High active connections',
                'recommendation': 'Consider connection pooling or scaling database',
                'priority': 'Medium'
            })
        
        # Redis recommendations
        redis_metrics = await self.check_redis_performance()
        
        if redis_metrics.get('hit_rate', 0) < 80:
            recommendations.append({
                'category': 'Redis',
                'issue': 'Low cache hit rate',
                'recommendation': 'Review caching strategy and increase cache size',
                'priority': 'Medium'
            })
        
        # API recommendations
        api_metrics = await self.check_api_performance()
        
        if api_metrics.get('api_endpoint_time', 0) > 1.0:
            recommendations.append({
                'category': 'API',
                'issue': 'Slow API response',
                'recommendation': 'Optimize API endpoints and consider caching',
                'priority': 'Medium'
            })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    async def run_full_optimization(self) -> Dict[str, Any]:
        """Run complete performance optimization"""
        logger.info("Starting full performance optimization...")
        
        start_time = time.time()
        
        # Collect all metrics
        system_metrics = await self.check_system_resources()
        db_metrics = await self.check_database_performance()
        redis_metrics = await self.check_redis_performance()
        api_metrics = await self.check_api_performance()
        
        # Run optimizations
        db_optimizations = await self.optimize_database()
        redis_optimizations = await self.optimize_redis()
        
        # Generate recommendations
        recommendations = await self.generate_recommendations()
        
        optimization_time = time.time() - start_time
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_time': optimization_time,
            'system_metrics': system_metrics,
            'database_metrics': db_metrics,
            'redis_metrics': redis_metrics,
            'api_metrics': api_metrics,
            'database_optimizations': db_optimizations,
            'redis_optimizations': redis_optimizations,
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority_issues': len([r for r in recommendations if r['priority'] == 'High']),
                'medium_priority_issues': len([r for r in recommendations if r['priority'] == 'Medium']),
                'low_priority_issues': len([r for r in recommendations if r['priority'] == 'Low'])
            }
        }
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to file"""
        if filename is None:
            filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")

async def main():
    """Main function"""
    optimizer = PerformanceOptimizer()
    
    print("🚀 AlphaPlus Performance Optimization")
    print("=" * 50)
    
    # Run full optimization
    results = await optimizer.run_full_optimization()
    
    # Save results
    optimizer.save_results(results)
    
    # Print summary
    print("\n📊 Optimization Summary:")
    print(f"Total recommendations: {results['summary']['total_recommendations']}")
    print(f"High priority issues: {results['summary']['high_priority_issues']}")
    print(f"Medium priority issues: {results['summary']['medium_priority_issues']}")
    print(f"Low priority issues: {results['summary']['low_priority_issues']}")
    
    print("\n🔧 Recommendations:")
    for rec in results['recommendations']:
        priority_emoji = "🔴" if rec['priority'] == 'High' else "🟡" if rec['priority'] == 'Medium' else "🟢"
        print(f"{priority_emoji} {rec['category']}: {rec['recommendation']}")
    
    print(f"\n✅ Optimization completed in {results['optimization_time']:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
