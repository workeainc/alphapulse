#!/usr/bin/env python3
"""
Deployment Script for Advanced Database Optimizations
Applies hybrid storage, materialized views, and advanced indexing
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from database.connection import TimescaleDBConnection
from database.migrations.migration_010_advanced_optimizations import AdvancedOptimizationsMigration
from database.optimized_data_service import OptimizedDataService

logger = logging.getLogger(__name__)

class AdvancedOptimizationsDeployer:
    """Deployer for advanced database optimizations"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.db_connection = None
        self.optimized_service = None
        self.deployment_start = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('deployment_advanced_optimizations.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("🚀 Advanced Optimizations Deployer initialized")
    
    async def deploy(self):
        """Deploy all advanced optimizations"""
        try:
            self.deployment_start = datetime.now()
            logger.info("🔄 Starting advanced optimizations deployment...")
            
            # Step 1: Initialize database connection
            await self._initialize_database()
            
            # Step 2: Run migration
            await self._run_migration()
            
            # Step 3: Initialize optimized data service
            await self._initialize_optimized_service()
            
            # Step 4: Warm up cache
            await self._warm_up_cache()
            
            # Step 5: Run performance tests
            await self._run_performance_tests()
            
            # Step 6: Generate deployment report
            await self._generate_deployment_report()
            
            deployment_time = (datetime.now() - self.deployment_start).total_seconds()
            logger.info(f"✅ Advanced optimizations deployment completed in {deployment_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._rollback_deployment()
            raise
    
    async def _initialize_database(self):
        """Initialize database connection"""
        try:
            logger.info("📊 Initializing database connection...")
            
            self.db_connection = TimescaleDBConnection(self.config)
            await self.db_connection.initialize()
            
            logger.info("✅ Database connection initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def _run_migration(self):
        """Run the advanced optimizations migration"""
        try:
            logger.info("📊 Running advanced optimizations migration...")
            
            migration = AdvancedOptimizationsMigration(self.db_connection.async_session)
            await migration.upgrade()
            
            logger.info("✅ Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Error running migration: {e}")
            raise
    
    async def _initialize_optimized_service(self):
        """Initialize the optimized data service"""
        try:
            logger.info("📊 Initializing optimized data service...")
            
            self.optimized_service = OptimizedDataService(
                db_session_factory=self.db_connection.async_session,
                redis_url=self.config.get('redis_url', 'redis://localhost:6379'),
                max_memory_mb=self.config.get('max_memory_mb', 1024)
            )
            
            await self.optimized_service.initialize()
            
            logger.info("✅ Optimized data service initialized")
            
        except Exception as e:
            logger.error(f"Error initializing optimized service: {e}")
            raise
    
    async def _warm_up_cache(self):
        """Warm up cache with frequently accessed data"""
        try:
            logger.info("📊 Warming up cache...")
            
            symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'])
            await self.optimized_service.warm_cache(symbols)
            
            logger.info("✅ Cache warming completed")
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            raise
    
    async def _run_performance_tests(self):
        """Run performance tests to validate optimizations"""
        try:
            logger.info("📊 Running performance tests...")
            
            # Test market data retrieval
            await self._test_market_data_performance()
            
            # Test signals retrieval
            await self._test_signals_performance()
            
            # Test patterns retrieval
            await self._test_patterns_performance()
            
            # Test UI data retrieval
            await self._test_ui_data_performance()
            
            logger.info("✅ Performance tests completed")
            
        except Exception as e:
            logger.error(f"Error running performance tests: {e}")
            raise
    
    async def _test_market_data_performance(self):
        """Test market data retrieval performance"""
        try:
            logger.info("Testing market data performance...")
            
            import time
            
            # Test with cache
            start_time = time.time()
            data_with_cache = await self.optimized_service.get_market_data(
                'BTCUSDT', '1h', hours=24, use_cache=True
            )
            cache_time = (time.time() - start_time) * 1000
            
            # Test without cache
            start_time = time.time()
            data_without_cache = await self.optimized_service.get_market_data(
                'BTCUSDT', '1h', hours=24, use_cache=False
            )
            no_cache_time = (time.time() - start_time) * 1000
            
            logger.info(f"Market data performance:")
            logger.info(f"  - With cache: {cache_time:.2f}ms ({len(data_with_cache)} rows)")
            logger.info(f"  - Without cache: {no_cache_time:.2f}ms ({len(data_without_cache)} rows)")
            logger.info(f"  - Speedup: {no_cache_time/cache_time:.2f}x")
            
        except Exception as e:
            logger.error(f"Error testing market data performance: {e}")
    
    async def _test_signals_performance(self):
        """Test signals retrieval performance"""
        try:
            logger.info("Testing signals performance...")
            
            import time
            
            # Test with cache
            start_time = time.time()
            signals_with_cache = await self.optimized_service.get_signals(
                hours=24, use_cache=True
            )
            cache_time = (time.time() - start_time) * 1000
            
            # Test without cache
            start_time = time.time()
            signals_without_cache = await self.optimized_service.get_signals(
                hours=24, use_cache=False
            )
            no_cache_time = (time.time() - start_time) * 1000
            
            logger.info(f"Signals performance:")
            logger.info(f"  - With cache: {cache_time:.2f}ms ({len(signals_with_cache)} signals)")
            logger.info(f"  - Without cache: {no_cache_time:.2f}ms ({len(signals_without_cache)} signals)")
            logger.info(f"  - Speedup: {no_cache_time/cache_time:.2f}x")
            
        except Exception as e:
            logger.error(f"Error testing signals performance: {e}")
    
    async def _test_patterns_performance(self):
        """Test patterns retrieval performance"""
        try:
            logger.info("Testing patterns performance...")
            
            import time
            
            start_time = time.time()
            patterns = await self.optimized_service.get_patterns(hours=24)
            patterns_time = (time.time() - start_time) * 1000
            
            logger.info(f"Patterns performance:")
            logger.info(f"  - Query time: {patterns_time:.2f}ms ({len(patterns)} patterns)")
            
        except Exception as e:
            logger.error(f"Error testing patterns performance: {e}")
    
    async def _test_ui_data_performance(self):
        """Test UI data retrieval performance"""
        try:
            logger.info("Testing UI data performance...")
            
            import time
            
            start_time = time.time()
            ui_data = await self.optimized_service.get_ui_data()
            ui_time = (time.time() - start_time) * 1000
            
            logger.info(f"UI data performance:")
            logger.info(f"  - Query time: {ui_time:.2f}ms")
            logger.info(f"  - Data size: {len(str(ui_data))} characters")
            
        except Exception as e:
            logger.error(f"Error testing UI data performance: {e}")
    
    async def _generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        try:
            logger.info("📊 Generating deployment report...")
            
            # Get performance report
            performance_report = await self.optimized_service.get_performance_report()
            
            # Get cache stats
            cache_stats = await self.optimized_service.get_cache_stats()
            
            # Calculate deployment metrics
            deployment_time = (datetime.now() - self.deployment_start).total_seconds()
            
            report = {
                'deployment_info': {
                    'start_time': self.deployment_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': deployment_time,
                    'status': 'success'
                },
                'performance_report': performance_report,
                'cache_stats': cache_stats,
                'recommendations': self._generate_recommendations(performance_report, cache_stats)
            }
            
            # Save report to file
            import json
            report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"✅ Deployment report saved to {report_file}")
            
            # Print summary
            self._print_deployment_summary(report)
            
        except Exception as e:
            logger.error(f"Error generating deployment report: {e}")
    
    def _generate_recommendations(self, performance_report: dict, cache_stats: dict) -> list:
        """Generate recommendations based on performance data"""
        recommendations = []
        
        # Check cache hit rates
        if cache_stats.get('cache_hit_rate', 0) < 0.8:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'high',
                'description': 'Cache hit rate is below 80%. Consider increasing cache size or adjusting TTL.',
                'action': 'Increase max_memory_mb or adjust cache TTL settings'
            })
        
        # Check query performance
        query_metrics = cache_stats.get('query_metrics', {})
        for query_type, metrics in query_metrics.items():
            avg_time = metrics.get('avg_time_ms', 0)
            if avg_time > 100:  # Queries taking more than 100ms
                recommendations.append({
                    'type': 'query_optimization',
                    'priority': 'medium',
                    'description': f'{query_type} queries are taking {avg_time:.2f}ms on average.',
                    'action': 'Consider adding indexes or optimizing the query'
                })
        
        # Check memory usage
        memory_usage = cache_stats.get('memory_usage_mb', 0)
        max_memory = self.config.get('max_memory_mb', 1024)
        if memory_usage > max_memory * 0.8:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'description': f'Memory usage is {memory_usage:.1f}MB out of {max_memory}MB.',
                'action': 'Consider increasing max_memory_mb or optimizing cache eviction'
            })
        
        return recommendations
    
    def _print_deployment_summary(self, report: dict):
        """Print deployment summary to console"""
        print("\n" + "="*60)
        print("🚀 ADVANCED OPTIMIZATIONS DEPLOYMENT SUMMARY")
        print("="*60)
        
        # Deployment info
        deployment_info = report['deployment_info']
        print(f"✅ Status: {deployment_info['status']}")
        print(f"⏱️  Duration: {deployment_info['duration_seconds']:.2f} seconds")
        print(f"🕐 Started: {deployment_info['start_time']}")
        print(f"🕐 Completed: {deployment_info['end_time']}")
        
        # Cache stats
        cache_stats = report['cache_stats']
        print(f"\n📊 CACHE PERFORMANCE:")
        print(f"   Memory hits: {cache_stats.get('memory_hits', 0)}")
        print(f"   Redis hits: {cache_stats.get('redis_hits', 0)}")
        print(f"   Cache misses: {cache_stats.get('cache_misses', 0)}")
        print(f"   Memory usage: {cache_stats.get('memory_usage_mb', 0):.1f}MB")
        
        # Query performance
        query_metrics = cache_stats.get('query_metrics', {})
        if query_metrics:
            print(f"\n📈 QUERY PERFORMANCE:")
            for query_type, metrics in query_metrics.items():
                avg_time = metrics.get('avg_time_ms', 0)
                cache_hit_rate = metrics.get('cache_hit_rate', 0)
                print(f"   {query_type}: {avg_time:.2f}ms avg, {cache_hit_rate:.1%} cache hit rate")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. [{rec['priority'].upper()}] {rec['description']}")
        
        print("="*60)
        print("🎉 Deployment completed successfully!")
        print("="*60 + "\n")
    
    async def _rollback_deployment(self):
        """Rollback deployment in case of failure"""
        try:
            logger.info("🔄 Rolling back deployment...")
            
            if self.optimized_service:
                await self.optimized_service.shutdown()
            
            if self.db_connection:
                # Run migration rollback
                migration = AdvancedOptimizationsMigration(self.db_connection.async_session)
                await migration.downgrade()
            
            logger.info("✅ Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"Error rolling back deployment: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.optimized_service:
                await self.optimized_service.shutdown()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main deployment function"""
    # Configuration
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'alphapulse'),
        'username': os.getenv('DB_USER', 'alpha_emon'),
        'password': os.getenv('DB_PASSWORD', 'Emon_@17711'),
        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
        'max_memory_mb': int(os.getenv('MAX_MEMORY_MB', '1024')),
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    }
    
    deployer = AdvancedOptimizationsDeployer(config)
    
    try:
        await deployer.deploy()
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)
    finally:
        await deployer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
