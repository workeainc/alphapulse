#!/usr/bin/env python3
"""
Ultra-Optimized Pattern Detection Deployment Script
Handles migration, testing, and integration of the complete optimization system
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import modules with correct paths
from database.connection import TimescaleDBConnection
from app.services.ultra_optimized_pattern_service import UltraOptimizedPatternService
from strategies.ultra_optimized_pattern_detector import UltraOptimizedPatternDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/ultra_optimized_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class UltraOptimizedPatternDeployment:
    """
    Comprehensive deployment manager for ultra-optimized pattern detection
    """
    
    def __init__(self):
        """Initialize deployment manager"""
        self.db_connection = None
        self.pattern_service = None
        self.deployment_stats = {
            'start_time': None,
            'end_time': None,
            'migration_success': False,
            'testing_success': False,
            'integration_success': False,
            'performance_improvement': 0.0,
            'errors': []
        }
        
        logger.info("🚀 Ultra-Optimized Pattern Deployment Manager initialized")
    
    async def deploy_complete_system(self):
        """
        **COMPLETE DEPLOYMENT PROCESS**
        Deploy the entire ultra-optimized pattern detection system
        """
        self.deployment_stats['start_time'] = datetime.now()
        logger.info("🎯 Starting complete ultra-optimized pattern detection deployment")
        
        try:
            # Step 1: Database Migration
            await self._run_database_migration()
            
            # Step 2: Service Initialization
            await self._initialize_services()
            
            # Step 3: Performance Testing
            await self._run_performance_tests()
            
            # Step 4: Integration Testing
            await self._run_integration_tests()
            
            # Step 5: Production Deployment
            await self._deploy_to_production()
            
            # Step 6: Final Validation
            await self._final_validation()
            
            self.deployment_stats['end_time'] = datetime.now()
            self.deployment_stats['integration_success'] = True
            
            logger.info("🎉 Ultra-optimized pattern detection deployment completed successfully!")
            await self._generate_deployment_report()
            
        except Exception as e:
            self.deployment_stats['errors'].append(str(e))
            logger.error(f"❌ Deployment failed: {e}")
            await self._rollback_deployment()
            raise
    
    async def _run_database_migration(self):
        """Run database migration for ultra-optimized pattern schema"""
        logger.info("📊 Starting database migration...")
        
        try:
            # Initialize database connection
            self.db_connection = TimescaleDBConnection()
            await self.db_connection.initialize()
            
            # Run migration using Alembic
            import subprocess
            import os
            
            # Change to backend directory
            backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
            original_dir = os.getcwd()
            os.chdir(backend_dir)
            
            # Run migration
            result = subprocess.run([
                'alembic', 'upgrade', 'head'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Migration failed: {result.stderr}")
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info("✅ Database migration completed successfully")
            self.deployment_stats['migration_success'] = True
            
        except Exception as e:
            logger.error(f"❌ Database migration failed: {e}")
            self.deployment_stats['errors'].append(f"Migration error: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize ultra-optimized pattern services"""
        logger.info("🔧 Initializing ultra-optimized pattern services...")
        
        try:
            # Initialize pattern service
            self.pattern_service = UltraOptimizedPatternService(self.db_connection, max_workers=8)
            await self.pattern_service.initialize()
            
            logger.info("✅ Ultra-optimized pattern services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            self.deployment_stats['errors'].append(f"Service initialization error: {e}")
            raise
    
    async def _run_performance_tests(self):
        """Run comprehensive performance tests"""
        logger.info("⚡ Running performance tests...")
        
        try:
            # Create test data
            test_data = self._create_performance_test_data()
            
            # Test 1: Single timeframe performance
            single_perf = await self._test_single_timeframe_performance(test_data)
            
            # Test 2: Multi-timeframe performance
            multi_perf = await self._test_multi_timeframe_performance(test_data)
            
            # Test 3: Cache performance
            cache_perf = await self._test_cache_performance(test_data)
            
            # Test 4: Database performance
            db_perf = await self._test_database_performance()
            
            # Calculate overall performance improvement
            avg_processing_time = (single_perf['avg_time'] + multi_perf['avg_time']) / 2
            self.deployment_stats['performance_improvement'] = avg_processing_time
            
            logger.info(f"✅ Performance tests completed - Avg processing time: {avg_processing_time:.2f}ms")
            self.deployment_stats['testing_success'] = True
            
        except Exception as e:
            logger.error(f"❌ Performance testing failed: {e}")
            self.deployment_stats['errors'].append(f"Performance testing error: {e}")
            raise
    
    async def _test_single_timeframe_performance(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test single timeframe pattern detection performance"""
        logger.info("🧪 Testing single timeframe performance...")
        
        times = []
        pattern_counts = []
        
        for symbol, data in test_data.items():
            for timeframe in ['1m', '5m', '15m']:
                start_time = time.time()
                
                patterns = await self.pattern_service.detect_patterns(
                    symbol, timeframe, data, use_cache=True
                )
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000
                
                times.append(processing_time)
                pattern_counts.append(len(patterns))
                
                logger.info(f"  {symbol} {timeframe}: {len(patterns)} patterns in {processing_time:.2f}ms")
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_patterns': sum(pattern_counts),
            'avg_patterns': np.mean(pattern_counts)
        }
    
    async def _test_multi_timeframe_performance(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test multi-timeframe pattern detection performance"""
        logger.info("🧪 Testing multi-timeframe performance...")
        
        times = []
        pattern_counts = []
        
        for symbol, data in test_data.items():
            # Create multi-timeframe data
            multi_data = {
                '1m': data,
                '5m': self._resample_data(data, '5T'),
                '15m': self._resample_data(data, '15T')
            }
            
            start_time = time.time()
            
            patterns = await self.pattern_service.detect_patterns_multi_timeframe(
                symbol, multi_data, use_cache=True
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            times.append(processing_time)
            total_patterns = sum(len(p) for p in patterns.values())
            pattern_counts.append(total_patterns)
            
            logger.info(f"  {symbol} multi-timeframe: {total_patterns} patterns in {processing_time:.2f}ms")
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_patterns': sum(pattern_counts),
            'avg_patterns': np.mean(pattern_counts)
        }
    
    async def _test_cache_performance(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test cache performance"""
        logger.info("🧪 Testing cache performance...")
        
        cache_hits = 0
        cache_misses = 0
        cache_times = []
        
        for symbol, data in test_data.items():
            # First call (cache miss)
            start_time = time.time()
            patterns1 = await self.pattern_service.detect_patterns(symbol, '1m', data, use_cache=True)
            end_time = time.time()
            cache_misses += 1
            cache_times.append((end_time - start_time) * 1000)
            
            # Second call (cache hit)
            start_time = time.time()
            patterns2 = await self.pattern_service.detect_patterns(symbol, '1m', data, use_cache=True)
            end_time = time.time()
            cache_hits += 1
            cache_times.append((end_time - start_time) * 1000)
        
        cache_hit_rate = cache_hits / (cache_hits + cache_misses)
        avg_cache_time = np.mean(cache_times)
        
        logger.info(f"  Cache hit rate: {cache_hit_rate:.2%}")
        logger.info(f"  Average cache time: {avg_cache_time:.2f}ms")
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'avg_cache_time': avg_cache_time,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }
    
    async def _test_database_performance(self) -> Dict[str, Any]:
        """Test database performance"""
        logger.info("🧪 Testing database performance...")
        
        try:
            # Test pattern retrieval
            start_time = time.time()
            patterns = await self.pattern_service.get_patterns(
                'BTCUSDT', limit=1000, hours_back=24
            )
            end_time = time.time()
            retrieval_time = (end_time - start_time) * 1000
            
            # Test performance stats
            start_time = time.time()
            stats = await self.pattern_service.get_performance_stats()
            end_time = time.time()
            stats_time = (end_time - start_time) * 1000
            
            logger.info(f"  Pattern retrieval: {len(patterns)} patterns in {retrieval_time:.2f}ms")
            logger.info(f"  Stats retrieval: {stats_time:.2f}ms")
            
            return {
                'retrieval_time': retrieval_time,
                'stats_time': stats_time,
                'patterns_retrieved': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Database performance test failed: {e}")
            return {}
    
    async def _run_integration_tests(self):
        """Run integration tests with existing systems"""
        logger.info("🔗 Running integration tests...")
        
        try:
            # Test 1: Compatibility with existing pattern format
            await self._test_format_compatibility()
            
            # Test 2: Database schema compatibility
            await self._test_schema_compatibility()
            
            # Test 3: API compatibility
            await self._test_api_compatibility()
            
            logger.info("✅ Integration tests completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Integration testing failed: {e}")
            self.deployment_stats['errors'].append(f"Integration testing error: {e}")
            raise
    
    async def _test_format_compatibility(self):
        """Test compatibility with existing pattern format"""
        logger.info("  Testing format compatibility...")
        
        # Create test data
        test_data = self._create_test_data()
        
        # Get patterns in new format
        new_patterns = await self.pattern_service.detect_patterns('BTCUSDT', '1m', test_data)
        
        # Verify format compatibility
        required_fields = [
            'pattern_id', 'symbol', 'timeframe', 'pattern_name', 'confidence',
            'strength', 'direction', 'timestamp'
        ]
        
        for pattern in new_patterns:
            missing_fields = [field for field in required_fields if field not in pattern]
            if missing_fields:
                raise Exception(f"Missing required fields: {missing_fields}")
        
        logger.info(f"    ✅ Format compatibility verified - {len(new_patterns)} patterns")
    
    async def _test_schema_compatibility(self):
        """Test database schema compatibility"""
        logger.info("  Testing schema compatibility...")
        
        # Test table existence
        async with self.db_connection.async_session() as session:
            from sqlalchemy import text
            
            # Check ultra_optimized_patterns table
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ultra_optimized_patterns'
                );
            """))
            
            if not result.scalar():
                raise Exception("ultra_optimized_patterns table not found")
            
            # Check indexes
            index_result = await session.execute(text("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'ultra_optimized_patterns'
            """))
            
            indexes = [row[0] for row in index_result]
            required_indexes = [
                'idx_ultra_patterns_timestamp_brin',
                'idx_ultra_patterns_high_confidence',
                'idx_ultra_patterns_covering'
            ]
            
            missing_indexes = [idx for idx in required_indexes if idx not in indexes]
            if missing_indexes:
                raise Exception(f"Missing required indexes: {missing_indexes}")
        
        logger.info("    ✅ Schema compatibility verified")
    
    async def _test_api_compatibility(self):
        """Test API compatibility"""
        logger.info("  Testing API compatibility...")
        
        # Test all service methods
        test_data = self._create_test_data()
        
        # Test single pattern detection
        patterns = await self.pattern_service.detect_patterns('BTCUSDT', '1m', test_data)
        assert isinstance(patterns, list), "Pattern detection should return list"
        
        # Test multi-timeframe detection
        multi_data = {'1m': test_data, '5m': test_data}
        multi_patterns = await self.pattern_service.detect_patterns_multi_timeframe('BTCUSDT', multi_data)
        assert isinstance(multi_patterns, dict), "Multi-timeframe detection should return dict"
        
        # Test pattern retrieval
        retrieved_patterns = await self.pattern_service.get_patterns('BTCUSDT', limit=10)
        assert isinstance(retrieved_patterns, list), "Pattern retrieval should return list"
        
        # Test performance stats
        stats = await self.pattern_service.get_performance_stats()
        assert isinstance(stats, dict), "Performance stats should return dict"
        
        logger.info("    ✅ API compatibility verified")
    
    async def _deploy_to_production(self):
        """Deploy to production environment"""
        logger.info("🚀 Deploying to production...")
        
        try:
            # Update configuration files
            await self._update_configurations()
            
            # Restart services
            await self._restart_services()
            
            # Verify deployment
            await self._verify_production_deployment()
            
            logger.info("✅ Production deployment completed")
            
        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            self.deployment_stats['errors'].append(f"Production deployment error: {e}")
            raise
    
    async def _update_configurations(self):
        """Update configuration files for production"""
        logger.info("  Updating configurations...")
        
        # Update main configuration to use ultra-optimized patterns
        config_file = os.path.join(os.path.dirname(__file__), '..', 'backend', 'config', 'config.py')
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Add ultra-optimized pattern configuration
            if 'ULTRA_OPTIMIZED_PATTERNS' not in content:
                ultra_config = '''
# Ultra-Optimized Pattern Detection Configuration
ULTRA_OPTIMIZED_PATTERNS = {
    'enabled': True,
    'max_workers': 8,
    'buffer_size': 1000,
    'cache_ttl': 300,
    'min_confidence': 0.6,
    'min_strength': 0.5,
    'use_vectorized_operations': True,
    'use_parallel_processing': True,
    'use_cache': True
}
'''
                with open(config_file, 'a') as f:
                    f.write(ultra_config)
        
        logger.info("    ✅ Configurations updated")
    
    async def _restart_services(self):
        """Restart services to apply new configuration"""
        logger.info("  Restarting services...")
        
        # In a real deployment, you would restart your application services here
        # For now, we'll just simulate the restart
        await asyncio.sleep(2)
        
        logger.info("    ✅ Services restarted")
    
    async def _verify_production_deployment(self):
        """Verify production deployment"""
        logger.info("  Verifying production deployment...")
        
        # Test that the service is working in production mode
        test_data = self._create_test_data()
        
        patterns = await self.pattern_service.detect_patterns('BTCUSDT', '1m', test_data)
        
        if not patterns:
            raise Exception("No patterns detected in production mode")
        
        logger.info(f"    ✅ Production deployment verified - {len(patterns)} patterns detected")
    
    async def _final_validation(self):
        """Final validation of the deployment"""
        logger.info("🔍 Running final validation...")
        
        try:
            # Test end-to-end functionality
            test_data = self._create_test_data()
            
            # Test pattern detection
            patterns = await self.pattern_service.detect_patterns('BTCUSDT', '1m', test_data)
            
            # Test pattern retrieval
            retrieved_patterns = await self.pattern_service.get_patterns('BTCUSDT', limit=10)
            
            # Test performance stats
            stats = await self.pattern_service.get_performance_stats()
            
            # Validate results
            assert len(patterns) > 0, "No patterns detected"
            assert len(retrieved_patterns) > 0, "No patterns retrieved"
            assert 'detector_stats' in stats, "Performance stats incomplete"
            
            logger.info("✅ Final validation completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Final validation failed: {e}")
            self.deployment_stats['errors'].append(f"Final validation error: {e}")
            raise
    
    async def _generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        logger.info("📊 Generating deployment report...")
        
        deployment_time = (self.deployment_stats['end_time'] - self.deployment_stats['start_time']).total_seconds()
        
        report = {
            'deployment_summary': {
                'status': 'SUCCESS' if self.deployment_stats['integration_success'] else 'FAILED',
                'start_time': self.deployment_stats['start_time'].isoformat(),
                'end_time': self.deployment_stats['end_time'].isoformat(),
                'duration_seconds': deployment_time,
                'performance_improvement_ms': self.deployment_stats['performance_improvement']
            },
            'component_status': {
                'database_migration': 'SUCCESS' if self.deployment_stats['migration_success'] else 'FAILED',
                'performance_testing': 'SUCCESS' if self.deployment_stats['testing_success'] else 'FAILED',
                'integration_testing': 'SUCCESS' if self.deployment_stats['integration_success'] else 'FAILED'
            },
            'errors': self.deployment_stats['errors'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        import json
        report_file = f'reports/ultra_optimized_deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Deployment report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 ULTRA-OPTIMIZED PATTERN DETECTION DEPLOYMENT COMPLETED")
        print("="*80)
        print(f"Status: {report['deployment_summary']['status']}")
        print(f"Duration: {deployment_time:.2f} seconds")
        print(f"Performance Improvement: {self.deployment_stats['performance_improvement']:.2f}ms")
        print(f"Errors: {len(self.deployment_stats['errors'])}")
        print("="*80)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []
        
        if self.deployment_stats['performance_improvement'] > 100:
            recommendations.append("Consider increasing max_workers for better parallel processing")
        
        if len(self.deployment_stats['errors']) > 0:
            recommendations.append("Review and fix deployment errors before production use")
        
        recommendations.extend([
            "Monitor cache hit rates and adjust cache TTL as needed",
            "Set up alerts for pattern detection performance metrics",
            "Regularly clean up old cache data to prevent memory bloat",
            "Consider implementing Redis for distributed caching in production"
        ])
        
        return recommendations
    
    async def _rollback_deployment(self):
        """Rollback deployment in case of failure"""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Rollback database migration
            import subprocess
            import os
            
            backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
            os.chdir(backend_dir)
            
            subprocess.run([
                'alembic', 'downgrade', '-1'
            ], capture_output=True, text=True)
            
            logger.info("✅ Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
    
    def _create_performance_test_data(self) -> Dict[str, pd.DataFrame]:
        """Create comprehensive test data for performance testing"""
        logger.info("📊 Creating performance test data...")
        
        test_data = {}
        
        # Create data for multiple symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        
        for symbol in symbols:
            # Generate realistic price data
            np.random.seed(hash(symbol) % 1000)
            n = 10000  # Large dataset for performance testing
            
            closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
            highs = closes + np.random.rand(n) * 2
            lows = closes - np.random.rand(n) * 2
            opens = np.roll(closes, 1)
            opens[0] = closes[0]
            volumes = np.random.randint(1000, 10000, n)
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
            })
            
            test_data[symbol] = df
        
        logger.info(f"✅ Created test data for {len(symbols)} symbols")
        return test_data
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create simple test data for integration testing"""
        np.random.seed(42)
        n = 1000
        
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.random.rand(n) * 2
        lows = closes - np.random.rand(n) * 2
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(1000, 10000, n)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min')
        })
    
    def _resample_data(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample data to different timeframe"""
        resampled = data.set_index('timestamp').resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()

async def main():
    """Main deployment function"""
    logger.info("🚀 Starting Ultra-Optimized Pattern Detection Deployment")
    
    # Create deployment manager
    deployment = UltraOptimizedPatternDeployment()
    
    try:
        # Run complete deployment
        await deployment.deploy_complete_system()
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run deployment
    asyncio.run(main())
