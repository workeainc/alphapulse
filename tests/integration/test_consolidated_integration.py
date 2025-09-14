#!/usr/bin/env python3
"""
Consolidated Integration Test Suite for AlphaPulse
Combines essential testing functionality from multiple duplicate test files
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Update import paths for new structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ..database.connection import TimescaleDBConnection, get_enhanced_connection
from ..database.models import create_tables, drop_tables, Trade, MarketData, SentimentData
from ..core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsolidatedTestSuite:
    """Consolidated test suite for AlphaPulse core functionality"""
    
    def __init__(self):
        self.db_connection = None
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    async def setup(self):
        """Setup test environment"""
        try:
            logger.info("🚀 Setting up Consolidated Test Suite...")
            
            # Initialize database connection
            self.db_connection = get_enhanced_connection()
            await self.db_connection.initialize()
            
            # Setup TimescaleDB
            await self.db_connection.setup_timescaledb()
            
            # Create test tables
            create_tables()
            
            logger.info("✅ Test environment setup completed")
            
        except Exception as e:
            logger.error(f"❌ Test setup failed: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment"""
        try:
            logger.info("🧹 Cleaning up test environment...")
            
            if self.db_connection:
                await self.db_connection.close()
            
            logger.info("✅ Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Test cleanup failed: {e}")
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        self.test_results['total_tests'] += 1
        
        try:
            logger.info(f"🧪 Running test: {test_name}")
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            
            execution_time = time.time() - start_time
            self.test_results['passed'] += 1
            logger.info(f"✅ Test passed: {test_name} ({execution_time:.2f}s)")
            
        except Exception as e:
            self.test_results['failed'] += 1
            error_msg = f"Test failed: {test_name} - {str(e)}"
            self.test_results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    async def test_database_connection(self):
        """Test database connection functionality"""
        # Test basic connectivity
        health_status = await self.db_connection.health_check()
        assert health_status['healthy'], f"Database health check failed: {health_status}"
        
        # Test connection pool
        pool_config = self.db_connection.get_pool_config()
        assert pool_config.min_connections > 0, "Invalid pool configuration"
        
        # Test session creation
        async with self.db_connection.get_async_session() as session:
            result = await session.execute("SELECT 1")
            assert result is not None, "Session execution failed"
    
    async def test_timescaledb_features(self):
        """Test TimescaleDB-specific features"""
        # Test hypertable creation (should already exist from setup)
        async with self.db_connection.get_async_session() as session:
            # Check if TimescaleDB extension is enabled
            result = await session.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb'")
            extensions = await result.fetchall()
            assert len(extensions) > 0, "TimescaleDB extension not enabled"
            
            # Check if hypertables exist
            result = await session.execute("SELECT * FROM timescaledb_information.hypertables")
            hypertables = await result.fetchall()
            assert len(hypertables) > 0, "No hypertables found"
    
    async def test_data_models(self):
        """Test database models and CRUD operations"""
        async with self.db_connection.get_async_session() as session:
            # Test Trade model
            trade = Trade(
                symbol="BTCUSDT",
                side="long",
                entry_price=50000.0,
                quantity=0.1,
                strategy_name="test_strategy",
                status="open"
            )
            session.add(trade)
            await session.commit()
            
            # Verify trade was created
            result = await session.execute("SELECT * FROM trades WHERE symbol = 'BTCUSDT'")
            trades = await result.fetchall()
            assert len(trades) > 0, "Trade creation failed"
            
            # Test MarketData model
            market_data = MarketData(
                symbol="BTCUSDT",
                timeframe="1h",
                timestamp=datetime.utcnow(),
                open_price=50000.0,
                high_price=51000.0,
                low_price=49000.0,
                close_price=50500.0,
                volume=1000.0
            )
            session.add(market_data)
            await session.commit()
            
            # Verify market data was created
            result = await session.execute("SELECT * FROM market_data WHERE symbol = 'BTCUSDT'")
            market_data_list = await result.fetchall()
            assert len(market_data_list) > 0, "Market data creation failed"
    
    async def test_connection_pooling(self):
        """Test connection pooling functionality"""
        # Test multiple concurrent connections
        async def test_connection():
            async with self.db_connection.get_async_session() as session:
                await session.execute("SELECT 1")
                await asyncio.sleep(0.1)  # Simulate work
        
        # Run multiple concurrent connections
        tasks = [test_connection() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Check pool statistics
        stats = self.db_connection.get_connection_stats()
        assert stats['total_connections_created'] > 0, "No connections were created"
    
    async def test_health_monitoring(self):
        """Test health monitoring functionality"""
        # Get health status
        health_status = self.db_connection.get_health_status()
        assert health_status.state.value in ['healthy', 'degraded', 'unknown'], f"Invalid health state: {health_status.state.value}"
        
        # Get connection stats
        stats = self.db_connection.get_connection_stats()
        assert 'total_health_checks' in stats, "Health check stats missing"
    
    async def test_performance_optimization(self):
        """Test performance optimization features"""
        async with self.db_connection.get_async_session() as session:
            # Test bulk insert performance
            start_time = time.time()
            
            # Insert multiple market data records
            market_data_records = []
            for i in range(100):
                market_data = MarketData(
                    symbol="ETHUSDT",
                    timeframe="1m",
                    timestamp=datetime.utcnow() + timedelta(minutes=i),
                    open_price=3000.0 + i,
                    high_price=3000.0 + i + 10,
                    low_price=3000.0 + i - 10,
                    close_price=3000.0 + i + 5,
                    volume=100.0 + i
                )
                market_data_records.append(market_data)
            
            session.add_all(market_data_records)
            await session.commit()
            
            execution_time = time.time() - start_time
            logger.info(f"Bulk insert of 100 records completed in {execution_time:.2f}s")
            
            # Verify bulk insert
            result = await session.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'ETHUSDT'")
            count = await result.scalar()
            assert count >= 100, f"Bulk insert failed, expected >=100, got {count}"
    
    async def test_error_handling(self):
        """Test error handling and resilience"""
        # Test invalid query handling
        try:
            async with self.db_connection.get_async_session() as session:
                await session.execute("SELECT * FROM non_existent_table")
                assert False, "Should have raised an error"
        except Exception as e:
            logger.info(f"✅ Expected error caught: {e}")
        
        # Test connection recovery
        health_status = await self.db_connection.health_check()
        assert health_status['healthy'], "Connection should recover from errors"
    
    async def run_all_tests(self):
        """Run all consolidated tests"""
        self.test_results['start_time'] = datetime.utcnow()
        
        try:
            await self.setup()
            
            # Run all tests
            tests = [
                ("Database Connection", self.test_database_connection),
                ("TimescaleDB Features", self.test_timescaledb_features),
                ("Data Models", self.test_data_models),
                ("Connection Pooling", self.test_connection_pooling),
                ("Health Monitoring", self.test_health_monitoring),
                ("Performance Optimization", self.test_performance_optimization),
                ("Error Handling", self.test_error_handling),
            ]
            
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
            
        finally:
            await self.teardown()
            self.test_results['end_time'] = datetime.utcnow()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if self.test_results['start_time'] and self.test_results['end_time']:
            duration = (self.test_results['end_time'] - self.test_results['start_time']).total_seconds()
        else:
            duration = 0
        
        report = {
            'test_suite': 'Consolidated Integration Test Suite',
            'timestamp': datetime.utcnow().isoformat(),
            'duration_seconds': duration,
            'results': self.test_results.copy(),
            'summary': {
                'total_tests': self.test_results['total_tests'],
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'success_rate': f"{(self.test_results['passed'] / self.test_results['total_tests'] * 100):.1f}%" if self.test_results['total_tests'] > 0 else "0%"
            }
        }
        
        return report
    
    def print_report(self):
        """Print test report to console"""
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("🧪 CONSOLIDATED INTEGRATION TEST SUITE RESULTS")
        print("="*80)
        print(f"📊 Test Suite: {report['test_suite']}")
        print(f"⏰ Timestamp: {report['timestamp']}")
        print(f"⏱️  Duration: {report['duration_seconds']:.2f} seconds")
        print(f"📈 Total Tests: {report['summary']['total_tests']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"🎯 Success Rate: {report['summary']['success_rate']}")
        
        if report['results']['errors']:
            print("\n❌ ERRORS:")
            for error in report['results']['errors']:
                print(f"   • {error}")
        
        print("="*80)
        
        # Save report to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_filename = f"consolidated_test_report_{timestamp}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {report_filename}")


async def main():
    """Main test execution function"""
    test_suite = ConsolidatedTestSuite()
    
    try:
        await test_suite.run_all_tests()
        test_suite.print_report()
        
        # Exit with appropriate code
        if test_suite.test_results['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
