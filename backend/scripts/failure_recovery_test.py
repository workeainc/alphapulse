#!/usr/bin/env python3
"""
Critical Failure Recovery Testing for Streaming Infrastructure
Validates system resilience under various failure scenarios
"""
import asyncio
import time
import random
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from streaming.stream_processor import StreamProcessor
from streaming.stream_metrics import StreamMetrics
from streaming.failover_manager import FailoverManager
from streaming.backpressure_handler import BackpressureHandler
from core.config import STREAMING_CONFIG, settings
from database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FailureRecoveryTest:
    def __init__(self):
        self.stream_processor = None
        self.stream_metrics = None
        self.failover_manager = None
        self.backpressure_handler = None
        self.db_connection = None
        self.test_results = {
            'total_failures_simulated': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time_ms': 0,
            'max_recovery_time_ms': 0,
            'min_recovery_time_ms': float('inf'),
            'data_loss_events': 0,
            'consistency_checks_passed': 0,
            'consistency_checks_failed': 0,
            'failover_events': 0
        }
        self.recovery_times = []
        self.test_messages = []
        self.original_config = STREAMING_CONFIG.copy()

    async def initialize_components(self):
        """Initialize all streaming components"""
        logger.info("🔧 Initializing streaming components for failure recovery test...")
        try:
            # Initialize components
            self.stream_processor = StreamProcessor(STREAMING_CONFIG)
            self.stream_metrics = StreamMetrics(STREAMING_CONFIG)
            self.failover_manager = FailoverManager(STREAMING_CONFIG)
            self.backpressure_handler = BackpressureHandler(STREAMING_CONFIG)
            
            # Initialize components
            await self.stream_processor.initialize()
            await self.stream_metrics.initialize()
            await self.failover_manager.initialize()
            await self.backpressure_handler.initialize()
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection()
            await self.db_connection.initialize()
            
            logger.info("✅ All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False

    def generate_test_messages(self, count=100):
        """Generate test messages for consistency checking"""
        messages = []
        for i in range(count):
            message = {
                'message_id': f"test_msg_{i:04d}_{int(time.time())}",
                'symbol': f"TEST{i:03d}",
                'data_type': 'tick',
                'source': 'failure_recovery_test',
                'data': {
                    'price': round(random.uniform(1, 1000), 8),
                    'volume': round(random.uniform(0.1, 100), 8),
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': datetime.now()
            }
            messages.append(message)
        return messages

    @asynccontextmanager
    async def simulate_network_interruption(self, duration_seconds=30):
        """Simulate network interruption by temporarily disabling Redis connection"""
        logger.info(f"🌐 Simulating network interruption for {duration_seconds} seconds...")
        
        # Store original Redis config
        original_redis_host = STREAMING_CONFIG.get('redis_host', 'localhost')
        original_redis_port = STREAMING_CONFIG.get('redis_port', 6379)
        
        try:
            # Simulate network failure by changing Redis host to unreachable
            STREAMING_CONFIG['redis_host'] = 'unreachable-host'
            STREAMING_CONFIG['redis_port'] = 9999
            
            logger.info("🌐 Network interruption started - Redis connection should fail")
            yield
            
            # Wait for the specified duration
            await asyncio.sleep(duration_seconds)
            
        finally:
            # Restore original Redis config
            STREAMING_CONFIG['redis_host'] = original_redis_host
            STREAMING_CONFIG['redis_port'] = original_redis_port
            logger.info("🌐 Network interruption ended - Redis connection restored")

    @asynccontextmanager
    async def simulate_redis_downtime(self, duration_seconds=60):
        """Simulate Redis downtime by stopping Redis service (if possible)"""
        logger.info(f"🔴 Simulating Redis downtime for {duration_seconds} seconds...")
        
        try:
            # In a real scenario, this would stop the Redis service
            # For testing, we'll simulate by changing connection parameters
            original_redis_host = STREAMING_CONFIG.get('redis_host', 'localhost')
            original_redis_port = STREAMING_CONFIG.get('redis_port', 6379)
            
            STREAMING_CONFIG['redis_host'] = 'localhost'
            STREAMING_CONFIG['redis_port'] = 9999  # Invalid port
            
            logger.info("🔴 Redis downtime simulation started")
            yield
            
            await asyncio.sleep(duration_seconds)
            
        finally:
            # Restore original Redis config
            STREAMING_CONFIG['redis_host'] = original_redis_host
            STREAMING_CONFIG['redis_port'] = original_redis_port
            logger.info("🔴 Redis downtime simulation ended")

    @asynccontextmanager
    async def simulate_database_unavailability(self, duration_seconds=45):
        """Simulate TimescaleDB unavailability"""
        logger.info(f"🗄️ Simulating TimescaleDB unavailability for {duration_seconds} seconds...")
        
        try:
            # Store original database config
            original_db_config = {
                'host': settings.DATABASE_HOST,
                'port': settings.DATABASE_PORT,
                'database': settings.DATABASE_NAME
            }
            
            # Simulate database failure by changing connection parameters
            settings.DATABASE_HOST = 'unreachable-db-host'
            settings.DATABASE_PORT = 9999
            
            logger.info("🗄️ Database unavailability simulation started")
            yield
            
            await asyncio.sleep(duration_seconds)
            
        finally:
            # Restore original database config
            settings.DATABASE_HOST = original_db_config['host']
            settings.DATABASE_PORT = original_db_config['port']
            settings.DATABASE_NAME = original_db_config['database']
            logger.info("🗄️ Database unavailability simulation ended")

    async def test_failover_recovery(self, failure_type, failure_duration=30):
        """Test failover and recovery for a specific failure type"""
        logger.info(f"🔄 Testing {failure_type} failover and recovery...")
        
        self.test_results['total_failures_simulated'] += 1
        start_time = time.time()
        
        # Generate test messages before failure
        test_messages = self.generate_test_messages(50)
        self.test_messages.extend(test_messages)
        
        # Process messages before failure
        for message in test_messages:
            try:
                await self.stream_processor.process_message(message)
            except Exception as e:
                logger.warning(f"⚠️ Pre-failure message processing failed: {e}")
        
        # Simulate failure
        if failure_type == 'network_interruption':
            async with self.simulate_network_interruption(failure_duration):
                await self.monitor_failure_impact()
        elif failure_type == 'redis_downtime':
            async with self.simulate_redis_downtime(failure_duration):
                await self.monitor_failure_impact()
        elif failure_type == 'database_unavailability':
            async with self.simulate_database_unavailability(failure_duration):
                await self.monitor_failure_impact()
        
        # Measure recovery time
        recovery_start = time.time()
        
        # Attempt to process messages after failure
        recovery_success = await self.test_recovery_processing()
        
        recovery_time = (time.time() - recovery_start) * 1000  # Convert to milliseconds
        self.recovery_times.append(recovery_time)
        
        # Check data consistency
        consistency_ok = await self.check_data_consistency()
        
        # Update test results
        if recovery_success and consistency_ok:
            self.test_results['successful_recoveries'] += 1
            logger.info(f"✅ {failure_type} recovery successful in {recovery_time:.2f}ms")
        else:
            self.test_results['failed_recoveries'] += 1
            logger.error(f"❌ {failure_type} recovery failed after {recovery_time:.2f}ms")
        
        if not consistency_ok:
            self.test_results['data_loss_events'] += 1
        
        if consistency_ok:
            self.test_results['consistency_checks_passed'] += 1
        else:
            self.test_results['consistency_checks_failed'] += 1

    async def monitor_failure_impact(self):
        """Monitor system behavior during failure"""
        logger.info("📊 Monitoring failure impact...")
        
        try:
            # Check failover manager status
            failover_status = await self.failover_manager.get_status()
            if failover_status.get('failover_active'):
                self.test_results['failover_events'] += 1
                logger.info("🔄 Failover event detected")
            
            # Check backpressure status
            backpressure_status = await self.backpressure_handler.get_status()
            if backpressure_status.get('backpressure_active'):
                logger.info("⚠️ Backpressure handling activated")
            
            # Check system metrics
            metrics = await self.stream_metrics.get_current_metrics()
            logger.info(f"📊 System metrics during failure: {metrics}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to monitor failure impact: {e}")

    async def test_recovery_processing(self):
        """Test message processing after failure recovery"""
        logger.info("🔄 Testing message processing after recovery...")
        
        try:
            # Generate new test messages
            recovery_messages = self.generate_test_messages(20)
            
            # Process messages
            success_count = 0
            for message in recovery_messages:
                try:
                    result = await self.stream_processor.process_message(message)
                    if result:
                        success_count += 1
                except Exception as e:
                    logger.error(f"❌ Recovery message processing failed: {e}")
            
            recovery_rate = success_count / len(recovery_messages) if recovery_messages else 0
            logger.info(f"🔄 Recovery processing success rate: {recovery_rate:.2%}")
            
            return recovery_rate > 0.8  # 80% success rate threshold
            
        except Exception as e:
            logger.error(f"❌ Recovery processing test failed: {e}")
            return False

    async def check_data_consistency(self):
        """Check data consistency after failure recovery"""
        logger.info("🔍 Checking data consistency...")
        
        try:
            # Check if all test messages were processed
            async with self.db_connection.get_session() as session:
                # Count messages in stream_messages table
                result = await session.execute("""
                    SELECT COUNT(*) as count
                    FROM stream_messages 
                    WHERE source = 'failure_recovery_test'
                    AND timestamp > NOW() - INTERVAL '1 hour'
                """)
                message_count = result.fetchone().count if result.fetchone() else 0
                
                # Check for data gaps
                result = await session.execute("""
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(timestamp) as earliest_timestamp,
                        MAX(timestamp) as latest_timestamp
                    FROM stream_messages 
                    WHERE source = 'failure_recovery_test'
                    AND timestamp > NOW() - INTERVAL '1 hour'
                """)
                consistency_data = result.fetchone()
                
                if consistency_data:
                    logger.info(f"🔍 Data consistency check:")
                    logger.info(f"   - Total messages: {consistency_data.total_messages}")
                    logger.info(f"   - Unique symbols: {consistency_data.unique_symbols}")
                    logger.info(f"   - Time range: {consistency_data.earliest_timestamp} to {consistency_data.latest_timestamp}")
                    
                    # Basic consistency checks
                    expected_messages = len(self.test_messages)
                    actual_messages = consistency_data.total_messages
                    
                    if actual_messages >= expected_messages * 0.9:  # Allow 10% loss
                        logger.info("✅ Data consistency check passed")
                        return True
                    else:
                        logger.warning(f"⚠️ Data consistency check failed: expected {expected_messages}, got {actual_messages}")
                        return False
                else:
                    logger.warning("⚠️ No data found for consistency check")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Data consistency check failed: {e}")
            return False

    async def test_graceful_degradation(self):
        """Test graceful degradation when components are unavailable"""
        logger.info("🔄 Testing graceful degradation...")
        
        try:
            # Test with limited functionality
            test_message = {
                'message_id': f"degradation_test_{int(time.time())}",
                'symbol': 'DEGRADATION_TEST',
                'data_type': 'tick',
                'source': 'graceful_degradation_test',
                'data': {
                    'price': 100.0,
                    'volume': 1.0,
                    'timestamp': datetime.now().isoformat()
                },
                'timestamp': datetime.now()
            }
            
            # Process message with degraded components
            result = await self.stream_processor.process_message(test_message)
            
            if result:
                logger.info("✅ Graceful degradation test passed")
                return True
            else:
                logger.warning("⚠️ Graceful degradation test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Graceful degradation test failed: {e}")
            return False

    def generate_test_report(self):
        """Generate comprehensive failure recovery test report"""
        logger.info("📋 Generating failure recovery test report...")
        
        # Calculate recovery statistics
        if self.recovery_times:
            self.test_results['avg_recovery_time_ms'] = sum(self.recovery_times) / len(self.recovery_times)
            self.test_results['max_recovery_time_ms'] = max(self.recovery_times)
            self.test_results['min_recovery_time_ms'] = min(self.recovery_times)
        
        report = {
            'test_summary': {
                'test_name': 'Streaming Infrastructure Failure Recovery Test',
                'test_date': datetime.now().isoformat(),
                'failure_scenarios_tested': [
                    'network_interruption',
                    'redis_downtime', 
                    'database_unavailability'
                ],
                'total_test_messages': len(self.test_messages)
            },
            'recovery_metrics': self.test_results,
            'pass_fail_criteria': {
                'recovery_success_rate': 'successful_recoveries / total_failures > 0.9',
                'avg_recovery_time': 'avg_recovery_time_ms < 30000',  # 30 seconds
                'data_consistency': 'consistency_checks_passed / total_failures > 0.9',
                'data_loss_tolerance': 'data_loss_events / total_failures < 0.1'
            },
            'recommendations': []
        }
        
        # Evaluate results against targets
        total_failures = self.test_results['total_failures_simulated']
        if total_failures > 0:
            recovery_success_rate = self.test_results['successful_recoveries'] / total_failures
            consistency_success_rate = self.test_results['consistency_checks_passed'] / total_failures
            data_loss_rate = self.test_results['data_loss_events'] / total_failures
            
            if recovery_success_rate < 0.9:
                report['recommendations'].append("❌ Recovery success rate below 90% - improve failover mechanisms")
            
            if self.test_results['avg_recovery_time_ms'] > 30000:
                report['recommendations'].append("⚠️ Average recovery time exceeds 30 seconds - optimize recovery procedures")
            
            if consistency_success_rate < 0.9:
                report['recommendations'].append("❌ Data consistency rate below 90% - improve data integrity mechanisms")
            
            if data_loss_rate > 0.1:
                report['recommendations'].append("❌ Data loss rate exceeds 10% - implement better data protection")
        
        if not report['recommendations']:
            report['recommendations'].append("✅ All failure recovery targets met - system is resilient")
        
        return report

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up failure recovery test resources...")
        try:
            if self.stream_processor:
                await self.stream_processor.shutdown()
            if self.stream_metrics:
                await self.stream_metrics.shutdown()
            if self.failover_manager:
                await self.failover_manager.shutdown()
            if self.backpressure_handler:
                await self.backpressure_handler.shutdown()
            if self.db_connection:
                await self.db_connection.close()
            
            # Restore original configuration
            STREAMING_CONFIG.clear()
            STREAMING_CONFIG.update(self.original_config)
            
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main failure recovery test execution"""
    logger.info("=" * 80)
    logger.info("🔄 STREAMING INFRASTRUCTURE FAILURE RECOVERY TEST")
    logger.info("=" * 80)
    
    failure_test = FailureRecoveryTest()
    
    try:
        # Initialize components
        if not await failure_test.initialize_components():
            logger.error("❌ Failed to initialize components - aborting test")
            return False
        
        # Test different failure scenarios
        failure_scenarios = [
            ('network_interruption', 30),
            ('redis_downtime', 60),
            ('database_unavailability', 45)
        ]
        
        for failure_type, duration in failure_scenarios:
            await failure_test.test_failover_recovery(failure_type, duration)
            # Wait between tests
            await asyncio.sleep(10)
        
        # Test graceful degradation
        await failure_test.test_graceful_degradation()
        
        # Generate report
        report = failure_test.generate_test_report()
        
        # Print results
        logger.info("=" * 80)
        logger.info("📊 FAILURE RECOVERY TEST RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Total Failures Simulated: {report['recovery_metrics']['total_failures_simulated']}")
        logger.info(f"Successful Recoveries: {report['recovery_metrics']['successful_recoveries']}")
        logger.info(f"Failed Recoveries: {report['recovery_metrics']['failed_recoveries']}")
        logger.info(f"Recovery Success Rate: {(report['recovery_metrics']['successful_recoveries'] / report['recovery_metrics']['total_failures_simulated'] * 100):.1f}%" if report['recovery_metrics']['total_failures_simulated'] > 0 else "N/A")
        logger.info(f"Average Recovery Time: {report['recovery_metrics']['avg_recovery_time_ms']:.2f}ms")
        logger.info(f"Max Recovery Time: {report['recovery_metrics']['max_recovery_time_ms']:.2f}ms")
        logger.info(f"Data Loss Events: {report['recovery_metrics']['data_loss_events']}")
        logger.info(f"Consistency Checks Passed: {report['recovery_metrics']['consistency_checks_passed']}")
        logger.info(f"Consistency Checks Failed: {report['recovery_metrics']['consistency_checks_failed']}")
        logger.info(f"Failover Events: {report['recovery_metrics']['failover_events']}")
        
        logger.info("\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            logger.info(f"  {rec}")
        
        # Save detailed report
        report_file = backend_path / "failure_recovery_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        # Determine overall success
        total_failures = report['recovery_metrics']['total_failures_simulated']
        if total_failures > 0:
            recovery_success_rate = report['recovery_metrics']['successful_recoveries'] / total_failures
            consistency_success_rate = report['recovery_metrics']['consistency_checks_passed'] / total_failures
            data_loss_rate = report['recovery_metrics']['data_loss_events'] / total_failures
            
            success = (
                recovery_success_rate > 0.9 and
                consistency_success_rate > 0.9 and
                data_loss_rate < 0.1 and
                report['recovery_metrics']['avg_recovery_time_ms'] < 30000
            )
        else:
            success = False
        
        if success:
            logger.info("🎉 FAILURE RECOVERY TEST PASSED - System is resilient!")
        else:
            logger.error("❌ FAILURE RECOVERY TEST FAILED - System needs improvement")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failure recovery test failed with exception: {e}")
        return False
    finally:
        await failure_test.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
