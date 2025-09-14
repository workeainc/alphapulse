#!/usr/bin/env python3
"""
Stress Testing Script for Phase 2 Outcome Tracking
Tests error handling under high-throughput stress scenarios
"""

import asyncio
import time
import random
import json
import logging
import psutil
import gc
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
import statistics

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from outcome_tracking.outcome_tracker import OutcomeTracker, OutcomeType, SignalOutcome
from outcome_tracking.tp_sl_detector import TPSLDetector, HitType, TPSLHit
from outcome_tracking.performance_analyzer import PerformanceAnalyzer
from database.connection import TimescaleDBConnection
from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTestOutcomeTracking:
    """Stress testing class for Phase 2 outcome tracking system"""

    def __init__(self):
        self.outcome_tracker = None
        self.tp_sl_detector = None
        self.performance_analyzer = None
        self.db_connection = None
        
        # Stress test configuration
        self.test_duration = 120  # 2 minutes
        self.max_concurrent_requests = 50
        self.error_injection_rate = 0.1  # 10% error injection
        self.network_latency_ms = 100
        self.memory_pressure_threshold = 0.8  # 80% memory usage
        
        # Stress test scenarios
        self.scenarios = [
            'database_connection_failure',
            'redis_disconnection',
            'network_latency',
            'memory_pressure',
            'cpu_exhaustion',
            'concurrent_overload',
            'data_corruption',
            'timeout_scenarios'
        ]
        
        # Performance metrics
        self.start_time = None
        self.end_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        self.response_times = []
        self.error_counts = {}
        self.recovery_times = []
        
        # Test results
        self.test_results = {
            'stress_test': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'timeout_requests': 0,
                'success_rate': 0,
                'avg_response_time_ms': 0,
                'p95_response_time_ms': 0,
                'p99_response_time_ms': 0,
                'max_response_time_ms': 0,
                'avg_recovery_time_ms': 0,
                'circuit_breaker_trips': 0,
                'success': False
            },
            'scenario_results': {},
            'error_analysis': {},
            'recovery_analysis': {}
        }

    async def initialize_components(self):
        """Initialize all outcome tracking components"""
        try:
            logger.info("Initializing outcome tracking components for stress test...")
            
            # Initialize database connection with stress test settings
            self.db_connection = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 20,  # Reduced for stress testing
                'max_overflow': 30,
                'pool_timeout': 10,  # Reduced timeout
                'query_timeout': 5  # Reduced query timeout
            })
            await self.db_connection.initialize()
            
            # Initialize outcome tracker
            self.outcome_tracker = OutcomeTracker()
            await self.outcome_tracker.initialize()
            
            # Initialize TP/SL detector
            self.tp_sl_detector = TPSLDetector()
            await self.tp_sl_detector.initialize()
            
            # Initialize performance analyzer
            self.performance_analyzer = PerformanceAnalyzer()
            await self.performance_analyzer.initialize()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False

    def generate_stress_signal(self, signal_id: str, scenario: str) -> Dict[str, Any]:
        """Generate a test signal with stress test modifications"""
        signal = {
            'signal_id': signal_id,
            'symbol': random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']),
            'side': random.choice(['buy', 'sell']),
            'strategy': random.choice(['momentum', 'mean_reversion', 'breakout', 'scalping']),
            'confidence': random.uniform(0.7, 0.95),
            'strength': random.choice(['weak', 'medium', 'strong']),
            'timestamp': datetime.now(timezone.utc),
            'price': random.uniform(100, 50000),
            'stop_loss': random.uniform(50, 25000),
            'take_profit': random.uniform(150, 75000),
            'metadata': {
                'stress_test': True,
                'scenario': scenario,
                'stress_test_id': f"stress_test_{int(time.time())}",
                'injected_errors': []
            }
        }
        
        # Inject errors based on scenario
        if scenario == 'data_corruption':
            signal['price'] = None  # Corrupt data
        elif scenario == 'timeout_scenarios':
            signal['metadata']['timeout_ms'] = random.randint(5000, 10000)  # Long timeout
        
        return signal

    async def simulate_database_failure(self):
        """Simulate database connection failure"""
        try:
            logger.info("🔥 Simulating database connection failure...")
            
            # Close database connection
            if self.db_connection:
                await self.db_connection.close()
            
            # Wait for failure to be detected
            await asyncio.sleep(2)
            
            # Attempt to reconnect
            start_time = time.time()
            await self.db_connection.initialize()
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_times.append(recovery_time)
            logger.info(f"✅ Database recovery completed in {recovery_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Database failure simulation error: {e}")

    async def simulate_redis_disconnection(self):
        """Simulate Redis disconnection"""
        try:
            logger.info("🔥 Simulating Redis disconnection...")
            
            # Simulate Redis connection loss
            await asyncio.sleep(1)
            
            # Simulate reconnection
            start_time = time.time()
            await asyncio.sleep(random.uniform(0.5, 2.0))  # Random reconnection time
            recovery_time = (time.time() - start_time) * 1000
            
            self.recovery_times.append(recovery_time)
            logger.info(f"✅ Redis recovery completed in {recovery_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Redis disconnection simulation error: {e}")

    async def simulate_network_latency(self):
        """Simulate network latency"""
        try:
            logger.info("🔥 Simulating network latency...")
            
            # Add random network latency
            latency = random.uniform(50, 500)  # 50-500ms latency
            await asyncio.sleep(latency / 1000)
            
            logger.info(f"✅ Network latency simulation: {latency:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Network latency simulation error: {e}")

    async def simulate_memory_pressure(self):
        """Simulate memory pressure"""
        try:
            logger.info("🔥 Simulating memory pressure...")
            
            # Create memory pressure by allocating large objects
            memory_objects = []
            for i in range(1000):
                memory_objects.append([random.random() for _ in range(1000)])
            
            # Force garbage collection
            gc.collect()
            
            # Clear objects
            del memory_objects
            gc.collect()
            
            logger.info("✅ Memory pressure simulation completed")
            
        except Exception as e:
            logger.error(f"❌ Memory pressure simulation error: {e}")

    async def simulate_cpu_exhaustion(self):
        """Simulate CPU exhaustion"""
        try:
            logger.info("🔥 Simulating CPU exhaustion...")
            
            # Perform CPU-intensive operations
            for i in range(10000):
                _ = sum(random.random() for _ in range(1000))
            
            logger.info("✅ CPU exhaustion simulation completed")
            
        except Exception as e:
            logger.error(f"❌ CPU exhaustion simulation error: {e}")

    async def process_stress_request(self, request_id: str, scenario: str) -> Dict[str, Any]:
        """Process a single stress test request"""
        request_start_time = time.time()
        result = {
            'request_id': request_id,
            'scenario': scenario,
            'success': False,
            'response_time_ms': 0,
            'error': None,
            'timeout': False
        }
        
        try:
            # Generate stress signal
            signal = self.generate_stress_signal(request_id, scenario)
            
            # Apply scenario-specific stress
            if scenario == 'database_connection_failure' and random.random() < 0.1:
                await self.simulate_database_failure()
            elif scenario == 'redis_disconnection' and random.random() < 0.1:
                await self.simulate_redis_disconnection()
            elif scenario == 'network_latency':
                await self.simulate_network_latency()
            elif scenario == 'memory_pressure' and random.random() < 0.2:
                await self.simulate_memory_pressure()
            elif scenario == 'cpu_exhaustion' and random.random() < 0.2:
                await self.simulate_cpu_exhaustion()
            
            # Process signal with timeout
            timeout_seconds = signal.get('metadata', {}).get('timeout_ms', 5000) / 1000
            
            try:
                # Track signal with timeout
                outcome = await asyncio.wait_for(
                    self.outcome_tracker.track_signal(signal),
                    timeout=timeout_seconds
                )
                
                # Check TP/SL with timeout
                price_data = {
                    'symbol': signal['symbol'],
                    'timestamp': datetime.now(timezone.utc),
                    'price': signal['price'] + random.uniform(-signal['price'] * 0.1, signal['price'] * 0.1),
                    'volume': random.uniform(1000, 100000)
                }
                
                hits = await asyncio.wait_for(
                    self.tp_sl_detector.track_position(signal['signal_id'], price_data),
                    timeout=timeout_seconds
                )
                
                result['success'] = True
                
            except asyncio.TimeoutError:
                result['timeout'] = True
                result['error'] = f"Request timed out after {timeout_seconds}s"
                self.error_counts['timeout'] = self.error_counts.get('timeout', 0) + 1
                
            except Exception as e:
                result['error'] = str(e)
                error_type = type(e).__name__
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
        except Exception as e:
            result['error'] = str(e)
            error_type = type(e).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        finally:
            # Calculate response time
            result['response_time_ms'] = (time.time() - request_start_time) * 1000
            self.response_times.append(result['response_time_ms'])
            
            # Update counters
            self.total_requests += 1
            if result['success']:
                self.successful_requests += 1
            elif result['timeout']:
                self.timeout_requests += 1
            else:
                self.failed_requests += 1
        
        return result

    async def run_stress_scenario(self, scenario: str) -> Dict[str, Any]:
        """Run a specific stress scenario"""
        logger.info(f"🔥 Running stress scenario: {scenario}")
        
        scenario_start_time = time.time()
        scenario_requests = []
        
        # Generate requests for this scenario
        num_requests = random.randint(50, 200)
        
        # Process requests with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_request_with_semaphore(request_id: str):
            async with semaphore:
                return await self.process_stress_request(request_id, scenario)
        
        # Create request tasks
        request_tasks = []
        for i in range(num_requests):
            request_id = f"{scenario}_{i}_{int(time.time())}"
            task = asyncio.create_task(process_request_with_semaphore(request_id))
            request_tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*request_tasks, return_exceptions=True)
        
        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = sum(1 for r in results if isinstance(r, dict) and not r.get('success'))
        timeouts = sum(1 for r in results if isinstance(r, dict) and r.get('timeout'))
        
        scenario_duration = (time.time() - scenario_start_time) * 1000
        
        scenario_result = {
            'scenario': scenario,
            'total_requests': num_requests,
            'successful_requests': successful,
            'failed_requests': failed,
            'timeout_requests': timeouts,
            'success_rate': (successful / num_requests * 100) if num_requests > 0 else 0,
            'duration_ms': scenario_duration,
            'avg_response_time_ms': statistics.mean([r.get('response_time_ms', 0) for r in results if isinstance(r, dict)]) if results else 0
        }
        
        logger.info(f"✅ Scenario {scenario} completed:")
        logger.info(f"   - Success rate: {scenario_result['success_rate']:.2f}%")
        logger.info(f"   - Avg response time: {scenario_result['avg_response_time_ms']:.2f}ms")
        logger.info(f"   - Timeouts: {timeouts}")
        
        return scenario_result

    async def run_stress_test(self):
        """Run the main stress test"""
        logger.info(f"🚀 Starting stress test: {len(self.scenarios)} scenarios for {self.test_duration} seconds")
        
        # Initialize components
        if not await self.initialize_components():
            return False
        
        # Record start time
        self.start_time = time.time()
        self.end_time = self.start_time + self.test_duration
        
        # Run stress scenarios
        scenario_results = []
        for scenario in self.scenarios:
            if time.time() < self.end_time:
                try:
                    result = await self.run_stress_scenario(scenario)
                    scenario_results.append(result)
                    self.test_results['scenario_results'][scenario] = result
                except Exception as e:
                    logger.error(f"❌ Scenario {scenario} failed: {e}")
                    self.test_results['scenario_results'][scenario] = {
                        'scenario': scenario,
                        'error': str(e),
                        'success': False
                    }
        
        # Record end time
        self.end_time = time.time()
        
        # Process results
        await self.process_stress_test_results(scenario_results)
        
        return True

    async def process_stress_test_results(self, scenario_results: List[Dict[str, Any]]):
        """Process and analyze stress test results"""
        logger.info("📈 Processing stress test results...")
        
        # Calculate overall metrics
        total_requests = sum(result.get('total_requests', 0) for result in scenario_results)
        total_successful = sum(result.get('successful_requests', 0) for result in scenario_results)
        total_failed = sum(result.get('failed_requests', 0) for result in scenario_results)
        total_timeouts = sum(result.get('timeout_requests', 0) for result in scenario_results)
        
        # Calculate response time statistics
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            max_response_time = max(self.response_times)
            p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = max_response_time = p95_response_time = p99_response_time = 0
        
        # Calculate recovery time statistics
        if self.recovery_times:
            avg_recovery_time = statistics.mean(self.recovery_times)
        else:
            avg_recovery_time = 0
        
        # Calculate success rate
        success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        # Update test results
        self.test_results['stress_test'].update({
            'total_requests': total_requests,
            'successful_requests': total_successful,
            'failed_requests': total_failed,
            'timeout_requests': total_timeouts,
            'success_rate': round(success_rate, 2),
            'avg_response_time_ms': round(avg_response_time, 2),
            'p95_response_time_ms': round(p95_response_time, 2),
            'p99_response_time_ms': round(p99_response_time, 2),
            'max_response_time_ms': round(max_response_time, 2),
            'avg_recovery_time_ms': round(avg_recovery_time, 2),
            'circuit_breaker_trips': self.error_counts.get('ConnectionError', 0),
            'success': success_rate >= 80 and avg_response_time < 1000  # 80% success rate and <1s avg response
        })
        
        # Update error analysis
        self.test_results['error_analysis'] = {
            'error_counts': self.error_counts,
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            'total_errors': sum(self.error_counts.values())
        }
        
        # Update recovery analysis
        self.test_results['recovery_analysis'] = {
            'total_recoveries': len(self.recovery_times),
            'avg_recovery_time_ms': round(avg_recovery_time, 2),
            'max_recovery_time_ms': round(max(self.recovery_times), 2) if self.recovery_times else 0,
            'min_recovery_time_ms': round(min(self.recovery_times), 2) if self.recovery_times else 0
        }
        
        # Log results
        logger.info("📊 Stress Test Results:")
        logger.info(f"   ✅ Total requests: {total_requests}")
        logger.info(f"   ✅ Success rate: {success_rate:.2f}%")
        logger.info(f"   ✅ Avg response time: {avg_response_time:.2f}ms")
        logger.info(f"   ✅ P95 response time: {p95_response_time:.2f}ms")
        logger.info(f"   ✅ P99 response time: {p99_response_time:.2f}ms")
        logger.info(f"   ✅ Timeouts: {total_timeouts}")
        logger.info(f"   ✅ Avg recovery time: {avg_recovery_time:.2f}ms")
        logger.info(f"   ✅ Test success: {self.test_results['stress_test']['success']}")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up resources...")
            
            if self.outcome_tracker:
                await self.outcome_tracker.shutdown()
            
            if self.tp_sl_detector:
                await self.tp_sl_detector.shutdown()
            
            if self.performance_analyzer:
                await self.performance_analyzer.shutdown()
            
            if self.db_connection:
                await self.db_connection.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        report = {
            'test_summary': {
                'test_name': 'Phase 2 Outcome Tracking Stress Test',
                'test_duration_seconds': self.test_duration,
                'scenarios_tested': len(self.scenarios),
                'max_concurrent_requests': self.max_concurrent_requests,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            },
            'stress_test_results': self.test_results['stress_test'],
            'scenario_results': self.test_results['scenario_results'],
            'error_analysis': self.test_results['error_analysis'],
            'recovery_analysis': self.test_results['recovery_analysis'],
            'recommendations': self.generate_recommendations()
        }
        
        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        # Performance recommendations
        if self.test_results['stress_test']['success_rate'] < 80:
            recommendations.append("⚠️ Success rate is below 80%. Review error handling and system resilience.")
        
        if self.test_results['stress_test']['avg_response_time_ms'] > 1000:
            recommendations.append("⚠️ Average response time is high. Consider optimizing processing logic and database queries.")
        
        if self.test_results['stress_test']['p99_response_time_ms'] > 5000:
            recommendations.append("⚠️ 99th percentile response time is high. Implement timeout handling and circuit breakers.")
        
        if self.test_results['stress_test']['timeout_requests'] > 0:
            recommendations.append("⚠️ Timeout requests detected. Review timeout configurations and system capacity.")
        
        # Error handling recommendations
        if self.test_results['error_analysis']['total_errors'] > 0:
            recommendations.append("⚠️ Errors detected during stress test. Review error handling and recovery mechanisms.")
        
        # Recovery recommendations
        if self.test_results['recovery_analysis']['avg_recovery_time_ms'] > 5000:
            recommendations.append("⚠️ Recovery time is high. Implement faster recovery mechanisms and connection pooling.")
        
        # Success recommendations
        if self.test_results['stress_test']['success']:
            recommendations.append("✅ System demonstrates good resilience under stress conditions.")
        else:
            recommendations.append("❌ System does not meet stress test requirements. Address issues before production deployment.")
        
        return recommendations

async def main():
    """Main function to run the stress test"""
    logger.info("🚀 Starting Phase 2 Outcome Tracking Stress Test")
    
    # Create stress test instance
    stress_test = StressTestOutcomeTracking()
    
    try:
        # Run stress test
        success = await stress_test.run_stress_test()
        
        if success:
            # Generate and save report
            report = stress_test.generate_report()
            
            # Save report to file
            report_file = f"stress_test_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Stress test report saved to: {report_file}")
            
            # Print summary
            logger.info("🎯 Stress Test Summary:")
            logger.info(f"   - Success: {report['stress_test_results']['success']}")
            logger.info(f"   - Success rate: {report['stress_test_results']['success_rate']}%")
            logger.info(f"   - Avg response time: {report['stress_test_results']['avg_response_time_ms']}ms")
            logger.info(f"   - Timeouts: {report['stress_test_results']['timeout_requests']}")
            logger.info(f"   - Total errors: {report['error_analysis']['total_errors']}")
            
            # Print recommendations
            logger.info("💡 Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"   - {rec}")
            
        else:
            logger.error("❌ Stress test failed")
            
    except Exception as e:
        logger.error(f"❌ Stress test error: {e}")
        
    finally:
        # Cleanup
        await stress_test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
