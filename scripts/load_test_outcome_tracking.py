#!/usr/bin/env python3
"""
Load Testing Script for Phase 2 Outcome Tracking
Simulates 1000+ signals/sec load conditions to validate performance
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
from typing import Dict, List, Any
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

class LoadTestOutcomeTracking:
    """Load testing class for Phase 2 outcome tracking system"""

    def __init__(self):
        self.outcome_tracker = None
        self.tp_sl_detector = None
        self.performance_analyzer = None
        self.db_connection = None
        
        # Load test configuration
        self.test_duration = 60  # 60 seconds
        self.signals_per_second = 1000
        self.concurrent_workers = 10
        self.batch_size = 100
        
        # Performance metrics
        self.start_time = None
        self.end_time = None
        self.total_signals_processed = 0
        self.total_tp_sl_checks = 0
        self.processing_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.errors = []
        
        # Test results
        self.test_results = {
            'load_test': {
                'total_signals': 0,
                'signals_per_second': 0,
                'avg_processing_time_ms': 0,
                'max_processing_time_ms': 0,
                'min_processing_time_ms': 0,
                'p95_processing_time_ms': 0,
                'p99_processing_time_ms': 0,
                'error_rate': 0,
                'memory_peak_mb': 0,
                'cpu_peak_percent': 0,
                'gc_collections': 0,
                'success': False
            },
            'database_performance': {
                'total_queries': 0,
                'avg_query_time_ms': 0,
                'max_query_time_ms': 0,
                'connection_pool_usage': 0,
                'success': False
            },
            'system_resources': {
                'memory_usage_mb': [],
                'cpu_usage_percent': [],
                'disk_io_mb': 0,
                'network_io_mb': 0,
                'success': False
            }
        }

    async def initialize_components(self):
        """Initialize all outcome tracking components"""
        try:
            logger.info("Initializing outcome tracking components for load test...")
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD,
                'pool_size': 50,  # Increased for load testing
                'max_overflow': 100
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

    def generate_test_signal(self, signal_id: str) -> Dict[str, Any]:
        """Generate a test signal for load testing"""
        return {
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
                'test_signal': True,
                'load_test_id': f"load_test_{int(time.time())}",
                'batch_id': signal_id.split('_')[0]
            }
        }

    def generate_test_price_data(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test price data for TP/SL detection"""
        base_price = signal['price']
        return {
            'symbol': signal['symbol'],
            'timestamp': datetime.now(timezone.utc),
            'price': base_price + random.uniform(-base_price * 0.1, base_price * 0.1),
            'volume': random.uniform(1000, 100000),
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'bid_volume': random.uniform(500, 50000),
            'ask_volume': random.uniform(500, 50000)
        }

    async def process_signal_batch(self, batch_id: int, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of signals"""
        batch_start_time = time.time()
        batch_results = {
            'batch_id': batch_id,
            'signals_processed': 0,
            'tp_sl_checks': 0,
            'processing_time_ms': 0,
            'errors': []
        }
        
        try:
            for signal in signals:
                signal_start_time = time.time()
                
                # Track signal
                try:
                    outcome = await self.outcome_tracker.track_signal(signal)
                    batch_results['signals_processed'] += 1
                except Exception as e:
                    batch_results['errors'].append(f"Signal tracking error: {e}")
                
                # Check TP/SL
                try:
                    price_data = self.generate_test_price_data(signal)
                    hits = await self.tp_sl_detector.track_position(signal['signal_id'], price_data)
                    batch_results['tp_sl_checks'] += 1
                except Exception as e:
                    batch_results['errors'].append(f"TP/SL check error: {e}")
                
                # Record processing time
                signal_processing_time = (time.time() - signal_start_time) * 1000
                self.processing_times.append(signal_processing_time)
                
            batch_results['processing_time_ms'] = (time.time() - batch_start_time) * 1000
            
        except Exception as e:
            batch_results['errors'].append(f"Batch processing error: {e}")
        
        return batch_results

    async def monitor_system_resources(self):
        """Monitor system resources during load test"""
        while self.start_time and time.time() < self.end_time:
            try:
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage.append(memory_info.used / 1024 / 1024)  # MB
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                
                # Force garbage collection
                gc.collect()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")

    async def run_load_test(self):
        """Run the main load test"""
        logger.info(f"🚀 Starting load test: {self.signals_per_second} signals/sec for {self.test_duration} seconds")
        
        # Initialize components
        if not await self.initialize_components():
            return False
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self.monitor_system_resources())
        
        # Record start time
        self.start_time = time.time()
        self.end_time = self.start_time + self.test_duration
        
        # Calculate signals per batch
        total_signals = self.signals_per_second * self.test_duration
        signals_per_batch = self.batch_size
        total_batches = total_signals // signals_per_batch
        
        logger.info(f"📊 Load test parameters:")
        logger.info(f"   - Total signals: {total_signals}")
        logger.info(f"   - Signals per batch: {signals_per_batch}")
        logger.info(f"   - Total batches: {total_batches}")
        logger.info(f"   - Concurrent workers: {self.concurrent_workers}")
        
        # Create signal batches
        batches = []
        for batch_id in range(total_batches):
            batch_signals = []
            for i in range(signals_per_batch):
                signal_id = f"batch_{batch_id}_signal_{i}"
                signal = self.generate_test_signal(signal_id)
                batch_signals.append(signal)
            batches.append((batch_id, batch_signals))
        
        # Process batches with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrent_workers)
        
        async def process_batch_with_semaphore(batch_id: int, signals: List[Dict[str, Any]]):
            async with semaphore:
                return await self.process_signal_batch(batch_id, signals)
        
        # Start batch processing
        batch_tasks = []
        for batch_id, signals in batches:
            task = asyncio.create_task(process_batch_with_semaphore(batch_id, signals))
            batch_tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Record end time
        self.end_time = time.time()
        
        # Cancel monitoring task
        monitor_task.cancel()
        
        # Process results
        await self.process_load_test_results(batch_results)
        
        return True

    async def process_load_test_results(self, batch_results: List[Dict[str, Any]]):
        """Process and analyze load test results"""
        logger.info("📈 Processing load test results...")
        
        # Calculate performance metrics
        total_signals_processed = sum(result.get('signals_processed', 0) for result in batch_results if isinstance(result, dict))
        total_tp_sl_checks = sum(result.get('tp_sl_checks', 0) for result in batch_results if isinstance(result, dict))
        total_errors = sum(len(result.get('errors', [])) for result in batch_results if isinstance(result, dict))
        
        # Calculate processing time statistics
        if self.processing_times:
            avg_processing_time = statistics.mean(self.processing_times)
            max_processing_time = max(self.processing_times)
            min_processing_time = min(self.processing_times)
            p95_processing_time = statistics.quantiles(self.processing_times, n=20)[18]  # 95th percentile
            p99_processing_time = statistics.quantiles(self.processing_times, n=100)[98]  # 99th percentile
        else:
            avg_processing_time = max_processing_time = min_processing_time = p95_processing_time = p99_processing_time = 0
        
        # Calculate system resource usage
        memory_peak = max(self.memory_usage) if self.memory_usage else 0
        cpu_peak = max(self.cpu_usage) if self.cpu_usage else 0
        
        # Calculate actual signals per second
        actual_duration = self.end_time - self.start_time
        actual_signals_per_second = total_signals_processed / actual_duration if actual_duration > 0 else 0
        
        # Calculate error rate
        error_rate = (total_errors / total_signals_processed * 100) if total_signals_processed > 0 else 0
        
        # Update test results
        self.test_results['load_test'].update({
            'total_signals': total_signals_processed,
            'signals_per_second': round(actual_signals_per_second, 2),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'max_processing_time_ms': round(max_processing_time, 2),
            'min_processing_time_ms': round(min_processing_time, 2),
            'p95_processing_time_ms': round(p95_processing_time, 2),
            'p99_processing_time_ms': round(p99_processing_time, 2),
            'error_rate': round(error_rate, 2),
            'memory_peak_mb': round(memory_peak, 2),
            'cpu_peak_percent': round(cpu_peak, 2),
            'gc_collections': gc.get_count()[0],
            'success': error_rate < 5 and actual_signals_per_second >= self.signals_per_second * 0.9
        })
        
        # Log results
        logger.info("📊 Load Test Results:")
        logger.info(f"   ✅ Total signals processed: {total_signals_processed}")
        logger.info(f"   ✅ Actual signals/sec: {actual_signals_per_second:.2f}")
        logger.info(f"   ✅ TP/SL checks: {total_tp_sl_checks}")
        logger.info(f"   ✅ Avg processing time: {avg_processing_time:.2f}ms")
        logger.info(f"   ✅ P95 processing time: {p95_processing_time:.2f}ms")
        logger.info(f"   ✅ P99 processing time: {p99_processing_time:.2f}ms")
        logger.info(f"   ✅ Error rate: {error_rate:.2f}%")
        logger.info(f"   ✅ Memory peak: {memory_peak:.2f}MB")
        logger.info(f"   ✅ CPU peak: {cpu_peak:.2f}%")
        logger.info(f"   ✅ Test success: {self.test_results['load_test']['success']}")

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
        """Generate comprehensive load test report"""
        report = {
            'test_summary': {
                'test_name': 'Phase 2 Outcome Tracking Load Test',
                'test_duration_seconds': self.test_duration,
                'target_signals_per_second': self.signals_per_second,
                'concurrent_workers': self.concurrent_workers,
                'batch_size': self.batch_size,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            },
            'performance_metrics': self.test_results['load_test'],
            'system_resources': {
                'memory_usage_mb': self.memory_usage,
                'cpu_usage_percent': self.cpu_usage,
                'memory_peak_mb': max(self.memory_usage) if self.memory_usage else 0,
                'cpu_peak_percent': max(self.cpu_usage) if self.cpu_usage else 0
            },
            'recommendations': self.generate_recommendations()
        }
        
        return report

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        if self.test_results['load_test']['signals_per_second'] < self.signals_per_second * 0.9:
            recommendations.append("⚠️ System cannot handle target load. Consider optimizing database queries and connection pooling.")
        
        if self.test_results['load_test']['avg_processing_time_ms'] > 100:
            recommendations.append("⚠️ Average processing time is high. Consider optimizing signal processing logic.")
        
        if self.test_results['load_test']['p99_processing_time_ms'] > 500:
            recommendations.append("⚠️ 99th percentile processing time is high. Consider implementing caching and optimization.")
        
        if self.test_results['load_test']['error_rate'] > 1:
            recommendations.append("⚠️ Error rate is high. Review error handling and system stability.")
        
        # Memory recommendations
        if self.test_results['load_test']['memory_peak_mb'] > 1000:
            recommendations.append("⚠️ Memory usage is high. Consider implementing memory management and cleanup.")
        
        # CPU recommendations
        if self.test_results['load_test']['cpu_peak_percent'] > 80:
            recommendations.append("⚠️ CPU usage is high. Consider optimizing algorithms and implementing parallel processing.")
        
        # Success recommendations
        if self.test_results['load_test']['success']:
            recommendations.append("✅ System meets performance requirements for production deployment.")
        else:
            recommendations.append("❌ System does not meet performance requirements. Address issues before production deployment.")
        
        return recommendations

async def main():
    """Main function to run the load test"""
    logger.info("🚀 Starting Phase 2 Outcome Tracking Load Test")
    
    # Create load test instance
    load_test = LoadTestOutcomeTracking()
    
    try:
        # Run load test
        success = await load_test.run_load_test()
        
        if success:
            # Generate and save report
            report = load_test.generate_report()
            
            # Save report to file
            report_file = f"load_test_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Load test report saved to: {report_file}")
            
            # Print summary
            logger.info("🎯 Load Test Summary:")
            logger.info(f"   - Success: {report['performance_metrics']['success']}")
            logger.info(f"   - Signals/sec: {report['performance_metrics']['signals_per_second']}")
            logger.info(f"   - Error rate: {report['performance_metrics']['error_rate']}%")
            logger.info(f"   - Memory peak: {report['performance_metrics']['memory_peak_mb']}MB")
            logger.info(f"   - CPU peak: {report['performance_metrics']['cpu_peak_percent']}%")
            
            # Print recommendations
            logger.info("💡 Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"   - {rec}")
            
        else:
            logger.error("❌ Load test failed")
            
    except Exception as e:
        logger.error(f"❌ Load test error: {e}")
        
    finally:
        # Cleanup
        await load_test.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
