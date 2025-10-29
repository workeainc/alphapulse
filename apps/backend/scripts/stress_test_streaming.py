#!/usr/bin/env python3
"""
Critical Stress Testing for Streaming Infrastructure
Validates system performance under high load conditions
"""
import asyncio
import time
import random
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.streaming.stream_processor import StreamProcessor
from src.streaming.stream_metrics import StreamMetrics
from src.streaming.backpressure_handler import BackpressureHandler
from src.core.config import STREAMING_CONFIG, settings
from src.database.connection import TimescaleDBConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingStressTest:
    def __init__(self):
        self.stream_processor = None
        self.stream_metrics = None
        self.backpressure_handler = None
        self.db_connection = None
        self.test_results = {
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0,
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'min_latency_ms': float('inf'),
            'throughput_msgs_per_sec': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'backpressure_events': 0,
            'error_rate_percent': 0
        }
        self.latencies = []
        self.start_time = None
        self.end_time = None

    async def initialize_components(self):
        """Initialize all streaming components"""
        logger.info("🔧 Initializing streaming components for stress test...")
        try:
            # Initialize components
            self.stream_processor = StreamProcessor(STREAMING_CONFIG)
            self.stream_metrics = StreamMetrics(STREAMING_CONFIG)
            self.backpressure_handler = BackpressureHandler(STREAMING_CONFIG)
            
            # Initialize components
            await self.stream_processor.initialize()
            await self.stream_metrics.initialize()
            await self.backpressure_handler.initialize()
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection({})
            await self.db_connection.initialize()
            
            logger.info("✅ All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False

    def generate_test_symbols(self, count=1000):
        """Generate test symbols for stress testing"""
        symbols = []
        for i in range(count):
            # Generate realistic symbol names
            if i < 500:
                # Crypto pairs
                base = random.choice(['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'LTC', 'BCH'])
                quote = random.choice(['USDT', 'USD', 'EUR', 'GBP'])
                symbol = f"{base}{quote}"
            elif i < 800:
                # Stock symbols
                symbol = f"STOCK{i:03d}"
            else:
                # Forex pairs
                base = random.choice(['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF'])
                quote = random.choice(['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF'])
                if base != quote:
                    symbol = f"{base}{quote}"
                else:
                    symbol = f"FX{i:03d}"
            symbols.append(symbol)
        return symbols

    def generate_market_data(self, symbol, timestamp):
        """Generate realistic market data"""
        base_price = random.uniform(1, 100000)
        price_change = random.uniform(-0.05, 0.05)  # ±5% change
        current_price = base_price * (1 + price_change)
        volume = random.uniform(0.1, 1000)
        
        return {
            'message_id': f"{symbol}_{timestamp.timestamp()}_{random.randint(1000, 9999)}",
            'symbol': symbol,
            'data_type': 'tick',
            'source': 'stress_test',
            'data': {
                'price': round(current_price, 8),
                'volume': round(volume, 8),
                'bid': round(current_price * 0.999, 8),
                'ask': round(current_price * 1.001, 8),
                'timestamp': timestamp.isoformat()
            },
            'timestamp': timestamp
        }

    async def simulate_peak_market_conditions(self, symbols, duration_seconds=300):
        """Simulate peak market conditions with high-frequency data"""
        logger.info(f"🚀 Starting peak market simulation with {len(symbols)} symbols for {duration_seconds} seconds")
        
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(seconds=duration_seconds)
        
        # Simulate different market conditions
        phases = [
            ('normal', 1, 60),      # Normal trading: 1 msg/sec per symbol
            ('high_activity', 5, 60),  # High activity: 5 msg/sec per symbol
            ('peak', 10, 60),       # Peak activity: 10 msg/sec per symbol
            ('extreme', 20, 60),    # Extreme activity: 20 msg/sec per symbol
            ('normal', 1, 60)       # Back to normal
        ]
        
        total_messages = 0
        
        for phase_name, messages_per_sec, phase_duration in phases:
            logger.info(f"📈 Phase: {phase_name} - {messages_per_sec} msg/sec per symbol for {phase_duration}s")
            phase_end = datetime.now() + timedelta(seconds=phase_duration)
            
            while datetime.now() < phase_end and datetime.now() < end_time:
                # Generate messages for all symbols
                tasks = []
                for symbol in symbols:
                    for _ in range(messages_per_sec):
                        data = self.generate_market_data(symbol, datetime.now())
                        tasks.append(self.process_message_with_timing(data))
                
                # Process messages concurrently
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    total_messages += len(results)
                    
                    # Update metrics
                    for result in results:
                        if isinstance(result, dict) and 'latency' in result:
                            self.latencies.append(result['latency'])
                            self.test_results['total_messages'] += 1
                            if result.get('success'):
                                self.test_results['successful_messages'] += 1
                            else:
                                self.test_results['failed_messages'] += 1
                
                # Small delay to control rate
                await asyncio.sleep(1)
        
        self.end_time = datetime.now()
        logger.info(f"✅ Peak market simulation completed. Total messages: {total_messages}")

    async def process_message_with_timing(self, message):
        """Process message and measure latency"""
        start_time = time.time()
        try:
            result = await self.stream_processor.process_message(message)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                'success': True,
                'latency': latency,
                'result': result
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"❌ Message processing failed: {e}")
            return {
                'success': False,
                'latency': latency,
                'error': str(e)
            }

    async def collect_system_metrics(self):
        """Collect system performance metrics"""
        logger.info("📊 Collecting system performance metrics...")
        
        try:
            # Get streaming metrics
            metrics = await self.stream_metrics.get_current_metrics()
            
            # Get backpressure status
            backpressure_status = await self.backpressure_handler.get_status()
            
            # Calculate test statistics
            if self.latencies:
                self.test_results['avg_latency_ms'] = sum(self.latencies) / len(self.latencies)
                self.test_results['max_latency_ms'] = max(self.latencies)
                self.test_results['min_latency_ms'] = min(self.latencies)
            
            # Calculate throughput
            if self.start_time and self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
                self.test_results['throughput_msgs_per_sec'] = self.test_results['total_messages'] / duration
            
            # Calculate error rate
            if self.test_results['total_messages'] > 0:
                self.test_results['error_rate_percent'] = (
                    self.test_results['failed_messages'] / self.test_results['total_messages']
                ) * 100
            
            # Add system metrics
            self.test_results['memory_usage_mb'] = metrics.get('memory_usage_mb', 0)
            self.test_results['cpu_usage_percent'] = metrics.get('cpu_usage_percent', 0)
            self.test_results['backpressure_events'] = backpressure_status.get('backpressure_events', 0)
            
            logger.info("✅ System metrics collected successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect system metrics: {e}")

    async def validate_database_performance(self):
        """Validate TimescaleDB performance under load"""
        logger.info("🗄️ Validating TimescaleDB performance...")
        
        try:
            # Test query performance
            start_time = time.time()
            
            # Test streaming tables queries
            async with self.db_connection.get_session() as session:
                # Test stream_messages table
                result = await session.execute("""
                    SELECT COUNT(*) as count, 
                           AVG(EXTRACT(EPOCH FROM (NOW() - timestamp))) as avg_age_seconds
                    FROM stream_messages 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                stream_stats = result.fetchone()
                
                # Test normalized_data table
                result = await session.execute("""
                    SELECT COUNT(*) as count
                    FROM normalized_data 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                normalized_stats = result.fetchone()
                
                # Test realtime_candles table
                result = await session.execute("""
                    SELECT COUNT(*) as count
                    FROM realtime_candles 
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
                """)
                candles_stats = result.fetchone()
            
            query_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Database queries completed in {query_time:.2f}ms")
            logger.info(f"   - Stream messages: {stream_stats.count if stream_stats else 0}")
            logger.info(f"   - Normalized data: {normalized_stats.count if normalized_stats else 0}")
            logger.info(f"   - Real-time candles: {candles_stats.count if candles_stats else 0}")
            
            return query_time < 1000  # Should complete within 1 second
            
        except Exception as e:
            logger.error(f"❌ Database performance validation failed: {e}")
            return False

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📋 Generating stress test report...")
        
        report = {
            'test_summary': {
                'test_name': 'Streaming Infrastructure Stress Test',
                'test_date': datetime.now().isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
                'symbols_tested': 1000,
                'phases_simulated': ['normal', 'high_activity', 'peak', 'extreme', 'normal']
            },
            'performance_metrics': self.test_results,
            'pass_fail_criteria': {
                'latency_target': 'avg_latency_ms < 100',
                'throughput_target': 'throughput_msgs_per_sec > 1000',
                'error_rate_target': 'error_rate_percent < 1',
                'memory_target': 'memory_usage_mb < 2048',
                'database_target': 'query_time_ms < 1000'
            },
            'recommendations': []
        }
        
        # Evaluate results against targets
        if self.test_results['avg_latency_ms'] > 100:
            report['recommendations'].append("⚠️ Average latency exceeds 100ms target - consider optimization")
        
        if self.test_results['throughput_msgs_per_sec'] < 1000:
            report['recommendations'].append("⚠️ Throughput below 1000 msg/sec target - consider scaling")
        
        if self.test_results['error_rate_percent'] > 1:
            report['recommendations'].append("❌ Error rate exceeds 1% target - investigate failures")
        
        if self.test_results['memory_usage_mb'] > 2048:
            report['recommendations'].append("⚠️ Memory usage high - consider memory optimization")
        
        if not report['recommendations']:
            report['recommendations'].append("✅ All performance targets met - system ready for production")
        
        return report

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up test resources...")
        try:
            if self.stream_processor:
                await self.stream_processor.shutdown()
            if self.stream_metrics:
                await self.stream_metrics.shutdown()
            if self.backpressure_handler:
                await self.backpressure_handler.shutdown()
            if self.db_connection:
                await self.db_connection.close()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main stress test execution"""
    logger.info("=" * 80)
    logger.info("🔥 STREAMING INFRASTRUCTURE STRESS TEST")
    logger.info("=" * 80)
    
    stress_test = StreamingStressTest()
    
    try:
        # Initialize components
        if not await stress_test.initialize_components():
            logger.error("❌ Failed to initialize components - aborting test")
            return False
        
        # Generate test symbols
        symbols = stress_test.generate_test_symbols(1000)
        logger.info(f"📊 Generated {len(symbols)} test symbols")
        
        # Run peak market simulation
        await stress_test.simulate_peak_market_conditions(symbols, duration_seconds=300)
        
        # Collect system metrics
        await stress_test.collect_system_metrics()
        
        # Validate database performance
        db_performance_ok = await stress_test.validate_database_performance()
        
        # Generate report
        report = stress_test.generate_test_report()
        
        # Print results
        logger.info("=" * 80)
        logger.info("📊 STRESS TEST RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Total Messages: {report['performance_metrics']['total_messages']:,}")
        logger.info(f"Successful: {report['performance_metrics']['successful_messages']:,}")
        logger.info(f"Failed: {report['performance_metrics']['failed_messages']:,}")
        logger.info(f"Error Rate: {report['performance_metrics']['error_rate_percent']:.2f}%")
        logger.info(f"Average Latency: {report['performance_metrics']['avg_latency_ms']:.2f}ms")
        logger.info(f"Max Latency: {report['performance_metrics']['max_latency_ms']:.2f}ms")
        logger.info(f"Throughput: {report['performance_metrics']['throughput_msgs_per_sec']:.2f} msg/sec")
        logger.info(f"Memory Usage: {report['performance_metrics']['memory_usage_mb']:.2f} MB")
        logger.info(f"CPU Usage: {report['performance_metrics']['cpu_usage_percent']:.2f}%")
        logger.info(f"Backpressure Events: {report['performance_metrics']['backpressure_events']}")
        logger.info(f"Database Performance: {'✅ PASS' if db_performance_ok else '❌ FAIL'}")
        
        logger.info("\n📋 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            logger.info(f"  {rec}")
        
        # Save detailed report
        report_file = backend_path / "stress_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        # Determine overall success
        success = (
            report['performance_metrics']['error_rate_percent'] < 1 and
            report['performance_metrics']['avg_latency_ms'] < 100 and
            db_performance_ok
        )
        
        if success:
            logger.info("🎉 STRESS TEST PASSED - System ready for production!")
        else:
            logger.error("❌ STRESS TEST FAILED - System needs optimization")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Stress test failed with exception: {e}")
        return False
    finally:
        await stress_test.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
