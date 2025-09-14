#!/usr/bin/env python3
"""
Test Real-Time Signal Processing System
Comprehensive validation of the enhanced AlphaPlus real-time system
"""

import asyncio
import logging
import asyncpg
import json
from datetime import datetime, timedelta
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeSystemTester:
    """Test the real-time signal processing system"""
    
    def __init__(self):
        self.db_pool = None
        self.test_results = {}
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='alpha_emon',
                password='Emon_@17711',
                database='alphapulse'
            )
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def test_database_schema(self):
        """Test database schema and tables"""
        try:
            logger.info("🔍 Testing database schema...")
            
            async with self.db_pool.acquire() as conn:
                # Test signals table
                signals_columns = await conn.fetch("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'signals' 
                    ORDER BY column_name
                """)
                
                required_columns = [
                    'confidence', 'health_score', 'ensemble_votes', 'confidence_breakdown',
                    'news_impact_score', 'sentiment_score', 'signal_priority', 'is_active',
                    'expires_at', 'cancelled_reason', 'real_time_processing_time_ms',
                    'notification_sent', 'external_alert_sent'
                ]
                
                existing_columns = [col['column_name'] for col in signals_columns]
                missing_columns = [col for col in required_columns if col not in existing_columns]
                
                if missing_columns:
                    logger.error(f"❌ Missing columns in signals table: {missing_columns}")
                    self.test_results['database_schema'] = False
                    return False
                
                # Test real-time tables
                tables_to_check = [
                    'real_time_signal_queue',
                    'signal_notifications', 
                    'ensemble_model_votes',
                    'performance_metrics'
                ]
                
                for table in tables_to_check:
                    exists = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = $1
                        )
                    """, table)
                    
                    if not exists:
                        logger.error(f"❌ Table {table} does not exist")
                        self.test_results['database_schema'] = False
                        return False
                
                logger.info("✅ Database schema validation passed")
                self.test_results['database_schema'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Database schema test failed: {e}")
            self.test_results['database_schema'] = False
            return False
    
    async def test_signal_insertion(self):
        """Test inserting a test signal with real-time fields"""
        try:
            logger.info("🔍 Testing signal insertion...")
            
            async with self.db_pool.acquire() as conn:
                # Create test signal data
                test_signal = {
                    'signal_id': f'test_signal_{datetime.now().timestamp()}',
                    'symbol': 'BTC/USDT',
                    'timeframe': '1h',
                    'direction': 'long',
                    'confidence': 0.89,
                    'health_score': 0.87,
                    'ensemble_votes': json.dumps({
                        'technical_ml': {'vote_confidence': 0.85, 'vote_direction': 'long', 'model_weight': 0.4},
                        'price_action_ml': {'vote_confidence': 0.82, 'vote_direction': 'long', 'model_weight': 0.2},
                        'sentiment_score': {'vote_confidence': 0.78, 'vote_direction': 'long', 'model_weight': 0.2},
                        'market_regime': {'vote_confidence': 0.90, 'vote_direction': 'long', 'model_weight': 0.2}
                    }),
                    'confidence_breakdown': json.dumps({
                        'pattern_analysis': 0.85,
                        'technical_analysis': 0.82,
                        'sentiment_analysis': 0.78,
                        'volume_analysis': 0.88,
                        'market_regime_analysis': 0.90,
                        'risk_reward_ratio': 0.85
                    }),
                    'news_impact_score': 0.75,
                    'sentiment_score': 0.78,
                    'signal_priority': 85,
                    'is_active': True,
                    'expires_at': datetime.now() + timedelta(hours=2),
                    'real_time_processing_time_ms': 125.5,
                    'entry_price': 45000.0,
                    'stop_loss': 44500.0,
                    'tp1': 45500.0,
                    'ts': datetime.now()
                }
                
                # Insert test signal
                await conn.execute("""
                    INSERT INTO signals (
                        signal_id, symbol, timeframe, direction, confidence, health_score,
                        ensemble_votes, confidence_breakdown, news_impact_score, sentiment_score,
                        signal_priority, is_active, expires_at, real_time_processing_time_ms,
                        entry_price, stop_loss, tp1, ts
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                    )
                """, *test_signal.values())
                
                # Verify insertion
                inserted_signal = await conn.fetchrow("""
                    SELECT * FROM signals WHERE signal_id = $1
                """, test_signal['signal_id'])
                
                if not inserted_signal:
                    logger.error("❌ Test signal was not inserted")
                    self.test_results['signal_insertion'] = False
                    return False
                
                logger.info(f"✅ Test signal inserted successfully: {test_signal['signal_id']}")
                self.test_results['signal_insertion'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Signal insertion test failed: {e}")
            self.test_results['signal_insertion'] = False
            return False
    
    async def test_signal_queue(self):
        """Test real-time signal queue functionality"""
        try:
            logger.info("🔍 Testing signal queue...")
            
            async with self.db_pool.acquire() as conn:
                # Add test signal to queue
                test_queue_item = {
                    'signal_id': f'test_queue_{datetime.now().timestamp()}',
                    'symbol': 'ETH/USDT',
                    'priority': 90,
                    'confidence': 0.92,
                    'health_score': 0.89,
                    'status': 'pending'
                }
                
                await conn.execute("""
                    INSERT INTO real_time_signal_queue 
                    (signal_id, symbol, priority, confidence, health_score, status)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, *test_queue_item.values())
                
                # Test queue retrieval
                queue_items = await conn.fetch("""
                    SELECT * FROM real_time_signal_queue 
                    WHERE status = 'pending' 
                    ORDER BY priority DESC, confidence DESC
                """)
                
                if not queue_items:
                    logger.error("❌ No items found in signal queue")
                    self.test_results['signal_queue'] = False
                    return False
                
                # Test queue processing
                await conn.execute("""
                    UPDATE real_time_signal_queue 
                    SET status = 'processed', processed_at = NOW()
                    WHERE signal_id = $1
                """, test_queue_item['signal_id'])
                
                logger.info("✅ Signal queue test passed")
                self.test_results['signal_queue'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Signal queue test failed: {e}")
            self.test_results['signal_queue'] = False
            return False
    
    async def test_notifications(self):
        """Test notification system"""
        try:
            logger.info("🔍 Testing notification system...")
            
            async with self.db_pool.acquire() as conn:
                # Create test notification
                test_notification = {
                    'signal_id': f'test_notification_{datetime.now().timestamp()}',
                    'notification_type': 'signal_generated',
                    'channel': 'dashboard',
                    'delivery_status': 'sent',
                    'delivery_time_ms': 45.2
                }
                
                await conn.execute("""
                    INSERT INTO signal_notifications 
                    (signal_id, notification_type, channel, delivery_status, delivery_time_ms)
                    VALUES ($1, $2, $3, $4, $5)
                """, *test_notification.values())
                
                # Test notification retrieval
                notifications = await conn.fetch("""
                    SELECT * FROM signal_notifications 
                    WHERE sent_at > NOW() - INTERVAL '1 hour'
                    ORDER BY sent_at DESC
                """)
                
                if not notifications:
                    logger.error("❌ No notifications found")
                    self.test_results['notifications'] = False
                    return False
                
                logger.info("✅ Notification system test passed")
                self.test_results['notifications'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Notification test failed: {e}")
            self.test_results['notifications'] = False
            return False
    
    async def test_ensemble_votes(self):
        """Test ensemble model voting system"""
        try:
            logger.info("🔍 Testing ensemble voting system...")
            
            async with self.db_pool.acquire() as conn:
                # Create test ensemble votes
                test_signal_id = f'test_ensemble_{datetime.now().timestamp()}'
                
                test_votes = [
                    (test_signal_id, 'technical_ml', 0.85, 'long', 0.4, 25.5),
                    (test_signal_id, 'price_action_ml', 0.82, 'long', 0.2, 18.2),
                    (test_signal_id, 'sentiment_score', 0.78, 'long', 0.2, 12.8),
                    (test_signal_id, 'market_regime', 0.90, 'long', 0.2, 22.1)
                ]
                
                for vote in test_votes:
                    await conn.execute("""
                        INSERT INTO ensemble_model_votes 
                        (signal_id, model_name, vote_confidence, vote_direction, model_weight, processing_time_ms)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, vote[0], vote[1], vote[2], vote[3], vote[4], vote[5])
                
                # Test vote retrieval
                votes = await conn.fetch("""
                    SELECT * FROM ensemble_model_votes 
                    WHERE signal_id = $1
                    ORDER BY model_weight DESC
                """, test_signal_id)
                
                if len(votes) != 4:
                    logger.error(f"❌ Expected 4 votes, got {len(votes)}")
                    self.test_results['ensemble_votes'] = False
                    return False
                
                logger.info("✅ Ensemble voting system test passed")
                self.test_results['ensemble_votes'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Ensemble voting test failed: {e}")
            self.test_results['ensemble_votes'] = False
            return False
    
    async def test_performance_metrics(self):
        """Test performance metrics tracking"""
        try:
            logger.info("🔍 Testing performance metrics...")
            
            async with self.db_pool.acquire() as conn:
                # Create test performance metrics
                test_metrics = {
                    'test_run_id': f'perf_test_{datetime.now().timestamp()}',
                    'latency_avg_ms': 125.5,
                    'latency_max_ms': 250.0,
                    'throughput_signals_sec': 2.5,
                    'accuracy': 0.87,
                    'filter_rate': 0.15,
                    'real_time_latency_avg_ms': 45.2,
                    'real_time_latency_max_ms': 89.5,
                    'signal_generation_rate_per_min': 3.2,
                    'ensemble_accuracy': 0.89,
                    'news_reaction_time_ms': 125.0,
                    'notification_delivery_time_ms': 45.5
                }
                
                await conn.execute("""
                    INSERT INTO performance_metrics (
                        test_run_id, latency_avg_ms, latency_max_ms, throughput_signals_sec,
                        accuracy, filter_rate, real_time_latency_avg_ms, real_time_latency_max_ms,
                        signal_generation_rate_per_min, ensemble_accuracy, news_reaction_time_ms,
                        notification_delivery_time_ms
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                    )
                """, *test_metrics.values())
                
                # Test metrics retrieval
                metrics = await conn.fetchrow("""
                    SELECT * FROM performance_metrics 
                    WHERE test_run_id = $1
                """, test_metrics['test_run_id'])
                
                if not metrics:
                    logger.error("❌ Performance metrics not found")
                    self.test_results['performance_metrics'] = False
                    return False
                
                logger.info("✅ Performance metrics test passed")
                self.test_results['performance_metrics'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            self.test_results['performance_metrics'] = False
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting comprehensive real-time system tests...")
        
        # Initialize
        if not await self.initialize():
            logger.error("❌ Failed to initialize test system")
            return
        
        # Run tests
        tests = [
            ('Database Schema', self.test_database_schema),
            ('Signal Insertion', self.test_signal_insertion),
            ('Signal Queue', self.test_signal_queue),
            ('Notifications', self.test_notifications),
            ('Ensemble Votes', self.test_ensemble_votes),
            ('Performance Metrics', self.test_performance_metrics)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        # Print results
        self.print_results()
        
        # Cleanup
        await self.cleanup()
    
    def print_results(self):
        """Print test results"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 REAL-TIME SYSTEM TEST RESULTS")
        logger.info(f"{'='*60}")
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title():<25} {status}")
            if result:
                passed += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Real-time system is ready.")
        else:
            logger.info("⚠️ Some tests failed. Please check the implementation.")
        logger.info(f"{'='*60}")
    
    async def cleanup(self):
        """Clean up test data"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    # Clean up test data
                    await conn.execute("""
                        DELETE FROM signals WHERE signal_id LIKE 'test_%'
                    """)
                    
                    await conn.execute("""
                        DELETE FROM real_time_signal_queue WHERE signal_id LIKE 'test_%'
                    """)
                    
                    await conn.execute("""
                        DELETE FROM signal_notifications WHERE signal_id LIKE 'test_%'
                    """)
                    
                    await conn.execute("""
                        DELETE FROM ensemble_model_votes WHERE signal_id LIKE 'test_%'
                    """)
                    
                    await conn.execute("""
                        DELETE FROM performance_metrics WHERE test_run_id LIKE 'test_%'
                    """)
                
                await self.db_pool.close()
                logger.info("✅ Test cleanup completed")
                
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main test function"""
    tester = RealTimeSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
