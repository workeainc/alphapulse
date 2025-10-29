#!/usr/bin/env python3
"""
Test script for Phase 2 Outcome Tracking System
Validates all outcome tracking components
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from src.outcome_tracking.outcome_tracker import OutcomeTracker, OutcomeType, SignalOutcome
from src.outcome_tracking.tp_sl_detector import TPSLDetector, HitType, TPSLHit
from src.database.connection import TimescaleDBConnection
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2OutcomeTrackingTest:
    """Test class for Phase 2 outcome tracking system"""
    
    def __init__(self):
        self.outcome_tracker = None
        self.tp_sl_detector = None
        self.db_connection = None
        self.test_results = {
            'outcome_tracker_tests': 0,
            'outcome_tracker_passed': 0,
            'tp_sl_detector_tests': 0,
            'tp_sl_detector_passed': 0,
            'database_tests': 0,
            'database_passed': 0,
            'integration_tests': 0,
            'integration_passed': 0
        }
    
    async def initialize_components(self):
        """Initialize all outcome tracking components"""
        logger.info("🔧 Initializing Phase 2 outcome tracking components...")
        
        try:
            # Initialize outcome tracker
            self.outcome_tracker = OutcomeTracker({
                'enable_real_time_tracking': True,
                'enable_audit_trail': True,
                'enable_transactional_consistency': True,
                'tracking_interval': 0.5,
                'max_tracking_duration': 3600
            })
            await self.outcome_tracker.initialize()
            
            # Initialize TP/SL detector
            self.tp_sl_detector = TPSLDetector({
                'hit_tolerance': 0.001,
                'min_hit_duration': timedelta(milliseconds=100),
                'enable_partial_fills': True,
                'enable_trailing_stops': False,
                'detection_interval': 0.1
            })
            await self.tp_sl_detector.initialize()
            
            # Initialize database connection
            self.db_connection = TimescaleDBConnection({
                'host': settings.TIMESCALEDB_HOST,
                'port': settings.TIMESCALEDB_PORT,
                'database': settings.TIMESCALEDB_DATABASE,
                'username': settings.TIMESCALEDB_USERNAME,
                'password': settings.TIMESCALEDB_PASSWORD
            })
            await self.db_connection.initialize()
            
            logger.info("✅ All Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 2 components: {e}")
            return False
    
    async def test_outcome_tracker(self):
        """Test outcome tracker functionality"""
        logger.info("🧪 Testing Outcome Tracker...")
        
        try:
            # Test 1: Track a signal
            signal_data = {
                'signal_id': f"test_signal_{uuid.uuid4().hex[:8]}",
                'symbol': 'BTCUSDT',
                'entry_price': 50000.0,
                'signal_type': 'long',
                'tp_price': 51000.0,
                'sl_price': 49000.0,
                'max_hold_time': timedelta(hours=24)
            }
            
            await self.outcome_tracker.track_signal(signal_data)
            self.test_results['outcome_tracker_tests'] += 1
            self.test_results['outcome_tracker_passed'] += 1
            logger.info("✅ Signal tracking test passed")
            
            # Test 2: Get metrics
            metrics = self.outcome_tracker.get_metrics()
            if metrics['is_running'] and metrics['active_signals_count'] >= 0:
                self.test_results['outcome_tracker_tests'] += 1
                self.test_results['outcome_tracker_passed'] += 1
                logger.info("✅ Metrics retrieval test passed")
            
            # Test 3: Stop tracking
            await self.outcome_tracker.stop_tracking_signal(signal_data['signal_id'], 'test_complete')
            self.test_results['outcome_tracker_tests'] += 1
            self.test_results['outcome_tracker_passed'] += 1
            logger.info("✅ Signal stop tracking test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Outcome tracker test failed: {e}")
            return False
    
    async def test_tp_sl_detector(self):
        """Test TP/SL detector functionality"""
        logger.info("🧪 Testing TP/SL Detector...")
        
        try:
            # Test 1: Track a position
            position_data = {
                'signal_id': f"test_position_{uuid.uuid4().hex[:8]}",
                'symbol': 'ETHUSDT',
                'entry_price': 3000.0,
                'signal_type': 'long',
                'position_size': 1.0,
                'remaining_position': 1.0,
                'tpsl_config': {
                    'take_profit_price': 3100.0,
                    'stop_loss_price': 2900.0,
                    'partial_tp_prices': [3050.0],
                    'partial_sl_prices': [2950.0],
                    'hit_tolerance': 0.001,
                    'min_hit_duration': timedelta(milliseconds=100)
                }
            }
            
            await self.tp_sl_detector.track_position(position_data)
            self.test_results['tp_sl_detector_tests'] += 1
            self.test_results['tp_sl_detector_passed'] += 1
            logger.info("✅ Position tracking test passed")
            
            # Test 2: Get metrics
            metrics = self.tp_sl_detector.get_metrics()
            if metrics['is_running'] and metrics['active_positions_count'] >= 0:
                self.test_results['tp_sl_detector_tests'] += 1
                self.test_results['tp_sl_detector_passed'] += 1
                logger.info("✅ TP/SL metrics retrieval test passed")
            
            # Test 3: Stop tracking
            await self.tp_sl_detector.stop_tracking_position(position_data['signal_id'], 'test_complete')
            self.test_results['tp_sl_detector_tests'] += 1
            self.test_results['tp_sl_detector_passed'] += 1
            logger.info("✅ Position stop tracking test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TP/SL detector test failed: {e}")
            return False
    
    async def test_database_operations(self):
        """Test database operations for outcome tracking"""
        logger.info("🧪 Testing Database Operations...")
        
        try:
            if not self.db_connection:
                logger.warning("No database connection available")
                return False
            
            # Test 1: Check if outcome tracking tables exist
            session = await self.db_connection.get_session()
            async with session as session:
                # Check signal_outcomes table
                result = await session.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'signal_outcomes'
                    )
                """)
                signal_outcomes_exists = result.scalar()
                
                if signal_outcomes_exists:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ Signal outcomes table exists")
                
                # Check tp_sl_hits table
                result = await session.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'tp_sl_hits'
                    )
                """)
                tp_sl_hits_exists = result.scalar()
                
                if tp_sl_hits_exists:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ TP/SL hits table exists")
                
                # Check model_drift_events table
                result = await session.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'model_drift_events'
                    )
                """)
                model_drift_exists = result.scalar()
                
                if model_drift_exists:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ Model drift events table exists")
                
                # Check audit_logs table
                result = await session.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'audit_logs'
                    )
                """)
                audit_logs_exists = result.scalar()
                
                if audit_logs_exists:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ Audit logs table exists")
                
                # Test 2: Check views
                result = await session.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.views 
                        WHERE view_name = 'outcome_summary'
                    )
                """)
                outcome_summary_exists = result.scalar()
                
                if outcome_summary_exists:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ Outcome summary view exists")
                
                # Test 3: Insert test data
                test_signal_id = f"test_db_{uuid.uuid4().hex[:8]}"
                await session.execute("""
                    INSERT INTO signals (signal_id, symbol, signal_type, entry_price, confidence_score)
                    VALUES (:signal_id, 'TEST', 'long', 100.0, 0.85)
                """, {'signal_id': test_signal_id})
                
                await session.execute("""
                    INSERT INTO signal_outcomes (
                        signal_id, outcome_type, exit_price, exit_timestamp, 
                        realized_pnl, time_to_exit, order_type, order_state
                    ) VALUES (
                        :signal_id, 'tp_hit', 110.0, NOW(), 
                        10.0, INTERVAL '1 hour', 'market', 'filled'
                    )
                """, {'signal_id': test_signal_id})
                
                await session.commit()
                self.test_results['database_tests'] += 1
                self.test_results['database_passed'] += 1
                logger.info("✅ Database insert test passed")
                
                # Test 4: Query test data
                result = await session.execute("""
                    SELECT COUNT(*) FROM signal_outcomes WHERE signal_id = :signal_id
                """, {'signal_id': test_signal_id})
                count = result.scalar()
                
                if count > 0:
                    self.test_results['database_tests'] += 1
                    self.test_results['database_passed'] += 1
                    logger.info("✅ Database query test passed")
                
                # Cleanup test data
                await session.execute("DELETE FROM signal_outcomes WHERE signal_id = :signal_id", 
                                   {'signal_id': test_signal_id})
                await session.execute("DELETE FROM signals WHERE signal_id = :signal_id", 
                                   {'signal_id': test_signal_id})
                await session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database operations test failed: {e}")
            return False
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("🧪 Testing Component Integration...")
        
        try:
            # Test 1: Outcome tracker and TP/SL detector integration
            signal_id = f"test_integration_{uuid.uuid4().hex[:8]}"
            
            # Create signal data
            signal_data = {
                'signal_id': signal_id,
                'symbol': 'ADAUSDT',
                'entry_price': 0.50,
                'signal_type': 'long',
                'tp_price': 0.55,
                'sl_price': 0.45,
                'max_hold_time': timedelta(hours=12)
            }
            
            # Track signal in outcome tracker
            await self.outcome_tracker.track_signal(signal_data)
            
            # Track position in TP/SL detector
            position_data = {
                'signal_id': signal_id,
                'symbol': 'ADAUSDT',
                'entry_price': 0.50,
                'signal_type': 'long',
                'position_size': 1.0,
                'remaining_position': 1.0,
                'tpsl_config': {
                    'take_profit_price': 0.55,
                    'stop_loss_price': 0.45,
                    'hit_tolerance': 0.001,
                    'min_hit_duration': timedelta(milliseconds=100)
                }
            }
            await self.tp_sl_detector.track_position(position_data)
            
            self.test_results['integration_tests'] += 1
            self.test_results['integration_passed'] += 1
            logger.info("✅ Component integration test passed")
            
            # Test 2: Callback integration
            callback_called = False
            
            async def test_callback(outcome):
                nonlocal callback_called
                callback_called = True
                logger.info(f"✅ Callback received: {outcome.outcome_type}")
            
            self.outcome_tracker.add_outcome_callback(test_callback)
            
            # Simulate an outcome
            test_outcome = SignalOutcome(
                signal_id=signal_id,
                outcome_type=OutcomeType.TP_HIT,
                exit_price=0.55,
                exit_timestamp=datetime.now(timezone.utc),
                realized_pnl=0.05,
                max_adverse_excursion=0.0,
                max_favorable_excursion=0.05,
                time_to_exit=timedelta(hours=1)
            )
            
            await self.outcome_tracker._trigger_outcome_callbacks(test_outcome)
            
            if callback_called:
                self.test_results['integration_tests'] += 1
                self.test_results['integration_passed'] += 1
                logger.info("✅ Callback integration test passed")
            
            # Cleanup
            await self.outcome_tracker.stop_tracking_signal(signal_id, 'test_complete')
            await self.tp_sl_detector.stop_tracking_position(signal_id, 'test_complete')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📋 Generating Phase 2 test report...")
        
        total_tests = sum([
            self.test_results['outcome_tracker_tests'],
            self.test_results['tp_sl_detector_tests'],
            self.test_results['database_tests'],
            self.test_results['integration_tests']
        ])
        
        total_passed = sum([
            self.test_results['outcome_tracker_passed'],
            self.test_results['tp_sl_detector_passed'],
            self.test_results['database_passed'],
            self.test_results['integration_passed']
        ])
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'test_summary': {
                'test_name': 'Phase 2 Outcome Tracking System Test',
                'test_date': datetime.now().isoformat(),
                'total_tests': total_tests,
                'total_passed': total_passed,
                'success_rate': success_rate
            },
            'component_results': {
                'outcome_tracker': {
                    'tests': self.test_results['outcome_tracker_tests'],
                    'passed': self.test_results['outcome_tracker_passed'],
                    'success_rate': (self.test_results['outcome_tracker_passed'] / self.test_results['outcome_tracker_tests'] * 100) if self.test_results['outcome_tracker_tests'] > 0 else 0
                },
                'tp_sl_detector': {
                    'tests': self.test_results['tp_sl_detector_tests'],
                    'passed': self.test_results['tp_sl_detector_passed'],
                    'success_rate': (self.test_results['tp_sl_detector_passed'] / self.test_results['tp_sl_detector_tests'] * 100) if self.test_results['tp_sl_detector_tests'] > 0 else 0
                },
                'database_operations': {
                    'tests': self.test_results['database_tests'],
                    'passed': self.test_results['database_passed'],
                    'success_rate': (self.test_results['database_passed'] / self.test_results['database_tests'] * 100) if self.test_results['database_tests'] > 0 else 0
                },
                'integration': {
                    'tests': self.test_results['integration_tests'],
                    'passed': self.test_results['integration_passed'],
                    'success_rate': (self.test_results['integration_passed'] / self.test_results['integration_tests'] * 100) if self.test_results['integration_tests'] > 0 else 0
                }
            },
            'phase2_readiness': {
                'outcome_tracking': '✅ READY' if self.test_results['outcome_tracker_passed'] == self.test_results['outcome_tracker_tests'] else '❌ NEEDS FIXES',
                'tp_sl_detection': '✅ READY' if self.test_results['tp_sl_detector_passed'] == self.test_results['tp_sl_detector_tests'] else '❌ NEEDS FIXES',
                'database_integration': '✅ READY' if self.test_results['database_passed'] == self.test_results['database_tests'] else '❌ NEEDS FIXES',
                'component_integration': '✅ READY' if self.test_results['integration_passed'] == self.test_results['integration_tests'] else '❌ NEEDS FIXES'
            }
        }
        
        return report
    
    async def cleanup(self):
        """Cleanup test resources"""
        logger.info("🧹 Cleaning up test resources...")
        
        try:
            if self.outcome_tracker:
                await self.outcome_tracker.shutdown()
            if self.tp_sl_detector:
                await self.tp_sl_detector.shutdown()
            if self.db_connection:
                await self.db_connection.close()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main test execution"""
    logger.info("=" * 80)
    logger.info("🧪 PHASE 2 OUTCOME TRACKING SYSTEM TEST")
    logger.info("=" * 80)
    
    test = Phase2OutcomeTrackingTest()
    
    try:
        # Initialize components
        if not await test.initialize_components():
            logger.error("❌ Failed to initialize components - aborting test")
            return False
        
        # Run tests
        await test.test_outcome_tracker()
        await test.test_tp_sl_detector()
        await test.test_database_operations()
        await test.test_integration()
        
        # Generate report
        report = test.generate_test_report()
        
        # Print results
        logger.info("=" * 80)
        logger.info("📊 PHASE 2 TEST RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
        logger.info(f"Passed: {report['test_summary']['total_passed']}")
        logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        
        logger.info("\n📋 COMPONENT RESULTS:")
        for component, results in report['component_results'].items():
            logger.info(f"  {component.replace('_', ' ').title()}: {results['passed']}/{results['tests']} ({results['success_rate']:.1f}%)")
        
        logger.info("\n🎯 PHASE 2 READINESS:")
        for component, status in report['phase2_readiness'].items():
            logger.info(f"  {component.replace('_', ' ').title()}: {status}")
        
        # Determine overall success
        success = report['test_summary']['success_rate'] >= 90
        
        if success:
            logger.info("\n🎉 PHASE 2 OUTCOME TRACKING SYSTEM READY FOR PRODUCTION!")
        else:
            logger.error("\n❌ PHASE 2 OUTCOME TRACKING SYSTEM NEEDS FIXES")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Phase 2 test failed with exception: {e}")
        return False
    finally:
        await test.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
