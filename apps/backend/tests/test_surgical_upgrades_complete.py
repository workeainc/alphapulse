#!/usr/bin/env python3
"""
Comprehensive Test for Surgical Upgrades Implementation
Tests all phases of the surgical upgrades including interface standardization,
confidence calibration, hard gating, and enhanced signal generation.
"""

import asyncio
import asyncpg
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.onnx_inference import ONNXInferenceEngine
from src.ai.feature_drift_detector import FeatureDriftDetector
from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalUpgradesTester:
    """Comprehensive tester for surgical upgrades"""
    
    def __init__(self):
        self.db_pool = None
        self.test_results = {}
        
    async def setup(self):
        """Setup database connection and test environment"""
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
    
    async def test_phase1_interface_standardization(self):
        """Test Phase 1: Interface Standardization"""
        logger.info("🧪 Testing Phase 1: Interface Standardization...")
        
        try:
            # Test ONNX Inference Engine with standardized interface
            onnx_engine = ONNXInferenceEngine(db_pool=self.db_pool)
            
            # Test interface registration
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM component_interface_registry 
                    WHERE component_name = 'ONNXInferenceEngine'
                """)
                if result['count'] > 0:
                    logger.info("✅ ONNX interface registration verified")
                else:
                    logger.error("❌ ONNX interface registration failed")
                    return False
            
            # Test Feature Drift Detector with standardized interface
            drift_detector = FeatureDriftDetector(db_pool=self.db_pool)
            
            # Test drift detection with standardized interface
            feature_vector = np.array([0.5, 0.7, 0.3, 0.9, 0.6])
            feature_names = ['rsi', 'macd', 'volume', 'sentiment', 'volatility']
            
            drift_result = await drift_detector.score(feature_vector, feature_names)
            
            if hasattr(drift_result, 'psi') and hasattr(drift_result, 'alert'):
                logger.info(f"✅ Feature drift detection working: PSI={drift_result.psi:.3f}, Alert={drift_result.alert}")
            else:
                logger.error("❌ Feature drift detection failed")
                return False
            
            # Test interface performance tracking
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM interface_performance_metrics 
                    WHERE component_name IN ('ONNXInferenceEngine', 'FeatureDriftDetector')
                """)
                if result['count'] > 0:
                    logger.info("✅ Interface performance tracking working")
                else:
                    logger.warning("⚠️ No interface performance metrics found")
            
            self.test_results['phase1'] = True
            logger.info("✅ Phase 1: Interface Standardization - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 1 test failed: {e}")
            self.test_results['phase1'] = False
            return False
    
    async def test_phase2_confidence_calibration(self):
        """Test Phase 2: Confidence Calibration"""
        logger.info("🧪 Testing Phase 2: Confidence Calibration...")
        
        try:
            # Test calibration tables exist
            async with self.db_pool.acquire() as conn:
                tables_to_check = [
                    'calibrated_signal_generation',
                    'confidence_fusion_components',
                    'ensemble_voting_results',
                    'calibration_training_data',
                    'calibration_model_performance',
                    'dynamic_threshold_adjustment'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        logger.error(f"❌ Table {table} not found")
                        return False
                
                logger.info("✅ All calibration tables exist")
            
            # Test dynamic threshold adjustment
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT current_threshold, target_win_rate 
                    FROM dynamic_threshold_adjustment 
                    WHERE threshold_type = 'confidence'
                """)
                if result:
                    logger.info(f"✅ Dynamic threshold: {result['current_threshold']}, Target: {result['target_win_rate']}")
                else:
                    logger.error("❌ Dynamic threshold not found")
                    return False
            
            # Test reliability buckets
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM reliability_buckets
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Reliability buckets: {result['count']} buckets configured")
                else:
                    logger.warning("⚠️ No reliability buckets found")
            
            self.test_results['phase2'] = True
            logger.info("✅ Phase 2: Confidence Calibration - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 2 test failed: {e}")
            self.test_results['phase2'] = False
            return False
    
    async def test_hard_gating_system(self):
        """Test Hard Gating System"""
        logger.info("🧪 Testing Hard Gating System...")
        
        try:
            # Test gate configuration
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM signal_gates_config 
                    WHERE is_enabled = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Gate configuration: {result['count']} active gates")
                else:
                    logger.error("❌ No active gates configured")
                    return False
            
            # Test gate dependencies
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM gate_dependencies 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Gate dependencies: {result['count']} dependencies configured")
                else:
                    logger.warning("⚠️ No gate dependencies found")
            
            # Test validation rules
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM real_time_validation_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Validation rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No validation rules found")
            
            self.test_results['hard_gating'] = True
            logger.info("✅ Hard Gating System - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hard gating test failed: {e}")
            self.test_results['hard_gating'] = False
            return False
    
    async def test_signal_lifecycle_management(self):
        """Test Signal Lifecycle Management"""
        logger.info("🧪 Testing Signal Lifecycle Management...")
        
        try:
            # Test lifecycle tables
            async with self.db_pool.acquire() as conn:
                tables_to_check = [
                    'signal_state_machine',
                    'signal_expiry_management',
                    'signal_cooldown_management',
                    'active_signal_mutex'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        logger.error(f"❌ Table {table} not found")
                        return False
                
                logger.info("✅ All lifecycle management tables exist")
            
            # Test quota management
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM quota_configuration 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Quota configuration: {result['count']} active quotas")
                else:
                    logger.warning("⚠️ No quota configuration found")
            
            self.test_results['lifecycle_management'] = True
            logger.info("✅ Signal Lifecycle Management - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lifecycle management test failed: {e}")
            self.test_results['lifecycle_management'] = False
            return False
    
    async def test_enhanced_signal_generation(self):
        """Test Enhanced Signal Generation"""
        logger.info("🧪 Testing Enhanced Signal Generation...")
        
        try:
            # Test enhanced signal tables
            async with self.db_pool.acquire() as conn:
                tables_to_check = [
                    'enhanced_signal_payloads',
                    'signal_reasoning_chain',
                    'mtf_agreement_details'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        logger.error(f"❌ Table {table} not found")
                        return False
                
                logger.info("✅ All enhanced signal tables exist")
            
            # Test news override system
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM news_override_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ News override rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No news override rules found")
            
            self.test_results['enhanced_signals'] = True
            logger.info("✅ Enhanced Signal Generation - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced signal generation test failed: {e}")
            self.test_results['enhanced_signals'] = False
            return False
    
    async def test_performance_monitoring(self):
        """Test Performance Monitoring"""
        logger.info("🧪 Testing Performance Monitoring...")
        
        try:
            # Test performance monitoring tables
            async with self.db_pool.acquire() as conn:
                tables_to_check = [
                    'enhanced_performance_metrics',
                    'performance_alerting',
                    'system_health_monitoring'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        logger.error(f"❌ Table {table} not found")
                        return False
                
                logger.info("✅ All performance monitoring tables exist")
            
            # Test system health monitoring
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM system_health_monitoring
                """)
                if result['count'] > 0:
                    logger.info(f"✅ System health monitoring: {result['count']} components tracked")
                else:
                    logger.warning("⚠️ No system health data found")
            
            self.test_results['performance_monitoring'] = True
            logger.info("✅ Performance Monitoring - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            self.test_results['performance_monitoring'] = False
            return False
    
    async def test_data_sanity_validation(self):
        """Test Data Sanity Validation"""
        logger.info("🧪 Testing Data Sanity Validation...")
        
        try:
            # Test data sanity tables
            async with self.db_pool.acquire() as conn:
                tables_to_check = [
                    'data_sanity_rules',
                    'data_sanity_validation_results',
                    'orderbook_health_metrics'
                ]
                
                for table in tables_to_check:
                    result = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    """)
                    if result['count'] == 0:
                        logger.error(f"❌ Table {table} not found")
                        return False
                
                logger.info("✅ All data sanity tables exist")
            
            # Test data sanity rules
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count FROM data_sanity_rules 
                    WHERE is_active = TRUE
                """)
                if result['count'] > 0:
                    logger.info(f"✅ Data sanity rules: {result['count']} active rules")
                else:
                    logger.warning("⚠️ No data sanity rules found")
            
            self.test_results['data_sanity'] = True
            logger.info("✅ Data Sanity Validation - PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data sanity validation test failed: {e}")
            self.test_results['data_sanity'] = False
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("🚀 Starting Comprehensive Surgical Upgrades Test...")
        
        if not await self.setup():
            return False
        
        test_functions = [
            self.test_phase1_interface_standardization,
            self.test_phase2_confidence_calibration,
            self.test_hard_gating_system,
            self.test_signal_lifecycle_management,
            self.test_enhanced_signal_generation,
            self.test_performance_monitoring,
            self.test_data_sanity_validation
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            try:
                if await test_func():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed with exception: {e}")
        
        # Generate test summary
        logger.info("\n" + "="*60)
        logger.info("📊 SURGICAL UPGRADES TEST SUMMARY")
        logger.info("="*60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL SURGICAL UPGRADES TESTS PASSED!")
            return True
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()

async def main():
    """Main test function"""
    tester = SurgicalUpgradesTester()
    try:
        success = await tester.run_comprehensive_test()
        return success
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
