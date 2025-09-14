#!/usr/bin/env python3
"""
Phase 1: Model Versioning & Rollback System Test
Comprehensive test for the enhanced ML auto-retraining system with versioning and rollback capabilities
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Phase 1 components
from ai.ml_auto_retraining.model_versioning_manager import ModelVersioningManager, ModelVersion
from ai.ml_auto_retraining.rollback_manager import RollbackManager, RollbackDecision
from ai.ml_auto_retraining.train_model import MLModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

class Phase1TestSuite:
    """Comprehensive test suite for Phase 1 enhancements"""
    
    def __init__(self):
        self.versioning_manager = ModelVersioningManager(DB_CONFIG)
        self.rollback_manager = RollbackManager(DB_CONFIG)
        self.trainer = MLModelTrainer(DB_CONFIG)
        self.test_results = {}
        
    async def setup_test_environment(self):
        """Setup test environment with sample data"""
        logger.info("🔧 Setting up Phase 1 test environment...")
        
        try:
            # Initialize trainer components
            await self.trainer.initialize_components()
            
            # Generate sample OHLCV data if needed
            await self._ensure_sample_data()
            
            logger.info("✅ Phase 1 test environment setup completed")
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            raise
    
    async def _ensure_sample_data(self):
        """Ensure sample OHLCV data exists for testing"""
        logger.info("📊 Ensuring sample OHLCV data exists...")
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            # Check if we have enough data
            cursor.execute("""
                SELECT COUNT(*) FROM ohlcv WHERE symbol = 'BTCUSDT' 
                AND time >= '2024-12-01' AND time <= '2024-12-31'
            """)
            
            count = cursor.fetchone()[0]
            
            if count < 1000:
                logger.info("📝 Generating additional sample OHLCV data...")
                
                # Generate sample data for December 2024
                start_date = datetime(2024, 12, 1)
                end_date = datetime(2024, 12, 31)
                
                # Create sample OHLCV data
                dates = pd.date_range(start=start_date, end=end_date, freq='1h')
                
                for date in dates:
                    # Generate realistic OHLCV data
                    base_price = 45000 + np.random.normal(0, 1000)
                    open_price = base_price
                    high_price = open_price + np.random.uniform(0, 500)
                    low_price = open_price - np.random.uniform(0, 500)
                    close_price = np.random.uniform(low_price, high_price)
                    volume = np.random.uniform(100, 1000)
                    
                    cursor.execute("""
                        INSERT INTO ohlcv (time, symbol, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (time, symbol) DO NOTHING
                    """, (date, 'BTCUSDT', open_price, high_price, low_price, close_price, volume))
                
                conn.commit()
                logger.info(f"✅ Generated {len(dates)} sample OHLCV records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure sample data: {e}")
    
    async def test_model_versioning_manager(self):
        """Test Model Versioning Manager functionality"""
        logger.info("🧪 Testing Model Versioning Manager...")
        
        test_name = "Model Versioning Manager"
        test_passed = True
        test_details = []
        
        try:
            # Test 1: Create model lineage
            logger.info("  📝 Test 1: Creating model lineage...")
            
            lineage_id = await self.versioning_manager.create_model_lineage(
                model_name="test_model",
                model_version=1,
                training_data=pd.DataFrame({'test': [1, 2, 3]}),
                feature_set={'features': ['feature1', 'feature2']},
                hyperparameters={'param1': 0.1, 'param2': 100},
                training_environment="test_env",
                training_duration_seconds=60,
                training_samples=1000,
                validation_samples=200,
                lineage_metadata={'test': True}
            )
            
            if lineage_id:
                test_details.append("✅ Model lineage creation: PASS")
            else:
                test_details.append("❌ Model lineage creation: FAIL")
                test_passed = False
            
            # Test 2: Register model version
            logger.info("  📝 Test 2: Registering model version...")
            
            model_version = await self.versioning_manager.register_model_version(
                model_name="test_model",
                version=1,
                status="staging",
                regime="trending",
                symbol="BTCUSDT",
                model_artifact_path="/tmp/test_model.joblib",
                model_artifact_size_mb=1.5,
                model_artifact_hash="test_hash",
                training_metrics={'accuracy': 0.85, 'f1_score': 0.82},
                validation_metrics={'accuracy': 0.83, 'f1_score': 0.80},
                version_metadata={'test': True}
            )
            
            if model_version:
                test_details.append("✅ Model version registration: PASS")
            else:
                test_details.append("❌ Model version registration: FAIL")
                test_passed = False
            
            # Test 3: Get production model
            logger.info("  📝 Test 3: Getting production model...")
            
            # First, create a production model
            await self.versioning_manager.register_model_version(
                model_name="test_production_model",
                version=1,
                status="production",
                regime="trending",
                symbol="BTCUSDT",
                model_artifact_path="/tmp/test_production_model.joblib",
                model_artifact_size_mb=1.5,
                model_artifact_hash="test_production_hash",
                training_metrics={'accuracy': 0.87, 'f1_score': 0.84},
                validation_metrics={'accuracy': 0.85, 'f1_score': 0.82},
                version_metadata={'test': True}
            )
            
            production_model = await self.versioning_manager.get_production_model(
                "test_production_model", "trending", "BTCUSDT"
            )
            
            if production_model and production_model.status == "production":
                test_details.append("✅ Get production model: PASS")
            else:
                test_details.append("❌ Get production model: FAIL")
                test_passed = False
            
            # Test 4: Update model usage
            logger.info("  📝 Test 4: Updating model usage...")
            
            await self.versioning_manager.update_model_usage(
                "test_production_model", 1, inference_time_ms=15.5
            )
            
            test_details.append("✅ Update model usage: PASS")
            
            # Test 5: Mark model for rollback
            logger.info("  📝 Test 5: Marking model for rollback...")
            
            await self.versioning_manager.mark_model_for_rollback(
                "test_production_model", 1, "Test rollback reason"
            )
            
            rollback_candidates = await self.versioning_manager.get_rollback_candidates(
                "test_production_model", "trending", "BTCUSDT"
            )
            
            if rollback_candidates:
                test_details.append("✅ Mark model for rollback: PASS")
            else:
                test_details.append("❌ Mark model for rollback: FAIL")
                test_passed = False
            
        except Exception as e:
            logger.error(f"❌ Model Versioning Manager test failed: {e}")
            test_details.append(f"❌ Test failed with error: {e}")
            test_passed = False
        
        self.test_results[test_name] = {
            'passed': test_passed,
            'details': test_details
        }
        
        logger.info(f"✅ Model Versioning Manager test completed: {'PASS' if test_passed else 'FAIL'}")
    
    async def test_rollback_manager(self):
        """Test Rollback Manager functionality"""
        logger.info("🧪 Testing Rollback Manager...")
        
        test_name = "Rollback Manager"
        test_passed = True
        test_details = []
        
        try:
            # Test 1: Assess rollback needs
            logger.info("  📝 Test 1: Assessing rollback needs...")
            
            rollback_decision = await self.rollback_manager.assess_rollback_needs(
                model_name="test_production_model",
                regime="trending",
                symbol="BTCUSDT",
                current_performance={'accuracy': 0.70, 'f1_score': 0.65},  # Degraded performance
                current_drift_metrics={'psi': 0.30},  # High drift
                current_error_rate=0.15  # High error rate
            )
            
            if rollback_decision:
                test_details.append("✅ Rollback assessment: PASS")
                logger.info(f"    Rollback decision: {rollback_decision.rollback_reason}")
            else:
                test_details.append("❌ Rollback assessment: FAIL")
                test_passed = False
            
            # Test 2: Find rollback candidate
            logger.info("  📝 Test 2: Finding rollback candidate...")
            
            # Create an archived model for rollback
            await self.versioning_manager.register_model_version(
                model_name="test_rollback_model",
                version=1,
                status="archived",
                regime="trending",
                symbol="BTCUSDT",
                model_artifact_path="/tmp/test_archived_model.joblib",
                model_artifact_size_mb=1.5,
                model_artifact_hash="test_archived_hash",
                training_metrics={'accuracy': 0.85, 'f1_score': 0.82},
                validation_metrics={'accuracy': 0.83, 'f1_score': 0.80},
                performance_metrics={'accuracy': 0.84, 'f1_score': 0.81},
                version_metadata={'test': True}
            )
            
            rollback_candidate = await self.rollback_manager.find_rollback_candidate(
                "test_rollback_model", "trending", "BTCUSDT"
            )
            
            if rollback_candidate:
                test_details.append("✅ Find rollback candidate: PASS")
            else:
                test_details.append("❌ Find rollback candidate: FAIL")
                test_passed = False
            
            # Test 3: Get rollback history
            logger.info("  📝 Test 3: Getting rollback history...")
            
            rollback_history = await self.rollback_manager.get_rollback_history(
                "test_rollback_model", days=30
            )
            
            test_details.append("✅ Get rollback history: PASS")
            
        except Exception as e:
            logger.error(f"❌ Rollback Manager test failed: {e}")
            test_details.append(f"❌ Test failed with error: {e}")
            test_passed = False
        
        self.test_results[test_name] = {
            'passed': test_passed,
            'details': test_details
        }
        
        logger.info(f"✅ Rollback Manager test completed: {'PASS' if test_passed else 'FAIL'}")
    
    async def test_enhanced_training_integration(self):
        """Test enhanced training with versioning integration"""
        logger.info("🧪 Testing Enhanced Training Integration...")
        
        test_name = "Enhanced Training Integration"
        test_passed = True
        test_details = []
        
        try:
            # Test 1: Train model with versioning
            logger.info("  📝 Test 1: Training model with versioning...")
            
            end_date = datetime(2024, 12, 31)
            start_date = end_date - timedelta(days=7)  # Short period for testing
            
            model_path = await self.trainer.train_model(
                symbol="BTCUSDT",
                regime="trending",
                start_date=start_date,
                end_date=end_date,
                model_name="test_enhanced_model",
                horizon=5
            )
            
            if model_path:
                test_details.append("✅ Enhanced training: PASS")
            else:
                test_details.append("❌ Enhanced training: FAIL")
                test_passed = False
            
            # Test 2: Verify model lineage creation
            logger.info("  📝 Test 2: Verifying model lineage...")
            
            # Check if model lineage was created
            import psycopg2
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_lineage 
                WHERE model_name = 'test_enhanced_model'
            """)
            
            lineage_count = cursor.fetchone()[0]
            
            if lineage_count > 0:
                test_details.append("✅ Model lineage verification: PASS")
            else:
                test_details.append("❌ Model lineage verification: FAIL")
                test_passed = False
            
            # Test 3: Verify model version registration
            logger.info("  📝 Test 3: Verifying model version registration...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_versions 
                WHERE model_name = 'test_enhanced_model'
            """)
            
            version_count = cursor.fetchone()[0]
            
            if version_count > 0:
                test_details.append("✅ Model version registration: PASS")
            else:
                test_details.append("❌ Model version registration: FAIL")
                test_passed = False
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Enhanced Training Integration test failed: {e}")
            test_details.append(f"❌ Test failed with error: {e}")
            test_passed = False
        
        self.test_results[test_name] = {
            'passed': test_passed,
            'details': test_details
        }
        
        logger.info(f"✅ Enhanced Training Integration test completed: {'PASS' if test_passed else 'FAIL'}")
    
    async def test_database_schema(self):
        """Test Phase 1 database schema"""
        logger.info("🧪 Testing Phase 1 Database Schema...")
        
        test_name = "Database Schema"
        test_passed = True
        test_details = []
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            
            # Test 1: Check model_lineage table
            logger.info("  📝 Test 1: Checking model_lineage table...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_lineage
            """)
            
            lineage_count = cursor.fetchone()[0]
            test_details.append(f"✅ model_lineage table: {lineage_count} records")
            
            # Test 2: Check model_versions table
            logger.info("  📝 Test 2: Checking model_versions table...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_versions
            """)
            
            versions_count = cursor.fetchone()[0]
            test_details.append(f"✅ model_versions table: {versions_count} records")
            
            # Test 3: Check rollback_events table
            logger.info("  📝 Test 3: Checking rollback_events table...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM rollback_events
            """)
            
            rollback_count = cursor.fetchone()[0]
            test_details.append(f"✅ rollback_events table: {rollback_count} records")
            
            # Test 4: Check model_performance_history table
            logger.info("  📝 Test 4: Checking model_performance_history table...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_performance_history
            """)
            
            performance_count = cursor.fetchone()[0]
            test_details.append(f"✅ model_performance_history table: {performance_count} records")
            
            # Test 5: Check model_comparison table
            logger.info("  📝 Test 5: Checking model_comparison table...")
            
            cursor.execute("""
                SELECT COUNT(*) FROM model_comparison
            """)
            
            comparison_count = cursor.fetchone()[0]
            test_details.append(f"✅ model_comparison table: {comparison_count} records")
            
            # Test 6: Check enhanced ml_models table
            logger.info("  📝 Test 6: Checking enhanced ml_models table...")
            
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'ml_models' 
                AND column_name IN ('lineage_id', 'parent_model_name', 'rollback_candidate')
            """)
            
            enhanced_columns = [row[0] for row in cursor.fetchall()]
            
            if len(enhanced_columns) >= 3:
                test_details.append("✅ Enhanced ml_models table: PASS")
            else:
                test_details.append("❌ Enhanced ml_models table: FAIL")
                test_passed = False
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Database Schema test failed: {e}")
            test_details.append(f"❌ Test failed with error: {e}")
            test_passed = False
        
        self.test_results[test_name] = {
            'passed': test_passed,
            'details': test_details
        }
        
        logger.info(f"✅ Database Schema test completed: {'PASS' if test_passed else 'FAIL'}")
    
    async def run_all_tests(self):
        """Run all Phase 1 tests"""
        logger.info("🚀 Starting Phase 1: Model Versioning & Rollback System Tests")
        
        try:
            # Setup test environment
            await self.setup_test_environment()
            
            # Run all tests
            await self.test_database_schema()
            await self.test_model_versioning_manager()
            await self.test_rollback_manager()
            await self.test_enhanced_training_integration()
            
            # Print test results
            self.print_test_results()
            
        except Exception as e:
            logger.error(f"❌ Phase 1 test suite failed: {e}")
        finally:
            # Cleanup
            await self.cleanup()
    
    def print_test_results(self):
        """Print comprehensive test results"""
        logger.info("\n" + "="*80)
        logger.info("📊 PHASE 1: MODEL VERSIONING & ROLLBACK SYSTEM TEST RESULTS")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"\n{test_name}: {status}")
            
            for detail in result['details']:
                logger.info(f"  {detail}")
        
        logger.info(f"\n📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL PHASE 1 TESTS PASSED! Model Versioning & Rollback System is ready.")
        else:
            logger.info("⚠️ Some tests failed. Please review the details above.")
        
        logger.info("="*80)
    
    async def cleanup(self):
        """Cleanup test resources"""
        logger.info("🧹 Cleaning up Phase 1 test resources...")
        
        try:
            if self.trainer:
                await self.trainer.cleanup()
            
            logger.info("✅ Phase 1 test cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main test function"""
    logger.info("🎯 Phase 1: Model Versioning & Rollback System Test Suite")
    
    test_suite = Phase1TestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
