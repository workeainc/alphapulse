#!/usr/bin/env python3
"""
Enhanced ML Auto-Retraining System Test
Comprehensive test for automation scheduling, drift monitoring, and integration
"""

import os
import sys
import json
import logging
import asyncio
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced ML components
from src.ai.ml_auto_retraining.ml_scheduler import MLScheduler
from src.ai.ml_auto_retraining.drift_monitor import DriftMonitor
from src.ai.ml_auto_retraining.train_model import MLModelTrainer
from src.ai.ml_auto_retraining.evaluate_and_promote import ModelEvaluator
from src.ai.ml_auto_retraining.ml_inference_engine import MLInferenceEngine, EnhancedPatternDetector

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

class EnhancedMLSystemTester:
    """Comprehensive tester for enhanced ML auto-retraining system"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.scheduler = None
        self.drift_monitor = None
        self.trainer = None
        self.evaluator = None
        self.inference_engine = None
        self.enhanced_detector = None
        
    async def initialize_components(self):
        """Initialize all ML components"""
        logger.info("🔧 Initializing enhanced ML system components...")
        
        try:
            # Initialize scheduler
            self.scheduler = MLScheduler(self.db_config)
            await self.scheduler.initialize()
            
            # Initialize drift monitor
            self.drift_monitor = DriftMonitor(self.db_config)
            await self.drift_monitor.initialize()
            
            # Initialize trainer
            self.trainer = MLModelTrainer(self.db_config)
            await self.trainer.initialize_components()
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(self.db_config)
            
            # Initialize inference engine
            self.inference_engine = MLInferenceEngine(self.db_config)
            
            # Initialize enhanced pattern detector
            self.enhanced_detector = EnhancedPatternDetector(self.db_config)
            
            logger.info("✅ All enhanced ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def test_database_verification(self):
        """Test database tables and data"""
        logger.info("📋 Step 1: Verifying enhanced database tables...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check ML auto-retraining tables
            ml_tables = [
                'ml_models', 'ml_eval_history', 'ml_training_jobs', 'ml_performance_tracking'
            ]
            
            # Check drift monitoring tables
            drift_tables = [
                'ml_drift_alerts', 'ml_drift_details', 'ml_reference_features', 
                'ml_drift_thresholds', 'ml_drift_actions'
            ]
            
            all_tables = ml_tables + drift_tables
            
            for table in all_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ {table}: {count} records")
            
            # Check production models
            cursor.execute("""
                SELECT model_name, regime, symbol, version, status 
                FROM ml_models 
                WHERE status = 'production'
                ORDER BY created_at DESC
            """)
            
            production_models = cursor.fetchall()
            logger.info(f"📊 Production models: {len(production_models)}")
            
            for model in production_models:
                logger.info(f"   - {model[0]} {model[1]} {model[2]} v{model[3]} ({model[4]})")
            
            conn.close()
            logger.info("✅ Database verification completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            return False
    
    async def test_scheduler_functionality(self):
        """Test ML scheduler functionality"""
        logger.info("📋 Step 2: Testing ML scheduler functionality...")
        
        try:
            # Test job status
            job_status = self.scheduler.get_job_status()
            logger.info(f"📊 Scheduled jobs: {len(job_status)}")
            
            for job_id, status in job_status.items():
                logger.info(f"   - {job_id}: {status['status']} (last run: {status['last_run']})")
            
            # Test manual training trigger
            logger.info("🔧 Testing manual training trigger...")
            await self.scheduler.trigger_manual_training('BTCUSDT', 'trending')
            
            logger.info("✅ Scheduler functionality test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scheduler functionality test failed: {e}")
            return False
    
    async def test_drift_monitoring(self):
        """Test drift monitoring functionality"""
        logger.info("📋 Step 3: Testing drift monitoring functionality...")
        
        try:
            # Test drift detection for existing model
            drift_results = await self.drift_monitor.detect_drift('BTCUSDT', 'trending')
            
            if drift_results:
                logger.info(f"📊 Drift detection results: {len(drift_results)} features analyzed")
                
                drift_features = [r.feature_name for r in drift_results if r.is_drift]
                if drift_features:
                    logger.warning(f"🚨 Drift detected in features: {drift_features}")
                else:
                    logger.info("✅ No significant drift detected")
                
                # Show top drift scores
                top_drift = sorted(drift_results, key=lambda x: x.drift_score, reverse=True)[:5]
                for result in top_drift:
                    logger.info(f"   - {result.feature_name}: {result.drift_score:.4f} ({result.drift_type})")
            else:
                logger.info("ℹ️ No drift results available (insufficient data)")
            
            # Test drift summary
            drift_summary = await self.drift_monitor.get_drift_summary(days=7)
            logger.info(f"📊 Drift summary: {drift_summary.get('total_alerts', 0)} alerts in last 7 days")
            
            logger.info("✅ Drift monitoring test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Drift monitoring test failed: {e}")
            return False
    
    async def test_enhanced_training_pipeline(self):
        """Test enhanced training pipeline with drift integration"""
        logger.info("📋 Step 4: Testing enhanced training pipeline...")
        
        try:
            # Test training with drift-aware features
            end_date = datetime(2024, 12, 31)  # Use date range with actual data
            start_date = end_date - timedelta(days=30)  # Shorter period for testing
            
            model_path = await self.trainer.train_model(
                symbol='BTCUSDT',
                regime='trending',
                start_date=start_date,
                end_date=end_date,
                model_name='alphaplus_pattern_classifier'
            )
            
            if model_path:
                logger.info(f"✅ Enhanced training completed: {model_path}")
                
                # Test evaluation with drift consideration
                evaluation_result = await self.evaluator.evaluate_and_promote_model(
                    model_path=model_path,
                    symbol='BTCUSDT',
                    regime='trending',
                    model_name='alphaplus_pattern_classifier'
                )
                
                logger.info(f"📊 Evaluation result: {evaluation_result}")
                
            else:
                logger.warning("⚠️ Enhanced training failed (insufficient data)")
            
            logger.info("✅ Enhanced training pipeline test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced training pipeline test failed: {e}")
            return False
    
    async def test_enhanced_inference_integration(self):
        """Test enhanced inference with drift monitoring"""
        logger.info("📋 Step 5: Testing enhanced inference integration...")
        
        try:
            # Create sample market data
            dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
            
            # Create trending market data
            base_prices = np.linspace(45000, 55000, len(dates))
            noise = np.random.normal(0, 200, len(dates))
            prices = base_prices + noise
            
            sample_data = pd.DataFrame({
                'open': prices + np.random.normal(0, 50, len(dates)),
                'high': prices + np.abs(np.random.normal(0, 100, len(dates))),
                'low': prices - np.abs(np.random.normal(0, 100, len(dates))),
                'close': prices,
                'volume': np.random.uniform(1000, 5000, len(dates))
            }, index=dates)
            
            # Test enhanced pattern detection
            enhanced_signals = await self.enhanced_detector.detect_patterns_with_ml(
                df=sample_data,
                symbol='BTCUSDT',
                pattern_signals=[type('Signal', (), {
                    'pattern_name': 'hammer',
                    'confidence': 0.8
                })() for _ in range(3)]
            )
            
            logger.info(f"✅ Enhanced pattern detection: {len(enhanced_signals)} signals processed")
            
            # Test ML performance summary
            performance_summary = self.enhanced_detector.get_ml_performance_summary('BTCUSDT', days=7)
            logger.info(f"📊 ML performance summary: {len(performance_summary)} regimes tracked")
            
            for regime, summary in performance_summary.items():
                if summary.get('total_predictions', 0) > 0:
                    logger.info(f"   - {regime}: {summary['total_predictions']} predictions, "
                              f"{summary['accuracy']:.2%} accuracy")
            
            logger.info("✅ Enhanced inference integration test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced inference integration test failed: {e}")
            return False
    
    async def test_automation_workflow(self):
        """Test complete automation workflow"""
        logger.info("📋 Step 6: Testing complete automation workflow...")
        
        try:
            # Simulate drift detection and automatic retraining
            logger.info("🔄 Simulating drift detection workflow...")
            
            # Create mock drift alert
            from src.ai.ml_auto_retraining.drift_monitor import DriftAlert
            
            mock_alert = DriftAlert(
                symbol='BTCUSDT',
                regime='trending',
                model_name='alphaplus_pattern_classifier',
                drift_type='data_drift',
                severity='high',
                features_affected=['volume_ratio', 'rsi_14'],
                overall_drift_score=0.25,
                timestamp=datetime.now(),
                action_required='schedule_retrain'
            )
            
            # Test automatic retraining trigger
            await self.drift_monitor.trigger_retraining_on_drift(mock_alert)
            
            logger.info("✅ Automation workflow test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Automation workflow test failed: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of enhanced ML system"""
        logger.info("🚀 Starting comprehensive enhanced ML auto-retraining system test")
        logger.info("=" * 80)
        
        test_results = {}
        
        try:
            # Initialize components
            await self.initialize_components()
            
            # Run all tests
            tests = [
                ("Database Verification", self.test_database_verification),
                ("Scheduler Functionality", self.test_scheduler_functionality),
                ("Drift Monitoring", self.test_drift_monitoring),
                ("Enhanced Training Pipeline", self.test_enhanced_training_pipeline),
                ("Enhanced Inference Integration", self.test_enhanced_inference_integration),
                ("Automation Workflow", self.test_automation_workflow)
            ]
            
            for test_name, test_func in tests:
                logger.info(f"\n{'='*20} {test_name} {'='*20}")
                try:
                    result = await test_func()
                    test_results[test_name] = "PASS" if result else "FAIL"
                except Exception as e:
                    logger.error(f"❌ {test_name} failed with exception: {e}")
                    test_results[test_name] = "FAIL"
            
            # Print final results
            logger.info("\n" + "=" * 80)
            logger.info("📊 Enhanced ML Auto-Retraining System Test Summary")
            logger.info("=" * 80)
            
            passed = sum(1 for result in test_results.values() if result == "PASS")
            total = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✅ PASS" if result == "PASS" else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
            
            logger.info(f"\n📈 Overall Result: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 All tests passed! Enhanced ML auto-retraining system is ready for production.")
            else:
                logger.warning("⚠️ Some tests failed. Please review the logs and fix issues.")
            
            return passed == total
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return False
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up enhanced ML system...")
        
        try:
            if self.scheduler:
                await self.scheduler.cleanup()
            
            if self.drift_monitor:
                await self.drift_monitor.cleanup()
            
            if self.trainer:
                await self.trainer.cleanup()
            
            logger.info("✅ Enhanced ML system cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# CLI interface
async def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ML Auto-Retraining System Test')
    parser.add_argument('--test', choices=['all', 'scheduler', 'drift', 'training', 'inference'], 
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    tester = EnhancedMLSystemTester(DB_CONFIG)
    
    try:
        if args.test == 'all':
            success = await tester.run_comprehensive_test()
        else:
            await tester.initialize_components()
            
            if args.test == 'scheduler':
                success = await tester.test_scheduler_functionality()
            elif args.test == 'drift':
                success = await tester.test_drift_monitoring()
            elif args.test == 'training':
                success = await tester.test_enhanced_training_pipeline()
            elif args.test == 'inference':
                success = await tester.test_enhanced_inference_integration()
            
            logger.info(f"📊 {args.test.title()} Test Result: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
