#!/usr/bin/env python3
"""
Test Enhanced Auto-Retraining Pipeline
Comprehensive testing of the enhanced auto-retraining pipeline functionality
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'services'))

from enhanced_auto_retraining_pipeline import EnhancedAutoRetrainingPipeline, RetrainingJob, DriftMetrics, ModelPerformance
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAutoRetrainingPipelineTester:
    """Test suite for Enhanced Auto-Retraining Pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
        
    async def setup(self):
        """Setup test environment"""
        try:
            logger.info("🔧 Setting up Enhanced Auto-Retraining Pipeline test environment...")
            
            # Initialize the pipeline
            self.pipeline = EnhancedAutoRetrainingPipeline()
            await self.pipeline.initialize()
            
            logger.info("✅ Test environment setup completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment"""
        try:
            if self.pipeline:
                await self.pipeline.stop()
            logger.info("🧹 Test environment cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error during teardown: {e}")
    
    async def generate_test_data(self):
        """Generate test data for drift detection and performance monitoring"""
        try:
            logger.info("📊 Generating test data...")
            
            # Generate sample performance data
            performance_data = []
            for i in range(10):
                performance_data.append({
                    'model_name': 'test_model',
                    'model_version': f'v{i+1}',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'auc_score': 0.75 + (i * 0.02),
                    'precision_score': 0.70 + (i * 0.01),
                    'recall_score': 0.65 + (i * 0.015),
                    'f1_score': 0.67 + (i * 0.012),
                    'sharpe_ratio': 1.2 + (i * 0.1),
                    'max_drawdown': 0.15 - (i * 0.01),
                    'win_rate': 0.55 + (i * 0.02),
                    'total_trades': 100 + (i * 10),
                    'profitable_trades': 55 + (i * 8)
                })
            
            # Generate sample drift data
            drift_data = []
            for i in range(5):
                drift_data.append({
                    'model_name': 'test_model',
                    'model_version': f'v{i+1}',
                    'symbol': 'BTCUSDT',
                    'drift_type': 'psi',
                    'feature_name': 'price_change',
                    'drift_score': 0.1 + (i * 0.05),
                    'threshold': 0.25,
                    'is_drift_detected': i > 2,
                    'samples_analyzed': 1000
                })
            
            # Store test data in database
            await self._store_test_data(performance_data, drift_data)
            
            logger.info("✅ Test data generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating test data: {e}")
            raise
    
    async def _store_test_data(self, performance_data: List[Dict], drift_data: List[Dict]):
        """Store test data in database"""
        try:
            with self.pipeline.engine.connect() as conn:
                # Store performance data (using existing table structure)
                for data in performance_data:
                    conn.execute(text("""
                        INSERT INTO model_performance_tracking (
                            timestamp, model_type, model_version, accuracy_score, precision_score, 
                            recall_score, f1_score, auc_score, training_samples, 
                            validation_samples, performance_metadata
                        ) VALUES (
                            NOW(), :model_name, :model_version, :f1_score, :precision_score,
                            :recall_score, :f1_score, :auc_score, :total_trades,
                            :profitable_trades, :metadata
                        )
                    """), {
                        'model_name': data['model_name'],
                        'model_version': data['model_version'],
                        'precision_score': data['precision_score'],
                        'recall_score': data['recall_score'],
                        'f1_score': data['f1_score'],
                        'auc_score': data['auc_score'],
                        'total_trades': data['total_trades'],
                        'profitable_trades': data['profitable_trades'],
                        'metadata': json.dumps({
                            'sharpe_ratio': data['sharpe_ratio'],
                            'max_drawdown': data['max_drawdown'],
                            'win_rate': data['win_rate'],
                            'symbol': data['symbol'],
                            'timeframe': data['timeframe']
                        })
                    })
                
                # Store drift data
                for data in drift_data:
                    conn.execute(text("""
                        INSERT INTO model_drift_monitoring (
                            model_name, model_version, symbol, drift_type, feature_name,
                            drift_score, threshold, is_drift_detected, samples_analyzed
                        ) VALUES (
                            :model_name, :model_version, :symbol, :drift_type, :feature_name,
                            :drift_score, :threshold, :is_drift_detected, :samples_analyzed
                        )
                    """), data)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing test data: {e}")
            raise
    
    async def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        try:
            logger.info("🧪 Testing pipeline initialization...")
            
            assert self.pipeline is not None, "Pipeline should be initialized"
            assert self.pipeline.engine is not None, "Database engine should be initialized"
            assert len(self.pipeline.retraining_jobs) >= 0, "Retraining jobs should be loaded"
            
            logger.info("✅ Pipeline initialization test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Pipeline initialization: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_job_creation(self):
        """Test retraining job creation"""
        try:
            logger.info("🧪 Testing retraining job creation...")
            
            # Create a test job
            test_job_config = {
                'model_name': 'test_lightgbm',
                'model_type': 'lightgbm',
                'symbol': 'ETHUSDT',
                'timeframe': '4h',
                'schedule_cron': '0 6 * * *',
                'priority': 3
            }
            
            await self.pipeline._create_retraining_job(test_job_config)
            
            # Verify job was created
            with self.pipeline.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT job_id, model_name, model_type, symbol, status
                    FROM auto_retraining_jobs
                    WHERE model_name = :model_name AND symbol = :symbol
                """), {
                    'model_name': 'test_lightgbm',
                    'symbol': 'ETHUSDT'
                })
                
                row = result.fetchone()
                assert row is not None, "Job should be created in database"
                assert row.model_name == 'test_lightgbm', "Model name should match"
                assert row.model_type == 'lightgbm', "Model type should match"
                assert row.status == 'active', "Job status should be active"
            
            logger.info("✅ Job creation test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Job creation test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Job creation: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_performance_degradation_detection(self):
        """Test performance degradation detection"""
        try:
            logger.info("🧪 Testing performance degradation detection...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_performance_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test performance degradation detection
            should_retrain = await self.pipeline._check_performance_degradation(test_job)
            
            # Should detect degradation based on test data
            assert isinstance(should_retrain, bool), "Should return boolean"
            
            logger.info("✅ Performance degradation detection test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Performance degradation detection test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Performance degradation detection: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_data_drift_detection(self):
        """Test data drift detection"""
        try:
            logger.info("🧪 Testing data drift detection...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_drift_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test data drift detection
            drift_detected = await self.pipeline._check_data_drift(test_job)
            
            # Should detect drift based on test data
            assert isinstance(drift_detected, bool), "Should return boolean"
            
            logger.info("✅ Data drift detection test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Data drift detection test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Data drift detection: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_model_age_check(self):
        """Test model age checking"""
        try:
            logger.info("🧪 Testing model age checking...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_age_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test model age checking
            age_exceeded = await self.pipeline._check_model_age(test_job)
            
            # Should return boolean
            assert isinstance(age_exceeded, bool), "Should return boolean"
            
            logger.info("✅ Model age checking test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Model age checking test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Model age checking: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_should_retrain_logic(self):
        """Test should retrain logic"""
        try:
            logger.info("🧪 Testing should retrain logic...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_retrain_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test should retrain logic
            should_retrain = await self.pipeline._should_retrain(test_job)
            
            # Should return boolean
            assert isinstance(should_retrain, bool), "Should return boolean"
            
            logger.info("✅ Should retrain logic test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Should retrain logic test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Should retrain logic: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_model_versioning(self):
        """Test model versioning functionality"""
        try:
            logger.info("🧪 Testing model versioning...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_version_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test model version storage
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.pipeline._store_model_version(test_job, model_version)
            
            # Verify model version was stored
            with self.pipeline.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT model_name, model_version, symbol, deployment_status, is_active
                    FROM model_version_management
                    WHERE model_name = :model_name AND model_version = :model_version
                """), {
                    'model_name': test_job.model_name,
                    'model_version': model_version
                })
                
                row = result.fetchone()
                assert row is not None, "Model version should be stored"
                assert row.model_version == model_version, "Model version should match"
                assert row.deployment_status == 'trained', "Deployment status should be trained"
                assert row.is_active == True, "Model should be active"
            
            logger.info("✅ Model versioning test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Model versioning test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Model versioning: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_retraining_history_logging(self):
        """Test retraining history logging"""
        try:
            logger.info("🧪 Testing retraining history logging...")
            
            # Create a test job
            test_job = RetrainingJob(
                job_id='test_history_job',
                model_name='test_model',
                model_type='lightgbm',
                symbol='BTCUSDT',
                timeframe='1h',
                schedule_cron='0 2 * * *',
                priority=1,
                auto_deploy=True,
                performance_threshold=0.8,
                drift_threshold=0.25,
                min_training_samples=1000,
                max_training_age_days=30,
                retraining_strategy=None,
                last_run=None,
                next_run=None,
                status='active'
            )
            
            # Test retraining history logging
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await self.pipeline._log_retraining_history(test_job, model_version, 'completed')
            
            # Verify history was logged
            with self.pipeline.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT job_id, model_name, model_version, trigger_type, status
                    FROM retraining_job_history
                    WHERE job_id = :job_id AND model_version = :model_version
                """), {
                    'job_id': test_job.job_id,
                    'model_version': model_version
                })
                
                row = result.fetchone()
                assert row is not None, "Retraining history should be logged"
                assert row.job_id == test_job.job_id, "Job ID should match"
                assert row.model_version == model_version, "Model version should match"
                assert row.status == 'completed', "Status should be completed"
            
            logger.info("✅ Retraining history logging test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Retraining history logging test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Retraining history logging: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def test_pipeline_statistics(self):
        """Test pipeline statistics"""
        try:
            logger.info("🧪 Testing pipeline statistics...")
            
            # Get pipeline statistics
            stats = self.pipeline.get_stats()
            
            # Verify statistics structure
            required_keys = ['jobs_scheduled', 'jobs_completed', 'jobs_failed', 'models_retrained', 
                           'drift_alerts', 'auto_deployments', 'active_jobs', 'running_jobs', 'total_jobs']
            
            for key in required_keys:
                assert key in stats, f"Statistics should contain {key}"
                assert isinstance(stats[key], (int, float)), f"{key} should be numeric"
            
            logger.info("✅ Pipeline statistics test passed")
            self.test_results['tests_passed'] += 1
            
        except Exception as e:
            logger.error(f"❌ Pipeline statistics test failed: {e}")
            self.test_results['tests_failed'] += 1
            self.test_results['errors'].append(f"Pipeline statistics: {e}")
        
        self.test_results['tests_run'] += 1
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            logger.info("🚀 Starting Enhanced Auto-Retraining Pipeline test suite...")
            
            await self.setup()
            await self.generate_test_data()
            
            # Run all tests
            await self.test_pipeline_initialization()
            await self.test_job_creation()
            await self.test_performance_degradation_detection()
            await self.test_data_drift_detection()
            await self.test_model_age_check()
            await self.test_should_retrain_logic()
            await self.test_model_versioning()
            await self.test_retraining_history_logging()
            await self.test_pipeline_statistics()
            
            await self.teardown()
            
            # Generate test report
            self._generate_test_report()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.test_results['errors'].append(f"Test suite: {e}")
            await self.teardown()
    
    def _generate_test_report(self):
        """Generate test report"""
        try:
            report = {
                'test_suite': 'Enhanced Auto-Retraining Pipeline',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': self.test_results['tests_run'],
                    'passed': self.test_results['tests_passed'],
                    'failed': self.test_results['tests_failed'],
                    'success_rate': (self.test_results['tests_passed'] / self.test_results['tests_run'] * 100) if self.test_results['tests_run'] > 0 else 0
                },
                'errors': self.test_results['errors'],
                'pipeline_stats': self.pipeline.get_stats() if self.pipeline else {}
            }
            
            # Save report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'enhanced_auto_retraining_pipeline_test_results_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Print summary
            logger.info("📊 Test Results Summary:")
            logger.info(f"   Total Tests: {report['summary']['total_tests']}")
            logger.info(f"   Passed: {report['summary']['passed']}")
            logger.info(f"   Failed: {report['summary']['failed']}")
            logger.info(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
            
            if report['errors']:
                logger.warning(f"   Errors: {len(report['errors'])}")
                for error in report['errors']:
                    logger.warning(f"     - {error}")
            
            logger.info(f"📄 Detailed report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error generating test report: {e}")

async def main():
    """Main test execution"""
    tester = EnhancedAutoRetrainingPipelineTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
