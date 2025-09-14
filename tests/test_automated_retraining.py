"""
Test script for Automated Model Retraining Engine
Phase 3: Automated Model Retraining and Deployment Pipeline
"""

import asyncio
import asyncpg
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from ai.automated_model_retraining import (
    AutomatedModelRetrainingEngine, 
    AutomatedRetrainingScheduler,
    TrainingJob, 
    TriggerType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedRetrainingTest:
    """Test the automated retraining engine"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'alpha_emon',
            'password': 'Emon_@17711',
            'database': 'alphapulse'
        }
        self.db_pool = None
        self.engine = None
        self.scheduler = None
    
    async def setup(self):
        """Setup test environment"""
        try:
            self.db_pool = await asyncpg.create_pool(**self.db_config)
            self.engine = AutomatedModelRetrainingEngine(self.db_pool)
            self.scheduler = AutomatedRetrainingScheduler(self.engine)
            logger.info("✅ Test environment setup completed")
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup test environment"""
        if self.db_pool:
            await self.db_pool.close()
        logger.info("✅ Test environment cleaned up")
    
    async def test_trigger_evaluation(self):
        """Test trigger evaluation functionality"""
        logger.info("🔍 Testing Trigger Evaluation...")
        
        try:
            # Test drift trigger evaluation
            async with self.db_pool.acquire() as conn:
                # Insert mock drift data
                await conn.execute("""
                    INSERT INTO model_drift_detection (
                        model_id, detection_timestamp, feature_drift_score, 
                        concept_drift_score, data_drift_score, drift_severity
                    ) VALUES (
                        (SELECT id FROM model_training_jobs WHERE model_name = 'catboost_signal_predictor' LIMIT 1),
                        $1, $2, $3, $4, $5
                    )
                """, datetime.now(), 0.35, 0.30, 0.25, 'high')
                
                # Check triggers
                new_jobs = await self.engine.check_retraining_triggers()
                logger.info(f"📊 Trigger evaluation found {len(new_jobs)} new jobs to create")
                
                return len(new_jobs) > 0
                
        except Exception as e:
            logger.error(f"❌ Trigger evaluation test failed: {e}")
            return False
    
    async def test_job_submission(self):
        """Test job submission functionality"""
        logger.info("📝 Testing Job Submission...")
        
        try:
            # Create a test job
            test_job = TrainingJob(
                job_id=str(uuid.uuid4()),
                model_name="catboost_signal_predictor",
                model_version="v20241201_120000",
                training_type="retrain",
                priority=8,
                training_config={
                    'training_type': 'retrain',
                    'hyperparameters': {'iterations': 1000, 'learning_rate': 0.1},
                    'validation_split': 0.2
                },
                hyperparameters={'iterations': 1000, 'learning_rate': 0.1},
                data_sources={'historical_data': True, 'real_time_data': True},
                trigger_type=TriggerType.MANUAL,
                trigger_metadata={'test': True}
            )
            
            # Submit the job
            success = await self.engine.submit_training_job(test_job)
            logger.info(f"📊 Job submission: {'SUCCESS' if success else 'FAILED'}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Job submission test failed: {e}")
            return False
    
    async def test_job_management(self):
        """Test job management functionality"""
        logger.info("⚙️ Testing Job Management...")
        
        try:
            # Get pending jobs
            pending_jobs = await self.engine.get_pending_jobs()
            logger.info(f"📊 Found {len(pending_jobs)} pending jobs")
            
            # Get running jobs
            running_jobs = await self.engine.get_running_jobs()
            logger.info(f"📊 Found {len(running_jobs)} running jobs")
            
            # Test job status retrieval
            if pending_jobs:
                job_status = await self.engine.get_job_status(pending_jobs[0]['job_id'])
                logger.info(f"📊 Job status: {job_status['status'] if job_status else 'NOT_FOUND'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Job management test failed: {e}")
            return False
    
    async def test_scheduler_functionality(self):
        """Test scheduler functionality"""
        logger.info("⏰ Testing Scheduler Functionality...")
        
        try:
            # Start scheduler for a short time
            scheduler_task = asyncio.create_task(self.scheduler.start())
            
            # Let it run for a few seconds
            await asyncio.sleep(5)
            
            # Stop scheduler
            await self.scheduler.stop()
            
            # Wait for scheduler to stop
            try:
                await asyncio.wait_for(scheduler_task, timeout=10)
            except asyncio.TimeoutError:
                scheduler_task.cancel()
            
            logger.info("📊 Scheduler test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scheduler test failed: {e}")
            return False
    
    async def test_database_integration(self):
        """Test database integration"""
        logger.info("🗄️ Testing Database Integration...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Check if tables exist and have data
                tables_to_check = [
                    'model_training_jobs',
                    'automated_retraining_triggers',
                    'model_drift_detection',
                    'model_performance_tracking'
                ]
                
                for table in tables_to_check:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    logger.info(f"📊 Table {table}: {count} records")
                
                # Check recent training jobs
                recent_jobs = await conn.fetch("""
                    SELECT model_name, status, created_at 
                    FROM model_training_jobs 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                
                logger.info(f"📊 Recent training jobs: {len(recent_jobs)} found")
                for job in recent_jobs:
                    logger.info(f"  - {job['model_name']}: {job['status']} ({job['created_at']})")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Database integration test failed: {e}")
            return False
    
    async def test_manual_job_creation(self):
        """Test manual job creation"""
        logger.info("🛠️ Testing Manual Job Creation...")
        
        try:
            # Create a manual trigger
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO automated_retraining_triggers (
                        model_name, trigger_type, trigger_conditions, trigger_thresholds,
                        retraining_config, priority, is_active
                    ) VALUES (
                        'test_model', 'manual', '{"manual": true}', '{"manual": true}',
                        '{"training_type": "retrain", "hyperparameters": {"iterations": 500}}', 10, true
                    )
                """)
                
                logger.info("📊 Manual trigger created")
                
                # Check if trigger is detected
                new_jobs = await self.engine.check_retraining_triggers()
                logger.info(f"📊 Manual trigger created {len(new_jobs)} jobs")
                
                return len(new_jobs) > 0
                
        except Exception as e:
            logger.error(f"❌ Manual job creation test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            
            logger.info("🚀 Starting Automated Retraining Engine Tests...")
            logger.info("=" * 60)
            
            test_results = {}
            
            # Run individual tests
            test_results['trigger_evaluation'] = await self.test_trigger_evaluation()
            test_results['job_submission'] = await self.test_job_submission()
            test_results['job_management'] = await self.test_job_management()
            test_results['scheduler_functionality'] = await self.test_scheduler_functionality()
            test_results['database_integration'] = await self.test_database_integration()
            test_results['manual_job_creation'] = await self.test_manual_job_creation()
            
            # Summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("📈 AUTOMATED RETRAINING ENGINE TEST SUMMARY:")
            logger.info("=" * 60)
            
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{status} {test_name}")
            
            logger.info("")
            logger.info(f"Overall Result: {passed_tests}/{total_tests} tests passed")
            
            if passed_tests == total_tests:
                logger.info("🎉 ALL TESTS PASSED! Automated retraining engine is working correctly.")
            else:
                logger.warning("⚠️ Some tests failed. Please check the logs above.")
            
            return passed_tests == total_tests
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return False
        finally:
            await self.cleanup()

async def main():
    """Main test function"""
    test = AutomatedRetrainingTest()
    success = await test.run_all_tests()
    
    if success:
        logger.info("✅ Automated retraining engine test completed successfully!")
    else:
        logger.error("❌ Automated retraining engine test failed!")

if __name__ == "__main__":
    asyncio.run(main())
