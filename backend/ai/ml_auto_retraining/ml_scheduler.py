#!/usr/bin/env python3
"""
ML Auto-Retraining Scheduler
Automates daily/weekly model retraining with drift detection and performance monitoring
"""

import os
import sys
import json
import logging
import argparse
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing ML components
from ai.ml_auto_retraining.train_model import MLModelTrainer
from ai.ml_auto_retraining.evaluate_and_promote import ModelEvaluator
from ai.ml_auto_retraining.ml_inference_engine import MLInferenceEngine
from ai.noise_filter_engine import NoiseFilterEngine
from ai.market_regime_classifier import MarketRegimeClassifier
from ai.adaptive_learning_engine import AdaptiveLearningEngine

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

@dataclass
class TrainingJob:
    """Training job configuration"""
    symbol: str
    regime: str
    model_name: str
    training_days: int
    schedule: str  # cron expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = 'pending'

class MLScheduler:
    """ML Auto-Retraining Scheduler"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.scheduler = BackgroundScheduler()
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.trainer = None
        self.evaluator = None
        
    async def initialize(self):
        """Initialize scheduler components"""
        logger.info("🔧 Initializing ML Scheduler...")
        
        try:
            # Initialize ML components
            self.trainer = MLModelTrainer(self.db_config)
            await self.trainer.initialize_components()
            
            self.evaluator = ModelEvaluator(self.db_config)
            
            # Load training jobs from database
            await self._load_training_jobs()
            
            # Setup scheduler events
            self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
            
            logger.info("✅ ML Scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Scheduler: {e}")
            raise
    
    async def _load_training_jobs(self):
        """Load training jobs from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Load default training jobs
            default_jobs = [
                TrainingJob('BTCUSDT', 'trending', 'alphaplus_pattern_classifier', 120, '0 3 * * *'),
                TrainingJob('BTCUSDT', 'sideways', 'alphaplus_pattern_classifier', 120, '0 4 * * *'),
                TrainingJob('BTCUSDT', 'volatile', 'alphaplus_pattern_classifier', 120, '0 5 * * *'),
                TrainingJob('BTCUSDT', 'consolidation', 'alphaplus_pattern_classifier', 120, '0 6 * * *'),
                TrainingJob('ETHUSDT', 'trending', 'alphaplus_pattern_classifier', 120, '0 7 * * *'),
                TrainingJob('ETHUSDT', 'sideways', 'alphaplus_pattern_classifier', 120, '0 8 * * *'),
            ]
            
            for job in default_jobs:
                job_id = f"{job.symbol}_{job.regime}"
                self.training_jobs[job_id] = job
                
                # Add job to scheduler
                self.scheduler.add_job(
                    func=self._execute_training_job,
                    trigger=CronTrigger.from_crontab(job.schedule),
                    args=[job_id],
                    id=job_id,
                    name=f"Train {job.symbol} {job.regime}",
                    replace_existing=True
                )
                
                logger.info(f"✅ Scheduled training job: {job_id} at {job.schedule}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to load training jobs: {e}")
            raise
    
    async def _execute_training_job(self, job_id: str):
        """Execute a training job"""
        job = self.training_jobs.get(job_id)
        if not job:
            logger.error(f"❌ Training job not found: {job_id}")
            return
        
        logger.info(f"🚀 Starting training job: {job_id}")
        
        try:
            # Update job status
            job.status = 'running'
            job.last_run = datetime.now()
            await self._update_job_status(job_id, 'running')
            
            # Execute training
            # Use date range with actual data (2024 data available)
            end_date = datetime(2024, 12, 31)
            start_date = end_date - timedelta(days=job.training_days)
            
            # Train model
            model_path = await self.trainer.train_model(
                symbol=job.symbol,
                regime=job.regime,
                start_date=start_date,
                end_date=end_date,
                model_name=job.model_name
            )
            
            if model_path:
                # Evaluate and potentially promote model
                promotion_result = await self.evaluator.evaluate_and_promote_model(
                    model_path=model_path,
                    symbol=job.symbol,
                    regime=job.regime,
                    model_name=job.model_name
                )
                
                # Update job status
                job.status = 'completed'
                await self._update_job_status(job_id, 'completed', {
                    'model_path': model_path,
                    'promotion_result': promotion_result
                })
                
                logger.info(f"✅ Training job completed: {job_id}")
                
            else:
                job.status = 'failed'
                await self._update_job_status(job_id, 'failed', {'error': 'Training failed'})
                logger.error(f"❌ Training job failed: {job_id}")
                
        except Exception as e:
            job.status = 'failed'
            await self._update_job_status(job_id, 'failed', {'error': str(e)})
            logger.error(f"❌ Training job error: {job_id} - {e}")
            raise
    
    async def _update_job_status(self, job_id: str, status: str, metadata: Dict[str, Any] = None):
        """Update job status in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_training_jobs (
                    job_id, symbol, regime, model_name, status, 
                    last_run, metadata, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s::jsonb, NOW()
                )
            """, (
                job_id,
                self.training_jobs[job_id].symbol,
                self.training_jobs[job_id].regime,
                self.training_jobs[job_id].model_name,
                status,
                datetime.now(),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update job status: {e}")
    
    def _job_executed(self, event):
        """Handle successful job execution"""
        logger.info(f"✅ Job executed successfully: {event.job_id}")
    
    def _job_error(self, event):
        """Handle job execution error"""
        logger.error(f"❌ Job execution failed: {event.job_id} - {event.exception}")
    
    def start(self):
        """Start the scheduler"""
        logger.info("🚀 Starting ML Scheduler...")
        self.scheduler.start()
        logger.info("✅ ML Scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("🛑 Stopping ML Scheduler...")
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("✅ ML Scheduler stopped")
        else:
            logger.info("ℹ️ ML Scheduler was not running")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all training jobs"""
        status = {}
        for job_id, job in self.training_jobs.items():
            status[job_id] = {
                'symbol': job.symbol,
                'regime': job.regime,
                'status': job.status,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'enabled': job.enabled
            }
        return status
    
    async def trigger_manual_training(self, symbol: str, regime: str, model_name: str = 'alphaplus_pattern_classifier'):
        """Trigger manual training for a specific symbol and regime"""
        job_id = f"{symbol}_{regime}"
        
        if job_id not in self.training_jobs:
            # Create temporary job
            job = TrainingJob(symbol, regime, model_name, 120, 'manual')
            self.training_jobs[job_id] = job
        
        logger.info(f"🔧 Triggering manual training: {job_id}")
        await self._execute_training_job(job_id)
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up ML Scheduler...")
        
        if self.trainer:
            await self.trainer.cleanup()
        
        self.stop()
        logger.info("✅ ML Scheduler cleanup completed")

# CLI interface
async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='ML Auto-Retraining Scheduler')
    parser.add_argument('--action', choices=['start', 'stop', 'status', 'manual'], required=True)
    parser.add_argument('--symbol', help='Symbol for manual training')
    parser.add_argument('--regime', help='Regime for manual training')
    parser.add_argument('--model_name', default='alphaplus_pattern_classifier', help='Model name')
    
    args = parser.parse_args()
    
    scheduler = MLScheduler(DB_CONFIG)
    
    try:
        if args.action == 'start':
            await scheduler.initialize()
            scheduler.start()
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                
        elif args.action == 'stop':
            scheduler.stop()
            logger.info("✅ Scheduler stopped")
            
        elif args.action == 'status':
            await scheduler.initialize()
            status = scheduler.get_job_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.action == 'manual':
            if not args.symbol or not args.regime:
                logger.error("❌ Symbol and regime required for manual training")
                return
                
            await scheduler.initialize()
            await scheduler.trigger_manual_training(args.symbol, args.regime, args.model_name)
            
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")
    finally:
        await scheduler.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
