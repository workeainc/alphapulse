"""
Automated Model Retraining Engine for AlphaPlus
Phase 3: Automated Model Retraining and Deployment Pipeline
"""

import asyncio
import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TriggerType(Enum):
    DRIFT_THRESHOLD = "drift_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TIME_BASED = "time_based"
    MANUAL = "manual"

@dataclass
class TrainingJob:
    """Training job configuration"""
    job_id: str
    model_name: str
    model_version: str
    training_type: str
    priority: int
    training_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    data_sources: Dict[str, Any]
    trigger_type: TriggerType
    trigger_metadata: Dict[str, Any]

@dataclass
class TrainingResult:
    """Training result"""
    job_id: str
    model_name: str
    model_version: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_artifacts: Dict[str, Any]
    training_duration: float
    status: TrainingStatus
    error_message: Optional[str] = None

class AutomatedModelRetrainingEngine:
    """Automated model retraining engine with database integration"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logger
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.job_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Training configurations for different models
        self.model_configs = {
            'catboost_signal_predictor': {
                'training_type': 'retrain',
                'hyperparameters': {
                    'iterations': 1000,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'random_seed': 42
                },
                'validation_split': 0.2,
                'early_stopping_rounds': 50
            },
            'xgboost_signal_predictor': {
                'training_type': 'incremental',
                'hyperparameters': {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'validation_split': 0.2,
                'early_stopping_rounds': 50
            },
            'lightgbm_signal_predictor': {
                'training_type': 'retrain',
                'hyperparameters': {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'validation_split': 0.2,
                'early_stopping_rounds': 50
            }
        }
    
    async def check_retraining_triggers(self) -> List[TrainingJob]:
        """Check all retraining triggers and return jobs that need to be created"""
        jobs_to_create = []
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get all active triggers
                triggers = await conn.fetch("""
                    SELECT * FROM automated_retraining_triggers 
                    WHERE is_active = TRUE
                    ORDER BY priority DESC
                """)
                
                for trigger in triggers:
                    should_trigger = await self._evaluate_trigger(conn, trigger)
                    if should_trigger:
                        job = await self._create_training_job(trigger)
                        jobs_to_create.append(job)
                        
        except Exception as e:
            self.logger.error(f"Error checking retraining triggers: {e}")
        
        return jobs_to_create
    
    async def _evaluate_trigger(self, conn: asyncpg.Connection, trigger: asyncpg.Record) -> bool:
        """Evaluate if a trigger should fire"""
        trigger_type = trigger['trigger_type']
        
        # Parse JSON fields
        try:
            conditions = trigger['trigger_conditions'] if isinstance(trigger['trigger_conditions'], dict) else json.loads(trigger['trigger_conditions'])
            thresholds = trigger['trigger_thresholds'] if isinstance(trigger['trigger_thresholds'], dict) else json.loads(trigger['trigger_thresholds'])
        except (json.JSONDecodeError, TypeError):
            self.logger.error(f"Invalid JSON in trigger {trigger['trigger_id']}")
            return False
        
        try:
            if trigger_type == TriggerType.DRIFT_THRESHOLD.value:
                return await self._evaluate_drift_trigger(conn, conditions, thresholds)
            elif trigger_type == TriggerType.PERFORMANCE_DEGRADATION.value:
                return await self._evaluate_performance_trigger(conn, conditions, thresholds)
            elif trigger_type == TriggerType.TIME_BASED.value:
                return await self._evaluate_time_trigger(conn, conditions, thresholds)
            elif trigger_type == 'manual':
                return True  # Manual triggers always fire
            else:
                self.logger.warning(f"Unknown trigger type: {trigger_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating trigger {trigger['trigger_id']}: {e}")
            return False
    
    async def _evaluate_drift_trigger(self, conn: asyncpg.Connection, conditions: Dict, thresholds: Dict) -> bool:
        """Evaluate drift-based trigger"""
        try:
            # Get latest drift scores for the model
            drift_scores = await conn.fetch("""
                SELECT feature_drift_score, concept_drift_score, data_drift_score
                FROM model_drift_detection 
                WHERE model_id IN (
                    SELECT id FROM model_training_jobs 
                    WHERE model_name = $1 
                    ORDER BY created_at DESC 
                    LIMIT 1
                )
                ORDER BY detection_timestamp DESC 
                LIMIT 1
            """, conditions.get('model_name', 'catboost_signal_predictor'))
            
            if not drift_scores:
                return False
            
            latest_drift = drift_scores[0]
            
            # Check if any drift score exceeds threshold
            feature_threshold = thresholds.get('feature_drift_threshold', 0.3)
            concept_threshold = thresholds.get('concept_drift_threshold', 0.25)
            
            if (latest_drift['feature_drift_score'] > feature_threshold or 
                latest_drift['concept_drift_score'] > concept_threshold):
                self.logger.info(f"Drift trigger fired: feature={latest_drift['feature_drift_score']:.3f}, concept={latest_drift['concept_drift_score']:.3f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating drift trigger: {e}")
            return False
    
    async def _evaluate_performance_trigger(self, conn: asyncpg.Connection, conditions: Dict, thresholds: Dict) -> bool:
        """Evaluate performance-based trigger"""
        try:
            # Get latest performance metrics
            performance = await conn.fetch("""
                SELECT accuracy, f1_score, precision, recall
                FROM model_performance_tracking 
                WHERE model_id IN (
                    SELECT id FROM model_training_jobs 
                    WHERE model_name = $1 
                    ORDER BY created_at DESC 
                    LIMIT 1
                )
                ORDER BY evaluation_timestamp DESC 
                LIMIT 1
            """, conditions.get('model_name', 'catboost_signal_predictor'))
            
            if not performance:
                return False
            
            latest_perf = performance[0]
            
            # Check if performance is below thresholds
            accuracy_threshold = thresholds.get('accuracy_threshold', 0.85)
            f1_threshold = thresholds.get('f1_threshold', 0.80)
            
            if (latest_perf['accuracy'] < accuracy_threshold or 
                latest_perf['f1_score'] < f1_threshold):
                self.logger.info(f"Performance trigger fired: accuracy={latest_perf['accuracy']:.3f}, f1={latest_perf['f1_score']:.3f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance trigger: {e}")
            return False
    
    async def _evaluate_time_trigger(self, conn: asyncpg.Connection, conditions: Dict, thresholds: Dict) -> bool:
        """Evaluate time-based trigger"""
        try:
            # Get last training time
            last_training = await conn.fetchval("""
                SELECT created_at FROM model_training_jobs 
                WHERE model_name = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """, conditions.get('model_name', 'catboost_signal_predictor'))
            
            if not last_training:
                return True  # No previous training, trigger immediately
            
            days_since_training = (datetime.now() - last_training).days
            max_days = thresholds.get('max_days_between_training', 7)
            
            if days_since_training > max_days:
                self.logger.info(f"Time trigger fired: {days_since_training} days since last training")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating time trigger: {e}")
            return False
    
    async def _create_training_job(self, trigger: asyncpg.Record) -> TrainingJob:
        """Create a training job from a trigger"""
        model_name = trigger['model_name']
        job_id = str(uuid.uuid4())
        
        # Get model configuration
        model_config = self.model_configs.get(model_name, self.model_configs['catboost_signal_predictor'])
        
        # Generate new version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"v{timestamp}"
        
        # Parse retraining config
        try:
            retraining_config = trigger['retraining_config'] if isinstance(trigger['retraining_config'], dict) else json.loads(trigger['retraining_config'])
        except (json.JSONDecodeError, TypeError):
            retraining_config = model_config
        
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            model_version=model_version,
            training_type=model_config['training_type'],
            priority=trigger['priority'],
            training_config=retraining_config,
            hyperparameters=model_config['hyperparameters'],
            data_sources={
                'historical_data': True,
                'real_time_data': True,
                'synthetic_data': False
            },
            trigger_type=TriggerType(trigger['trigger_type']) if trigger['trigger_type'] != 'manual' else TriggerType.MANUAL,
            trigger_metadata={
                'trigger_id': str(trigger['trigger_id']),
                'trigger_conditions': trigger['trigger_conditions'],
                'trigger_thresholds': trigger['trigger_thresholds']
            }
        )
        
        return job
    
    async def submit_training_job(self, job: TrainingJob) -> bool:
        """Submit a training job to the database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_training_jobs (
                        job_id, model_name, model_version, training_type, status, priority,
                        training_config, hyperparameters, feature_config, data_sources,
                        trigger_type, trigger_metadata, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, 
                job.job_id, job.model_name, job.model_version, job.training_type,
                TrainingStatus.PENDING.value, job.priority, json.dumps(job.training_config),
                json.dumps(job.hyperparameters), json.dumps({}), json.dumps(job.data_sources),
                job.trigger_type.value, json.dumps(job.trigger_metadata), datetime.now()
                )
                
                self.logger.info(f"Training job submitted: {job.job_id} for {job.model_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error submitting training job: {e}")
            return False
    
    async def start_training_job(self, job_id: str) -> bool:
        """Start a training job"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get job details
                job_record = await conn.fetchrow("""
                    SELECT * FROM model_training_jobs WHERE job_id = $1
                """, job_id)
                
                if not job_record:
                    self.logger.error(f"Job not found: {job_id}")
                    return False
                
                # Update status to running
                await conn.execute("""
                    UPDATE model_training_jobs 
                    SET status = $1, started_at = $2 
                    WHERE job_id = $3
                """, TrainingStatus.RUNNING.value, datetime.now(), job_id)
                
                # Add to running jobs
                with self.job_lock:
                    self.running_jobs[job_id] = TrainingJob(
                        job_id=job_record['job_id'],
                        model_name=job_record['model_name'],
                        model_version=job_record['model_version'],
                        training_type=job_record['training_type'],
                        priority=job_record['priority'],
                        training_config=job_record['training_config'],
                        hyperparameters=job_record['hyperparameters'],
                        data_sources=job_record['data_sources'],
                        trigger_type=TriggerType(job_record['trigger_type']),
                        trigger_metadata=job_record['trigger_metadata']
                    )
                
                # Start training in background
                asyncio.create_task(self._execute_training_job(job_id))
                
                self.logger.info(f"Training job started: {job_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error starting training job: {e}")
            return False
    
    async def _execute_training_job(self, job_id: str):
        """Execute a training job"""
        start_time = time.time()
        
        try:
            # Simulate training process
            self.logger.info(f"Starting training execution for job: {job_id}")
            
            # Update progress periodically
            for epoch in range(1, 11):
                await asyncio.sleep(2)  # Simulate training time
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE model_training_jobs 
                        SET current_epoch = $1, current_metric = $2, training_loss = $3
                        WHERE job_id = $4
                    """, epoch, 0.85 + (epoch * 0.01), 0.15 - (epoch * 0.01), job_id)
            
            # Training completed successfully
            training_duration = time.time() - start_time
            
            # Generate mock results
            result = TrainingResult(
                job_id=job_id,
                model_name="catboost_signal_predictor",  # Will be updated from DB
                model_version="v20241201_120000",
                training_metrics={
                    'accuracy': 0.89,
                    'precision': 0.87,
                    'recall': 0.91,
                    'f1_score': 0.89,
                    'auc_roc': 0.92
                },
                validation_metrics={
                    'accuracy': 0.87,
                    'precision': 0.85,
                    'recall': 0.89,
                    'f1_score': 0.87,
                    'auc_roc': 0.90
                },
                model_artifacts={
                    'model_path': f"/models/{job_id}/model.onnx",
                    'feature_importance': {"feature1": 0.3, "feature2": 0.25},
                    'model_size_mb': 15.5
                },
                training_duration=training_duration,
                status=TrainingStatus.COMPLETED
            )
            
            await self._complete_training_job(result)
            
        except Exception as e:
            self.logger.error(f"Training job failed: {job_id}, error: {e}")
            
            result = TrainingResult(
                job_id=job_id,
                model_name="unknown",
                model_version="unknown",
                training_metrics={},
                validation_metrics={},
                model_artifacts={},
                training_duration=time.time() - start_time,
                status=TrainingStatus.FAILED,
                error_message=str(e)
            )
            
            await self._complete_training_job(result)
    
    async def _complete_training_job(self, result: TrainingResult):
        """Complete a training job and update database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Update job status
                await conn.execute("""
                    UPDATE model_training_jobs 
                    SET status = $1, completed_at = $2, training_metrics = $3,
                        validation_metrics = $4, model_artifacts = $5, error_message = $6
                    WHERE job_id = $7
                """, 
                result.status.value, datetime.now(), json.dumps(result.training_metrics),
                json.dumps(result.validation_metrics), json.dumps(result.model_artifacts),
                result.error_message, result.job_id
                )
                
                # Store performance tracking
                if result.status == TrainingStatus.COMPLETED:
                    await conn.execute("""
                        INSERT INTO model_performance_tracking (
                            model_id, evaluation_timestamp, accuracy, precision, recall,
                            f1_score, auc_roc, evaluation_config
                        ) VALUES (
                            (SELECT id FROM model_training_jobs WHERE job_id = $1),
                            $2, $3, $4, $5, $6, $7, $8
                        )
                    """,
                    result.job_id, datetime.now(), result.validation_metrics['accuracy'],
                    result.validation_metrics['precision'], result.validation_metrics['recall'],
                    result.validation_metrics['f1_score'], result.validation_metrics['auc_roc'],
                    json.dumps({'evaluation_type': 'validation', 'split_ratio': 0.2})
                    )
                
                # Remove from running jobs
                with self.job_lock:
                    self.running_jobs.pop(result.job_id, None)
                
                self.logger.info(f"Training job completed: {result.job_id} with status {result.status.value}")
                
        except Exception as e:
            self.logger.error(f"Error completing training job: {e}")
    
    async def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get all pending training jobs"""
        try:
            async with self.db_pool.acquire() as conn:
                jobs = await conn.fetch("""
                    SELECT * FROM model_training_jobs 
                    WHERE status = 'pending' 
                    ORDER BY priority DESC, created_at ASC
                """)
                
                return [dict(job) for job in jobs]
                
        except Exception as e:
            self.logger.error(f"Error getting pending jobs: {e}")
            return []
    
    async def get_running_jobs(self) -> List[Dict[str, Any]]:
        """Get all running training jobs"""
        try:
            async with self.db_pool.acquire() as conn:
                jobs = await conn.fetch("""
                    SELECT * FROM model_training_jobs 
                    WHERE status = 'running'
                    ORDER BY started_at ASC
                """)
                
                return [dict(job) for job in jobs]
                
        except Exception as e:
            self.logger.error(f"Error getting running jobs: {e}")
            return []
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        try:
            async with self.db_pool.acquire() as conn:
                job = await conn.fetchrow("""
                    SELECT * FROM model_training_jobs WHERE job_id = $1
                """, job_id)
                
                return dict(job) if job else None
                
        except Exception as e:
            self.logger.error(f"Error getting job status: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE model_training_jobs 
                    SET status = $1, completed_at = $2 
                    WHERE job_id = $3 AND status = 'running'
                """, TrainingStatus.CANCELLED.value, datetime.now(), job_id)
                
                if result == "UPDATE 1":
                    with self.job_lock:
                        self.running_jobs.pop(job_id, None)
                    
                    self.logger.info(f"Job cancelled: {job_id}")
                    return True
                else:
                    self.logger.warning(f"Job not found or not running: {job_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error cancelling job: {e}")
            return False

class AutomatedRetrainingScheduler:
    """Scheduler for automated retraining"""
    
    def __init__(self, retraining_engine: AutomatedModelRetrainingEngine):
        self.engine = retraining_engine
        self.logger = logger
        self.running = False
        self.check_interval = 300  # 5 minutes
    
    async def start(self):
        """Start the scheduler"""
        self.running = True
        self.logger.info("Automated retraining scheduler started")
        
        while self.running:
            try:
                await self._check_and_submit_jobs()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        self.logger.info("Automated retraining scheduler stopped")
    
    async def _check_and_submit_jobs(self):
        """Check triggers and submit new jobs"""
        try:
            # Check for new jobs to create
            new_jobs = await self.engine.check_retraining_triggers()
            
            for job in new_jobs:
                # Submit the job
                success = await self.engine.submit_training_job(job)
                if success:
                    self.logger.info(f"New training job submitted: {job.job_id}")
            
            # Start pending jobs if we have capacity
            pending_jobs = await self.engine.get_pending_jobs()
            running_jobs = await self.engine.get_running_jobs()
            
            # Start jobs if we have capacity (max 3 concurrent)
            max_concurrent = 3
            if len(running_jobs) < max_concurrent and pending_jobs:
                for job in pending_jobs[:max_concurrent - len(running_jobs)]:
                    await self.engine.start_training_job(job['job_id'])
                    
        except Exception as e:
            self.logger.error(f"Error in check and submit jobs: {e}")
