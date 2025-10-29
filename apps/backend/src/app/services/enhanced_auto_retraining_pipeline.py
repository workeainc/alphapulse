#!/usr/bin/env python3
"""
Enhanced Auto-Retraining Pipeline Service
Comprehensive automated model retraining with drift detection, performance monitoring, and model versioning
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import os
from sqlalchemy import create_engine, text
import schedule
import time
import threading

# ML imports for drift detection
try:
    from scipy.stats import ks_2samp, entropy
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available for drift detection")

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL = "manual"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"

class DriftType(Enum):
    PSI = "psi"
    KL_DIVERGENCE = "kl_divergence"
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"

class RetrainingStrategy(Enum):
    INCREMENTAL = "incremental"
    FULL = "full"
    TRANSFER_LEARNING = "transfer_learning"

@dataclass
class RetrainingJob:
    """Retraining job configuration"""
    job_id: str
    model_name: str
    model_type: str
    symbol: str
    timeframe: str
    schedule_cron: str
    priority: int
    auto_deploy: bool
    performance_threshold: float
    drift_threshold: float
    min_training_samples: int
    max_training_age_days: int
    retraining_strategy: RetrainingStrategy
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    status: str

@dataclass
class DriftMetrics:
    """Data drift detection metrics"""
    drift_type: DriftType
    feature_name: str
    drift_score: float
    threshold: float
    is_drift_detected: bool
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    model_version: str
    auc_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int

class EnhancedAutoRetrainingPipeline:
    """Enhanced Auto-Retraining Pipeline with comprehensive monitoring and automation"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
        self.engine = create_engine(self.database_url)
        self.logger = logger
        
        # Pipeline configuration
        self.check_frequency = 3600  # 1 hour
        self.max_concurrent_jobs = 3
        self.is_running = False
        
        # Job management
        self.retraining_jobs = {}
        self.running_jobs = {}
        self.job_history = []
        
        # Performance tracking
        self.stats = {
            'jobs_scheduled': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'models_retrained': 0,
            'drift_alerts': 0,
            'auto_deployments': 0
        }
        
        self.logger.info("🔄 Enhanced Auto-Retraining Pipeline initialized")
    
    async def initialize(self):
        """Initialize the auto-retraining pipeline"""
        try:
            self.logger.info("Initializing Enhanced Auto-Retraining Pipeline...")
            
            # Load existing retraining jobs
            await self._load_retraining_jobs()
            
            # Setup default jobs for all model types
            await self._setup_default_jobs()
            
            # Initialize drift monitoring
            await self._initialize_drift_monitoring()
            
            self.logger.info("✅ Enhanced Auto-Retraining Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize auto-retraining pipeline: {e}")
            raise
    
    async def start(self):
        """Start the auto-retraining pipeline"""
        try:
            if self.is_running:
                self.logger.warning("Auto-retraining pipeline is already running")
                return
            
            self.logger.info("Starting Enhanced Auto-Retraining Pipeline...")
            self.is_running = True
            
            # Start main pipeline loop
            asyncio.create_task(self._pipeline_loop())
            
            # Start drift monitoring loop
            asyncio.create_task(self._drift_monitoring_loop())
            
            # Start performance monitoring loop
            asyncio.create_task(self._performance_monitoring_loop())
            
            self.logger.info("✅ Enhanced Auto-Retraining Pipeline started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start auto-retraining pipeline: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the auto-retraining pipeline"""
        try:
            self.logger.info("Stopping Enhanced Auto-Retraining Pipeline...")
            self.is_running = False
            
            # Wait for running jobs to complete
            if self.running_jobs:
                self.logger.info(f"Waiting for {len(self.running_jobs)} running jobs to complete...")
                await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
            
            self.logger.info("✅ Enhanced Auto-Retraining Pipeline stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping auto-retraining pipeline: {e}")
    
    async def _pipeline_loop(self):
        """Main pipeline loop"""
        while self.is_running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(self.check_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in pipeline loop: {e}")
                await asyncio.sleep(60)
    
    async def _drift_monitoring_loop(self):
        """Drift monitoring loop"""
        while self.is_running:
            try:
                await self._monitor_model_drift()
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in drift monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                await self._monitor_model_performance()
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(600)
    
    async def _load_retraining_jobs(self):
        """Load existing retraining jobs from database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT job_id, model_name, model_type, symbol, timeframe, schedule_cron,
                           priority, auto_deploy, performance_threshold, drift_threshold,
                           min_training_samples, max_training_age_days, status, last_run, next_run
                    FROM auto_retraining_jobs 
                    WHERE status = 'active'
                    ORDER BY priority DESC, next_run ASC
                """))
                
                for row in result:
                    job = RetrainingJob(
                        job_id=row.job_id,
                        model_name=row.model_name,
                        model_type=row.model_type,
                        symbol=row.symbol,
                        timeframe=row.timeframe,
                        schedule_cron=row.schedule_cron,
                        priority=row.priority,
                        auto_deploy=row.auto_deploy,
                        performance_threshold=float(row.performance_threshold),
                        drift_threshold=float(row.drift_threshold),
                        min_training_samples=row.min_training_samples,
                        max_training_age_days=row.max_training_age_days,
                        retraining_strategy=RetrainingStrategy.INCREMENTAL,
                        last_run=row.last_run,
                        next_run=row.next_run,
                        status=row.status
                    )
                    self.retraining_jobs[row.job_id] = job
                
                self.logger.info(f"Loaded {len(self.retraining_jobs)} active retraining jobs")
                
        except Exception as e:
            self.logger.error(f"Error loading retraining jobs: {e}")
    
    async def _setup_default_jobs(self):
        """Setup default retraining jobs for all model types"""
        try:
            default_jobs = [
                {
                    'model_name': 'lightgbm_ensemble',
                    'model_type': 'lightgbm',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'schedule_cron': '0 2 * * *',
                    'priority': 1
                },
                {
                    'model_name': 'lstm_time_series',
                    'model_type': 'lstm',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'schedule_cron': '0 3 * * *',
                    'priority': 2
                },
                {
                    'model_name': 'transformer_model',
                    'model_type': 'transformer',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'schedule_cron': '0 4 * * *',
                    'priority': 2
                },
                {
                    'model_name': 'ensemble_system',
                    'model_type': 'ensemble',
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'schedule_cron': '0 5 * * *',
                    'priority': 1
                }
            ]
            
            for job_config in default_jobs:
                await self._create_retraining_job(job_config)
                
        except Exception as e:
            self.logger.error(f"Error setting up default jobs: {e}")
    
    async def _create_retraining_job(self, job_config: Dict[str, Any]):
        """Create a new retraining job"""
        try:
            job_id = f"{job_config['model_name']}_{job_config['symbol']}_{job_config['timeframe']}"
            
            with self.engine.connect() as conn:
                # Check if job already exists
                result = conn.execute(text("""
                    SELECT job_id FROM auto_retraining_jobs WHERE job_id = :job_id
                """), {'job_id': job_id})
                
                if result.fetchone():
                    self.logger.info(f"Retraining job {job_id} already exists")
                    return
            
                # Insert new job
                conn.execute(text("""
                    INSERT INTO auto_retraining_jobs (
                        job_id, model_name, model_type, symbol, timeframe, schedule_cron,
                        priority, auto_deploy, performance_threshold, drift_threshold,
                        min_training_samples, max_training_age_days, status, next_run
                    ) VALUES (
                        :job_id, :model_name, :model_type, :symbol, :timeframe, :schedule_cron,
                        :priority, TRUE, 0.8, 0.25, 1000, 30, 'active', NOW()
                    )
                """), {
                    'job_id': job_id,
                    'model_name': job_config['model_name'],
                    'model_type': job_config['model_type'],
                    'symbol': job_config['symbol'],
                    'timeframe': job_config['timeframe'],
                    'schedule_cron': job_config['schedule_cron'],
                    'priority': job_config['priority']
                })
                conn.commit()
                
                self.logger.info(f"Created retraining job: {job_id}")
                
        except Exception as e:
            self.logger.error(f"Error creating retraining job: {e}")
    
    async def _check_and_run_jobs(self):
        """Check and run scheduled jobs"""
        try:
            current_time = datetime.now()
            
            for job_id, job in self.retraining_jobs.items():
                if job.status != 'active':
                    continue
                
                if job.next_run and current_time >= job.next_run:
                    if len(self.running_jobs) < self.max_concurrent_jobs:
                        # Start the job
                        self.running_jobs[job_id] = asyncio.create_task(
                            self._execute_retraining_job(job)
                        )
                        self.stats['jobs_scheduled'] += 1
                        
                        self.logger.info(f"Started retraining job: {job_id}")
                    else:
                        self.logger.warning(f"Max concurrent jobs reached, skipping: {job_id}")
            
            # Clean up completed jobs
            completed_jobs = []
            for job_id, task in self.running_jobs.items():
                if task.done():
                    completed_jobs.append(job_id)
                    try:
                        await task
                    except Exception as e:
                        self.logger.error(f"Job {job_id} failed: {e}")
                        self.stats['jobs_failed'] += 1
            
            for job_id in completed_jobs:
                del self.running_jobs[job_id]
                
        except Exception as e:
            self.logger.error(f"Error checking and running jobs: {e}")
    
    async def _execute_retraining_job(self, job: RetrainingJob):
        """Execute a retraining job"""
        try:
            job_id = job.job_id
            
            # Update job status
            await self._update_job_status(job_id, 'running')
            
            # Check if retraining is needed
            if not await self._should_retrain(job):
                self.logger.info(f"Retraining not needed for job: {job_id}")
                await self._update_job_status(job_id, 'completed')
                return
            
            # Execute retraining based on model type
            success = await self._retrain_model(job)
            
            if success:
                self.stats['jobs_completed'] += 1
                self.stats['models_retrained'] += 1
                await self._update_job_status(job_id, 'completed')
                
                # Auto-deploy if enabled
                if job.auto_deploy:
                    await self._auto_deploy_model(job)
                    self.stats['auto_deployments'] += 1
            else:
                self.stats['jobs_failed'] += 1
                await self._update_job_status(job_id, 'failed')
            
            # Update next run time
            await self._update_next_run(job_id, job.schedule_cron)
            
        except Exception as e:
            self.logger.error(f"Error executing retraining job {job.job_id}: {e}")
            await self._update_job_status(job.job_id, 'failed')
            self.stats['jobs_failed'] += 1
    
    async def _should_retrain(self, job: RetrainingJob) -> bool:
        """Check if model should be retrained"""
        try:
            # Check performance degradation
            performance_degraded = await self._check_performance_degradation(job)
            if performance_degraded:
                self.logger.info(f"Performance degradation detected for {job.job_id}")
                return True
            
            # Check data drift
            drift_detected = await self._check_data_drift(job)
            if drift_detected:
                self.logger.info(f"Data drift detected for {job.job_id}")
                return True
            
            # Check model age
            model_age_exceeded = await self._check_model_age(job)
            if model_age_exceeded:
                self.logger.info(f"Model age exceeded for {job.job_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if should retrain: {e}")
            return False
    
    async def _check_performance_degradation(self, job: RetrainingJob) -> bool:
        """Check for performance degradation"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT auc_score, precision_score, recall_score, f1_score
                    FROM model_performance_tracking
                    WHERE model_type = :model_name
                    ORDER BY timestamp DESC
                    LIMIT 2
                """), {
                    'model_name': job.model_name
                })
                
                rows = result.fetchall()
                if len(rows) < 2:
                    return False
                
                current_metrics = rows[0]
                previous_metrics = rows[1]
                
                # Check if any metric degraded significantly
                for metric in ['auc_score', 'precision_score', 'recall_score', 'f1_score']:
                    current = getattr(current_metrics, metric)
                    previous = getattr(previous_metrics, metric)
                    
                    if current and previous:
                        degradation = (previous - current) / previous
                        if degradation > 0.1:  # 10% degradation
                            return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking performance degradation: {e}")
            return False
    
    async def _check_data_drift(self, job: RetrainingJob) -> bool:
        """Check for data drift"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT drift_score, threshold, is_drift_detected
                    FROM model_drift_monitoring
                    WHERE model_name = :model_name
                    ORDER BY timestamp DESC
                    LIMIT 1
                """), {
                    'model_name': job.model_name
                })
                
                row = result.fetchone()
                if row and row.is_drift_detected:
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking data drift: {e}")
            return False
    
    async def _check_model_age(self, job: RetrainingJob) -> bool:
        """Check if model is too old"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT created_at
                    FROM model_version_management
                    WHERE model_name = :model_name AND symbol = :symbol AND is_active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """), {
                    'model_name': job.model_name,
                    'symbol': job.symbol
                })
                
                row = result.fetchone()
                if row:
                    model_age = datetime.now() - row.created_at
                    return model_age.days > job.max_training_age_days
                
                return True  # No active model found
                
        except Exception as e:
            self.logger.error(f"Error checking model age: {e}")
            return False
    
    async def _retrain_model(self, job: RetrainingJob) -> bool:
        """Retrain the model"""
        try:
            self.logger.info(f"Starting retraining for {job.job_id}")
            
            # This would integrate with the existing ML training services
            # For now, we'll simulate the retraining process
            
            # Simulate training time
            await asyncio.sleep(5)
            
            # Create new model version
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store model version
            await self._store_model_version(job, model_version)
            
            # Log retraining history
            await self._log_retraining_history(job, model_version, 'completed')
            
            self.logger.info(f"Retraining completed for {job.job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error retraining model {job.job_id}: {e}")
            await self._log_retraining_history(job, None, 'failed', str(e))
            return False
    
    async def _store_model_version(self, job: RetrainingJob, model_version: str):
        """Store new model version"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO model_version_management (
                        model_name, model_version, symbol, model_type, deployment_status, is_active
                    ) VALUES (
                        :model_name, :model_version, :symbol, :model_type, 'trained', TRUE
                    )
                """), {
                    'model_name': job.model_name,
                    'model_version': model_version,
                    'symbol': job.symbol,
                    'model_type': job.model_type
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing model version: {e}")
    
    async def _log_retraining_history(self, job: RetrainingJob, model_version: str, status: str, error_message: str = None):
        """Log retraining history"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO retraining_job_history (
                        job_id, model_name, model_version, trigger_type, status, error_message
                    ) VALUES (
                        :job_id, :model_name, :model_version, 'scheduled', :status, :error_message
                    )
                """), {
                    'job_id': job.job_id,
                    'model_name': job.model_name,
                    'model_version': model_version,
                    'status': status,
                    'error_message': error_message
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging retraining history: {e}")
    
    async def _auto_deploy_model(self, job: RetrainingJob):
        """Auto-deploy the retrained model"""
        try:
            self.logger.info(f"Auto-deploying model for {job.job_id}")
            
            # This would integrate with the deployment system
            # For now, we'll just update the deployment status
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE model_version_management
                    SET deployment_status = 'deployed', deployed_at = NOW()
                    WHERE model_name = :model_name AND symbol = :symbol AND is_active = TRUE
                """), {
                    'model_name': job.model_name,
                    'symbol': job.symbol
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error auto-deploying model: {e}")
    
    async def _update_job_status(self, job_id: str, status: str):
        """Update job status"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE auto_retraining_jobs
                    SET status = :status, updated_at = NOW()
                    WHERE job_id = :job_id
                """), {
                    'job_id': job_id,
                    'status': status
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating job status: {e}")
    
    async def _update_next_run(self, job_id: str, schedule_cron: str):
        """Update next run time based on cron schedule"""
        try:
            # Simple cron parsing for demonstration
            # In production, use a proper cron library
            next_run = datetime.now() + timedelta(days=1)
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE auto_retraining_jobs
                    SET next_run = :next_run, last_run = NOW(), updated_at = NOW()
                    WHERE job_id = :job_id
                """), {
                    'job_id': job_id,
                    'next_run': next_run
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating next run: {e}")
    
    async def _initialize_drift_monitoring(self):
        """Initialize drift monitoring"""
        try:
            self.logger.info("Initializing drift monitoring...")
            # Drift monitoring initialization logic would go here
            self.logger.info("✅ Drift monitoring initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing drift monitoring: {e}")
    
    async def _monitor_model_drift(self):
        """Monitor model drift"""
        try:
            # Drift monitoring logic would go here
            pass
            
        except Exception as e:
            self.logger.error(f"Error monitoring model drift: {e}")
    
    async def _monitor_model_performance(self):
        """Monitor model performance"""
        try:
            # Performance monitoring logic would go here
            pass
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")

    # ==================== CLOSED-LOOP INTEGRATION METHODS ====================

    async def integrate_with_monitoring(self, monitoring_service):
        """Integrate with monitoring service for closed-loop automation"""
        try:
            self.monitoring_service = monitoring_service
            self.logger.info("✅ Integrated with monitoring service for closed-loop automation")
            
        except Exception as e:
            self.logger.error(f"Error integrating with monitoring service: {e}")

    async def handle_monitoring_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Handle monitoring alert and trigger appropriate retraining"""
        try:
            model_id = alert_data.get('model_id')
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity_level')
            current_value = alert_data.get('current_value', 0)
            
            self.logger.info(f"Handling monitoring alert: {alert_type} for {model_id} (severity: {severity})")
            
            # Check if we should trigger retraining based on alert
            if await self._should_trigger_retraining_from_alert(alert_data):
                # Create retraining job
                job_config = await self._create_retraining_job_from_alert(alert_data)
                if job_config:
                    await self._create_retraining_job(job_config)
                    
                    # Update alert status
                    if self.monitoring_service:
                        await self.monitoring_service.update_alert_trigger_status(
                            alert_data.get('alert_id'),
                            True,
                            job_config.get('job_id')
                        )
                    
                    self.logger.info(f"Triggered retraining from monitoring alert: {alert_type}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling monitoring alert: {e}")
            return False

    async def _should_trigger_retraining_from_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Determine if retraining should be triggered from monitoring alert"""
        try:
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity_level')
            current_value = alert_data.get('current_value', 0)
            
            # High severity alerts always trigger retraining
            if severity in ['high', 'critical']:
                return True
            
            # Check specific alert types
            if alert_type == 'drift':
                # Trigger if drift score is high
                return current_value >= 0.25
                
            elif alert_type == 'performance':
                # Trigger if performance degradation is significant
                return current_value >= 0.15
                
            elif alert_type == 'risk':
                # Trigger if risk score is very high
                return current_value >= 85
                
            elif alert_type == 'data_quality':
                # Trigger if data quality is poor
                return current_value <= 0.6
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if should trigger retraining from alert: {e}")
            return False

    async def _create_retraining_job_from_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create retraining job configuration from monitoring alert"""
        try:
            model_id = alert_data.get('model_id')
            alert_type = alert_data.get('alert_type')
            severity = alert_data.get('severity_level')
            
            # Determine retraining strategy based on alert type and severity
            if severity == 'critical':
                strategy = 'full'
                priority = 1
            elif severity == 'high':
                strategy = 'full'
                priority = 2
            elif severity == 'medium':
                strategy = 'incremental'
                priority = 3
            else:
                strategy = 'incremental'
                priority = 4
            
            # Create job configuration
            job_config = {
                'model_name': model_id,
                'model_type': self._get_model_type_from_id(model_id),
                'symbol': 'BTCUSDT',  # Default, could be extracted from model_id
                'timeframe': '1h',    # Default, could be extracted from model_id
                'schedule_cron': '0 * * * *',  # Immediate execution
                'priority': priority,
                'retraining_strategy': strategy,
                'trigger_source': f'monitoring_alert_{alert_type}',
                'alert_metadata': alert_data
            }
            
            return job_config
            
        except Exception as e:
            self.logger.error(f"Error creating retraining job from alert: {e}")
            return None

    def _get_model_type_from_id(self, model_id: str) -> str:
        """Extract model type from model ID"""
        try:
            if 'lightgbm' in model_id.lower():
                return 'lightgbm'
            elif 'lstm' in model_id.lower():
                return 'lstm'
            elif 'transformer' in model_id.lower():
                return 'transformer'
            elif 'ensemble' in model_id.lower():
                return 'ensemble'
            else:
                return 'lightgbm'  # Default
                
        except Exception as e:
            self.logger.error(f"Error getting model type from ID: {e}")
            return 'lightgbm'

    async def log_retraining_trigger(self, job_id: str, trigger_source: str, 
                                   alert_data: Dict[str, Any] = None) -> bool:
        """Log retraining trigger for feedback loop analysis"""
        try:
            with self.engine.connect() as conn:
                # Log trigger in retraining job history
                conn.execute(text("""
                    INSERT INTO retraining_job_history (
                        job_id, model_name, trigger_type, trigger_details, status
                    ) VALUES (
                        :job_id, :model_name, :trigger_type, :trigger_details, 'triggered'
                    )
                """), {
                    'job_id': job_id,
                    'model_name': alert_data.get('model_id') if alert_data else 'unknown',
                    'trigger_type': trigger_source,
                    'trigger_details': json.dumps(alert_data) if alert_data else '{}'
                })
                
                conn.commit()
                
                self.logger.info(f"Logged retraining trigger: {job_id} from {trigger_source}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging retraining trigger: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.stats,
            'active_jobs': len([j for j in self.retraining_jobs.values() if j.status == 'active']),
            'running_jobs': len(self.running_jobs),
            'total_jobs': len(self.retraining_jobs)
        }
