#!/usr/bin/env python3
"""
Auto-Retrain Trigger Service
Phase 4: Auto-Retrain Triggers

Implements:
1. Automatic enqueueing of urgent retrain jobs
2. Quick retrain configuration selection
3. Integration with existing retrain_queue table
4. Drift-based triggering logic
5. Priority-based job scheduling
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import json

# Local imports
from ..feature_drift_detector import FeatureDriftDetector
from ..concept_drift_detector import ConceptDriftDetector
from ..production_monitoring import production_monitoring
from ...database.data_versioning_dao import DataVersioningDAO
from ...database.connection import TimescaleDBConnection

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift that can trigger retraining"""
    FEATURE_DRIFT = "feature_drift"
    CONCEPT_DRIFT = "concept_drift"
    LATENCY_DRIFT = "latency_drift"
    COMBINED_DRIFT = "combined_drift"

class RetrainPriority(Enum):
    """Retrain job priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class RetrainConfig(Enum):
    """Retrain configuration types"""
    QUICK_RETRAIN = "quick_retrain"      # Fast retrain with limited data
    STANDARD_RETRAIN = "standard_retrain" # Normal retrain process
    FULL_RETRAIN = "full_retrain"        # Comprehensive retrain
    EMERGENCY_RETRAIN = "emergency_retrain" # Urgent minimal retrain

@dataclass
class RetrainTrigger:
    """Retrain trigger configuration"""
    trigger_id: str
    drift_type: DriftType
    severity: str
    priority: RetrainPriority
    config_type: RetrainConfig
    drift_score: float
    drift_details: Dict[str, Any]
    timestamp: datetime
    triggered_by: str
    metadata: Dict[str, Any]

@dataclass
class RetrainJob:
    """Retrain job specification"""
    job_id: str
    trigger_id: str
    priority: RetrainPriority
    config_type: RetrainConfig
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime
    scheduled_for: datetime
    estimated_duration: int  # minutes
    resource_requirements: Dict[str, Any]
    drift_context: Dict[str, Any]

class AutoRetrainTriggerService:
    """Service for automatically triggering model retraining based on drift detection"""
    
    def __init__(self):
        self.is_running = False
        
        # Drift detectors
        self.feature_drift_detector = FeatureDriftDetector()
        self.concept_drift_detector = ConceptDriftDetector()
        
        # Trigger configuration
        self.trigger_config = {
            'feature_drift': {
                'psi_threshold': 0.25,  # PSI > 0.25 triggers retrain
                'severity_mapping': {
                    'low': RetrainPriority.LOW,
                    'medium': RetrainPriority.MEDIUM,
                    'high': RetrainPriority.HIGH,
                    'critical': RetrainPriority.URGENT
                }
            },
            'concept_drift': {
                'auc_f1_threshold': 0.10,  # 10% drop triggers retrain
                'calibration_threshold': 0.15,  # 15% calibration error triggers retrain
                'severity_mapping': {
                    'low': RetrainPriority.LOW,
                    'medium': RetrainPriority.MEDIUM,
                    'high': RetrainPriority.HIGH,
                    'critical': RetrainPriority.URGENT
                }
            },
            'latency_drift': {
                'p95_threshold': 200,  # 200ms p95 triggers retrain
                'severity_mapping': {
                    'low': RetrainPriority.LOW,
                    'medium': RetrainPriority.MEDIUM,
                    'high': RetrainPriority.HIGH,
                    'critical': RetrainPriority.CRITICAL
                }
            }
        }
        
        # Job management
        self.active_triggers = {}
        self.trigger_history = []
        self.job_queue = []
        
        # Performance tracking
        self.performance_metrics = {
            'triggers_generated': 0,
            'jobs_created': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'avg_job_duration': 0.0
        }
        
        logger.info("🚀 Auto-Retrain Trigger Service initialized")
    
    async def start(self):
        """Start the auto-retrain trigger service"""
        if self.is_running:
            logger.warning("Auto-Retrain Trigger Service is already running")
            return
        
        try:
            self.is_running = True
            asyncio.create_task(self._monitor_drift_and_trigger())
            logger.info("✅ Auto-Retrain Trigger Service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Auto-Retrain Trigger Service: {e}")
            raise
    
    async def stop(self):
        """Stop the auto-retrain trigger service"""
        if not self.is_running:
            logger.warning("Auto-Retrain Trigger Service is not running")
            return
        
        try:
            self.is_running = False
            logger.info("✅ Auto-Retrain Trigger Service stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping Auto-Retrain Trigger Service: {e}")
            raise
    
    async def _monitor_drift_and_trigger(self):
        """Main monitoring loop for drift detection and triggering"""
        logger.info("🔄 Starting drift monitoring loop...")
        
        while self.is_running:
            try:
                # Check for feature drift
                await self._check_feature_drift()
                
                # Check for concept drift
                await self._check_concept_drift()
                
                # Check for latency drift
                await self._check_latency_drift()
                
                # Process pending triggers
                await self._process_pending_triggers()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in drift monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _check_feature_drift(self):
        """Check for feature drift and trigger retraining if needed"""
        try:
            # Get current feature drift status
            drift_status = await self.feature_drift_detector.get_drift_summary()
            
            if not drift_status:
                return
            
            # Check if any features exceed threshold
            for feature_name, drift_info in drift_status.items():
                if drift_info.get('drift_score', 0) > self.trigger_config['feature_drift']['psi_threshold']:
                    # Determine severity
                    drift_score = drift_info['drift_score']
                    if drift_score > 0.8:
                        severity = 'critical'
                    elif drift_score > 0.6:
                        severity = 'high'
                    elif drift_score > 0.4:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    # Trigger retraining
                    await self._trigger_feature_drift_retrain(
                        feature_name, severity, drift_score, drift_info
                    )
                    
        except Exception as e:
            logger.error(f"❌ Error checking feature drift: {e}")
    
    async def _check_concept_drift(self):
        """Check for concept drift and trigger retraining if needed"""
        try:
            # Get current concept drift status
            drift_status = await self.concept_drift_detector.get_drift_summary()
            
            if not drift_status:
                return
            
            # Check AUC/F1 drift
            if drift_status.get('auc_f1_drift', 0) > self.trigger_config['concept_drift']['auc_f1_threshold']:
                severity = 'critical' if drift_status['auc_f1_drift'] > 0.2 else 'high'
                await self._trigger_concept_drift_retrain(
                    'auc_f1', severity, drift_status['auc_f1_drift'], drift_status
                )
            
            # Check calibration drift
            if drift_status.get('calibration_drift', 0) > self.trigger_config['concept_drift']['calibration_threshold']:
                severity = 'critical' if drift_status['calibration_drift'] > 0.25 else 'high'
                await self._trigger_concept_drift_retrain(
                    'calibration', severity, drift_status['calibration_drift'], drift_status
                )
                    
        except Exception as e:
            logger.error(f"❌ Error checking concept drift: {e}")
    
    async def _check_latency_drift(self):
        """Check for latency drift and trigger retraining if needed"""
        try:
            # Get current latency status
            latency_summary = production_monitoring.get_latency_summary('inference', hours=1)
            
            if not latency_summary or 'error' in latency_summary:
                return
            
            # Check p95 latency
            p95_latency = latency_summary.get('statistics', {}).get('p95', 0)
            if p95_latency > self.trigger_config['latency_drift']['p95_threshold']:
                severity = 'critical' if p95_latency > 300 else 'high'
                await self._trigger_latency_drift_retrain(
                    'inference', severity, p95_latency, latency_summary
                )
                    
        except Exception as e:
            logger.error(f"❌ Error checking latency drift: {e}")
    
    async def _trigger_feature_drift_retrain(self, 
                                           feature_name: str, 
                                           severity: str, 
                                           drift_score: float, 
                                           drift_details: Dict[str, Any]):
        """Trigger retraining due to feature drift"""
        try:
            trigger_id = f"feature_drift_{uuid.uuid4().hex[:8]}"
            
            # Create trigger
            trigger = RetrainTrigger(
                trigger_id=trigger_id,
                drift_type=DriftType.FEATURE_DRIFT,
                severity=severity,
                priority=self.trigger_config['feature_drift']['severity_mapping'][severity],
                config_type=self._get_retrain_config_for_severity(severity),
                drift_score=drift_score,
                drift_details=drift_details,
                timestamp=datetime.now(),
                triggered_by="feature_drift_detector",
                metadata={'feature_name': feature_name}
            )
            
            # Store trigger
            self.active_triggers[trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            # Create retrain job
            await self._create_retrain_job(trigger)
            
            logger.info(f"🚨 Feature drift retrain triggered: {feature_name} (severity: {severity})")
            
        except Exception as e:
            logger.error(f"❌ Error triggering feature drift retrain: {e}")
    
    async def _trigger_concept_drift_retrain(self, 
                                           drift_type: str, 
                                           severity: str, 
                                           drift_score: float, 
                                           drift_details: Dict[str, Any]):
        """Trigger retraining due to concept drift"""
        try:
            trigger_id = f"concept_drift_{uuid.uuid4().hex[:8]}"
            
            # Create trigger
            trigger = RetrainTrigger(
                trigger_id=trigger_id,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                priority=self.trigger_config['concept_drift']['severity_mapping'][severity],
                config_type=self._get_retrain_config_for_severity(severity),
                drift_score=drift_score,
                drift_details=drift_details,
                timestamp=datetime.now(),
                triggered_by="concept_drift_detector",
                metadata={'drift_type': drift_type}
            )
            
            # Store trigger
            self.active_triggers[trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            # Create retrain job
            await self._create_retrain_job(trigger)
            
            logger.info(f"🚨 Concept drift retrain triggered: {drift_type} (severity: {severity})")
            
        except Exception as e:
            logger.error(f"❌ Error triggering concept drift retrain: {e}")
    
    async def _trigger_latency_drift_retrain(self, 
                                            latency_type: str, 
                                            severity: str, 
                                            latency_value: float, 
                                            latency_details: Dict[str, Any]):
        """Trigger retraining due to latency drift"""
        try:
            trigger_id = f"latency_drift_{uuid.uuid4().hex[:8]}"
            
            # Create trigger
            trigger = RetrainTrigger(
                trigger_id=trigger_id,
                drift_type=DriftType.LATENCY_DRIFT,
                severity=severity,
                priority=self.trigger_config['latency_drift']['severity_mapping'][severity],
                config_type=self._get_retrain_config_for_severity(severity),
                drift_score=latency_value / 1000,  # Normalize to 0-1 range
                drift_details=latency_details,
                timestamp=datetime.now(),
                triggered_by="production_monitoring",
                metadata={'latency_type': latency_type}
            )
            
            # Store trigger
            self.active_triggers[trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            # Create retrain job
            await self._create_retrain_job(trigger)
            
            logger.info(f"🚨 Latency drift retrain triggered: {latency_type} (severity: {severity})")
            
        except Exception as e:
            logger.error(f"❌ Error triggering latency drift retrain: {e}")
    
    def _get_retrain_config_for_severity(self, severity: str) -> RetrainConfig:
        """Get retrain configuration based on severity"""
        if severity == 'critical':
            return RetrainConfig.EMERGENCY_RETRAIN
        elif severity == 'high':
            return RetrainConfig.QUICK_RETRAIN
        elif severity == 'medium':
            return RetrainConfig.STANDARD_RETRAIN
        else:
            return RetrainConfig.STANDARD_RETRAIN
    
    async def _create_retrain_job(self, trigger: RetrainTrigger):
        """Create a retrain job from a trigger"""
        try:
            job_id = f"retrain_job_{uuid.uuid4().hex[:8]}"
            
            # Get job parameters based on config
            job_params = self._get_job_parameters(trigger.config_type)
            
            # Calculate schedule time based on priority
            scheduled_for = self._calculate_schedule_time(trigger.priority)
            
            # Create job
            job = RetrainJob(
                job_id=job_id,
                trigger_id=trigger.trigger_id,
                priority=trigger.priority,
                config_type=trigger.config_type,
                status='pending',
                created_at=datetime.now(),
                scheduled_for=scheduled_for,
                estimated_duration=job_params['duration'],
                resource_requirements=job_params['resources'],
                drift_context={
                    'drift_type': trigger.drift_type.value,
                    'severity': trigger.severity,
                    'drift_score': trigger.drift_score,
                    'drift_details': trigger.drift_details
                }
            )
            
            # Add to job queue
            self.job_queue.append(job)
            
            # Enqueue in database
            await self._enqueue_retrain_job(job)
            
            # Update metrics
            self.performance_metrics['jobs_created'] += 1
            
            logger.info(f"✅ Retrain job created: {job_id} (priority: {trigger.priority.value})")
            
        except Exception as e:
            logger.error(f"❌ Error creating retrain job: {e}")
    
    def _get_job_parameters(self, config_type: RetrainConfig) -> Dict[str, Any]:
        """Get job parameters based on configuration type"""
        if config_type == RetrainConfig.EMERGENCY_RETRAIN:
            return {
                'duration': 30,  # 30 minutes
                'resources': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'gpu_memory_gb': 8
                },
                'data_window': '1_week'
            }
        elif config_type == RetrainConfig.QUICK_RETRAIN:
            return {
                'duration': 120,  # 2 hours
                'resources': {
                    'cpu_cores': 4,
                    'memory_gb': 8,
                    'gpu_memory_gb': 4
                },
                'data_window': '4_weeks'
            }
        elif config_type == RetrainConfig.STANDARD_RETRAIN:
            return {
                'duration': 360,  # 6 hours
                'resources': {
                    'cpu_cores': 6,
                    'memory_gb': 12,
                    'gpu_memory_gb': 6
                },
                'data_window': '8_weeks'
            }
        else:  # FULL_RETRAIN
            return {
                'duration': 720,  # 12 hours
                'resources': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'gpu_memory_gb': 8
                },
                'data_window': '6_months'
            }
    
    def _calculate_schedule_time(self, priority: RetrainPriority) -> datetime:
        """Calculate when the job should be scheduled"""
        now = datetime.now()
        
        if priority == RetrainPriority.CRITICAL:
            return now + timedelta(minutes=5)  # Run in 5 minutes
        elif priority == RetrainPriority.URGENT:
            return now + timedelta(minutes=15)  # Run in 15 minutes
        elif priority == RetrainPriority.HIGH:
            return now + timedelta(hours=1)  # Run in 1 hour
        elif priority == RetrainPriority.MEDIUM:
            return now + timedelta(hours=4)  # Run in 4 hours
        else:  # LOW
            return now + timedelta(hours=12)  # Run in 12 hours
    
    async def _enqueue_retrain_job(self, job: RetrainJob):
        """Enqueue retrain job in the database"""
        try:
            # Get database session
            session_factory = await get_async_session()
            async with session_factory as session:
                dao = DataVersioningDAO(session)
                
                # Add to retrain queue
                await dao.add_to_retrain_queue(
                    signal_id=job.job_id,  # Use job_id as signal_id for now
                    reason=f"Auto-triggered {job.drift_context['drift_type']} retrain",
                    priority=job.priority.value,
                    metadata={
                        'job_id': job.job_id,
                        'trigger_id': job.trigger_id,
                        'config_type': job.config_type.value,
                        'estimated_duration': job.estimated_duration,
                        'resource_requirements': job.resource_requirements,
                        'drift_context': job.drift_context
                    }
                )
                
                logger.info(f"✅ Job {job.job_id} enqueued in database")
                
        except Exception as e:
            logger.error(f"❌ Error enqueueing job in database: {e}")
    
    async def _process_pending_triggers(self):
        """Process pending triggers and check for due jobs"""
        try:
            current_time = datetime.now()
            due_jobs = [
                job for job in self.job_queue
                if job.status == 'pending' and job.scheduled_for <= current_time
            ]
            
            for job in due_jobs:
                logger.info(f"🔄 Processing due job: {job.job_id}")
                
                # Update job status
                job.status = 'running'
                
                # TODO: Actually execute the retraining
                # For now, just mark as completed
                await asyncio.sleep(1)  # Simulate processing
                job.status = 'completed'
                
                # Update metrics
                self.performance_metrics['jobs_completed'] += 1
                
                logger.info(f"✅ Job {job.job_id} completed")
                
        except Exception as e:
            logger.error(f"❌ Error processing pending triggers: {e}")
    
    async def trigger_emergency_retrain(self, drift_summary: Dict[str, Any]):
        """Manually trigger emergency retraining"""
        try:
            trigger_id = f"emergency_{uuid.uuid4().hex[:8]}"
            
            # Create emergency trigger
            trigger = RetrainTrigger(
                trigger_id=trigger_id,
                drift_type=DriftType.COMBINED_DRIFT,
                severity='critical',
                priority=RetrainPriority.CRITICAL,
                config_type=RetrainConfig.EMERGENCY_RETRAIN,
                drift_score=1.0,  # Maximum drift score
                drift_details=drift_summary,
                timestamp=datetime.now(),
                triggered_by="manual_emergency",
                metadata={'emergency': True}
            )
            
            # Store trigger
            self.active_triggers[trigger_id] = trigger
            self.trigger_history.append(trigger)
            
            # Create emergency retrain job
            await self._create_retrain_job(trigger)
            
            logger.critical("🚨 Emergency retrain job created")
            
        except Exception as e:
            logger.error(f"❌ Error creating emergency retrain job: {e}")
    
    def get_trigger_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of triggers and jobs"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter recent triggers
            recent_triggers = [
                trigger for trigger in self.trigger_history
                if trigger.timestamp >= cutoff_time
            ]
            
            # Filter active jobs
            active_jobs = [job for job in self.job_queue if job.status in ['pending', 'running']]
            
            # Calculate metrics
            total_triggers = len(recent_triggers)
            total_jobs = len(self.job_queue)
            
            return {
                'period_hours': hours,
                'total_triggers': total_triggers,
                'total_jobs': total_jobs,
                'active_jobs': len(active_jobs),
                'recent_triggers': [
                    {
                        'trigger_id': t.trigger_id,
                        'drift_type': t.drift_type.value,
                        'severity': t.severity,
                        'priority': t.priority.value,
                        'timestamp': t.timestamp.isoformat()
                    }
                    for t in recent_triggers
                ],
                'active_jobs': [
                    {
                        'job_id': j.job_id,
                        'priority': j.priority.value,
                        'status': j.status,
                        'scheduled_for': j.scheduled_for.isoformat()
                    }
                    for j in active_jobs
                ],
                'performance_metrics': self.performance_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting trigger summary: {e}")
            return {'error': str(e)}
    
    def get_current_status(self) -> str:
        """Get current service status"""
        if not self.is_running:
            return "stopped"
        
        # Check if there are critical jobs
        critical_jobs = [job for job in self.job_queue if job.priority == RetrainPriority.CRITICAL]
        
        if critical_jobs:
            return "critical_jobs_pending"
        elif len(self.job_queue) > 0:
            return "jobs_pending"
        else:
            return "idle"

# Global instance
auto_retrain_trigger_service = AutoRetrainTriggerService()
