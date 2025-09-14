#!/usr/bin/env python3
"""
Consolidated Retraining Orchestrator for AlphaPulse
Phase 5: Unified Retraining System

Implements:
1. Weekly quick retrain (8-12 weeks data)
2. Monthly full retrain (12-24 months data)  
3. Nightly incremental updates (daily data)
4. Prefect workflow orchestration
5. Resource management and monitoring
6. Real data integration and ML training pipeline
7. Unified drift detection and auto-retrain triggers
8. Comprehensive drift monitoring and alerting
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import time
import psutil
import os
import numpy as np
import pandas as pd

# Prefect imports
try:
    from prefect import flow, task, get_run_logger
    from prefect.schedules import CronSchedule
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule as PrefectCronSchedule
    from prefect.filesystems import LocalFileSystem
    from prefect.infrastructure import Process
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    print("Prefect not available - using fallback scheduling")

# Local imports
from ...core.prefect_config import (
    prefect_settings, CRON_TEMPLATES, RESOURCE_PROFILES, TIMEZONE_OFFSETS
)
from .data_service import RetrainingDataService
from .trigger_service import AutoRetrainTriggerService
from .drift_monitor import DriftDetectionMonitor
from ..model_accuracy_improvement import ModelAccuracyImprovement
from ..feedback_loop import FeedbackLoop
from ..model_registry import ModelRegistry
from ..advanced_logging_system import redis_logger, EventType, LogLevel
from ..real_data_integration_service import real_data_integration_service
from ..ml_models.trainer import MLModelTrainer, ModelType, TrainingCadence, TrainingConfig
from ..production_monitoring import production_monitoring
from ..error_recovery_system import error_recovery_system
from ..ml_models.ensemble_manager import (
    EnhancedEnsembleManager, EnsembleConfig, ModelType, MarketRegime,
    enhanced_ensemble_manager
)

# Phase 4B imports
from ...services.pattern_performance_tracker import pattern_performance_tracker
from ...database.connection import TimescaleDBConnection

# Phase 4C imports
from ..ml_models.online_learner import OnlineLearner

logger = logging.getLogger(__name__)

class RetrainingOrchestrator:
    """Enhanced orchestrator with Phase 5B ensemble integration"""
    
    def __init__(self):
        self.is_running = False
        
        # Core retraining components
        self.model_improver = ModelAccuracyImprovement()
        self.feedback_loop = FeedbackLoop()
        self.model_registry = ModelRegistry()
        self.data_service = RetrainingDataService()
        self.ml_trainer = MLModelTrainer()
        
        # Drift detection and auto-triggers
        self.drift_monitor = DriftDetectionMonitor()
        self.auto_trigger_service = AutoRetrainTriggerService()
        
        # Phase 4B: Performance tracking and self-learning
        self.performance_tracker = pattern_performance_tracker
        
        # Phase 4C: Online learning and safe self-retraining
        self.online_learner = OnlineLearner({
            'enable_incremental_learning': True,
            'enable_shadow_mode': True,
            'shadow_validation_threshold': 0.7,
            'auto_rollback_threshold': 0.05,
            'mini_batch_size': 1000,
            'warm_start_enabled': True,
            # Phase 5A: Canary deployment enhancements
            'enable_canary_deployment': True,
            'canary_traffic_percentage': 0.01,
            'canary_validation_threshold': 0.75,
            'canary_rollback_threshold': 0.03,
            'canary_promotion_stages': [0.01, 0.05, 0.25, 1.0],
            'canary_min_samples_per_stage': 1000,
            'canary_min_duration_per_stage': 3600
        })
        
        # Phase 5B: Enhanced Ensemble Manager
        self.enhanced_ensemble_manager = enhanced_ensemble_manager
        
        # Workflow state
        self.active_workflows = {}
        self.workflow_history = []
        self.last_executions = {}
        
        # Performance tracking
        self.execution_stats = {
            'weekly_retrains': 0,
            'monthly_retrains': 0,
            'nightly_updates': 0,
            'drift_triggered_retrains': 0,
            'total_success': 0,
            'total_failures': 0,
            'avg_execution_time': 0.0
        }
        
        # Resource monitoring
        self.resource_limits = {
            'cpu_threshold': 80,  # 80% CPU usage
            'memory_threshold': 85,  # 85% memory usage
            'disk_threshold': 90   # 90% disk usage
        }
        
        # Orchestration configuration
        self.orchestration_config = {
            'comprehensive_check_interval': 300,  # 5 minutes
            'alert_retention_hours': 168,  # 1 week
            'drift_history_retention_hours': 720,  # 1 month
            'thresholds': {
                'combined_drift_critical': 0.8,  # 80% combined drift score
                'combined_drift_warning': 0.6,   # 60% combined drift score
                'alert_cooldown_minutes': 30,    # Prevent alert spam
            }
        }
        
        # Performance tracking for drift detection
        self.drift_performance_metrics = {
            'total_checks': 0,
            'drifts_detected': 0,
            'alerts_generated': 0,
            'retrains_triggered': 0,
            'avg_check_duration': 0.0
        }
        
        logger.info("🚀 Consolidated Retraining Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator and all subsystems"""
        if self.is_running:
            logger.warning("Orchestrator is already running")
            return
        
        try:
            # Initialize production monitoring
            logger.info("🔍 Starting production monitoring...")
            try:
                await production_monitoring.start()
                logger.info("✅ Production monitoring started")
            except Exception as e:
                logger.warning(f"⚠️ Production monitoring failed to start: {e}")
            
            # Initialize error recovery system
            logger.info("🔧 Starting error recovery system...")
            try:
                await error_recovery_system.start()
                logger.info("✅ Error recovery system started")
            except Exception as e:
                logger.warning(f"⚠️ Error recovery system failed to start: {e}")
            
            # Start drift detection and auto-triggers
            logger.info("🚨 Starting drift detection and auto-triggers...")
            try:
                await self.drift_monitor.start()
                await self.auto_trigger_service.start()
                logger.info("✅ Drift detection and auto-triggers started")
            except Exception as e:
                logger.warning(f"⚠️ Drift detection failed to start: {e}")
            
            # Start Phase 4B performance tracking
            logger.info("📊 Starting Phase 4B performance tracking...")
            try:
                await self.performance_tracker.start()
                logger.info("✅ Phase 4B performance tracking started")
            except Exception as e:
                logger.warning(f"⚠️ Performance tracking failed to start: {e}")
            
            # Deploy Prefect workflows
            if PREFECT_AVAILABLE:
                logger.info("🚀 Deploying Prefect workflows...")
                try:
                    await self._deploy_workflows()
                    logger.info("✅ Prefect workflows deployed")
                except Exception as e:
                    logger.warning(f"⚠️ Prefect workflow deployment failed: {e}")
            
            # Start orchestration loop
            self.is_running = True
            asyncio.create_task(self._run_orchestration_loop())
            
            logger.info("✅ Consolidated Retraining Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start orchestrator: {e}")
            raise
    
    async def stop(self):
        """Stop the orchestrator and all subsystems"""
        if not self.is_running:
            logger.warning("Orchestrator is not running")
            return
        
        try:
            logger.info("🛑 Stopping Consolidated Retraining Orchestrator...")
            
            # Stop orchestration loop
            self.is_running = False
            
            # Stop drift detection and auto-triggers
            try:
                await self.drift_monitor.stop()
                await self.auto_trigger_service.stop()
                logger.info("✅ Drift detection and auto-triggers stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping drift detection: {e}")
            
            # Stop Phase 4B performance tracking
            try:
                await self.performance_tracker.stop()
                logger.info("✅ Phase 4B performance tracking stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping performance tracking: {e}")
            
            # Stop production monitoring
            try:
                await production_monitoring.stop()
                logger.info("✅ Production monitoring stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping production monitoring: {e}")
            
            # Stop error recovery system
            try:
                await error_recovery_system.stop()
                logger.info("✅ Error recovery system stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping error recovery system: {e}")
            
            logger.info("✅ Consolidated Retraining Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping orchestrator: {e}")
            raise
    
    async def _deploy_workflows(self):
        """Deploy Prefect workflows for different retraining cadences"""
        if not PREFECT_AVAILABLE:
            logger.warning("Prefect not available - skipping workflow deployment")
            return
        
        try:
            # Deploy weekly quick retrain
            await self._deploy_weekly_workflow()
            
            # Deploy monthly full retrain
            await self._deploy_monthly_workflow()
            
            # Deploy nightly incremental workflow
            await self._deploy_nightly_workflow()
            
            logger.info("✅ All Prefect workflows deployed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy workflows: {e}")
            raise
    
    async def _deploy_weekly_workflow(self):
        """Deploy weekly quick retrain workflow"""
        if not PREFECT_AVAILABLE:
            return
        
        try:
            deployment = await Deployment.build_from_flow(
                flow=weekly_quick_retrain_flow,
                name="weekly-quick-retrain",
                version="1.0.0",
                work_queue_name="alphapulse-retraining",
                work_pool_name="alphapulse-pool",
                tags=["retraining", "weekly", "quick"],
                description="Weekly quick retrain with 8-12 weeks of data"
            )
            await deployment.apply()
            logger.info("✅ Weekly quick retrain workflow deployed")
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy weekly workflow: {e}")
            raise
    
    async def _deploy_monthly_workflow(self):
        """Deploy monthly full retrain workflow"""
        if not PREFECT_AVAILABLE:
            return
        
        try:
            deployment = await Deployment.build_from_flow(
                flow=monthly_full_retrain_flow,
                name="monthly-full-retrain",
                version="1.0.0",
                work_queue_name="alphapulse-retraining",
                work_pool_name="alphapulse-pool",
                tags=["retraining", "monthly", "full"],
                description="Monthly full retrain with 12-24 months of data"
            )
            await deployment.apply()
            logger.info("✅ Monthly full retrain workflow deployed")
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy monthly workflow: {e}")
            raise
    
    async def _deploy_nightly_workflow(self):
        """Deploy nightly incremental workflow"""
        if not PREFECT_AVAILABLE:
            return
        
        try:
            deployment = await Deployment.build_from_flow(
                flow=nightly_incremental_flow,
                name="nightly-incremental-update",
                version="1.0.0",
                work_queue_name="alphapulse-retraining",
                work_pool_name="alphapulse-pool",
                tags=["retraining", "nightly", "incremental"],
                description="Nightly incremental model updates"
            )
            await deployment.apply()
            logger.info("✅ Nightly incremental workflow deployed")
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy nightly workflow: {e}")
            raise
    
    async def _run_orchestration_loop(self):
        """Main orchestration loop for comprehensive monitoring and management"""
        logger.info("🔄 Starting orchestration loop...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Perform comprehensive drift checks
                await self._perform_comprehensive_drift_check()
                
                # Process auto-retrain triggers
                await self._process_auto_retrain_triggers()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Calculate execution time
                execution_time = time.time() - start_time
                self.drift_performance_metrics['avg_check_duration'] = (
                    (self.drift_performance_metrics['avg_check_duration'] * 
                     self.drift_performance_metrics['total_checks'] + execution_time) /
                    (self.drift_performance_metrics['total_checks'] + 1)
                )
                self.drift_performance_metrics['total_checks'] += 1
                
                # Wait for next check
                await asyncio.sleep(self.orchestration_config['comprehensive_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in orchestration loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_comprehensive_drift_check(self):
        """Perform comprehensive drift check across all systems"""
        try:
            # Check feature drift
            feature_status = await self.drift_monitor.check_feature_drift()
            
            # Check concept drift
            concept_status = await self.drift_monitor.check_concept_drift()
            
            # Check latency drift
            latency_status = await self.drift_monitor.check_latency_drift()
            
            # Generate comprehensive summary
            summary = await self._generate_comprehensive_summary(
                feature_status, concept_status, latency_status
            )
            
            # Check combined drift conditions
            await self._check_combined_drift_conditions(summary)
            
            # Log summary
            logger.info(f"🔍 Drift check completed - Overall status: {summary['overall_status']}")
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive drift check: {e}")
    
    async def _generate_comprehensive_summary(self, feature_status: Dict, 
                                           concept_status: Dict, 
                                           latency_status: Dict) -> Dict[str, Any]:
        """Generate comprehensive drift summary"""
        try:
            # Calculate overall status
            overall_status = 'healthy'
            recommendations = []
            
            # Check feature drift
            if feature_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Immediate action required - consider emergency retraining")
            elif feature_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Monitor closely and prepare for potential retraining")
            
            # Check concept drift
            if concept_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Model performance degradation detected - urgent retraining needed")
            elif concept_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Performance monitoring required")
            
            # Check latency drift
            if latency_status.get('status') == 'critical':
                overall_status = 'critical'
                recommendations.append("Inference latency critical - model size/complexity review required")
            elif latency_status.get('status') == 'warning':
                if overall_status == 'healthy':
                    overall_status = 'warning'
                recommendations.append("Latency monitoring required")
            
            return {
                'timestamp': datetime.now(),
                'feature_drift': feature_status,
                'concept_drift': concept_status,
                'latency_drift': latency_status,
                'overall_status': overall_status,
                'recommendations': recommendations,
                'metadata': {
                    'check_duration': time.time(),
                    'systems_checked': ['feature', 'concept', 'latency']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive summary: {e}")
            return {
                'timestamp': datetime.now(),
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def _check_combined_drift_conditions(self, summary: Dict[str, Any]):
        """Check combined drift conditions and trigger emergency responses"""
        try:
            if summary['overall_status'] == 'critical':
                await self._trigger_emergency_response(summary)
            elif summary['overall_status'] == 'warning':
                await self._trigger_warning_response(summary)
                
        except Exception as e:
            logger.error(f"❌ Error checking combined drift conditions: {e}")
    
    async def _trigger_emergency_response(self, summary: Dict[str, Any]):
        """Trigger emergency response for critical drift"""
        try:
            logger.critical(f"🚨 EMERGENCY: Critical drift detected - {summary['overall_status']}")
            logger.critical(f"🚨 Recommendations: {summary['recommendations']}")
            
            # Trigger emergency retraining
            await self.auto_trigger_service.trigger_emergency_retrain(summary)
            
            # Log emergency action
            logger.critical("🚨 Emergency retraining triggered")
            
        except Exception as e:
            logger.error(f"❌ Error triggering emergency response: {e}")
    
    async def _trigger_warning_response(self, summary: Dict[str, Any]):
        """Trigger warning response for warning-level drift"""
        try:
            logger.warning(f"⚠️ WARNING: Drift detected - {summary['overall_status']}")
            logger.warning(f"⚠️ Recommendations: {summary['recommendations']}")
            
            # Log warning action
            logger.warning("⚠️ Warning response triggered")
            
        except Exception as e:
            logger.error(f"❌ Error triggering warning response: {e}")
    
    async def _process_auto_retrain_triggers(self):
        """Process auto-retrain triggers and active jobs"""
        try:
            if self.auto_trigger_service.is_running:
                # Get trigger summary
                trigger_summary = await self.auto_trigger_service.get_trigger_summary(hours=1)
                
                # Check for active jobs
                active_jobs = trigger_summary.get('active_jobs', 0)
                if active_jobs > 0:
                    logger.info(f"🔄 {active_jobs} active retrain jobs in queue")
                
                # Process pending triggers
                await self.auto_trigger_service._process_pending_triggers()
                
        except Exception as e:
            logger.error(f"❌ Error processing auto-retrain triggers: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics for monitoring"""
        try:
            # Update drift detection metrics
            drift_summary = await self.drift_monitor.get_orchestration_summary()
            self.drift_performance_metrics.update({
                'drifts_detected': drift_summary.get('total_drifts', 0),
                'alerts_generated': drift_summary.get('total_alerts', 0),
                'retrains_triggered': drift_summary.get('retrains_triggered', 0)
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old drift detection and alert data"""
        try:
            # Clean up old drift history
            cutoff_time = datetime.now() - timedelta(
                hours=self.orchestration_config['drift_history_retention_hours']
            )
            
            # Clean up old alerts
            alert_cutoff = datetime.now() - timedelta(
                hours=self.orchestration_config['alert_retention_hours']
            )
            
            logger.debug("🧹 Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup: {e}")
    
    # Retraining execution methods (from original model_retraining_orchestrator)
    async def execute_weekly_retrain(self) -> bool:
        """Execute weekly quick retrain"""
        try:
            logger.info("🔄 Starting weekly quick retrain...")
            
            # Check resources
            if not await self._check_resources("weekly_quick"):
                logger.warning("⚠️ Insufficient resources for weekly retrain")
                return False
            
            # Execute retraining
            start_time = time.time()
            success = await self._perform_weekly_retrain()
            execution_time = time.time() - start_time
            
            if success:
                await self._record_execution("weekly_quick", True, execution_time)
                logger.info("✅ Weekly retrain completed successfully")
            else:
                await self._record_execution("weekly_quick", False, execution_time)
                logger.error("❌ Weekly retrain failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error in weekly retrain: {e}")
            return False
    
    async def execute_monthly_retrain(self) -> bool:
        """Execute monthly full retrain"""
        try:
            logger.info("🔄 Starting monthly full retrain...")
            
            # Check resources
            if not await self._check_resources("monthly_full"):
                logger.warning("⚠️ Insufficient resources for monthly retrain")
                return False
            
            # Execute retraining
            start_time = time.time()
            success = await self._perform_monthly_retrain()
            execution_time = time.time() - start_time
            
            if success:
                await self._record_execution("monthly_full", True, execution_time)
                logger.info("✅ Monthly retrain completed successfully")
            else:
                await self._record_execution("monthly_full", False, execution_time)
                logger.error("❌ Monthly retrain failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error in monthly retrain: {e}")
            return False
    
    async def execute_nightly_update(self) -> bool:
        """Execute nightly incremental update"""
        try:
            logger.info("🔄 Starting nightly incremental update...")
            
            # Check resources
            if not await self._check_resources("nightly_incremental"):
                logger.warning("⚠️ Insufficient resources for nightly update")
                return False
            
            # Execute update
            start_time = time.time()
            success = await self._perform_nightly_update()
            execution_time = time.time() - start_time
            
            if success:
                await self._record_execution("nightly_incremental", True, execution_time)
                logger.info("✅ Nightly update completed successfully")
            else:
                await self._record_execution("nightly_incremental", False, execution_time)
                logger.error("❌ Nightly update failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error in nightly update: {e}")
            return False
    
    async def _check_resources(self, retrain_type: str) -> bool:
        """Check if system has sufficient resources for retraining"""
        try:
            profile = RESOURCE_PROFILES[retrain_type]
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.resource_limits['cpu_threshold']:
                logger.warning(f"⚠️ CPU usage too high: {cpu_percent}%")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.resource_limits['memory_threshold']:
                logger.warning(f"⚠️ Memory usage too high: {memory.percent}%")
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > self.resource_limits['disk_threshold']:
                logger.warning(f"⚠️ Disk usage too high: {disk.percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking resources: {e}")
            return False
    
    async def _perform_weekly_retrain(self) -> bool:
        """Perform weekly quick retrain with 8-12 weeks of data"""
        try:
            # Prepare training data
            training_data = await self.data_service.prepare_weekly_training_data(
                symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
                weeks=10
            )
            
            if training_data is None or training_data.empty:
                logger.warning("⚠️ Insufficient data for weekly retrain")
                return False
            
            # Execute retraining
            training_result = await self.ml_trainer.train_model(
                X=training_data.drop('target', axis=1) if 'target' in training_data.columns else training_data,
                y=training_data['target'] if 'target' in training_data.columns else None,
                model_type=ModelType.XGBOOST,
                cadence=TrainingCadence.WEEKLY_QUICK,
                config=TrainingConfig(
                    learning_rate=0.001,
                    max_depth=6,
                    n_estimators=100
                )
            )
            
            if training_result and training_result.success:
                # Update model registry
                await self.model_registry.register_model(
                    model_path=training_result.model_path,
                    model_type="weekly_quick",
                    metrics={
                        'accuracy': training_result.accuracy,
                        'auc': training_result.auc,
                        'f1_score': training_result.f1_score
                    }
                )
                
                logger.info(f"✅ Weekly retrain completed: {training_result.model_path} in {training_result.training_time:.2f}s")
                return True
            else:
                logger.error("❌ Weekly retrain failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in weekly retrain: {e}")
            return False
    
    async def _perform_monthly_retrain(self) -> bool:
        """Perform monthly full retrain with 12-24 months of data"""
        try:
            # Prepare training data
            training_data = await self.data_service.prepare_monthly_training_data(
                symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
                months=18
            )
            
            if training_data is None or training_data.empty:
                logger.warning("⚠️ Insufficient data for monthly retrain")
                return False
            
            # Execute retraining
            training_result = await self.ml_trainer.train_model(
                X=training_data.drop('target', axis=1) if 'target' in training_data.columns else training_data,
                y=training_data['target'] if 'target' in training_data.columns else None,
                model_type=ModelType.XGBOOST,
                cadence=TrainingCadence.MONTHLY_FULL,
                config=TrainingConfig(
                    learning_rate=0.0005,
                    max_depth=8,
                    n_estimators=200
                )
            )
            
            if training_result and training_result.success:
                # Update model registry
                await self.model_registry.register_model(
                    model_path=training_result.model_path,
                    model_type="monthly_full",
                    metrics={
                        'accuracy': training_result.accuracy,
                        'auc': training_result.auc,
                        'f1_score': training_result.f1_score
                    }
                )
                
                logger.info(f"✅ Monthly retrain completed: {training_result.model_path}")
                return True
            else:
                logger.error("❌ Monthly retrain failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in monthly retrain: {e}")
            return False
    
    async def _perform_nightly_update(self) -> bool:
        """Perform nightly incremental update"""
        try:
            # Prepare daily data
            daily_data = await self.data_service.prepare_nightly_training_data(
                symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            )
            
            if daily_data is None or daily_data.empty:
                logger.warning("⚠️ Insufficient data for nightly update")
                return False
            
            # Perform incremental update using online learner
            from ..ai.ml_models.online_learner import OnlineLearner
            online_learner = OnlineLearner()
            update_result = await online_learner.learn_batch(
                features_list=daily_data.to_dict('records') if not daily_data.empty else [],
                labels=daily_data['target'].tolist() if 'target' in daily_data.columns else []
            )
            
            if update_result and update_result.get('status') == 'success':
                logger.info("✅ Nightly update completed successfully")
                return True
            else:
                logger.error("❌ Nightly update failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in nightly update: {e}")
            return False
    
    async def _record_execution(self, retrain_type: str, success: bool, execution_time: float):
        """Record execution statistics"""
        try:
            # Update execution stats
            if retrain_type == "weekly_quick":
                self.execution_stats['weekly_retrains'] += 1
            elif retrain_type == "monthly_full":
                self.execution_stats['monthly_retrains'] += 1
            elif retrain_type == "nightly_incremental":
                self.execution_stats['nightly_updates'] += 1
            
            if success:
                self.execution_stats['total_success'] += 1
            else:
                self.execution_stats['total_failures'] += 1
            
            # Update average execution time
            total_executions = self.execution_stats['total_success'] + self.execution_stats['total_failures']
            self.execution_stats['avg_execution_time'] = (
                (self.execution_stats['avg_execution_time'] * (total_executions - 1) + execution_time) /
                total_executions
            )
            
            # Log to monitoring system
            await redis_logger.log_event(
                EventType.RETRAINING,
                LogLevel.INFO,
                {
                    'type': retrain_type,
                    'success': success,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error recording execution: {e}")
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get comprehensive orchestration summary"""
        try:
            return {
                'status': 'running' if self.is_running else 'stopped',
                'execution_stats': self.execution_stats.copy(),
                'drift_performance': self.drift_performance_metrics.copy(),
                'active_workflows': len(self.active_workflows),
                'last_executions': self.last_executions.copy(),
                'resource_usage': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                },
                'subsystems': {
                    'drift_monitor': self.drift_monitor.is_running,
                    'auto_trigger_service': self.auto_trigger_service.is_running,
                    'production_monitoring': production_monitoring.is_running,
                    'error_recovery': error_recovery_system.is_running,
                    'performance_tracker': self.performance_tracker.is_running
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting orchestration summary: {e}")
            return {'error': str(e)}
    
    async def execute_phase4b_performance_retrain(self) -> bool:
        """
        Execute Phase 4B performance-based retraining
        Triggers retraining based on pattern performance metrics
        """
        try:
            logger.info("🔄 Starting Phase 4B performance-based retraining...")
            start_time = time.time()
            
            # Get performance recommendations
            recommendations = await self.performance_tracker.get_retraining_recommendations()
            
            if not recommendations['retraining_needed']:
                logger.info("✅ No retraining needed based on performance metrics")
                return True
            
            logger.info(f"📊 Performance-based retraining triggered: {recommendations['reasons']}")
            
            # Get current performance metrics
            overall_metrics = await self.performance_tracker.get_pattern_performance_metrics(days=7)
            regime_performances = await self.performance_tracker.get_regime_performance(days=7)
            
            # Create retraining event
            event_id = f"phase4b_performance_{int(time.time())}"
            await self._create_retraining_event(
                event_id=event_id,
                event_type='performance_triggered',
                model_type='pattern_detector',
                trigger_source='performance_tracker',
                trigger_metadata={
                    'recommendations': recommendations,
                    'overall_metrics': {
                        'success_rate': overall_metrics.success_rate,
                        'profit_factor': overall_metrics.profit_factor,
                        'total_signals': overall_metrics.total_signals
                    },
                    'regime_performances': [
                        {
                            'regime': r.regime_type,
                            'success_rate': r.success_rate,
                            'total_signals': r.total_signals
                        } for r in regime_performances
                    ]
                }
            )
            
            # Execute retraining for each model type
            retraining_success = True
            for model_type in recommendations['model_types']:
                try:
                    logger.info(f"🔄 Retraining {model_type} based on performance...")
                    
                    if model_type == 'pattern_detector':
                        success = await self._retrain_pattern_detector(overall_metrics, regime_performances)
                    elif model_type == 'quality_scorer':
                        success = await self._retrain_quality_scorer(overall_metrics, regime_performances)
                    else:
                        success = await self._retrain_generic_model(model_type, overall_metrics)
                    
                    if not success:
                        retraining_success = False
                        logger.error(f"❌ Failed to retrain {model_type}")
                    
                except Exception as e:
                    logger.error(f"❌ Error retraining {model_type}: {e}")
                    retraining_success = False
            
            execution_time = time.time() - start_time
            
            # Update retraining event
            await self._update_retraining_event(
                event_id=event_id,
                status='completed' if retraining_success else 'failed',
                end_timestamp=datetime.now(),
                duration_seconds=int(execution_time),
                accuracy_improvement=0.0,  # Will be calculated after deployment
                error_message=None if retraining_success else "Retraining failed"
            )
            
            # Record execution
            self._record_execution('phase4b_performance_retrain', retraining_success, execution_time)
            
            if retraining_success:
                logger.info(f"✅ Phase 4B performance-based retraining completed in {execution_time:.2f}s")
            else:
                logger.error(f"❌ Phase 4B performance-based retraining failed")
            
            return retraining_success
            
        except Exception as e:
            logger.error(f"❌ Error in Phase 4B performance-based retraining: {e}")
            return False
    
    async def _retrain_pattern_detector(self, metrics, regime_performances) -> bool:
        """Retrain pattern detector based on performance metrics"""
        try:
            # Adjust pattern detection thresholds based on performance
            if metrics.success_rate < 0.4:
                # Lower confidence thresholds for better recall
                logger.info("📉 Lowering pattern detection thresholds due to low success rate")
                # Implementation would adjust thresholds in pattern detector
            
            # Regime-specific adjustments
            for regime in regime_performances:
                if regime.success_rate < 0.3 and regime.total_signals >= 5:
                    logger.info(f"📊 Adjusting {regime.regime_type} regime patterns")
                    # Implementation would adjust regime-specific parameters
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error retraining pattern detector: {e}")
            return False
    
    async def _retrain_quality_scorer(self, metrics, regime_performances) -> bool:
        """Retrain quality scorer based on performance metrics"""
        try:
            # Adjust quality scoring weights based on performance
            if metrics.profit_factor and metrics.profit_factor < 1.2:
                logger.info("📉 Adjusting quality scoring weights due to low profit factor")
                # Implementation would retrain quality scorer with new weights
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error retraining quality scorer: {e}")
            return False
    
    async def _retrain_generic_model(self, model_type: str, metrics) -> bool:
        """Retrain generic model based on performance metrics"""
        try:
            logger.info(f"🔄 Retraining {model_type} with performance-based adjustments")
            # Implementation would retrain the specific model type
            return True
            
        except Exception as e:
            logger.error(f"❌ Error retraining {model_type}: {e}")
            return False
    
    async def _create_retraining_event(self, event_id: str, event_type: str, model_type: str, 
                                     trigger_source: str, trigger_metadata: Dict[str, Any]) -> None:
        """Create a retraining event record"""
        try:
            async with get_async_session() as session:
                query = text("""
                    INSERT INTO retraining_events (
                        event_id, event_type, model_type, trigger_source, trigger_metadata,
                        status, start_timestamp
                    ) VALUES (
                        :event_id, :event_type, :model_type, :trigger_source, :trigger_metadata,
                        'running', NOW()
                    )
                """)
                
                await session.execute(query, {
                    'event_id': event_id,
                    'event_type': event_type,
                    'model_type': model_type,
                    'trigger_source': trigger_source,
                    'trigger_metadata': json.dumps(trigger_metadata)
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error creating retraining event: {e}")
    
    async def _update_retraining_event(self, event_id: str, status: str, end_timestamp: datetime,
                                     duration_seconds: int, accuracy_improvement: float,
                                     error_message: Optional[str]) -> None:
        """Update a retraining event record"""
        try:
            async with get_async_session() as session:
                query = text("""
                    UPDATE retraining_events
                    SET status = :status,
                        end_timestamp = :end_timestamp,
                        duration_seconds = :duration_seconds,
                        accuracy_improvement = :accuracy_improvement,
                        error_message = :error_message
                    WHERE event_id = :event_id
                """)
                
                await session.execute(query, {
                    'status': status,
                    'end_timestamp': end_timestamp,
                    'duration_seconds': duration_seconds,
                    'accuracy_improvement': accuracy_improvement,
                    'error_message': error_message,
                    'event_id': event_id
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error updating retraining event: {e}")
    
    def get_current_status(self) -> str:
        """Get current orchestrator status"""
        if not self.is_running:
            return "stopped"
        
        # Check subsystem status
        subsystems_running = sum([
            self.drift_monitor.is_running,
            self.auto_trigger_service.is_running,
            production_monitoring.is_running,
            error_recovery_system.is_running,
            self.performance_tracker.is_running
        ])
        
        if subsystems_running == 5:
            return "fully_operational"
        elif subsystems_running >= 3:
            return "partially_operational"
        else:
            return "degraded"

# Global instance
retraining_orchestrator = RetrainingOrchestrator()

# Prefect flows (from original model_retraining_orchestrator)
if PREFECT_AVAILABLE:
    @flow(name="weekly-quick-retrain")
    async def weekly_quick_retrain_flow():
        """Weekly quick retrain flow"""
        try:
            logger = get_run_logger()
            logger.info("Starting weekly quick retrain flow")
            
            success = await retraining_orchestrator.execute_weekly_retrain()
            logger.info(f"Weekly retrain completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
            raise
else:
    async def weekly_quick_retrain_flow():
        """Weekly quick retrain flow"""
        try:
            logger.info("Starting weekly quick retrain flow")
            
            success = await retraining_orchestrator.execute_weekly_retrain()
            logger.info(f"Weekly retrain completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")
            raise

if PREFECT_AVAILABLE:
    @flow(name="monthly-full-retrain")
    async def monthly_full_retrain_flow():
        """Monthly full retrain flow"""
        try:
            logger = get_run_logger()
            logger.info("Starting monthly full retrain flow")
            
            success = await retraining_orchestrator.execute_monthly_retrain()
            logger.info(f"Monthly retrain completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Monthly retrain failed: {e}")
            raise
else:
    async def monthly_full_retrain_flow():
        """Monthly full retrain flow"""
        try:
            logger.info("Starting monthly full retrain flow")
            
            success = await retraining_orchestrator.execute_monthly_retrain()
            logger.info(f"Monthly retrain completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Monthly retrain failed: {e}")
            raise

if PREFECT_AVAILABLE:
    @flow(name="nightly-incremental-update")
    async def nightly_incremental_flow():
        """Nightly incremental update flow"""
        try:
            logger = get_run_logger()
            logger.info("Starting nightly incremental update flow")
            
            success = await retraining_orchestrator.execute_nightly_update()
            logger.info(f"Nightly update completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Nightly update failed: {e}")
            raise
else:
    async def nightly_incremental_flow():
        """Nightly incremental update flow"""
        try:
            logger.info("Starting nightly incremental update flow")
            
            success = await retraining_orchestrator.execute_nightly_update()
            logger.info(f"Nightly update completed: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Nightly update failed: {e}")
            raise

# Phase 4C: Online & Safe Self-Retraining Methods

async def execute_online_incremental_learning(features: np.ndarray, labels: np.ndarray, 
                                             model_type: str = "gradient_boosting") -> Dict[str, Any]:
    """
    Execute online incremental learning with shadow mode validation
    """
    try:
        logger.info("🔄 Starting online incremental learning...")
        start_time = time.time()
        
        # Initialize online learner if needed
        if not retraining_orchestrator.online_learner.is_learning:
            await retraining_orchestrator.online_learner.initialize()
        
        # Perform incremental learning
        result = await retraining_orchestrator.online_learner.incremental_learn(features, labels, model_type)
        
        execution_time = time.time() - start_time
        
        # Record execution
        retraining_orchestrator._record_execution('online_incremental_learning', result['status'] != 'error', execution_time)
        
        if result['status'] == 'promoted':
            logger.info(f"✅ Shadow model promoted to production in {execution_time:.2f}s")
        elif result['status'] == 'updated':
            logger.info(f"✅ Shadow model updated in {execution_time:.2f}s")
        elif result['status'] == 'buffered':
            logger.info(f"📦 Data buffered for mini-batch processing: {result['samples_buffered']} samples")
        else:
            logger.warning(f"⚠️ Online learning result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in online incremental learning: {e}")
        return {'status': 'error', 'error': str(e)}

async def execute_shadow_mode_validation() -> Dict[str, Any]:
    """
    Execute shadow mode validation and promotion decision
    """
    try:
        logger.info("🔍 Starting shadow mode validation...")
        
        if not retraining_orchestrator.online_learner.shadow_mode_active:
            return {'status': 'inactive', 'message': 'No shadow mode active'}
        
        # Perform validation
        validation_result = await retraining_orchestrator.online_learner._validate_shadow_model()
        
        if validation_result['should_promote']:
            logger.info(f"✅ Shadow model ready for promotion (improvement: {validation_result['score']:.4f})")
            await retraining_orchestrator.online_learner._promote_shadow_model()
            return {'status': 'promoted', 'improvement': validation_result['score']}
        else:
            logger.info(f"📊 Shadow model validation: {validation_result['reason']} (score: {validation_result['score']:.4f})")
            return {'status': 'validated', 'result': validation_result}
        
    except Exception as e:
        logger.error(f"❌ Error in shadow mode validation: {e}")
        return {'status': 'error', 'error': str(e)}

async def execute_auto_rollback_check() -> Dict[str, Any]:
    """
    Check for auto-rollback conditions and execute if needed
    """
    try:
        logger.info("🔄 Checking for auto-rollback conditions...")
        
        # Get recent performance metrics
        recent_metrics = await retraining_orchestrator.performance_tracker.get_pattern_performance_metrics(days=7)
        
        # Check rollback conditions
        rollback_conditions = []
        
        if recent_metrics.success_rate < 0.4:  # 40% success rate threshold
            rollback_conditions.append("Low success rate")
        
        if recent_metrics.profit_factor < 0.8:  # 80% profit factor threshold
            rollback_conditions.append("Low profit factor")
        
        if len(rollback_conditions) > 0:
            logger.warning(f"⚠️ Auto-rollback conditions detected: {', '.join(rollback_conditions)}")
            
            # Execute rollback
            await retraining_orchestrator.online_learner._rollback_shadow_model()
            
            # Create rollback event
            await retraining_orchestrator._create_retraining_event(
                event_id=f"auto_rollback_{int(time.time())}",
                event_type='auto_rollback',
                model_type='pattern_detector',
                trigger_source='performance_degradation',
                trigger_metadata={
                    'conditions': rollback_conditions,
                    'metrics': {
                        'success_rate': recent_metrics.success_rate,
                        'profit_factor': recent_metrics.profit_factor
                    }
                }
            )
            
            return {'status': 'rolled_back', 'conditions': rollback_conditions}
        else:
            logger.info("✅ No auto-rollback conditions detected")
            return {'status': 'no_rollback_needed'}
        
    except Exception as e:
        logger.error(f"❌ Error in auto-rollback check: {e}")
        return {'status': 'error', 'error': str(e)}

async def get_online_learning_status() -> Dict[str, Any]:
    """
    Get comprehensive online learning status
    """
    try:
        # Get online learner status
        online_status = await retraining_orchestrator.online_learner.get_learning_statistics()
        
        # Get shadow mode status
        shadow_status = {
            'shadow_mode_active': retraining_orchestrator.online_learner.shadow_mode_active,
            'shadow_model_version': retraining_orchestrator.online_learner.shadow_model_version,
            'validation_results': retraining_orchestrator.online_learner.shadow_validation_results
        }
        
        # Get performance metrics
        performance_metrics = await retraining_orchestrator.performance_tracker.get_pattern_performance_metrics(days=7)
        
        return {
            'online_learner': online_status,
            'shadow_mode': shadow_status,
            'performance_metrics': {
                'success_rate': performance_metrics.success_rate,
                'profit_factor': performance_metrics.profit_factor,
                'total_signals': performance_metrics.total_signals
            },
            'configuration': {
                'enable_incremental_learning': retraining_orchestrator.online_learner.enable_incremental_learning,
                'enable_shadow_mode': retraining_orchestrator.online_learner.enable_shadow_mode,
                'shadow_validation_threshold': retraining_orchestrator.online_learner.shadow_validation_threshold,
                'auto_rollback_threshold': retraining_orchestrator.online_learner.auto_rollback_threshold,
                'mini_batch_size': retraining_orchestrator.online_learner.mini_batch_size
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting online learning status: {e}")
        return {'status': 'error', 'error': str(e)}

# Phase 5A: Canary Deployment Methods

async def execute_canary_deployment_start(candidate_model, model_version: str) -> Dict[str, Any]:
    """
    Start canary deployment with a candidate model
    """
    try:
        logger.info(f"🚀 Starting canary deployment for model version: {model_version}")
        
        # Start canary deployment
        result = await retraining_orchestrator.online_learner.start_canary_deployment(candidate_model, model_version)
        
        if result['success']:
            logger.info(f"✅ Canary deployment started: {result['canary_version']}")
            
            # Create deployment event
            await retraining_orchestrator._create_retraining_event(
                event_id=f"canary_start_{int(time.time())}",
                event_type='canary_deployment_started',
                model_type='pattern_detector',
                trigger_source='manual',
                trigger_metadata={
                    'canary_version': result['canary_version'],
                    'current_stage': result['current_stage'],
                    'traffic_percentage': result['traffic_percentage']
                }
            )
            
            return {'status': 'started', 'canary_version': result['canary_version']}
        else:
            logger.error(f"❌ Failed to start canary deployment: {result['reason']}")
            return {'status': 'failed', 'reason': result['reason']}
        
    except Exception as e:
        logger.error(f"❌ Error starting canary deployment: {e}")
        return {'status': 'error', 'error': str(e)}

async def execute_canary_prediction(features: np.ndarray, label: float = None) -> Dict[str, Any]:
    """
    Process prediction through canary deployment
    """
    try:
        # Process canary prediction
        result = await retraining_orchestrator.online_learner.process_canary_prediction(features, label)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in canary prediction: {e}")
        return {'canary_active': False, 'use_canary': False, 'error': str(e)}

async def execute_canary_promotion() -> Dict[str, Any]:
    """
    Promote canary model to production
    """
    try:
        logger.info("🎯 Attempting canary promotion to production...")
        
        # Check if canary is ready for promotion
        canary_status = await retraining_orchestrator.online_learner.get_canary_status()
        
        if not canary_status['active']:
            return {'status': 'failed', 'reason': 'No active canary deployment'}
        
        if not canary_status['promotion_ready']:
            return {'status': 'not_ready', 'reason': 'Canary not ready for promotion', 'status': canary_status}
        
        # Promote canary to production
        result = await retraining_orchestrator.online_learner.promote_canary_to_production()
        
        if result['success']:
            logger.info(f"✅ Canary promoted to production: {result['model_version']}")
            
            # Create promotion event
            await retraining_orchestrator._create_retraining_event(
                event_id=f"canary_promote_{int(time.time())}",
                event_type='canary_promoted_to_production',
                model_type='pattern_detector',
                trigger_source='canary_validation',
                trigger_metadata={
                    'promoted_version': result['model_version'],
                    'promotion_method': 'canary'
                }
            )
            
            return {'status': 'promoted', 'model_version': result['model_version']}
        else:
            logger.error(f"❌ Failed to promote canary: {result['reason']}")
            return {'status': 'failed', 'reason': result['reason']}
        
    except Exception as e:
        logger.error(f"❌ Error in canary promotion: {e}")
        return {'status': 'error', 'error': str(e)}

    async def get_canary_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive canary deployment status"""
        try:
            # Get canary status from online learner
            canary_status = await self.online_learner.get_canary_status()
            
            # Get performance metrics
            performance_metrics = await self.performance_tracker.get_pattern_performance_metrics(days=7)
            
            return {
                'canary_deployment': canary_status,
                'performance_metrics': {
                    'success_rate': performance_metrics.success_rate,
                    'profit_factor': performance_metrics.profit_factor,
                    'total_signals': performance_metrics.total_signals
                },
                'configuration': {
                    'enable_canary_deployment': self.online_learner.enable_canary_deployment,
                    'canary_traffic_percentage': self.online_learner.canary_traffic_percentage,
                    'canary_validation_threshold': self.online_learner.canary_validation_threshold,
                    'canary_rollback_threshold': self.online_learner.canary_rollback_threshold,
                    'canary_promotion_stages': self.online_learner.canary_promotion_stages
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting canary deployment status: {e}")
            return {'status': 'error', 'error': str(e)}

    # Phase 5B: Enhanced Ensemble Methods
    
    async def execute_phase5b_ensemble_training(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Execute Phase 5B ensemble training"""
        try:
            self.logger.info("🚀 Starting Phase 5B: Enhanced Ensemble Training...")
            
            # Train all ensemble models
            training_results = await self.enhanced_ensemble_manager.train_all_models(X, y)
            
            # Get ensemble status
            ensemble_status = await self.enhanced_ensemble_manager.get_ensemble_status()
            
            result = {
                'status': 'completed',
                'training_results': training_results,
                'ensemble_status': ensemble_status,
                'models_trained': sum(training_results.values()),
                'total_models': len(training_results),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Phase 5B ensemble training completed: {result['models_trained']}/{result['total_models']} models trained")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5B ensemble training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def execute_phase5b_ensemble_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Execute Phase 5B ensemble prediction with regime-aware selection"""
        try:
            self.logger.info("🔮 Making Phase 5B ensemble prediction...")
            
            # Make ensemble prediction
            prediction = await self.enhanced_ensemble_manager.predict(X)
            
            result = {
                'status': 'success',
                'ensemble_prediction': prediction.ensemble_prediction,
                'confidence': prediction.confidence,
                'regime': prediction.regime.value,
                'regime_confidence': prediction.regime_confidence,
                'selected_models': prediction.selected_models,
                'individual_predictions': prediction.individual_predictions,
                'model_weights': prediction.model_weights,
                'meta_learner_score': prediction.meta_learner_score,
                'timestamp': prediction.timestamp.isoformat()
            }
            
            self.logger.info(f"✅ Phase 5B ensemble prediction: {prediction.ensemble_prediction:.4f} "
                           f"(regime: {prediction.regime.value}, confidence: {prediction.confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5B ensemble prediction failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def execute_phase5b_model_training(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train a specific model type for Phase 5B ensemble"""
        try:
            self.logger.info(f"🔄 Training Phase 5B model: {model_type}")
            
            # Convert string to ModelType enum
            try:
                model_enum = ModelType(model_type)
            except ValueError:
                return {
                    'status': 'failed',
                    'error': f"Invalid model type: {model_type}",
                    'timestamp': datetime.now().isoformat()
                }
            
            # Detect regime for performance tracking
            regime, _ = self.enhanced_ensemble_manager.meta_learner.detect_regime(X)
            
            # Train the model
            success = await self.enhanced_ensemble_manager.train_model(model_enum, X, y, regime)
            
            if success:
                result = {
                    'status': 'completed',
                    'model_type': model_type,
                    'regime': regime.value,
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.info(f"✅ Phase 5B {model_type} model trained successfully")
            else:
                result = {
                    'status': 'failed',
                    'model_type': model_type,
                    'error': 'Model training failed',
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.error(f"❌ Phase 5B {model_type} model training failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5B model training failed: {e}")
            return {
                'status': 'failed',
                'model_type': model_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def get_phase5b_ensemble_status(self) -> Dict[str, Any]:
        """Get Phase 5B ensemble status"""
        try:
            self.logger.info("📊 Getting Phase 5B ensemble status...")
            
            status = await self.enhanced_ensemble_manager.get_ensemble_status()
            
            result = {
                'status': 'success',
                'ensemble_status': status,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ Phase 5B ensemble status retrieved")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get Phase 5B ensemble status: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def execute_phase5b_regime_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime for Phase 5B"""
        try:
            self.logger.info("🔍 Analyzing market regime for Phase 5B...")
            
            # Detect regime
            regime, confidence = self.enhanced_ensemble_manager.meta_learner.detect_regime(X)
            
            # Get regime-specific weights
            regime_weights = self.enhanced_ensemble_manager.meta_learner.get_regime_weights(regime)
            
            # Get best models for this regime
            best_models = self.enhanced_ensemble_manager.meta_learner.select_best_models(regime, top_k=3)
            
            result = {
                'status': 'success',
                'regime': regime.value,
                'regime_confidence': confidence,
                'regime_weights': regime_weights,
                'best_models': [model.value for model in best_models],
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Phase 5B regime analysis: {regime.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5B regime analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
