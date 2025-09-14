#!/usr/bin/env python3
"""
Shadow/Canary Deployment System for AlphaPulse
Phase 2: Traffic routing, candidate vs production comparison, and promotion gates

Features:
- Route 10% of live traffic to candidate model
- Log candidate vs production results to TimescaleDB
- Promotion gate: candidate must beat baseline in live metrics for N trades
- Automatic rollback on performance degradation
- Real-time monitoring and alerting
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid

# Database imports
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from ..ai.model_registry import ModelRegistry
from ..ai.advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    ACTIVE = "active"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class TrafficSplit(Enum):
    """Traffic split configurations"""
    SHADOW_5 = 0.05    # 5% to candidate
    SHADOW_10 = 0.10   # 10% to candidate
    SHADOW_20 = 0.20   # 20% to candidate
    CANARY_50 = 0.50   # 50% to candidate

@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    model_path: str
    model_type: str
    created_at: datetime
    metrics: Dict[str, float]
    is_production: bool = False
    is_candidate: bool = False

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    candidate_model_id: str
    production_model_id: str
    traffic_split: TrafficSplit = TrafficSplit.SHADOW_10
    promotion_threshold: float = 0.7  # 70% better than baseline
    min_trades_for_promotion: int = 100
    max_trades_for_evaluation: int = 1000
    evaluation_window_hours: int = 24
    auto_rollback_threshold: float = 0.3  # 30% worse than baseline
    max_rollback_trades: int = 50
    created_at: datetime = field(default_factory=datetime.now)
    status: DeploymentStatus = DeploymentStatus.PENDING

@dataclass
class PredictionResult:
    """Prediction result for comparison"""
    request_id: str
    timestamp: datetime
    features: Dict[str, float]
    production_prediction: float
    candidate_prediction: float
    actual_outcome: Optional[float] = None
    production_confidence: float = 0.0
    candidate_confidence: float = 0.0
    latency_ms: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentMetrics:
    """Deployment performance metrics"""
    deployment_id: str
    total_requests: int = 0
    production_requests: int = 0
    candidate_requests: int = 0
    production_accuracy: float = 0.0
    candidate_accuracy: float = 0.0
    production_auc: float = 0.0
    candidate_auc: float = 0.0
    production_latency_p95: float = 0.0
    candidate_latency_p95: float = 0.0
    accuracy_improvement: float = 0.0
    auc_improvement: float = 0.0
    latency_improvement: float = 0.0
    overall_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class ShadowDeploymentService:
    """
    Shadow/Canary deployment service for model comparison and promotion
    """
    
    def __init__(self, db_session: AsyncSession = None):
        self.db_session = db_session
        self.model_registry = ModelRegistry()
        
        # Active deployments
        self.active_deployments: Dict[str, DeploymentConfig] = {}
        self.deployment_metrics: Dict[str, DeploymentMetrics] = {}
        
        # Model versions cache
        self.model_versions: Dict[str, ModelVersion] = {}
        
        # Performance tracking
        self.prediction_history: List[PredictionResult] = []
        self.max_history_size = 10000
        
        # Service state
        self.is_running = False
        self.monitoring_task = None
        
        logger.info("🚀 Shadow Deployment Service initialized")
    
    async def start(self):
        """Start the shadow deployment service"""
        if self.is_running:
            logger.warning("Shadow deployment service is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Shadow deployment service started")
    
    async def stop(self):
        """Stop the shadow deployment service"""
        if not self.is_running:
            logger.warning("Shadow deployment service is not running")
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Shadow deployment service stopped")
    
    async def create_deployment(self, 
                               candidate_model_id: str,
                               production_model_id: str,
                               traffic_split: TrafficSplit = TrafficSplit.SHADOW_10,
                               promotion_threshold: float = 0.7,
                               min_trades: int = 100) -> str:
        """
        Create a new shadow/canary deployment
        
        Returns:
            Deployment ID
        """
        try:
            # Validate models exist
            candidate_model = self.model_registry.get_model(candidate_model_id)
            production_model = self.model_registry.get_model(production_model_id)
            
            if not candidate_model or not production_model:
                raise ValueError("Invalid model IDs provided")
            
            # Generate deployment ID
            deployment_id = f"deployment_{uuid.uuid4().hex[:8]}"
            
            # Create deployment config
            deployment_config = DeploymentConfig(
                deployment_id=deployment_id,
                candidate_model_id=candidate_model_id,
                production_model_id=production_model_id,
                traffic_split=traffic_split,
                promotion_threshold=promotion_threshold,
                min_trades_for_promotion=min_trades
            )
            
            # Store deployment
            self.active_deployments[deployment_id] = deployment_config
            
            # Initialize metrics
            self.deployment_metrics[deployment_id] = DeploymentMetrics(
                deployment_id=deployment_id
            )
            
            # Log to database
            await self._log_deployment_created(deployment_config)
            
            logger.info(f"✅ Created deployment {deployment_id}: {candidate_model_id} -> {production_model_id}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"❌ Error creating deployment: {e}")
            raise
    
    async def predict_with_shadow(self, 
                                 features: Dict[str, float],
                                 deployment_id: str = None) -> Dict[str, Any]:
        """
        Make prediction with shadow deployment
        
        Returns:
            Dictionary with production prediction and optional candidate prediction
        """
        try:
            request_id = f"req_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            
            # Get production prediction
            production_model_id = None
            if deployment_id and deployment_id in self.active_deployments:
                deployment = self.active_deployments[deployment_id]
                production_model_id = deployment.production_model_id
            else:
                # Use default production model
                production_model_id = await self._get_default_production_model()
            
            # Get production prediction
            production_result = await self.model_registry.predict(
                model_id=production_model_id,
                features=features
            )
            
            production_prediction = production_result.get('prediction', 0.5)
            production_confidence = production_result.get('confidence', 0.0)
            
            # Determine if we should also get candidate prediction
            candidate_prediction = None
            candidate_confidence = None
            candidate_model_id = None
            
            if deployment_id and deployment_id in self.active_deployments:
                deployment = self.active_deployments[deployment_id]
                
                # Check if we should route to candidate based on traffic split
                if random.random() < deployment.traffic_split.value:
                    candidate_model_id = deployment.candidate_model_id
                    
                    # Get candidate prediction
                    candidate_result = await self.model_registry.predict(
                        model_id=candidate_model_id,
                        features=features
                    )
                    
                    candidate_prediction = candidate_result.get('prediction', 0.5)
                    candidate_confidence = candidate_result.get('confidence', 0.0)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create prediction result
            prediction_result = PredictionResult(
                request_id=request_id,
                timestamp=datetime.now(),
                features=features,
                production_prediction=production_prediction,
                candidate_prediction=candidate_prediction,
                production_confidence=production_confidence,
                candidate_confidence=candidate_confidence,
                latency_ms=latency_ms,
                model_versions={
                    'production': production_model_id,
                    'candidate': candidate_model_id
                }
            )
            
            # Store prediction result
            self.prediction_history.append(prediction_result)
            
            # Trim history if too large
            if len(self.prediction_history) > self.max_history_size:
                self.prediction_history = self.prediction_history[-self.max_history_size:]
            
            # Update metrics
            if deployment_id and deployment_id in self.deployment_metrics:
                await self._update_deployment_metrics(deployment_id, prediction_result)
            
            # Log to database
            await self._log_prediction_result(prediction_result, deployment_id)
            
            # Return results
            result = {
                'request_id': request_id,
                'prediction': production_prediction,
                'confidence': production_confidence,
                'latency_ms': latency_ms,
                'model_version': production_model_id
            }
            
            # Add candidate info if available
            if candidate_prediction is not None:
                result['candidate_prediction'] = candidate_prediction
                result['candidate_confidence'] = candidate_confidence
                result['candidate_model_version'] = candidate_model_id
                result['traffic_split'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in shadow prediction: {e}")
            # Return fallback prediction
            return {
                'request_id': f"req_{uuid.uuid4().hex[:8]}",
                'prediction': 0.5,
                'confidence': 0.0,
                'latency_ms': 0.0,
                'model_version': 'fallback',
                'error': str(e)
            }
    
    async def update_outcome(self, request_id: str, actual_outcome: float):
        """
        Update prediction result with actual outcome
        
        This is called when we have the actual result to compare against predictions
        """
        try:
            # Find prediction result
            for result in self.prediction_history:
                if result.request_id == request_id:
                    result.actual_outcome = actual_outcome
                    
                    # Update metrics for all active deployments
                    for deployment_id in self.active_deployments:
                        await self._update_deployment_metrics(deployment_id, result)
                    
                    # Log to database
                    await self._log_outcome_update(result)
                    break
            
        except Exception as e:
            logger.error(f"❌ Error updating outcome: {e}")
    
    async def evaluate_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Evaluate deployment performance and decide on promotion/rollback
        """
        try:
            if deployment_id not in self.active_deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.active_deployments[deployment_id]
            metrics = self.deployment_metrics[deployment_id]
            
            # Check if we have enough data
            if metrics.total_requests < deployment.min_trades_for_promotion:
                return {
                    'status': 'insufficient_data',
                    'message': f"Need {deployment.min_trades_for_promotion} trades, have {metrics.total_requests}",
                    'metrics': metrics.__dict__
                }
            
            # Calculate overall improvement score
            accuracy_weight = 0.4
            auc_weight = 0.4
            latency_weight = 0.2
            
            overall_score = (
                metrics.accuracy_improvement * accuracy_weight +
                metrics.auc_improvement * auc_weight +
                metrics.latency_improvement * latency_weight
            )
            
            # Update overall score
            metrics.overall_score = overall_score
            metrics.last_updated = datetime.now()
            
            # Check promotion criteria
            if overall_score >= deployment.promotion_threshold:
                # Promote candidate to production
                await self._promote_candidate(deployment_id)
                return {
                    'status': 'promoted',
                    'message': f"Candidate promoted with score {overall_score:.3f}",
                    'metrics': metrics.__dict__
                }
            
            # Check rollback criteria
            elif overall_score <= -deployment.auto_rollback_threshold:
                # Rollback deployment
                await self._rollback_deployment(deployment_id)
                return {
                    'status': 'rolled_back',
                    'message': f"Deployment rolled back with score {overall_score:.3f}",
                    'metrics': metrics.__dict__
                }
            
            # Continue monitoring
            return {
                'status': 'monitoring',
                'message': f"Continuing monitoring with score {overall_score:.3f}",
                'metrics': metrics.__dict__
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating deployment: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }
    
    async def _monitoring_loop(self):
        """Main monitoring loop for active deployments"""
        while self.is_running:
            try:
                # Evaluate all active deployments
                for deployment_id in list(self.active_deployments.keys()):
                    deployment = self.active_deployments[deployment_id]
                    
                    # Skip if deployment is not active
                    if deployment.status != DeploymentStatus.ACTIVE:
                        continue
                    
                    # Evaluate deployment
                    evaluation_result = await self.evaluate_deployment(deployment_id)
                    
                    # Log evaluation result
                    logger.info(f"Deployment {deployment_id} evaluation: {evaluation_result['status']}")
                    
                    # Handle promotion/rollback
                    if evaluation_result['status'] in ['promoted', 'rolled_back']:
                        await self._log_deployment_status_change(deployment_id, evaluation_result)
                
                # Wait before next evaluation
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # 1 minute on error
    
    async def _update_deployment_metrics(self, deployment_id: str, prediction_result: PredictionResult):
        """Update deployment metrics with new prediction result"""
        try:
            if deployment_id not in self.deployment_metrics:
                return
            
            metrics = self.deployment_metrics[deployment_id]
            deployment = self.active_deployments[deployment_id]
            
            # Update request counts
            metrics.total_requests += 1
            metrics.production_requests += 1
            
            if prediction_result.candidate_prediction is not None:
                metrics.candidate_requests += 1
            
            # Update accuracy if we have actual outcome
            if prediction_result.actual_outcome is not None:
                # Calculate production accuracy
                production_correct = abs(prediction_result.production_prediction - prediction_result.actual_outcome) < 0.1
                metrics.production_accuracy = (
                    (metrics.production_accuracy * (metrics.production_requests - 1) + production_correct) / 
                    metrics.production_requests
                )
                
                # Calculate candidate accuracy if available
                if prediction_result.candidate_prediction is not None:
                    candidate_correct = abs(prediction_result.candidate_prediction - prediction_result.actual_outcome) < 0.1
                    metrics.candidate_accuracy = (
                        (metrics.candidate_accuracy * (metrics.candidate_requests - 1) + candidate_correct) / 
                        metrics.candidate_requests
                    )
                
                # Calculate improvements
                if metrics.candidate_accuracy > 0 and metrics.production_accuracy > 0:
                    metrics.accuracy_improvement = (
                        metrics.candidate_accuracy - metrics.production_accuracy
                    ) / metrics.production_accuracy
            
            # Update latency metrics (simplified)
            if prediction_result.latency_ms > 0:
                # Simple moving average for latency
                if metrics.production_latency_p95 == 0:
                    metrics.production_latency_p95 = prediction_result.latency_ms
                else:
                    metrics.production_latency_p95 = (
                        metrics.production_latency_p95 * 0.95 + prediction_result.latency_ms * 0.05
                    )
            
            metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error updating deployment metrics: {e}")
    
    async def _promote_candidate(self, deployment_id: str):
        """Promote candidate model to production"""
        try:
            deployment = self.active_deployments[deployment_id]
            
            # Update model registry
            await self.model_registry.promote_model(
                model_id=deployment.candidate_model_id,
                production=True
            )
            
            # Update deployment status
            deployment.status = DeploymentStatus.PROMOTED
            
            # Log promotion
            await self._log_deployment_promotion(deployment)
            
            logger.info(f"🎉 Candidate {deployment.candidate_model_id} promoted to production!")
            
        except Exception as e:
            logger.error(f"❌ Error promoting candidate: {e}")
    
    async def _rollback_deployment(self, deployment_id: str):
        """Rollback deployment to production model"""
        try:
            deployment = self.active_deployments[deployment_id]
            
            # Update deployment status
            deployment.status = DeploymentStatus.ROLLED_BACK
            
            # Log rollback
            await self._log_deployment_rollback(deployment)
            
            logger.info(f"🔄 Deployment {deployment_id} rolled back to production model")
            
        except Exception as e:
            logger.error(f"❌ Error rolling back deployment: {e}")
    
    async def _get_default_production_model(self) -> str:
        """Get default production model ID"""
        # This would typically query the model registry for the current production model
        # For now, return a placeholder
        return "production_model_v1"
    
    async def _log_deployment_created(self, deployment: DeploymentConfig):
        """Log deployment creation to database"""
        try:
            if self.db_session:
                query = text("""
                    INSERT INTO shadow_deployments (
                        deployment_id, candidate_model_id, production_model_id,
                        traffic_split, promotion_threshold, min_trades,
                        status, created_at
                    ) VALUES (
                        :deployment_id, :candidate_model_id, :production_model_id,
                        :traffic_split, :promotion_threshold, :min_trades,
                        :status, :created_at
                    )
                """)
                
                await self.db_session.execute(query, {
                    'deployment_id': deployment.deployment_id,
                    'candidate_model_id': deployment.candidate_model_id,
                    'production_model_id': deployment.production_model_id,
                    'traffic_split': deployment.traffic_split.value,
                    'promotion_threshold': deployment.promotion_threshold,
                    'min_trades': deployment.min_trades_for_promotion,
                    'status': deployment.status.value,
                    'created_at': deployment.created_at
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging deployment creation: {e}")
    
    async def _log_prediction_result(self, result: PredictionResult, deployment_id: str = None):
        """Log prediction result to database"""
        try:
            if self.db_session:
                query = text("""
                    INSERT INTO shadow_predictions (
                        request_id, deployment_id, timestamp, features,
                        production_prediction, candidate_prediction,
                        production_confidence, candidate_confidence,
                        latency_ms, model_versions
                    ) VALUES (
                        :request_id, :deployment_id, :timestamp, :features,
                        :production_prediction, :candidate_prediction,
                        :production_confidence, :candidate_confidence,
                        :latency_ms, :model_versions
                    )
                """)
                
                await self.db_session.execute(query, {
                    'request_id': result.request_id,
                    'deployment_id': deployment_id,
                    'timestamp': result.timestamp,
                    'features': json.dumps(result.features),
                    'production_prediction': result.production_prediction,
                    'candidate_prediction': result.candidate_prediction,
                    'production_confidence': result.production_confidence,
                    'candidate_confidence': result.candidate_confidence,
                    'latency_ms': result.latency_ms,
                    'model_versions': json.dumps(result.model_versions)
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging prediction result: {e}")
    
    async def _log_outcome_update(self, result: PredictionResult):
        """Log outcome update to database"""
        try:
            if self.db_session and result.actual_outcome is not None:
                query = text("""
                    UPDATE shadow_predictions 
                    SET actual_outcome = :actual_outcome
                    WHERE request_id = :request_id
                """)
                
                await self.db_session.execute(query, {
                    'actual_outcome': result.actual_outcome,
                    'request_id': result.request_id
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging outcome update: {e}")
    
    async def _log_deployment_status_change(self, deployment_id: str, evaluation_result: Dict[str, Any]):
        """Log deployment status change to database"""
        try:
            if self.db_session:
                query = text("""
                    UPDATE shadow_deployments 
                    SET status = :status, updated_at = :updated_at
                    WHERE deployment_id = :deployment_id
                """)
                
                await self.db_session.execute(query, {
                    'status': evaluation_result['status'],
                    'updated_at': datetime.now(),
                    'deployment_id': deployment_id
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging deployment status change: {e}")
    
    async def _log_deployment_promotion(self, deployment: DeploymentConfig):
        """Log deployment promotion to database"""
        try:
            if self.db_session:
                query = text("""
                    INSERT INTO deployment_events (
                        deployment_id, event_type, event_data, timestamp
                    ) VALUES (
                        :deployment_id, 'promotion', :event_data, :timestamp
                    )
                """)
                
                await self.db_session.execute(query, {
                    'deployment_id': deployment.deployment_id,
                    'event_data': json.dumps({
                        'candidate_model_id': deployment.candidate_model_id,
                        'production_model_id': deployment.production_model_id
                    }),
                    'timestamp': datetime.now()
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging deployment promotion: {e}")
    
    async def _log_deployment_rollback(self, deployment: DeploymentConfig):
        """Log deployment rollback to database"""
        try:
            if self.db_session:
                query = text("""
                    INSERT INTO deployment_events (
                        deployment_id, event_type, event_data, timestamp
                    ) VALUES (
                        :deployment_id, 'rollback', :event_data, :timestamp
                    )
                """)
                
                await self.db_session.execute(query, {
                    'deployment_id': deployment.deployment_id,
                    'event_data': json.dumps({
                        'reason': 'performance_degradation',
                        'candidate_model_id': deployment.candidate_model_id
                    }),
                    'timestamp': datetime.now()
                })
                
                await self.db_session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error logging deployment rollback: {e}")
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of all active deployments"""
        try:
            summary = {
                'total_deployments': len(self.active_deployments),
                'active_deployments': [],
                'deployment_metrics': {}
            }
            
            for deployment_id, deployment in self.active_deployments.items():
                deployment_info = {
                    'deployment_id': deployment_id,
                    'candidate_model_id': deployment.candidate_model_id,
                    'production_model_id': deployment.production_model_id,
                    'traffic_split': deployment.traffic_split.value,
                    'status': deployment.status.value,
                    'created_at': deployment.created_at.isoformat()
                }
                
                summary['active_deployments'].append(deployment_info)
                
                # Add metrics if available
                if deployment_id in self.deployment_metrics:
                    metrics = self.deployment_metrics[deployment_id]
                    summary['deployment_metrics'][deployment_id] = {
                        'total_requests': metrics.total_requests,
                        'candidate_requests': metrics.candidate_requests,
                        'accuracy_improvement': metrics.accuracy_improvement,
                        'overall_score': metrics.overall_score,
                        'last_updated': metrics.last_updated.isoformat()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting deployment summary: {e}")
            return {'error': str(e)}

# Global shadow deployment service instance
shadow_deployment_service = ShadowDeploymentService()

# Export for use in other modules
__all__ = [
    'ShadowDeploymentService',
    'DeploymentConfig',
    'PredictionResult',
    'DeploymentMetrics',
    'TrafficSplit',
    'DeploymentStatus',
    'shadow_deployment_service'
]
