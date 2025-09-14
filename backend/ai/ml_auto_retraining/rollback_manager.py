#!/usr/bin/env python3
"""
Phase 1: Model Rollback Manager
Handles safe model rollbacks with validation, event tracking, and performance monitoring
"""

import os
import sys
import json
import logging
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the versioning manager
from ai.ml_auto_retraining.model_versioning_manager import ModelVersioningManager, ModelVersion, RollbackEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RollbackDecision:
    """Rollback decision information"""
    should_rollback: bool
    rollback_reason: str
    rollback_type: str
    performance_degradation: Optional[float]
    drift_severity: Optional[str]
    confidence: float
    recommended_version: Optional[int]
    risk_assessment: Dict[str, Any]

class RollbackManager:
    """Advanced model rollback manager with safety checks and validation"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.versioning_manager = ModelVersioningManager(db_config)
        
        # Rollback thresholds
        self.performance_degradation_threshold = 0.15  # 15% performance drop
        self.error_rate_threshold = 0.10  # 10% error rate
        self.drift_severity_threshold = 0.25  # 25% drift severity
        self.min_usage_count = 100  # Minimum usage count before considering rollback
        
    def _generate_rollback_id(self, model_name: str, from_version: int, to_version: int) -> str:
        """Generate unique rollback ID"""
        return f"{model_name}_rollback_{from_version}_to_{to_version}_{uuid.uuid4().hex[:8]}"
    
    async def assess_rollback_needs(self, 
                                  model_name: str, 
                                  regime: str, 
                                  symbol: str,
                                  current_performance: Optional[Dict[str, float]] = None,
                                  current_drift_metrics: Optional[Dict[str, float]] = None,
                                  current_error_rate: Optional[float] = None) -> RollbackDecision:
        """Assess if a model needs to be rolled back"""
        
        logger.info(f"🔍 Assessing rollback needs for {model_name} {regime} {symbol}")
        
        try:
            # Get current production model
            current_model = await self.versioning_manager.get_production_model(model_name, regime, symbol)
            if not current_model:
                logger.warning(f"⚠️ No production model found for {model_name} {regime} {symbol}")
                return RollbackDecision(
                    should_rollback=False,
                    rollback_reason="No production model found",
                    rollback_type="none",
                    performance_degradation=None,
                    drift_severity=None,
                    confidence=0.0,
                    recommended_version=None,
                    risk_assessment={"status": "no_model"}
                )
            
            # Check usage count
            if current_model.usage_count < self.min_usage_count:
                logger.info(f"ℹ️ Model {current_model.model_name} v{current_model.version} has insufficient usage ({current_model.usage_count})")
                return RollbackDecision(
                    should_rollback=False,
                    rollback_reason="Insufficient usage for rollback assessment",
                    rollback_type="none",
                    performance_degradation=None,
                    drift_severity=None,
                    confidence=0.0,
                    recommended_version=None,
                    risk_assessment={"status": "insufficient_usage", "usage_count": current_model.usage_count}
                )
            
            # Check error rate
            error_rate = current_model.error_count / max(current_model.total_inferences, 1)
            if error_rate > self.error_rate_threshold:
                logger.warning(f"🚨 High error rate detected: {error_rate:.2%} for {current_model.model_name} v{current_model.version}")
                return RollbackDecision(
                    should_rollback=True,
                    rollback_reason=f"High error rate: {error_rate:.2%}",
                    rollback_type="error",
                    performance_degradation=None,
                    drift_severity=None,
                    confidence=0.9,
                    recommended_version=None,
                    risk_assessment={"error_rate": error_rate, "threshold": self.error_rate_threshold}
                )
            
            # Check performance degradation if metrics provided
            if current_performance and current_model.performance_metrics:
                performance_degradation = self._calculate_performance_degradation(
                    current_model.performance_metrics, current_performance
                )
                
                if performance_degradation > self.performance_degradation_threshold:
                    logger.warning(f"🚨 Performance degradation detected: {performance_degradation:.2%}")
                    return RollbackDecision(
                        should_rollback=True,
                        rollback_reason=f"Performance degradation: {performance_degradation:.2%}",
                        rollback_type="performance",
                        performance_degradation=performance_degradation,
                        drift_severity=None,
                        confidence=0.8,
                        recommended_version=None,
                        risk_assessment={"performance_degradation": performance_degradation, "threshold": self.performance_degradation_threshold}
                    )
            
            # Check drift severity if metrics provided
            if current_drift_metrics:
                drift_severity = self._calculate_drift_severity(current_drift_metrics)
                
                if drift_severity > self.drift_severity_threshold:
                    logger.warning(f"🚨 High drift severity detected: {drift_severity:.2%}")
                    return RollbackDecision(
                        should_rollback=True,
                        rollback_reason=f"High drift severity: {drift_severity:.2%}",
                        rollback_type="drift",
                        performance_degradation=None,
                        drift_severity=drift_severity,
                        confidence=0.7,
                        recommended_version=None,
                        risk_assessment={"drift_severity": drift_severity, "threshold": self.drift_severity_threshold}
                    )
            
            # No rollback needed
            logger.info(f"✅ No rollback needed for {current_model.model_name} v{current_model.version}")
            return RollbackDecision(
                should_rollback=False,
                rollback_reason="Model performing within acceptable parameters",
                rollback_type="none",
                performance_degradation=None,
                drift_severity=None,
                confidence=0.95,
                recommended_version=None,
                risk_assessment={"status": "healthy"}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to assess rollback needs: {e}")
            return RollbackDecision(
                should_rollback=False,
                rollback_reason=f"Assessment failed: {str(e)}",
                rollback_type="none",
                performance_degradation=None,
                drift_severity=None,
                confidence=0.0,
                recommended_version=None,
                risk_assessment={"status": "error", "error": str(e)}
            )
    
    def _calculate_performance_degradation(self, 
                                         baseline_metrics: Dict[str, float], 
                                         current_metrics: Dict[str, float]) -> float:
        """Calculate performance degradation percentage"""
        
        try:
            # Use F1 score as primary metric, fallback to accuracy
            baseline_score = baseline_metrics.get('f1_score', baseline_metrics.get('accuracy', 0.0))
            current_score = current_metrics.get('f1_score', current_metrics.get('accuracy', 0.0))
            
            if baseline_score == 0:
                return 0.0
            
            degradation = (baseline_score - current_score) / baseline_score
            return max(0.0, degradation)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance degradation: {e}")
            return 0.0
    
    def _calculate_drift_severity(self, drift_metrics: Dict[str, float]) -> float:
        """Calculate overall drift severity"""
        
        try:
            # Use PSI as primary drift metric
            psi_values = [v for k, v in drift_metrics.items() if 'psi' in k.lower()]
            if psi_values:
                return max(psi_values)  # Return maximum PSI value
            
            # Fallback to any drift metric
            drift_values = list(drift_metrics.values())
            if drift_values:
                return max(drift_values)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate drift severity: {e}")
            return 0.0
    
    async def find_rollback_candidate(self, 
                                    model_name: str, 
                                    regime: str, 
                                    symbol: str,
                                    min_performance_threshold: float = 0.7) -> Optional[ModelVersion]:
        """Find a suitable model version for rollback"""
        
        logger.info(f"🔍 Finding rollback candidate for {model_name} {regime} {symbol}")
        
        try:
            # Get previous production models (archived)
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_name, version, status, regime, symbol, model_artifact_path,
                       model_artifact_size_mb, model_artifact_hash, training_metrics,
                       validation_metrics, test_metrics, performance_metrics, drift_metrics,
                       rollback_metrics, deployment_timestamp, last_used_timestamp,
                       usage_count, error_count, avg_inference_time_ms, total_inferences,
                       version_metadata, created_by
                FROM model_versions
                WHERE model_name = %s AND regime = %s AND symbol = %s 
                AND status IN ('archived', 'staging')
                AND performance_metrics IS NOT NULL
                ORDER BY version DESC
            """, (model_name, regime, symbol))
            
            rows = cursor.fetchall()
            conn.close()
            
            candidates = []
            for row in rows:
                model_version = ModelVersion(
                    model_name=row[0],
                    version=row[1],
                    status=row[2],
                    regime=row[3],
                    symbol=row[4],
                    model_artifact_path=row[5],
                    model_artifact_size_mb=row[6],
                    model_artifact_hash=row[7],
                    training_metrics=row[8] if row[8] else {},
                    validation_metrics=row[9] if row[9] else {},
                    test_metrics=row[10] if row[10] else None,
                    performance_metrics=row[11] if row[11] else None,
                    drift_metrics=row[12] if row[12] else None,
                    rollback_metrics=row[13] if row[13] else None,
                    deployment_timestamp=row[14],
                    last_used_timestamp=row[15],
                    usage_count=row[16],
                    error_count=row[17],
                    avg_inference_time_ms=row[18],
                    total_inferences=row[19],
                    version_metadata=row[20] if row[20] else {},
                    created_by=row[21]
                )
                candidates.append(model_version)
            
            # Filter candidates by performance threshold
            suitable_candidates = []
            for candidate in candidates:
                if candidate.performance_metrics:
                    f1_score = candidate.performance_metrics.get('f1_score', 0.0)
                    accuracy = candidate.performance_metrics.get('accuracy', 0.0)
                    
                    if f1_score >= min_performance_threshold or accuracy >= min_performance_threshold:
                        suitable_candidates.append(candidate)
            
            if not suitable_candidates:
                logger.warning(f"⚠️ No suitable rollback candidates found for {model_name} {regime} {symbol}")
                return None
            
            # Sort by performance and select the best candidate
            best_candidate = max(suitable_candidates, 
                               key=lambda x: x.performance_metrics.get('f1_score', 0.0) if x.performance_metrics else 0.0)
            
            logger.info(f"✅ Found rollback candidate: {best_candidate.model_name} v{best_candidate.version}")
            return best_candidate
            
        except Exception as e:
            logger.error(f"❌ Failed to find rollback candidate: {e}")
            return None
    
    async def execute_rollback(self, 
                             model_name: str, 
                             regime: str, 
                             symbol: str,
                             rollback_reason: str,
                             rollback_type: str,
                             performance_degradation: Optional[float] = None,
                             drift_severity: Optional[str] = None,
                             error_details: Optional[Dict[str, Any]] = None,
                             triggered_by: str = "auto_retraining_system") -> bool:
        """Execute a model rollback"""
        
        logger.info(f"🔄 Executing rollback for {model_name} {regime} {symbol}")
        
        rollback_start_time = datetime.now()
        
        try:
            # Get current production model
            current_model = await self.versioning_manager.get_production_model(model_name, regime, symbol)
            if not current_model:
                logger.error(f"❌ No production model found for rollback: {model_name} {regime} {symbol}")
                return False
            
            # Find rollback candidate
            rollback_candidate = await self.find_rollback_candidate(model_name, regime, symbol)
            if not rollback_candidate:
                logger.error(f"❌ No suitable rollback candidate found for {model_name} {regime} {symbol}")
                return False
            
            # Validate rollback candidate
            if not await self._validate_rollback_candidate(rollback_candidate):
                logger.error(f"❌ Rollback candidate validation failed for {rollback_candidate.model_name} v{rollback_candidate.version}")
                return False
            
            # Execute the rollback
            success = await self._perform_rollback(current_model, rollback_candidate)
            
            # Calculate rollback duration
            rollback_duration = (datetime.now() - rollback_start_time).total_seconds()
            
            # Record rollback event
            await self._record_rollback_event(
                model_name=model_name,
                from_version=current_model.version,
                to_version=rollback_candidate.version,
                rollback_reason=rollback_reason,
                rollback_type=rollback_type,
                performance_degradation=performance_degradation,
                drift_severity=drift_severity,
                error_details=error_details,
                rollback_triggered_by=triggered_by,
                rollback_duration_seconds=int(rollback_duration),
                rollback_success=success
            )
            
            if success:
                logger.info(f"✅ Rollback completed successfully: {current_model.model_name} v{current_model.version} -> v{rollback_candidate.version}")
            else:
                logger.error(f"❌ Rollback failed: {current_model.model_name} v{current_model.version} -> v{rollback_candidate.version}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Rollback execution failed: {e}")
            return False
    
    async def _validate_rollback_candidate(self, candidate: ModelVersion) -> bool:
        """Validate a rollback candidate"""
        
        try:
            # Check if model artifact exists
            if not os.path.exists(candidate.model_artifact_path):
                logger.error(f"❌ Model artifact not found: {candidate.model_artifact_path}")
                return False
            
            # Check performance metrics
            if not candidate.performance_metrics:
                logger.warning(f"⚠️ No performance metrics for rollback candidate: {candidate.model_name} v{candidate.version}")
                return False
            
            # Check error rate
            if candidate.total_inferences > 0:
                error_rate = candidate.error_count / candidate.total_inferences
                if error_rate > 0.2:  # 20% error rate threshold
                    logger.warning(f"⚠️ High error rate for rollback candidate: {error_rate:.2%}")
                    return False
            
            logger.info(f"✅ Rollback candidate validated: {candidate.model_name} v{candidate.version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback candidate validation failed: {e}")
            return False
    
    async def _perform_rollback(self, current_model: ModelVersion, rollback_candidate: ModelVersion) -> bool:
        """Perform the actual rollback operation"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Archive current production model
            cursor.execute("""
                UPDATE model_versions
                SET status = 'archived'
                WHERE model_name = %s AND version = %s AND status = 'production'
            """, (current_model.model_name, current_model.version))
            
            # Promote rollback candidate to production
            cursor.execute("""
                UPDATE model_versions
                SET status = 'production',
                    deployment_timestamp = NOW()
                WHERE model_name = %s AND version = %s
            """, (rollback_candidate.model_name, rollback_candidate.version))
            
            # Update ml_models table for compatibility
            cursor.execute("""
                UPDATE ml_models
                SET status = 'archived'
                WHERE model_name = %s AND version = %s AND status = 'production'
            """, (current_model.model_name, current_model.version))
            
            cursor.execute("""
                UPDATE ml_models
                SET status = 'production'
                WHERE model_name = %s AND version = %s
            """, (rollback_candidate.model_name, rollback_candidate.version))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Database rollback completed: {current_model.model_name} v{current_model.version} -> v{rollback_candidate.version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database rollback failed: {e}")
            return False
    
    async def _record_rollback_event(self,
                                   model_name: str,
                                   from_version: int,
                                   to_version: int,
                                   rollback_reason: str,
                                   rollback_type: str,
                                   performance_degradation: Optional[float],
                                   drift_severity: Optional[str],
                                   error_details: Optional[Dict[str, Any]],
                                   rollback_triggered_by: str,
                                   rollback_duration_seconds: int,
                                   rollback_success: bool):
        """Record rollback event in database"""
        
        try:
            rollback_id = self._generate_rollback_id(model_name, from_version, to_version)
            
            rollback_event = RollbackEvent(
                rollback_id=rollback_id,
                model_name=model_name,
                from_version=from_version,
                to_version=to_version,
                rollback_reason=rollback_reason,
                rollback_type=rollback_type,
                performance_degradation=performance_degradation,
                drift_severity=drift_severity,
                error_details=error_details,
                rollback_triggered_by=rollback_triggered_by,
                rollback_duration_seconds=rollback_duration_seconds,
                rollback_success=rollback_success,
                rollback_metadata={
                    "timestamp": datetime.now().isoformat(),
                    "phase": "phase1_enhancement"
                }
            )
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rollback_events (
                    rollback_id, model_name, from_version, to_version, rollback_reason,
                    rollback_type, performance_degradation, drift_severity, error_details,
                    rollback_triggered_by, rollback_duration_seconds, rollback_success, rollback_metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb
                )
            """, (
                rollback_event.rollback_id,
                rollback_event.model_name,
                rollback_event.from_version,
                rollback_event.to_version,
                rollback_event.rollback_reason,
                rollback_event.rollback_type,
                rollback_event.performance_degradation,
                rollback_event.drift_severity,
                json.dumps(rollback_event.error_details) if rollback_event.error_details else None,
                rollback_event.rollback_triggered_by,
                rollback_event.rollback_duration_seconds,
                rollback_event.rollback_success,
                json.dumps(rollback_event.rollback_metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Recorded rollback event: {rollback_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record rollback event: {e}")
    
    async def get_rollback_history(self, model_name: str, days: int = 30) -> List[RollbackEvent]:
        """Get rollback history for a model"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT rollback_id, model_name, from_version, to_version, rollback_reason,
                       rollback_type, performance_degradation, drift_severity, error_details,
                       rollback_triggered_by, rollback_duration_seconds, rollback_success, rollback_metadata
                FROM rollback_events
                WHERE model_name = %s AND created_at >= NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
            """, (model_name, days))
            
            rows = cursor.fetchall()
            conn.close()
            
            rollback_events = []
            for row in rows:
                rollback_event = RollbackEvent(
                    rollback_id=row[0],
                    model_name=row[1],
                    from_version=row[2],
                    to_version=row[3],
                    rollback_reason=row[4],
                    rollback_type=row[5],
                    performance_degradation=row[6],
                    drift_severity=row[7],
                    error_details=row[8] if row[8] else None,
                    rollback_triggered_by=row[9],
                    rollback_duration_seconds=row[10],
                    rollback_success=row[11],
                    rollback_metadata=row[12] if row[12] else {}
                )
                rollback_events.append(rollback_event)
            
            return rollback_events
            
        except Exception as e:
            logger.error(f"❌ Failed to get rollback history: {e}")
            return []
