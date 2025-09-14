#!/usr/bin/env python3
"""
Phase 1: Model Versioning & Rollback Manager
Handles model versioning, lineage tracking, and safe rollback capabilities
"""

import os
import sys
import json
import logging
import hashlib
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelLineage:
    """Model lineage information"""
    lineage_id: str
    model_name: str
    model_version: int
    parent_model_name: Optional[str]
    parent_model_version: Optional[int]
    training_data_hash: str
    feature_set_hash: str
    hyperparameters_hash: str
    training_environment: str
    git_commit_hash: Optional[str]
    docker_image_tag: Optional[str]
    training_duration_seconds: int
    training_samples: int
    validation_samples: int
    created_by: str
    lineage_metadata: Dict[str, Any]

@dataclass
class ModelVersion:
    """Model version information"""
    model_name: str
    version: int
    status: str  # staging, production, archived, failed, canary, rollback_candidate
    regime: str
    symbol: str
    model_artifact_path: str
    model_artifact_size_mb: float
    model_artifact_hash: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    performance_metrics: Optional[Dict[str, float]]
    drift_metrics: Optional[Dict[str, float]]
    rollback_metrics: Optional[Dict[str, float]]
    deployment_timestamp: datetime
    last_used_timestamp: Optional[datetime]
    usage_count: int
    error_count: int
    avg_inference_time_ms: Optional[float]
    total_inferences: int
    version_metadata: Dict[str, Any]
    created_by: str

@dataclass
class RollbackEvent:
    """Rollback event information"""
    rollback_id: str
    model_name: str
    from_version: int
    to_version: int
    rollback_reason: str
    rollback_type: str  # performance, drift, error, manual, automatic
    performance_degradation: Optional[float]
    drift_severity: Optional[str]
    error_details: Optional[Dict[str, Any]]
    rollback_triggered_by: str
    rollback_duration_seconds: int
    rollback_success: bool
    rollback_metadata: Dict[str, Any]

class ModelVersioningManager:
    """Advanced model versioning and rollback manager"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.model_cache: Dict[str, ModelVersion] = {}
        
    def _generate_hash(self, data: Any) -> str:
        """Generate hash for data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _generate_lineage_id(self, model_name: str, version: int) -> str:
        """Generate unique lineage ID"""
        return f"{model_name}_v{version}_lineage_{uuid.uuid4().hex[:8]}"
    
    def _generate_rollback_id(self, model_name: str, from_version: int, to_version: int) -> str:
        """Generate unique rollback ID"""
        return f"{model_name}_rollback_{from_version}_to_{to_version}_{uuid.uuid4().hex[:8]}"
    
    async def create_model_lineage(self, 
                                 model_name: str,
                                 model_version: int,
                                 parent_model_name: Optional[str] = None,
                                 parent_model_version: Optional[int] = None,
                                 training_data: pd.DataFrame = None,
                                 feature_set: Dict[str, Any] = None,
                                 hyperparameters: Dict[str, Any] = None,
                                 training_environment: str = "production",
                                 git_commit_hash: Optional[str] = None,
                                 docker_image_tag: Optional[str] = None,
                                 training_duration_seconds: int = 0,
                                 training_samples: int = 0,
                                 validation_samples: int = 0,
                                 created_by: str = "auto_retraining_system",
                                 lineage_metadata: Dict[str, Any] = None) -> str:
        """Create model lineage record"""
        
        logger.info(f"📝 Creating model lineage for {model_name} v{model_version}")
        
        try:
            # Generate hashes
            training_data_hash = self._generate_hash(training_data.to_dict() if training_data is not None else {})
            feature_set_hash = self._generate_hash(feature_set or {})
            hyperparameters_hash = self._generate_hash(hyperparameters or {})
            
            # Generate lineage ID
            lineage_id = self._generate_lineage_id(model_name, model_version)
            
            # Create lineage object
            lineage = ModelLineage(
                lineage_id=lineage_id,
                model_name=model_name,
                model_version=model_version,
                parent_model_name=parent_model_name,
                parent_model_version=parent_model_version,
                training_data_hash=training_data_hash,
                feature_set_hash=feature_set_hash,
                hyperparameters_hash=hyperparameters_hash,
                training_environment=training_environment,
                git_commit_hash=git_commit_hash,
                docker_image_tag=docker_image_tag,
                training_duration_seconds=training_duration_seconds,
                training_samples=training_samples,
                validation_samples=validation_samples,
                created_by=created_by,
                lineage_metadata=lineage_metadata or {}
            )
            
            # Store in database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_lineage (
                    lineage_id, model_name, model_version, parent_model_name, parent_model_version,
                    training_data_hash, feature_set_hash, hyperparameters_hash, training_environment,
                    git_commit_hash, docker_image_tag, training_duration_seconds, training_samples,
                    validation_samples, created_by, lineage_metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
            """, (
                lineage.lineage_id,
                lineage.model_name,
                lineage.model_version,
                lineage.parent_model_name,
                lineage.parent_model_version,
                lineage.training_data_hash,
                lineage.feature_set_hash,
                lineage.hyperparameters_hash,
                lineage.training_environment,
                lineage.git_commit_hash,
                lineage.docker_image_tag,
                lineage.training_duration_seconds,
                lineage.training_samples,
                lineage.validation_samples,
                lineage.created_by,
                json.dumps(lineage.lineage_metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Created model lineage: {lineage_id}")
            return lineage_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create model lineage: {e}")
            raise
    
    async def register_model_version(self,
                                   model_name: str,
                                   version: int,
                                   status: str,
                                   regime: str,
                                   symbol: str,
                                   model_artifact_path: str,
                                   model_artifact_size_mb: float,
                                   model_artifact_hash: str,
                                   training_metrics: Dict[str, float],
                                   validation_metrics: Dict[str, float],
                                   test_metrics: Optional[Dict[str, float]] = None,
                                   performance_metrics: Optional[Dict[str, float]] = None,
                                   drift_metrics: Optional[Dict[str, float]] = None,
                                   rollback_metrics: Optional[Dict[str, float]] = None,
                                   deployment_timestamp: Optional[datetime] = None,
                                   version_metadata: Optional[Dict[str, Any]] = None,
                                   created_by: str = "auto_retraining_system") -> ModelVersion:
        """Register a new model version"""
        
        logger.info(f"📝 Registering model version: {model_name} v{version} ({status})")
        
        try:
            if deployment_timestamp is None:
                deployment_timestamp = datetime.now()
            
            # Create model version object
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                status=status,
                regime=regime,
                symbol=symbol,
                model_artifact_path=model_artifact_path,
                model_artifact_size_mb=model_artifact_size_mb,
                model_artifact_hash=model_artifact_hash,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                performance_metrics=performance_metrics,
                drift_metrics=drift_metrics,
                rollback_metrics=rollback_metrics,
                deployment_timestamp=deployment_timestamp,
                last_used_timestamp=None,
                usage_count=0,
                error_count=0,
                avg_inference_time_ms=None,
                total_inferences=0,
                version_metadata=version_metadata or {},
                created_by=created_by
            )
            
            # Store in database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_versions (
                    model_name, version, status, regime, symbol, model_artifact_path,
                    model_artifact_size_mb, model_artifact_hash, training_metrics,
                    validation_metrics, test_metrics, performance_metrics, drift_metrics,
                    rollback_metrics, deployment_timestamp, version_metadata, created_by
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
                    %s::jsonb, %s::jsonb, %s, %s::jsonb, %s
                )
            """, (
                model_version.model_name,
                model_version.version,
                model_version.status,
                model_version.regime,
                model_version.symbol,
                model_version.model_artifact_path,
                model_version.model_artifact_size_mb,
                model_version.model_artifact_hash,
                json.dumps(model_version.training_metrics),
                json.dumps(model_version.validation_metrics),
                json.dumps(model_version.test_metrics) if model_version.test_metrics else None,
                json.dumps(model_version.performance_metrics) if model_version.performance_metrics else None,
                json.dumps(model_version.drift_metrics) if model_version.drift_metrics else None,
                json.dumps(model_version.rollback_metrics) if model_version.rollback_metrics else None,
                model_version.deployment_timestamp,
                json.dumps(model_version.version_metadata),
                model_version.created_by
            ))
            
            conn.commit()
            conn.close()
            
            # Cache the model version
            cache_key = f"{model_name}_{regime}_{symbol}"
            self.model_cache[cache_key] = model_version
            
            logger.info(f"✅ Registered model version: {model_name} v{version}")
            return model_version
            
        except Exception as e:
            logger.error(f"❌ Failed to register model version: {e}")
            raise
    
    async def get_production_model(self, model_name: str, regime: str, symbol: str) -> Optional[ModelVersion]:
        """Get the current production model for a specific configuration"""
        
        try:
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
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'production'
                ORDER BY version DESC
                LIMIT 1
            """, (model_name, regime, symbol))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
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
                return model_version
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get production model: {e}")
            return None
    
    async def get_model_lineage(self, model_name: str, version: int) -> Optional[ModelLineage]:
        """Get model lineage information"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT lineage_id, model_name, model_version, parent_model_name, parent_model_version,
                       training_data_hash, feature_set_hash, hyperparameters_hash, training_environment,
                       git_commit_hash, docker_image_tag, training_duration_seconds, training_samples,
                       validation_samples, created_by, lineage_metadata
                FROM model_lineage
                WHERE model_name = %s AND model_version = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (model_name, version))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                lineage = ModelLineage(
                    lineage_id=row[0],
                    model_name=row[1],
                    model_version=row[2],
                    parent_model_name=row[3],
                    parent_model_version=row[4],
                    training_data_hash=row[5],
                    feature_set_hash=row[6],
                    hyperparameters_hash=row[7],
                    training_environment=row[8],
                    git_commit_hash=row[9],
                    docker_image_tag=row[10],
                    training_duration_seconds=row[11],
                    training_samples=row[12],
                    validation_samples=row[13],
                    created_by=row[14],
                    lineage_metadata=row[15] if row[15] else {}
                )
                return lineage
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get model lineage: {e}")
            return None
    
    async def update_model_usage(self, model_name: str, version: int, inference_time_ms: float = None):
        """Update model usage statistics"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Update usage count and last used timestamp
            cursor.execute("""
                UPDATE model_versions
                SET usage_count = usage_count + 1,
                    last_used_timestamp = NOW(),
                    total_inferences = total_inferences + 1
                WHERE model_name = %s AND version = %s
            """, (model_name, version))
            
            # Update average inference time if provided
            if inference_time_ms is not None:
                cursor.execute("""
                    UPDATE model_versions
                    SET avg_inference_time_ms = CASE
                        WHEN avg_inference_time_ms IS NULL THEN %s
                        ELSE (avg_inference_time_ms * total_inferences + %s) / (total_inferences + 1)
                    END
                    WHERE model_name = %s AND version = %s
                """, (inference_time_ms, inference_time_ms, model_name, version))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update model usage: {e}")
    
    async def update_model_error(self, model_name: str, version: int, error_details: Dict[str, Any] = None):
        """Update model error statistics"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_versions
                SET error_count = error_count + 1
                WHERE model_name = %s AND version = %s
            """, (model_name, version))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update model error: {e}")
    
    async def mark_model_for_rollback(self, model_name: str, version: int, reason: str):
        """Mark a model as rollback candidate"""
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_versions
                SET status = 'rollback_candidate',
                    rollback_metrics = jsonb_set(
                        COALESCE(rollback_metrics, '{}'::jsonb),
                        '{rollback_reason}',
                        %s::jsonb
                    )
                WHERE model_name = %s AND version = %s
            """, (json.dumps(reason), model_name, version))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Marked model {model_name} v{version} for rollback: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to mark model for rollback: {e}")
    
    async def get_rollback_candidates(self, model_name: str, regime: str, symbol: str) -> List[ModelVersion]:
        """Get models marked for rollback"""
        
        try:
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
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'rollback_candidate'
                ORDER BY version DESC
            """, (model_name, regime, symbol))
            
            rows = cursor.fetchall()
            conn.close()
            
            rollback_candidates = []
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
                rollback_candidates.append(model_version)
            
            return rollback_candidates
            
        except Exception as e:
            logger.error(f"❌ Failed to get rollback candidates: {e}")
            return []
    
    async def get_previous_production_model(self, model_name: str, regime: str, symbol: str) -> Optional[ModelVersion]:
        """Get the previous production model for rollback"""
        
        try:
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
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'archived'
                ORDER BY version DESC
                LIMIT 1
            """, (model_name, regime, symbol))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
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
                return model_version
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get previous production model: {e}")
            return None
