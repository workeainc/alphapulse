#!/usr/bin/env python3
"""
Phase 5C: Feature Store Core Module
Implements:
1. Feature materialization and retrieval
2. Schema validation and contracts
3. Time-travel capabilities
4. Drift detection and monitoring
5. Integration with Phase 5B ensemble manager
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from uuid import UUID, uuid4

# Local imports
from ...database.connection import TimescaleDBConnection
from ..advanced_logging_system import redis_logger, EventType, LogLevel
from ..model_registry import ModelRegistry
from sqlalchemy import text

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"

class DriftType(Enum):
    """Types of feature drift"""
    SCHEMA = "schema"
    DISTRIBUTION = "distribution"
    MISSING = "missing"
    RANGE = "range"

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    id: UUID
    name: str
    version: str
    description: str
    schema: Dict[str, Any]
    data_type: FeatureType
    source_table: str
    computation_logic: str
    owner: str
    tags: List[str]
    validation_rules: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class FeatureSnapshot:
    """Feature snapshot for time-travel"""
    id: UUID
    feature_definition_id: UUID
    snapshot_timestamp: datetime
    data_hash: str
    feature_values: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class FeatureContract:
    """Feature contract for validation"""
    id: UUID
    name: str
    version: str
    description: str
    schema_contract: Dict[str, Any]
    validation_rules: Dict[str, Any]
    drift_thresholds: Dict[str, float]
    owner: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class DriftDetectionResult:
    """Result of drift detection"""
    feature_definition_id: UUID
    drift_type: DriftType
    drift_score: float
    baseline_snapshot_id: Optional[UUID]
    current_snapshot_id: Optional[UUID]
    drift_details: Dict[str, Any]
    is_drift_detected: bool
    threshold: float

class FeatureStore:
    """Phase 5C: Feature Store with time-travel and drift detection"""
    
    def __init__(self, db_connection: TimescaleDBConnection = None):
        self.db_connection = db_connection or TimescaleDBConnection()
        self.logger = logger
        self.model_registry = ModelRegistry()
        
        # Cache for performance
        self._feature_definitions_cache = {}
        self._contracts_cache = {}
        self._cache_ttl = timedelta(minutes=30)
        self._last_cache_update = None
    
    async def get_feature_definition(self, name: str, version: str = None) -> Optional[FeatureDefinition]:
        """Get feature definition by name and version"""
        try:
            cache_key = f"{name}:{version or 'latest'}"
            
            # Check cache first
            if self._is_cache_valid() and cache_key in self._feature_definitions_cache:
                return self._feature_definitions_cache[cache_key]
            
            # Query database
            query = """
                SELECT id, name, version, description, schema, data_type, source_table,
                       computation_logic, owner, tags, validation_rules, is_active,
                       created_at, updated_at
                FROM feature_definitions
                WHERE name = :name AND is_active = true
            """
            params = {'name': name}
            
            if version:
                query += " AND version = :version"
                params['version'] = version
            else:
                query += " ORDER BY version DESC LIMIT 1"
            
            async with self.db_connection.async_session() as conn:
                result = await conn.execute(text(query), **params)
                row = result.fetchone()
                if row:
                    result = dict(row._mapping)
                else:
                    result = None
                
                if result:
                    feature_def = FeatureDefinition(
                        id=result['id'],
                        name=result['name'],
                        version=result['version'],
                        description=result['description'],
                        schema=result['schema'],
                        data_type=FeatureType(result['data_type']),
                        source_table=result['source_table'],
                        computation_logic=result['computation_logic'],
                        owner=result['owner'],
                        tags=result['tags'],
                        validation_rules=result['validation_rules'],
                        is_active=result['is_active'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at']
                    )
                    
                    # Update cache
                    self._feature_definitions_cache[cache_key] = feature_def
                    return feature_def
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature definition {name}: {e}")
            return None
    
    async def get_feature_contract(self, name: str) -> Optional[FeatureContract]:
        """Get feature contract by name"""
        try:
            # Check cache first
            if self._is_cache_valid() and name in self._contracts_cache:
                return self._contracts_cache[name]
            
            query = """
                SELECT id, name, version, description, schema_contract, validation_rules,
                       drift_thresholds, owner, is_active, created_at, updated_at
                FROM feature_contracts
                WHERE name = :name AND is_active = true
                ORDER BY version DESC LIMIT 1
            """
            
            async with self.db_connection.async_session() as conn:
                result = await conn.execute(text(query), name=name)
                row = result.fetchone()
                if row:
                    result = dict(row._mapping)
                else:
                    result = None
                
                if result:
                    contract = FeatureContract(
                        id=result['id'],
                        name=result['name'],
                        version=result['version'],
                        description=result['description'],
                        schema_contract=result['schema_contract'],
                        validation_rules=result['validation_rules'],
                        drift_thresholds=result['drift_thresholds'],
                        owner=result['owner'],
                        is_active=result['is_active'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at']
                    )
                    
                    # Update cache
                    self._contracts_cache[name] = contract
                    return contract
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature contract {name}: {e}")
            return None
    
    async def materialize_features(self, feature_names: List[str], 
                                 timestamp: datetime = None) -> Dict[str, Any]:
        """Materialize features for given timestamp (time-travel)"""
        try:
            timestamp = timestamp or datetime.now()
            materialized_features = {}
            
            for feature_name in feature_names:
                # Get feature definition
                feature_def = await self.get_feature_definition(feature_name)
                if not feature_def:
                    self.logger.warning(f"Feature definition not found: {feature_name}")
                    continue
                
                # Get feature snapshot for timestamp
                snapshot = await self._get_feature_snapshot(feature_def.id, timestamp)
                if snapshot:
                    materialized_features[feature_name] = snapshot.feature_values
                else:
                    # Compute feature if no snapshot exists
                    computed_features = await self._compute_feature(feature_def, timestamp)
                    if computed_features:
                        materialized_features[feature_name] = computed_features
                        
                        # Create snapshot for future use
                        await self._create_feature_snapshot(feature_def.id, timestamp, computed_features)
            
            return materialized_features
            
        except Exception as e:
            self.logger.error(f"Error materializing features: {e}")
            return {}
    
    async def validate_features(self, features: Dict[str, Any], 
                              contract_name: str) -> Tuple[bool, List[str]]:
        """Validate features against contract"""
        try:
            contract = await self.get_feature_contract(contract_name)
            if not contract:
                return False, [f"Contract {contract_name} not found"]
            
            errors = []
            schema_contract = contract.schema_contract
            
            # Check required features
            required_features = schema_contract.get('required_features', [])
            for feature_name in required_features:
                if feature_name not in features:
                    errors.append(f"Required feature missing: {feature_name}")
                elif features[feature_name] is None:
                    errors.append(f"Required feature is null: {feature_name}")
            
            # Check feature types and validation rules
            feature_types = schema_contract.get('feature_types', {})
            validation_rules = schema_contract.get('validation_rules', {})
            
            for feature_name, feature_value in features.items():
                if feature_name in feature_types:
                    expected_type = feature_types[feature_name]
                    if not self._validate_feature_type(feature_value, expected_type):
                        errors.append(f"Feature {feature_name} has wrong type. Expected: {expected_type}")
                
                if feature_name in validation_rules:
                    rule_errors = self._validate_feature_rules(feature_value, validation_rules[feature_name])
                    errors.extend([f"Feature {feature_name}: {error}" for error in rule_errors])
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Error validating features: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def detect_drift(self, feature_name: str, 
                          baseline_timestamp: datetime = None) -> Optional[DriftDetectionResult]:
        """Detect drift for a feature"""
        try:
            feature_def = await self.get_feature_definition(feature_name)
            if not feature_def:
                return None
            
            # Get baseline and current snapshots
            baseline_timestamp = baseline_timestamp or (datetime.now() - timedelta(days=7))
            baseline_snapshot = await self._get_feature_snapshot(feature_def.id, baseline_timestamp)
            current_snapshot = await self._get_latest_feature_snapshot(feature_def.id)
            
            if not baseline_snapshot or not current_snapshot:
                return None
            
            # Calculate drift scores
            drift_results = []
            
            # Distribution drift
            dist_drift = self._calculate_distribution_drift(
                baseline_snapshot.feature_values, 
                current_snapshot.feature_values
            )
            drift_results.append(DriftDetectionResult(
                feature_definition_id=feature_def.id,
                drift_type=DriftType.DISTRIBUTION,
                drift_score=dist_drift,
                baseline_snapshot_id=baseline_snapshot.id,
                current_snapshot_id=current_snapshot.id,
                drift_details={'distribution_drift': dist_drift},
                is_drift_detected=dist_drift > 0.1,  # Default threshold
                threshold=0.1
            ))
            
            # Schema drift
            schema_drift = self._calculate_schema_drift(
                baseline_snapshot.feature_values,
                current_snapshot.feature_values
            )
            drift_results.append(DriftDetectionResult(
                feature_definition_id=feature_def.id,
                drift_type=DriftType.SCHEMA,
                drift_score=schema_drift,
                baseline_snapshot_id=baseline_snapshot.id,
                current_snapshot_id=current_snapshot.id,
                drift_details={'schema_drift': schema_drift},
                is_drift_detected=schema_drift > 0.05,  # Default threshold
                threshold=0.05
            ))
            
            # Log drift detection results
            for drift_result in drift_results:
                if drift_result.is_drift_detected:
                    await self._log_drift_detection(drift_result)
            
            return drift_results[0]  # Return first drift result for now
            
        except Exception as e:
            self.logger.error(f"Error detecting drift for {feature_name}: {e}")
            return None
    
    async def get_features_for_ensemble(self, timestamp: datetime = None) -> pd.DataFrame:
        """Get features for Phase 5B ensemble models"""
        try:
            # Get Phase 5B feature contract
            contract = await self.get_feature_contract('phase5b_ensemble_features')
            if not contract:
                self.logger.error("Phase 5B ensemble feature contract not found")
                return pd.DataFrame()
            
            # Get required features
            required_features = contract.schema_contract.get('required_features', [])
            
            # Materialize features
            materialized_features = await self.materialize_features(required_features, timestamp)
            
            # Validate features
            is_valid, errors = await self.validate_features(materialized_features, 'phase5b_ensemble_features')
            if not is_valid:
                self.logger.error(f"Feature validation failed: {errors}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([materialized_features])
            
            # Add required columns for regime detection
            if 'close' not in df.columns and 'close_price' in df.columns:
                df['close'] = df['close_price']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting features for ensemble: {e}")
            return pd.DataFrame()
    
    async def _get_feature_snapshot(self, feature_definition_id: UUID, 
                                   timestamp: datetime) -> Optional[FeatureSnapshot]:
        """Get feature snapshot for specific timestamp"""
        try:
            query = """
                SELECT id, feature_definition_id, snapshot_timestamp, data_hash,
                       feature_values, metadata, created_at
                FROM feature_snapshots
                WHERE feature_definition_id = :feature_definition_id
                  AND snapshot_timestamp <= :timestamp
                ORDER BY snapshot_timestamp DESC
                LIMIT 1
            """
            
            async with self.db_connection.get_connection() as conn:
                result = await conn.fetchrow(query, 
                                           feature_definition_id=feature_definition_id,
                                           timestamp=timestamp)
                
                if result:
                    return FeatureSnapshot(
                        id=result['id'],
                        feature_definition_id=result['feature_definition_id'],
                        snapshot_timestamp=result['snapshot_timestamp'],
                        data_hash=result['data_hash'],
                        feature_values=result['feature_values'],
                        metadata=result['metadata'],
                        created_at=result['created_at']
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting feature snapshot: {e}")
            return None
    
    async def _get_latest_feature_snapshot(self, feature_definition_id: UUID) -> Optional[FeatureSnapshot]:
        """Get latest feature snapshot"""
        try:
            query = """
                SELECT id, feature_definition_id, snapshot_timestamp, data_hash,
                       feature_values, metadata, created_at
                FROM feature_snapshots
                WHERE feature_definition_id = :feature_definition_id
                ORDER BY snapshot_timestamp DESC
                LIMIT 1
            """
            
            async with self.db_connection.get_connection() as conn:
                result = await conn.fetchrow(query, feature_definition_id=feature_definition_id)
                
                if result:
                    return FeatureSnapshot(
                        id=result['id'],
                        feature_definition_id=result['feature_definition_id'],
                        snapshot_timestamp=result['snapshot_timestamp'],
                        data_hash=result['data_hash'],
                        feature_values=result['feature_values'],
                        metadata=result['metadata'],
                        created_at=result['created_at']
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest feature snapshot: {e}")
            return None
    
    async def _compute_feature(self, feature_def: FeatureDefinition, 
                             timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Compute feature using computation logic"""
        try:
            # For now, implement basic feature computation
            # In production, this would execute the computation_logic SQL or Python code
            
            if feature_def.name == 'close_price':
                # Get latest close price
                query = """
                    SELECT close FROM candlestick_patterns
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            elif feature_def.name == 'volume':
                query = """
                    SELECT volume FROM candlestick_patterns
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            elif feature_def.name == 'btc_dominance':
                query = """
                    SELECT btc_dominance FROM market_intelligence
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            elif feature_def.name == 'market_correlation':
                query = """
                    SELECT market_correlation FROM market_intelligence
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            elif feature_def.name == 'volume_ratio':
                query = """
                    SELECT volume / AVG(volume) OVER (
                        ORDER BY timestamp ROWS 20 PRECEDING
                    ) as volume_ratio
                    FROM candlestick_patterns
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            elif feature_def.name == 'atr_percentage':
                query = """
                    SELECT (atr / close * 100) as atr_percentage
                    FROM candlestick_patterns
                    WHERE timestamp <= :timestamp
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
            else:
                self.logger.warning(f"Unknown feature computation: {feature_def.name}")
                return None
            
            async with self.db_connection.get_connection() as conn:
                result = await conn.fetchrow(query, timestamp=timestamp)
                
                if result:
                    return dict(result)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error computing feature {feature_def.name}: {e}")
            return None
    
    async def _create_feature_snapshot(self, feature_definition_id: UUID, 
                                     timestamp: datetime, feature_values: Dict[str, Any]):
        """Create feature snapshot"""
        try:
            # Calculate data hash
            data_str = json.dumps(feature_values, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            query = """
                INSERT INTO feature_snapshots 
                (id, feature_definition_id, snapshot_timestamp, data_hash, feature_values, metadata)
                VALUES (:id, :feature_definition_id, :snapshot_timestamp, :data_hash, :feature_values, :metadata)
            """
            
            async with self.db_connection.get_connection() as conn:
                await conn.execute(query, 
                                 id=uuid4(),
                                 feature_definition_id=feature_definition_id,
                                 snapshot_timestamp=timestamp,
                                 data_hash=data_hash,
                                 feature_values=feature_values,
                                 metadata={'computed_at': datetime.now().isoformat()})
            
        except Exception as e:
            self.logger.error(f"Error creating feature snapshot: {e}")
    
    async def _log_drift_detection(self, drift_result: DriftDetectionResult):
        """Log drift detection result"""
        try:
            query = """
                INSERT INTO feature_drift_logs 
                (id, feature_definition_id, detection_timestamp, drift_type, drift_score,
                 baseline_snapshot_id, current_snapshot_id, drift_details, is_resolved)
                VALUES (:id, :feature_definition_id, :detection_timestamp, :drift_type, :drift_score,
                        :baseline_snapshot_id, :current_snapshot_id, :drift_details, :is_resolved)
            """
            
            async with self.db_connection.get_connection() as conn:
                await conn.execute(query,
                                 id=uuid4(),
                                 feature_definition_id=drift_result.feature_definition_id,
                                 detection_timestamp=datetime.now(),
                                 drift_type=drift_result.drift_type.value,
                                 drift_score=drift_result.drift_score,
                                 baseline_snapshot_id=drift_result.baseline_snapshot_id,
                                 current_snapshot_id=drift_result.current_snapshot_id,
                                 drift_details=drift_result.drift_details,
                                 is_resolved=False)
            
            # Log to advanced logging system
            await redis_logger.log_event(
                EventType.FEATURE_DRIFT,
                LogLevel.WARNING,
                f"Feature drift detected: {drift_result.drift_type.value} score {drift_result.drift_score:.3f}",
                {'drift_result': drift_result.__dict__}
            )
            
        except Exception as e:
            self.logger.error(f"Error logging drift detection: {e}")
    
    def _validate_feature_type(self, value: Any, expected_type: str) -> bool:
        """Validate feature type"""
        try:
            if expected_type == 'numeric':
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected_type == 'categorical':
                return isinstance(value, str)
            elif expected_type == 'datetime':
                return isinstance(value, (datetime, str))  # Allow string for datetime
            elif expected_type == 'boolean':
                return isinstance(value, bool)
            return True
        except:
            return False
    
    def _validate_feature_rules(self, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate feature against rules"""
        errors = []
        
        try:
            if rules.get('not_null') and value is None:
                errors.append("Value cannot be null")
            
            if rules.get('positive') and isinstance(value, (int, float)) and value <= 0:
                errors.append("Value must be positive")
            
            if 'range' in rules and isinstance(value, (int, float)):
                min_val, max_val = rules['range']
                if value < min_val or value > max_val:
                    errors.append(f"Value must be between {min_val} and {max_val}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def _calculate_distribution_drift(self, baseline_values: Dict[str, Any], 
                                    current_values: Dict[str, Any]) -> float:
        """Calculate distribution drift between baseline and current values"""
        try:
            # Simple drift calculation - can be enhanced with statistical tests
            baseline_vals = list(baseline_values.values())
            current_vals = list(current_values.values())
            
            if not baseline_vals or not current_vals:
                return 0.0
            
            baseline_mean = np.mean(baseline_vals)
            current_mean = np.mean(current_vals)
            
            baseline_std = np.std(baseline_vals)
            current_std = np.std(current_vals)
            
            # Calculate drift as normalized difference
            mean_drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
            std_drift = abs(current_std - baseline_std) / (baseline_std + 1e-8)
            
            return (mean_drift + std_drift) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating distribution drift: {e}")
            return 0.0
    
    def _calculate_schema_drift(self, baseline_values: Dict[str, Any], 
                              current_values: Dict[str, Any]) -> float:
        """Calculate schema drift between baseline and current values"""
        try:
            baseline_keys = set(baseline_values.keys())
            current_keys = set(current_values.keys())
            
            # Calculate Jaccard distance
            intersection = len(baseline_keys.intersection(current_keys))
            union = len(baseline_keys.union(current_keys))
            
            if union == 0:
                return 0.0
            
            return 1 - (intersection / union)
            
        except Exception as e:
            self.logger.error(f"Error calculating schema drift: {e}")
            return 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._last_cache_update:
            return False
        return datetime.now() - self._last_cache_update < self._cache_ttl
    
    async def refresh_cache(self):
        """Refresh feature definitions and contracts cache"""
        try:
            self._feature_definitions_cache.clear()
            self._contracts_cache.clear()
            self._last_cache_update = datetime.now()
            self.logger.info("Feature store cache refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {e}")

# Global instance
feature_store = FeatureStore()
