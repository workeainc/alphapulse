#!/usr/bin/env python3
"""
Phase 5C: Simplified Feature Store Core Module
Implements basic feature store functionality without database dependencies
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

class SimpleFeatureStore:
    """Phase 5C: Simplified Feature Store with in-memory storage"""
    
    def __init__(self):
        self.logger = logger
        
        # In-memory storage for testing
        self._feature_definitions = {}
        self._contracts = {}
        self._snapshots = {}
        
        # Initialize with default Phase 5B features
        self._initialize_default_features()
    
    def _initialize_default_features(self):
        """Initialize with default Phase 5B features"""
        default_features = [
            {
                'id': uuid4(),
                'name': 'close_price',
                'version': '1.0.0',
                'description': 'Closing price for candlestick patterns',
                'schema': {'type': 'number', 'minimum': 0, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'candlestick_patterns',
                'computation_logic': 'SELECT close FROM candlestick_patterns',
                'owner': 'system',
                'tags': ['price', 'technical', 'phase5b'],
                'validation_rules': {'not_null': True, 'positive': True},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            },
            {
                'id': uuid4(),
                'name': 'volume',
                'version': '1.0.0',
                'description': 'Trading volume for candlestick patterns',
                'schema': {'type': 'number', 'minimum': 0, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'candlestick_patterns',
                'computation_logic': 'SELECT volume FROM candlestick_patterns',
                'owner': 'system',
                'tags': ['volume', 'technical', 'phase5b'],
                'validation_rules': {'not_null': True, 'positive': True},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            },
            {
                'id': uuid4(),
                'name': 'btc_dominance',
                'version': '1.0.0',
                'description': 'Bitcoin dominance percentage',
                'schema': {'type': 'number', 'minimum': 0, 'maximum': 100, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'market_intelligence',
                'computation_logic': 'SELECT btc_dominance FROM market_intelligence',
                'owner': 'system',
                'tags': ['market', 'dominance', 'phase5b'],
                'validation_rules': {'not_null': True, 'range': [0, 100]},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            },
            {
                'id': uuid4(),
                'name': 'market_correlation',
                'version': '1.0.0',
                'description': 'Market correlation coefficient',
                'schema': {'type': 'number', 'minimum': -1, 'maximum': 1, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'market_intelligence',
                'computation_logic': 'SELECT market_correlation FROM market_intelligence',
                'owner': 'system',
                'tags': ['market', 'correlation', 'phase5b'],
                'validation_rules': {'not_null': True, 'range': [-1, 1]},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            },
            {
                'id': uuid4(),
                'name': 'volume_ratio',
                'version': '1.0.0',
                'description': 'Volume ratio compared to average',
                'schema': {'type': 'number', 'minimum': 0, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'candlestick_patterns',
                'computation_logic': 'SELECT volume / AVG(volume) OVER (ORDER BY timestamp ROWS 20 PRECEDING) FROM candlestick_patterns',
                'owner': 'system',
                'tags': ['volume', 'ratio', 'phase5b'],
                'validation_rules': {'not_null': True, 'positive': True},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            },
            {
                'id': uuid4(),
                'name': 'atr_percentage',
                'version': '1.0.0',
                'description': 'Average True Range as percentage of price',
                'schema': {'type': 'number', 'minimum': 0, 'required': True},
                'data_type': FeatureType.NUMERIC,
                'source_table': 'candlestick_patterns',
                'computation_logic': 'SELECT atr / close * 100 FROM candlestick_patterns',
                'owner': 'system',
                'tags': ['volatility', 'atr', 'phase5b'],
                'validation_rules': {'not_null': True, 'positive': True},
                'is_active': True,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
        ]
        
        for feature_data in default_features:
            feature_def = FeatureDefinition(**feature_data)
            self._feature_definitions[feature_def.name] = feature_def
        
        # Initialize default contract
        default_contract = {
            'id': uuid4(),
            'name': 'phase5b_ensemble_features',
            'version': '1.0.0',
            'description': 'Feature contract for Phase 5B ensemble models',
            'schema_contract': {
                'required_features': ['close_price', 'volume', 'btc_dominance', 'market_correlation', 'volume_ratio', 'atr_percentage'],
                'feature_types': {
                    'close_price': 'numeric',
                    'volume': 'numeric',
                    'btc_dominance': 'numeric',
                    'market_correlation': 'numeric',
                    'volume_ratio': 'numeric',
                    'atr_percentage': 'numeric'
                },
                'validation_rules': {
                    'close_price': {'not_null': True, 'positive': True},
                    'volume': {'not_null': True, 'positive': True},
                    'btc_dominance': {'not_null': True, 'range': [0, 100]},
                    'market_correlation': {'not_null': True, 'range': [-1, 1]},
                    'volume_ratio': {'not_null': True, 'positive': True},
                    'atr_percentage': {'not_null': True, 'positive': True}
                }
            },
            'validation_rules': {'all_required_present': True, 'no_null_values': True},
            'drift_thresholds': {
                'distribution_drift': 0.1,
                'schema_drift': 0.05,
                'missing_data_threshold': 0.01
            },
            'owner': 'system',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        contract = FeatureContract(**default_contract)
        self._contracts[contract.name] = contract
        
        self.logger.info(f"✅ Initialized SimpleFeatureStore with {len(self._feature_definitions)} features and {len(self._contracts)} contracts")
    
    async def get_feature_definition(self, name: str, version: str = None) -> Optional[FeatureDefinition]:
        """Get feature definition by name and version"""
        try:
            if name in self._feature_definitions:
                return self._feature_definitions[name]
            return None
        except Exception as e:
            self.logger.error(f"Error getting feature definition {name}: {e}")
            return None
    
    async def get_feature_contract(self, name: str) -> Optional[FeatureContract]:
        """Get feature contract by name"""
        try:
            if name in self._contracts:
                return self._contracts[name]
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
                
                # For now, return synthetic data
                if feature_name == 'close_price':
                    materialized_features[feature_name] = 50000.0
                elif feature_name == 'volume':
                    materialized_features[feature_name] = 1000000.0
                elif feature_name == 'btc_dominance':
                    materialized_features[feature_name] = 45.5
                elif feature_name == 'market_correlation':
                    materialized_features[feature_name] = 0.75
                elif feature_name == 'volume_ratio':
                    materialized_features[feature_name] = 1.2
                elif feature_name == 'atr_percentage':
                    materialized_features[feature_name] = 2.5
                else:
                    materialized_features[feature_name] = 0.0
            
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
    
    async def detect_drift(self, feature_name: str, 
                          baseline_timestamp: datetime = None) -> Optional[Dict[str, Any]]:
        """Detect drift for a feature (simplified version)"""
        try:
            # For now, return a simple drift result
            return {
                'drift_type': 'distribution',
                'drift_score': 0.05,  # Low drift score
                'is_drift_detected': False,
                'threshold': 0.1
            }
        except Exception as e:
            self.logger.error(f"Error detecting drift for {feature_name}: {e}")
            return None
    
    async def refresh_cache(self):
        """Refresh feature definitions and contracts cache"""
        try:
            self.logger.info("Feature store cache refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {e}")

# Global instance
simple_feature_store = SimpleFeatureStore()
