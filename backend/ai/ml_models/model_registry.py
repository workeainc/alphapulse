"""
Advanced ML Model Registry for AlphaPlus
Manages model lifecycle, versioning, and deployment
"""

import logging
import json
import os
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import joblib
import numpy as np
import pandas as pd

# Import our enhanced components
try:
    from ...database.connection import TimescaleDBConnection
    from ...data.storage import DataStorage
except ImportError:
    # Fallback for testing
    TimescaleDBConnection = None
    DataStorage = None

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

class ModelType(Enum):
    """Model type enumeration"""
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    """Model metadata information"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    description: str
    author: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    data_schema: Dict[str, Any]
    dependencies: List[str]

class ModelRegistry:
    """Advanced ML model registry with versioning and lifecycle management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Registry configuration
        self.registry_path = self.config.get('registry_path', 'models/registry')
        self.model_storage_path = self.config.get('model_storage_path', 'models/storage')
        self.max_versions = self.config.get('max_versions', 10)
        self.auto_cleanup = self.config.get('auto_cleanup', True)
        
        # Component references
        self.db_connection = None
        self.storage = None
        
        # Registry state
        self.models: Dict[str, ModelMetadata] = {}
        self.active_models: Dict[str, str] = {}  # model_name -> active_version
        
        # Performance tracking
        self.stats = {
            'total_models': 0,
            'active_models': 0,
            'training_models': 0,
            'failed_models': 0
        }
        
        # Ensure directories exist
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.model_storage_path, exist_ok=True)
        
    async def initialize(self):
        """Initialize the model registry"""
        try:
            self.logger.info("Initializing Advanced Model Registry...")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Initialize storage if available
            if DataStorage:
                self.storage = DataStorage(
                    storage_path=self.registry_path,
                    db_config=self.config.get('db_config', {})
                )
                await self.storage.initialize()
            
            # Load existing registry
            await self._load_registry()
            
            # Update statistics
            self._update_stats()
            
            self.logger.info("Advanced Model Registry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Model Registry: {e}")
            raise
    
    async def _load_registry(self):
        """Load existing model registry from storage"""
        try:
            registry_file = os.path.join(self.registry_path, 'registry.json')
            
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                # Load models
                for model_data in registry_data.get('models', []):
                    model_id = model_data['model_id']
                    metadata = ModelMetadata(
                        model_id=model_id,
                        name=model_data['name'],
                        version=model_data['version'],
                        model_type=ModelType(model_data['model_type']),
                        status=ModelStatus(model_data['status']),
                        created_at=datetime.fromisoformat(model_data['created_at']),
                        updated_at=datetime.fromisoformat(model_data['updated_at']),
                        description=model_data.get('description', ''),
                        author=model_data.get('author', ''),
                        tags=model_data.get('tags', []),
                        hyperparameters=model_data.get('hyperparameters', {}),
                        training_metrics=model_data.get('training_metrics', {}),
                        validation_metrics=model_data.get('validation_metrics', {}),
                        feature_importance=model_data.get('feature_importance', {}),
                        data_schema=model_data.get('data_schema', {}),
                        dependencies=model_data.get('dependencies', [])
                    )
                    
                    self.models[model_id] = metadata
                
                # Load active models
                self.active_models = registry_data.get('active_models', {})
                
                self.logger.info(f"Loaded {len(self.models)} models from registry")
            
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
    
    async def _save_registry(self):
        """Save current registry to storage"""
        try:
            registry_data = {
                'models': [asdict(metadata) for metadata in self.models.values()],
                'active_models': self.active_models,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            registry_file = os.path.join(self.registry_path, 'registry.json')
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            self.logger.info("Registry saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    async def register_model(self, name: str, model_type: ModelType, 
                           description: str = "", author: str = "", 
                           tags: List[str] = None, hyperparameters: Dict[str, Any] = None) -> str:
        """Register a new model in the registry"""
        try:
            # Generate unique model ID
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            model_id = f"{name}_{model_type.value}_{timestamp}"
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version="1.0.0",
                model_type=model_type,
                status=ModelStatus.TRAINING,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                description=description,
                author=author,
                tags=tags or [],
                hyperparameters=hyperparameters or {},
                training_metrics={},
                validation_metrics={},
                feature_importance={},
                data_schema={},
                dependencies=[]
            )
            
            # Add to registry
            self.models[model_id] = metadata
            
            # Save registry
            await self._save_registry()
            
            # Update statistics
            self._update_stats()
            
            self.logger.info(f"Registered new model: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    async def save_model(self, model_id: str, model_object: Any, 
                        training_metrics: Dict[str, float] = None,
                        validation_metrics: Dict[str, float] = None,
                        feature_importance: Dict[str, float] = None,
                        data_schema: Dict[str, Any] = None) -> bool:
        """Save a trained model to storage"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.models[model_id]
            
            # Update metadata
            metadata.status = ModelStatus.TRAINED
            metadata.updated_at = datetime.now(timezone.utc)
            
            if training_metrics:
                metadata.training_metrics = training_metrics
            
            if validation_metrics:
                metadata.validation_metrics = validation_metrics
            
            if feature_importance:
                metadata.feature_importance = feature_importance
            
            if data_schema:
                metadata.data_schema = data_schema
            
            # Save model file
            model_file = os.path.join(self.model_storage_path, f"{model_id}.joblib")
            
            # Use joblib for better compatibility
            joblib.dump(model_object, model_file)
            
            # Save metadata
            await self._save_registry()
            
            self.logger.info(f"Model saved successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    async def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model from storage"""
        try:
            if model_id not in self.models:
                self.logger.error(f"Model {model_id} not found in registry")
                return None
            
            metadata = self.models[model_id]
            
            # Check if model is trained
            if metadata.status not in [ModelStatus.TRAINED, ModelStatus.VALIDATED, ModelStatus.DEPLOYED]:
                self.logger.error(f"Model {model_id} is not trained (status: {metadata.status})")
                return None
            
            # Load model file
            model_file = os.path.join(self.model_storage_path, f"{model_id}.joblib")
            
            if not os.path.exists(model_file):
                self.logger.error(f"Model file not found: {model_file}")
                return None
            
            # Load model
            model = joblib.load(model_file)
            
            self.logger.info(f"Model loaded successfully: {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    async def deploy_model(self, model_id: str) -> bool:
        """Deploy a model as the active version"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.models[model_id]
            
            # Check if model is ready for deployment
            if metadata.status not in [ModelStatus.TRAINED, ModelStatus.VALIDATED]:
                raise ValueError(f"Model {model_id} is not ready for deployment (status: {metadata.status})")
            
            # Update model status
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.now(timezone.utc)
            
            # Set as active model
            self.active_models[metadata.name] = model_id
            
            # Save registry
            await self._save_registry()
            
            # Update statistics
            self._update_stats()
            
            self.logger.info(f"Model deployed successfully: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return False
    
    async def get_active_model(self, model_name: str) -> Optional[str]:
        """Get the active version of a model"""
        try:
            return self.active_models.get(model_name)
            
        except Exception as e:
            self.logger.error(f"Error getting active model: {e}")
            return None
    
    async def list_models(self, name: str = None, model_type: ModelType = None, 
                         status: ModelStatus = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        try:
            filtered_models = []
            
            for metadata in self.models.values():
                # Apply filters
                if name and metadata.name != name:
                    continue
                
                if model_type and metadata.model_type != model_type:
                    continue
                
                if status and metadata.status != status:
                    continue
                
                filtered_models.append(metadata)
            
            # Sort by creation date (newest first)
            filtered_models.sort(key=lambda x: x.created_at, reverse=True)
            
            return filtered_models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    async def update_model_metadata(self, model_id: str, 
                                  updates: Dict[str, Any]) -> bool:
        """Update model metadata"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.models[model_id]
            
            # Update allowed fields
            allowed_fields = ['description', 'tags', 'hyperparameters', 'dependencies']
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(metadata, field):
                    setattr(metadata, field, value)
            
            # Update timestamp
            metadata.updated_at = datetime.now(timezone.utc)
            
            # Save registry
            await self._save_registry()
            
            self.logger.info(f"Model metadata updated: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model metadata: {e}")
            return False
    
    async def archive_model(self, model_id: str) -> bool:
        """Archive a model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.models[model_id]
            
            # Update status
            metadata.status = ModelStatus.ARCHIVED
            metadata.updated_at = datetime.now(timezone.utc)
            
            # Remove from active models if it was active
            if metadata.name in self.active_models and self.active_models[metadata.name] == model_id:
                del self.active_models[metadata.name]
            
            # Save registry
            await self._save_registry()
            
            # Update statistics
            self._update_stats()
            
            self.logger.info(f"Model archived: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error archiving model: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model completely"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.models[model_id]
            
            # Remove from active models if it was active
            if metadata.name in self.active_models and self.active_models[metadata.name] == model_id:
                del self.active_models[metadata.name]
            
            # Remove from registry
            del self.models[model_id]
            
            # Delete model file
            model_file = os.path.join(self.model_storage_path, f"{model_id}.joblib")
            if os.path.exists(model_file):
                os.remove(model_file)
            
            # Save registry
            await self._save_registry()
            
            # Update statistics
            self._update_stats()
            
            self.logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    async def cleanup_old_versions(self, model_name: str, keep_versions: int = None) -> int:
        """Clean up old versions of a model"""
        try:
            if keep_versions is None:
                keep_versions = self.max_versions
            
            # Get all versions of the model
            versions = await self.list_models(name=model_name)
            
            if len(versions) <= keep_versions:
                return 0
            
            # Sort by creation date and keep only the newest versions
            versions.sort(key=lambda x: x.created_at, reverse=True)
            versions_to_delete = versions[keep_versions:]
            
            deleted_count = 0
            for metadata in versions_to_delete:
                if await self.delete_model(metadata.model_id):
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old versions of {model_name}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old versions: {e}")
            return 0
    
    def _update_stats(self):
        """Update registry statistics"""
        try:
            self.stats['total_models'] = len(self.models)
            self.stats['active_models'] = len(self.active_models)
            self.stats['training_models'] = len([m for m in self.models.values() if m.status == ModelStatus.TRAINING])
            self.stats['failed_models'] = len([m for m in self.models.values() if m.status == ModelStatus.FAILED])
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            stats = self.stats.copy()
            
            # Add model type distribution
            type_distribution = {}
            for metadata in self.models.values():
                model_type = metadata.model_type.value
                type_distribution[model_type] = type_distribution.get(model_type, 0) + 1
            
            stats['model_type_distribution'] = type_distribution
            
            # Add status distribution
            status_distribution = {}
            for metadata in self.models.values():
                status = metadata.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            stats['status_distribution'] = status_distribution
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting registry stats: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for model registry"""
        try:
            health_status = {
                'status': 'healthy',
                'total_models': self.stats['total_models'],
                'active_models': self.stats['active_models'],
                'registry_path': self.registry_path,
                'model_storage_path': self.model_storage_path
            }
            
            # Check storage paths
            if not os.path.exists(self.registry_path):
                health_status['status'] = 'degraded'
                health_status['warnings'] = ['Registry path does not exist']
            
            if not os.path.exists(self.model_storage_path):
                health_status['status'] = 'degraded'
                if 'warnings' not in health_status:
                    health_status['warnings'] = []
                health_status['warnings'].append('Model storage path does not exist')
            
            # Check database health if available
            if self.db_connection:
                try:
                    db_health = await self.db_connection.health_check()
                    health_status['database_health'] = db_health
                    
                    if db_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        if 'warnings' not in health_status:
                            health_status['warnings'] = []
                        health_status['warnings'].append('Database connection issues')
                except Exception as e:
                    health_status['database_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close model registry"""
        try:
            # Save registry before closing
            await self._save_registry()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.storage:
                await self.storage.close()
            
            self.logger.info("Model registry closed")
            
        except Exception as e:
            self.logger.error(f"Error closing model registry: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
