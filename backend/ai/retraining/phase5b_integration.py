#!/usr/bin/env python3
"""
Phase 5B: Enhanced Ensemble Integration
Simple integration module for Phase 5B functionality
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..ml_models.ensemble_manager import (
    EnhancedEnsembleManager, EnsembleConfig, ModelType, MarketRegime,
    enhanced_ensemble_manager
)

logger = logging.getLogger(__name__)

class Phase5BIntegration:
    """Simple integration class for Phase 5B ensemble functionality"""
    
    def __init__(self):
        self.ensemble_manager = enhanced_ensemble_manager
        self.logger = logger
        
        logger.info("✅ Phase 5B Integration initialized")
    
    async def execute_ensemble_training(self, X: pd.DataFrame = None, y: pd.Series = None) -> Dict[str, Any]:
        """Execute Phase 5B ensemble training with feature store integration"""
        try:
            self.logger.info("🚀 Starting Phase 5B: Enhanced Ensemble Training with Feature Store...")
            
            # Use feature store if no data provided
            if X is None or y is None:
                self.logger.info("📊 Getting features from feature store...")
                X = await self.ensemble_manager.get_features_with_validation()
                
                if X.empty:
                    return {
                        'status': 'failed',
                        'error': 'No features available from feature store',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Create synthetic target for demonstration
                import numpy as np
                y = pd.Series(np.random.randint(0, 2, len(X)))
                self.logger.info("✅ Features retrieved from feature store")
            
            # Train all ensemble models
            training_results = await self.ensemble_manager.train_all_models(X, y)
            
            # Get ensemble status
            ensemble_status = await self.ensemble_manager.get_ensemble_status()
            
            result = {
                'status': 'completed',
                'training_results': training_results,
                'ensemble_status': ensemble_status,
                'models_trained': sum(training_results.values()) if training_results else 0,
                'total_models': len(training_results) if training_results else 0,
                'feature_store_used': X is not None,
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
    
    async def execute_ensemble_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Execute Phase 5B ensemble prediction with regime-aware selection"""
        try:
            self.logger.info("🔮 Making Phase 5B ensemble prediction...")
            
            # Make ensemble prediction
            prediction = await self.ensemble_manager.predict(X)
            
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
    
    async def execute_model_training(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
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
            regime, _ = self.ensemble_manager.meta_learner.detect_regime(X)
            
            # Train the model
            success = await self.ensemble_manager.train_model(model_enum, X, y, regime)
            
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
    
    async def get_ensemble_status(self) -> Dict[str, Any]:
        """Get Phase 5B ensemble status"""
        try:
            self.logger.info("📊 Getting Phase 5B ensemble status...")
            
            status = await self.ensemble_manager.get_ensemble_status()
            
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
    
    async def execute_regime_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime for Phase 5B"""
        try:
            self.logger.info("🔍 Analyzing market regime for Phase 5B...")
            
            # Detect regime
            regime, confidence = self.ensemble_manager.meta_learner.detect_regime(X)
            
            # Get regime-specific weights
            regime_weights = self.ensemble_manager.meta_learner.get_regime_weights(regime)
            
            # Get best models for this regime
            best_models = self.ensemble_manager.meta_learner.select_best_models(regime, top_k=3)
            
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
    
    async def execute_feature_drift_detection(self, feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect feature drift for Phase 5B"""
        try:
            self.logger.info("🔍 Detecting feature drift for Phase 5B...")
            
            # Detect drift using ensemble manager
            drift_results = await self.ensemble_manager.detect_feature_drift(feature_names)
            
            # Count drift incidents
            drift_count = sum(1 for result in drift_results.values() if result.get('is_drift_detected', False))
            total_features = len(drift_results)
            
            result = {
                'status': 'success',
                'drift_results': drift_results,
                'drift_count': drift_count,
                'total_features': total_features,
                'drift_percentage': (drift_count / total_features * 100) if total_features > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Phase 5B drift detection: {drift_count}/{total_features} features with drift")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase 5B drift detection failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global instance
phase5b_integration = Phase5BIntegration()

# Convenience functions
async def train_ensemble(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train the Phase 5B ensemble"""
    return await phase5b_integration.execute_ensemble_training(X, y)

async def predict_ensemble(X: pd.DataFrame) -> Dict[str, Any]:
    """Make ensemble prediction"""
    return await phase5b_integration.execute_ensemble_prediction(X)

async def train_model(model_type: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train a specific model"""
    return await phase5b_integration.execute_model_training(model_type, X, y)

async def get_status() -> Dict[str, Any]:
    """Get ensemble status"""
    return await phase5b_integration.get_ensemble_status()

async def analyze_regime(X: pd.DataFrame) -> Dict[str, Any]:
    """Analyze market regime"""
    return await phase5b_integration.execute_regime_analysis(X)

async def detect_feature_drift(feature_names: List[str] = None) -> Dict[str, Any]:
    """Detect feature drift"""
    return await phase5b_integration.execute_feature_drift_detection(feature_names)
