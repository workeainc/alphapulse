#!/usr/bin/env python3
"""
Ensemble System Service for AlphaPulse
Combines LightGBM, LSTM, and Transformer models for unified trading signals
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
import joblib
from sqlalchemy import create_engine, text

# Import individual model services
from app.services.predictive_analytics_service import PredictiveAnalyticsService, LiquidationPrediction
from app.services.lstm_time_series_service import LSTMTimeSeriesService, LSTMPrediction
from app.services.transformer_service import TransformerService, TransformerPrediction

logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    symbol: str
    timestamp: datetime
    unified_signal: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    model_contributions: Dict[str, float]  # Contribution of each model
    ensemble_weights: Dict[str, float]  # Dynamic weights for each model
    market_regime: str  # 'trending', 'ranging', 'volatile'
    prediction_horizon: int  # minutes
    metadata: Dict[str, Any] = None

class EnsembleSystemService:
    """Advanced ensemble system combining multiple ML models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.models_dir = self.config.get('models_dir', 'models/ensemble')
        self.update_frequency = self.config.get('update_frequency', 300)  # 5 minutes
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_voting')  # 'weighted_voting', 'stacking', 'blending'
        
        # Database connection
        self.database_url = self.config.get('database_url', os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"))
        self.engine = create_engine(self.database_url)
        
        # Initialize individual model services
        self.lightgbm_service = None
        self.lstm_service = None
        self.transformer_service = None
        
        # Ensemble weights (dynamic, updated based on performance)
        self.ensemble_weights = {
            'lightgbm': 0.4,  # Fast, interpretable
            'lstm': 0.35,     # Sequence modeling
            'transformer': 0.25  # Cross-timeframe analysis
        }
        
        # Meta-learner for stacking ensemble
        self.meta_learner = None
        self.meta_learner_trained = False
        
        # Regime-switching configuration
        self.regime_weights = {
            'trending': {
                'lightgbm': 0.3,
                'lstm': 0.45,      # LSTM excels in trending markets
                'transformer': 0.25
            },
            'ranging': {
                'lightgbm': 0.5,   # LightGBM good for ranging markets
                'lstm': 0.25,
                'transformer': 0.25
            },
            'volatile': {
                'lightgbm': 0.25,
                'lstm': 0.25,
                'transformer': 0.5  # Transformer good for volatile markets
            }
        }
        
        # Performance tracking per regime
        self.regime_performance = {
            'trending': {'lightgbm': 0.7, 'lstm': 0.8, 'transformer': 0.6},
            'ranging': {'lightgbm': 0.8, 'lstm': 0.6, 'transformer': 0.7},
            'volatile': {'lightgbm': 0.6, 'lstm': 0.7, 'transformer': 0.8}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'predictions_made': 0,
            'accuracy_scores': [],
            'model_performance': {
                'lightgbm': {'accuracy': 0.7, 'confidence': 0.8},
                'lstm': {'accuracy': 0.65, 'confidence': 0.75},
                'transformer': {'accuracy': 0.6, 'confidence': 0.7}
            },
            'ensemble_accuracy': 0.0,
            'last_weight_update': None
        }
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the ensemble system"""
        self.logger.info("Initializing Ensemble System Service...")
        
        try:
            # Initialize individual model services
            await self._initialize_model_services()
            
            # Load ensemble weights from disk
            await self._load_ensemble_weights()
            
            # Initialize performance tracking
            self.performance_metrics['last_weight_update'] = datetime.now()
            
            self.logger.info("Ensemble System Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ensemble system: {e}")
            raise
    
    async def _initialize_model_services(self):
        """Initialize individual model services"""
        try:
            # Initialize LightGBM service
            lightgbm_config = {
                'database_url': self.database_url,
                'model_storage_path': './ml_models',
                'retraining_interval_hours': 24
            }
            self.lightgbm_service = PredictiveAnalyticsService(lightgbm_config)
            await self.lightgbm_service.initialize()
            
            # Initialize LSTM service
            lstm_config = {
                'database_url': self.database_url,
                'models_dir': 'models/lstm',
                'sequence_length': 60,
                'prediction_horizons': [15, 30, 60]
            }
            self.lstm_service = LSTMTimeSeriesService(lstm_config)
            await self.lstm_service.initialize()
            
            # Initialize Transformer service
            transformer_config = {
                'database_url': self.database_url,
                'models_dir': 'models/transformer',
                'timeframes': ['15m', '1h', '4h'],
                'prediction_horizons': [30, 60, 240]
            }
            self.transformer_service = TransformerService(transformer_config)
            await self.transformer_service.initialize()
            
            self.logger.info("✅ All individual model services initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing model services: {e}")
            raise
    
    async def _load_ensemble_weights(self):
        """Load ensemble weights from disk"""
        try:
            weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    saved_weights = json.load(f)
                    self.ensemble_weights.update(saved_weights)
                    self.logger.info("✅ Loaded ensemble weights from disk")
            else:
                self.logger.info("Using default ensemble weights")
                
        except Exception as e:
            self.logger.error(f"Error loading ensemble weights: {e}")
    
    async def predict_unified_signal(self, symbol: str, market_data: Dict[str, Any]) -> EnsemblePrediction:
        """Generate unified trading signal from ensemble of models"""
        try:
            start_time = datetime.now()
            
            # Get predictions from all models
            predictions = await self._get_all_model_predictions(symbol, market_data)
            
            # Combine predictions using ensemble method
            unified_signal, confidence_score, model_contributions = await self._combine_predictions(predictions)
            
            # Determine risk level and market regime
            risk_level = self._determine_risk_level(predictions, confidence_score)
            market_regime = self._determine_market_regime(predictions)
            
            # Update ensemble weights based on recent performance
            # Update ensemble weights (use sync version to avoid async issues)
            self._update_ensemble_weights_sync(predictions, symbol)
            
            # Update performance metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['predictions_made'] += 1
            
            # Store ensemble prediction
            await self._store_ensemble_prediction(symbol, unified_signal, confidence_score, model_contributions)
            
            return EnsemblePrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                unified_signal=unified_signal,
                confidence_score=confidence_score,
                risk_level=risk_level,
                model_contributions=model_contributions,
                ensemble_weights=self.ensemble_weights.copy(),
                market_regime=market_regime,
                prediction_horizon=60,  # Default horizon
                metadata={
                    'individual_predictions': predictions,
                    'prediction_time': prediction_time,
                    'ensemble_method': self.ensemble_method
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating unified signal for {symbol}: {e}")
            return self._get_default_ensemble_prediction(symbol)
    
    async def _get_all_model_predictions(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get predictions from all individual models"""
        try:
            predictions = {}
            
            # LightGBM prediction
            try:
                lightgbm_pred = await self.lightgbm_service.predict_liquidations(symbol, market_data)
                predictions['lightgbm'] = {
                    'prediction': lightgbm_pred.liquidation_probability,
                    'confidence': lightgbm_pred.confidence_score,
                    'risk_level': lightgbm_pred.risk_level,
                    'metadata': lightgbm_pred.metadata
                }
            except Exception as e:
                self.logger.warning(f"LightGBM prediction failed for {symbol}: {e}")
                predictions['lightgbm'] = {
                    'prediction': 0.5,
                    'confidence': 0.3,
                    'risk_level': 'medium',
                    'metadata': {'error': str(e)}
                }
            
            # LSTM prediction
            try:
                lstm_pred = await self.lstm_service.predict_directional_bias(symbol, market_data)
                # Convert signal to probability for ensemble calculation
                signal_to_prob = {
                    'strong_buy': 0.9, 'buy': 0.7, 'hold': 0.5, 
                    'sell': 0.3, 'strong_sell': 0.1
                }
                signal_prob = signal_to_prob.get(lstm_pred.directional_bias, 0.5)
                
                predictions['lstm'] = {
                    'prediction': signal_prob,
                    'confidence': lstm_pred.confidence_score,
                    'directional_bias': lstm_pred.directional_bias,
                    'metadata': lstm_pred.metadata
                }
            except Exception as e:
                self.logger.warning(f"LSTM prediction failed for {symbol}: {e}")
                predictions['lstm'] = {
                    'prediction': 0.5,
                    'confidence': 0.3,
                    'directional_bias': 'neutral',
                    'metadata': {'error': str(e)}
                }
            
            # Transformer prediction
            try:
                transformer_pred = await self.transformer_service.predict_cross_timeframe_signal(symbol, market_data)
                # Convert signal to probability for ensemble calculation
                signal_to_prob = {
                    'strong_buy': 0.9, 'buy': 0.7, 'hold': 0.5, 
                    'sell': 0.3, 'strong_sell': 0.1
                }
                signal_prob = signal_to_prob.get(transformer_pred.cross_timeframe_signal, 0.5)
                
                predictions['transformer'] = {
                    'prediction': signal_prob,
                    'confidence': transformer_pred.confidence_score,
                    'cross_timeframe_signal': transformer_pred.cross_timeframe_signal,
                    'market_regime': transformer_pred.market_regime,
                    'metadata': transformer_pred.metadata
                }
            except Exception as e:
                self.logger.warning(f"Transformer prediction failed for {symbol}: {e}")
                predictions['transformer'] = {
                    'prediction': 0.5,
                    'confidence': 0.3,
                    'cross_timeframe_signal': 'hold',
                    'market_regime': 'ranging',
                    'metadata': {'error': str(e)}
                }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting model predictions: {e}")
            return {}
    
    async def _combine_predictions(self, predictions: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """Combine predictions using ensemble method with regime-switching"""
        try:
            if not predictions:
                return 'hold', 0.3, {}
            
            # Determine market regime
            market_regime = self._determine_market_regime(predictions)
            
            # Apply regime-switching weights
            regime_weights = self.regime_weights.get(market_regime, self.ensemble_weights)
            
            if self.ensemble_method == 'weighted_voting':
                return self._weighted_voting_ensemble(predictions, regime_weights)
            elif self.ensemble_method == 'stacking':
                return await self._stacking_ensemble(predictions, regime_weights)
            elif self.ensemble_method == 'blending':
                return self._blending_ensemble(predictions, regime_weights)
            else:
                return self._weighted_voting_ensemble(predictions, regime_weights)
                
        except Exception as e:
            self.logger.error(f"Error combining predictions: {e}")
            return 'hold', 0.3, {}
    
    def _weighted_voting_ensemble(self, predictions: Dict[str, Any], weights: Dict[str, float] = None) -> Tuple[str, float, Dict[str, float]]:
        """Weighted voting ensemble method with regime-switching support"""
        try:
            if weights is None:
                weights = self.ensemble_weights
                
            weighted_predictions = []
            model_contributions = {}
            
            for model_name, pred_data in predictions.items():
                if model_name in weights:
                    weight = weights[model_name]
                    confidence = pred_data.get('confidence', 0.5)
                    
                    # Adjust weight by confidence
                    adjusted_weight = weight * confidence
                    weighted_predictions.append(pred_data['prediction'] * adjusted_weight)
                    model_contributions[model_name] = adjusted_weight
                else:
                    weighted_predictions.append(pred_data['prediction'] * 0.1)  # Default weight
                    model_contributions[model_name] = 0.1
            
            if weighted_predictions:
                total_weight = sum(model_contributions.values())
                if total_weight > 0:
                    ensemble_prediction = sum(weighted_predictions) / total_weight
                    confidence_score = float(1 - np.std(weighted_predictions)) if len(weighted_predictions) > 1 else 0.5
                else:
                    ensemble_prediction = 0.5
                    confidence_score = 0.3
            else:
                ensemble_prediction = 0.5
                confidence_score = 0.3
            
            # Determine unified signal
            if ensemble_prediction > 0.8:
                unified_signal = 'strong_buy'
            elif ensemble_prediction > 0.6:
                unified_signal = 'buy'
            elif ensemble_prediction > 0.4:
                unified_signal = 'hold'
            elif ensemble_prediction > 0.2:
                unified_signal = 'sell'
            else:
                unified_signal = 'strong_sell'
            
            return unified_signal, confidence_score, model_contributions
            
        except Exception as e:
            self.logger.error(f"Error in weighted voting ensemble: {e}")
            return 'hold', 0.3, {}
    
    async def _stacking_ensemble(self, predictions: Dict[str, Any], weights: Dict[str, float] = None) -> Tuple[str, float, Dict[str, float]]:
        """Stacking ensemble method with meta-learner"""
        try:
            if not self.meta_learner_trained:
                # Fallback to weighted voting if meta-learner not trained
                return self._weighted_voting_ensemble(predictions, weights)
            
            # Prepare features for meta-learner
            meta_features = []
            for model_name in ['lightgbm', 'lstm', 'transformer']:
                if model_name in predictions:
                    meta_features.extend([
                        predictions[model_name]['prediction'],
                        predictions[model_name]['confidence']
                    ])
                else:
                    meta_features.extend([0.5, 0.3])  # Default values
            
            # Add market regime features
            market_regime = self._determine_market_regime(predictions)
            regime_encoding = {
                'trending': [1, 0, 0],
                'ranging': [0, 1, 0],
                'volatile': [0, 0, 1]
            }
            meta_features.extend(regime_encoding.get(market_regime, [0, 0, 0]))
            
            # Make prediction with meta-learner
            if self.meta_learner and len(meta_features) > 0:
                meta_prediction = self.meta_learner.predict([meta_features])[0]
                
                # Convert to signal
                if meta_prediction > 0.8:
                    unified_signal = 'strong_buy'
                elif meta_prediction > 0.6:
                    unified_signal = 'buy'
                elif meta_prediction > 0.4:
                    unified_signal = 'hold'
                elif meta_prediction > 0.2:
                    unified_signal = 'sell'
                else:
                    unified_signal = 'strong_sell'
                
                # Calculate confidence based on model agreement
                confidence_score = float(1 - np.std([pred['prediction'] for pred in predictions.values()])) if len(predictions) > 1 else 0.5
                
                # Calculate model contributions
                model_contributions = {name: pred['confidence'] for name, pred in predictions.items()}
                
                return unified_signal, confidence_score, model_contributions
            
            # Fallback to weighted voting
            return self._weighted_voting_ensemble(predictions, weights)
            
        except Exception as e:
            self.logger.error(f"Error in stacking ensemble: {e}")
            return self._weighted_voting_ensemble(predictions, weights)
    
    def _blending_ensemble(self, predictions: Dict[str, Any], weights: Dict[str, float] = None) -> Tuple[str, float, Dict[str, float]]:
        """Blending ensemble method with regime-switching support"""
        try:
            if weights is None:
                weights = self.ensemble_weights
                
            # Blending: weighted average with dynamic weights
            blended_prediction = 0.0
            total_weight = 0.0
            model_contributions = {}
            
            for model_name, pred_data in predictions.items():
                if model_name in weights:
                    weight = weights[model_name]
                    confidence = pred_data.get('confidence', 0.5)
                    
                    # Dynamic weight adjustment based on recent performance
                    performance_weight = self.performance_metrics['model_performance'].get(model_name, {}).get('accuracy', 0.5)
                    final_weight = weight * confidence * performance_weight
                    
                    blended_prediction += pred_data['prediction'] * final_weight
                    total_weight += final_weight
                    model_contributions[model_name] = final_weight
                else:
                    # Default contribution
                    blended_prediction += pred_data['prediction'] * 0.1
                    total_weight += 0.1
                    model_contributions[model_name] = 0.1
            
            if total_weight > 0:
                ensemble_prediction = blended_prediction / total_weight
                confidence_score = float(1 - np.std([pred_data['prediction'] for pred_data in predictions.values()])) if len(predictions) > 1 else 0.5
            else:
                ensemble_prediction = 0.5
                confidence_score = 0.3
            
            # Determine unified signal
            if ensemble_prediction > 0.8:
                unified_signal = 'strong_buy'
            elif ensemble_prediction > 0.6:
                unified_signal = 'buy'
            elif ensemble_prediction > 0.4:
                unified_signal = 'hold'
            elif ensemble_prediction > 0.2:
                unified_signal = 'sell'
            else:
                unified_signal = 'strong_sell'
            
            return unified_signal, confidence_score, model_contributions
            
        except Exception as e:
            self.logger.error(f"Error in blending ensemble: {e}")
            return 'hold', 0.3, {}
    
    def _determine_risk_level(self, predictions: Dict[str, Any], confidence_score: float) -> str:
        """Determine overall risk level"""
        try:
            # Calculate risk based on prediction variance and confidence
            prediction_values = [pred_data['prediction'] for pred_data in predictions.values()]
            
            if prediction_values:
                variance = np.var(prediction_values)
                
                if variance > 0.1 or confidence_score < 0.4:
                    return 'critical'
                elif variance > 0.05 or confidence_score < 0.6:
                    return 'high'
                elif variance > 0.02 or confidence_score < 0.8:
                    return 'medium'
                else:
                    return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return 'medium'
    
    def _determine_market_regime(self, predictions: Dict[str, Any]) -> str:
        """Determine market regime"""
        try:
            # Use Transformer's market regime if available
            if 'transformer' in predictions:
                transformer_regime = predictions['transformer'].get('market_regime', 'ranging')
                return transformer_regime
            
            # Fallback: determine from prediction variance
            prediction_values = [pred_data['prediction'] for pred_data in predictions.values()]
            
            if prediction_values:
                variance = np.var(prediction_values)
                
                if variance > 0.1:
                    return 'volatile'
                elif variance > 0.05:
                    return 'trending'
                else:
                    return 'ranging'
            else:
                return 'ranging'
                
        except Exception as e:
            self.logger.error(f"Error determining market regime: {e}")
            return 'ranging'
    
    async def _update_ensemble_weights(self, predictions: Dict[str, Any], symbol: str):
        """Update ensemble weights based on recent performance"""
        try:
            # Update weights every 100 predictions
            if self.performance_metrics['predictions_made'] % 100 == 0:
                await self._recalculate_weights(predictions, symbol)
                
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
    
    def _update_ensemble_weights_sync(self, predictions: Dict[str, Any], symbol: str):
        """Synchronous version for non-async contexts"""
        try:
            # Update weights every 100 predictions
            if self.performance_metrics['predictions_made'] % 100 == 0:
                # Use default weights if async recalculation fails
                self.logger.info("Weight update scheduled (async context required)")
                
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
    
    async def _recalculate_weights(self, predictions: Dict[str, Any], symbol: str):
        """Recalculate ensemble weights based on performance"""
        try:
            # Get recent performance metrics
            recent_performance = await self._get_recent_performance(symbol)
            
            if recent_performance:
                # Calculate new weights based on performance
                total_performance = sum(recent_performance.values())
                
                if total_performance > 0:
                    new_weights = {}
                    for model_name, performance in recent_performance.items():
                        new_weights[model_name] = performance / total_performance
                    
                    # Smooth weight transition
                    alpha = 0.1  # Learning rate
                    for model_name in self.ensemble_weights:
                        if model_name in new_weights:
                            self.ensemble_weights[model_name] = (
                                (1 - alpha) * self.ensemble_weights[model_name] + 
                                alpha * new_weights[model_name]
                            )
                    
                    # Save updated weights
                    await self._save_ensemble_weights()
                    
                    self.performance_metrics['last_weight_update'] = datetime.now()
                    self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
                    
        except Exception as e:
            self.logger.error(f"Error recalculating weights: {e}")
    
    async def _get_recent_performance(self, symbol: str) -> Dict[str, float]:
        """Get recent performance metrics for each model"""
        try:
            with self.engine.connect() as conn:
                # Get recent predictions and their accuracy
                result = conn.execute(text("""
                    SELECT model_type, AVG(confidence_score) as avg_confidence
                    FROM ml_predictions 
                    WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY model_type
                """), {"symbol": symbol})
                
                performance = {}
                for row in result:
                    performance[row[0]] = float(row[1]) if row[1] is not None else 0.5
                
                return performance
                
        except Exception as e:
            self.logger.error(f"Error getting recent performance: {e}")
            return {}
    
    async def _save_ensemble_weights(self):
        """Save ensemble weights to disk"""
        try:
            weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
            with open(weights_path, 'w') as f:
                json.dump(self.ensemble_weights, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving ensemble weights: {e}")
    
    async def train_meta_learner(self, training_data: List[Dict[str, Any]]):
        """Train meta-learner for stacking ensemble"""
        try:
            if len(training_data) < 100:
                self.logger.warning("Insufficient training data for meta-learner")
                return
            
            # Prepare training features and labels
            X = []
            y = []
            
            for sample in training_data:
                features = []
                # Add model predictions
                for model_name in ['lightgbm', 'lstm', 'transformer']:
                    if model_name in sample['predictions']:
                        features.extend([
                            sample['predictions'][model_name]['prediction'],
                            sample['predictions'][model_name]['confidence']
                        ])
                    else:
                        features.extend([0.5, 0.3])
                
                # Add market regime
                regime = sample.get('market_regime', 'ranging')
                regime_encoding = {
                    'trending': [1, 0, 0],
                    'ranging': [0, 1, 0],
                    'volatile': [0, 0, 1]
                }
                features.extend(regime_encoding.get(regime, [0, 0, 0]))
                
                X.append(features)
                y.append(sample['actual_outcome'])  # 1 for correct prediction, 0 for incorrect
            
            # Train meta-learner (using Logistic Regression for simplicity)
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.meta_learner = LogisticRegression(random_state=42)
            self.meta_learner.fit(X_train, y_train)
            
            # Evaluate meta-learner
            train_score = self.meta_learner.score(X_train, y_train)
            test_score = self.meta_learner.score(X_test, y_test)
            
            self.logger.info(f"Meta-learner trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
            # Save meta-learner
            meta_learner_path = os.path.join(self.models_dir, 'meta_learner.joblib')
            joblib.dump(self.meta_learner, meta_learner_path)
            
            self.meta_learner_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training meta-learner: {e}")
    
    async def update_regime_performance(self, symbol: str, market_regime: str, model_performance: Dict[str, float]):
        """Update performance metrics for each model in different market regimes"""
        try:
            if market_regime in self.regime_performance:
                # Update performance with exponential moving average
                alpha = 0.1  # Learning rate
                for model_name, performance in model_performance.items():
                    if model_name in self.regime_performance[market_regime]:
                        current_perf = self.regime_performance[market_regime][model_name]
                        self.regime_performance[market_regime][model_name] = (
                            (1 - alpha) * current_perf + alpha * performance
                        )
                
                # Save updated regime performance
                regime_perf_path = os.path.join(self.models_dir, 'regime_performance.json')
                with open(regime_perf_path, 'w') as f:
                    json.dump(self.regime_performance, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error updating regime performance: {e}")
    
    def get_regime_weights(self, market_regime: str) -> Dict[str, float]:
        """Get ensemble weights for specific market regime"""
        return self.regime_weights.get(market_regime, self.ensemble_weights)
    
    async def _store_ensemble_prediction(self, symbol: str, unified_signal: str, confidence_score: float, model_contributions: Dict[str, float]):
        """Store ensemble prediction in database"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO ml_predictions (
                        symbol, timestamp, model_type, prediction_type, prediction_value, confidence_score,
                        prediction_metadata, feature_vector
                    ) VALUES (
                        :symbol, :timestamp, :model_type, :prediction_type, :prediction_value, :confidence_score,
                        :prediction_metadata, :feature_vector
                    )
                """), {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "model_type": "ensemble",
                    "prediction_type": "price_direction",
                    "prediction_value": 0.5,  # Placeholder
                    "confidence_score": confidence_score,
                    "prediction_metadata": json.dumps({
                        "unified_signal": unified_signal,
                        "model_contributions": model_contributions,
                        "ensemble_weights": self.ensemble_weights,
                        "ensemble_method": self.ensemble_method
                    }),
                    "feature_vector": json.dumps(model_contributions)
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing ensemble prediction: {e}")
    
    def _get_default_ensemble_prediction(self, symbol: str) -> EnsemblePrediction:
        """Get default ensemble prediction"""
        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            unified_signal='hold',
            confidence_score=0.3,
            risk_level='medium',
            model_contributions={},
            ensemble_weights=self.ensemble_weights.copy(),
            market_regime='ranging',
            prediction_horizon=60,
            metadata={'error': 'fallback_prediction'}
        )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'ensemble_weights': self.ensemble_weights,
            'ensemble_method': self.ensemble_method,
            'individual_model_metrics': {
                'lightgbm': await self.lightgbm_service.get_performance_metrics() if self.lightgbm_service else {},
                'lstm': await self.lstm_service.get_performance_metrics() if self.lstm_service else {},
                'transformer': await self.transformer_service.get_performance_metrics() if self.transformer_service else {}
            }
        }
