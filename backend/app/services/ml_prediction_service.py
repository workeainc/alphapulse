#!/usr/bin/env python3
"""
ML Prediction Service
Handles real-time predictions using trained ML models
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("PyTorch not available for deep learning predictions")

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    BREAKOUT = "breakout"
    RETURN = "return"
    DIRECTION = "direction"
    ANOMALY = "anomaly"
    PRICE = "price"
    VOLATILITY = "volatility"
    REGIME = "regime"

@dataclass
class PredictionResult:
    """Result of a model prediction"""
    symbol: str
    timeframe: str
    timestamp: datetime
    model_version: str
    prediction_type: PredictionType
    prediction_value: float
    confidence_score: float
    feature_contributions: Dict[str, float]
    shap_values: Dict[str, float]
    model_metadata: Dict[str, Any]

class MLPredictionService:
    """Service for making real-time ML predictions"""
    
    def __init__(self, db_pool, model_storage_path: str = "./models"):
        self.db_pool = db_pool
        self.model_storage_path = model_storage_path
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded models
        self.model_cache = {}
        self.model_metadata_cache = {}
        
        # Feature engineering service
        from .ml_feature_engineering_service import MLFeatureEngineeringService
        self.feature_service = MLFeatureEngineeringService(db_pool)
        
        self.logger.info("🔮 ML Prediction Service initialized")
    
    async def predict(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> Optional[PredictionResult]:
        """Make a prediction for the given symbol and timeframe"""
        try:
            # Get active model for this symbol/timeframe
            active_model = await self._get_active_model(symbol, timeframe)
            if not active_model:
                self.logger.warning(f"⚠️ No active model found for {symbol} {timeframe}")
                return None
            
            # Generate features
            features = await self.feature_service.generate_comprehensive_features(symbol, timeframe, ohlcv_data)
            if not features:
                self.logger.error(f"❌ Failed to generate features for {symbol}")
                return None
            
            # Load model
            model = await self._load_model(active_model['version'])
            if not model:
                self.logger.error(f"❌ Failed to load model {active_model['version']}")
                return None
            
            # Make prediction
            prediction = await self._make_prediction(model, features, active_model)
            if not prediction:
                self.logger.error(f"❌ Failed to make prediction for {symbol}")
                return None
            
            # Store prediction
            await self._store_prediction(prediction)
            
            self.logger.info(f"✅ Prediction made for {symbol}: {prediction.prediction_type.value} = {prediction.prediction_value:.4f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error making prediction: {e}")
            return None
    
    async def _get_active_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get the active model for the given symbol and timeframe"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT model_name, version, model_type, validation_auc, validation_precision, 
                       validation_recall, feature_list, hyperparameters
                FROM model_versions
                WHERE model_name = $1 AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
                
                model_name = f"{symbol}_{timeframe}"
                row = await conn.fetchrow(query, model_name)
                
                if row:
                    return dict(row)
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting active model: {e}")
            return None
    
    async def _load_model(self, model_version: str) -> Optional[Any]:
        """Load a trained model from disk"""
        try:
            # Check cache first
            if model_version in self.model_cache:
                return self.model_cache[model_version]
            
            # Load model file
            model_path = f"{self.model_storage_path}/{model_version}.pkl"
            model = joblib.load(model_path)
            
            # Cache the model
            self.model_cache[model_version] = model
            
            # Load metadata
            metadata_path = f"{self.model_storage_path}/{model_version}_metadata.json"
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.model_metadata_cache[model_version] = metadata
            except FileNotFoundError:
                self.logger.warning(f"⚠️ Metadata file not found for {model_version}")
                self.model_metadata_cache[model_version] = {}
            
            self.logger.info(f"📦 Loaded model: {model_version}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model {model_version}: {e}")
            return None
    
    async def _make_prediction(self, model: Any, features: Any, model_info: Dict) -> Optional[PredictionResult]:
        """Make a prediction using the loaded model"""
        try:
            # Convert features to feature vector
            feature_vector = self._extract_feature_vector(features, model_info.get('feature_list', []))
            if feature_vector is None:
                return None
            
            # Make prediction
            prediction_value = model.predict([feature_vector])[0]
            
            # Get prediction probability for classification
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba([feature_vector])[0]
                confidence_score = max(prediction_proba)
            else:
                confidence_score = 0.8  # Default confidence for regression
            
            # Calculate feature contributions (simplified SHAP-like values)
            feature_contributions = self._calculate_feature_contributions(model, feature_vector, model_info.get('feature_list', []))
            
            # Determine prediction type
            prediction_type = self._determine_prediction_type(model_info.get('model_type', ''))
            
            # Create prediction result
            result = PredictionResult(
                symbol=features.symbol if hasattr(features, 'symbol') else 'unknown',
                timeframe=features.timeframe if hasattr(features, 'timeframe') else 'unknown',
                timestamp=datetime.now(),
                model_version=model_info['version'],
                prediction_type=prediction_type,
                prediction_value=float(prediction_value),
                confidence_score=float(confidence_score),
                feature_contributions=feature_contributions,
                shap_values=feature_contributions,  # Simplified - same as contributions
                model_metadata={
                    'model_type': model_info.get('model_type'),
                    'validation_auc': model_info.get('validation_auc'),
                    'validation_precision': model_info.get('validation_precision'),
                    'validation_recall': model_info.get('validation_recall')
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error making prediction: {e}")
            return None
    
    def _extract_feature_vector(self, features: Any, feature_list: List[str]) -> Optional[np.ndarray]:
        """Extract feature vector from MLFeatures object"""
        try:
            # Convert features to dictionary
            feature_dict = {
                'volume_ratio': features.volume_ratio,
                'volume_positioning_score': features.volume_positioning_score,
                'order_book_imbalance': features.order_book_imbalance,
                'vwap': features.vwap,
                'cumulative_volume_delta': features.cumulative_volume_delta,
                'relative_volume': features.relative_volume,
                'volume_flow_imbalance': features.volume_flow_imbalance,
                'ema_20': features.ema_20,
                'ema_50': features.ema_50,
                'ema_200': features.ema_200,
                'atr_14': features.atr_14,
                'obv': features.obv,
                'rsi_14': features.rsi_14,
                'macd': features.macd,
                'macd_signal': features.macd_signal,
                'macd_histogram': features.macd_histogram,
                'bid_depth_0_5': features.bid_depth_0_5,
                'bid_depth_1_0': features.bid_depth_1_0,
                'bid_depth_2_0': features.bid_depth_2_0,
                'ask_depth_0_5': features.ask_depth_0_5,
                'ask_depth_1_0': features.ask_depth_1_0,
                'ask_depth_2_0': features.ask_depth_2_0,
                'bid_ask_ratio': features.bid_ask_ratio,
                'spread_bps': features.spread_bps,
                'liquidity_score': features.liquidity_score,
                'minute_of_day': features.minute_of_day,
                'hour_of_day': features.hour_of_day,
                'day_of_week': features.day_of_week,
                'is_session_open': int(features.is_session_open),
                'session_volatility': features.session_volatility,
                'h1_return': features.h1_return,
                'h4_return': features.h4_return,
                'd1_return': features.d1_return,
                'h1_volume_ratio': features.h1_volume_ratio,
                'h4_volume_ratio': features.h4_volume_ratio,
                'd1_volume_ratio': features.d1_volume_ratio,
                'volume_pattern_confidence': features.volume_pattern_confidence,
                'volume_breakout': int(features.volume_breakout),
                'distance_to_support': features.distance_to_support,
                'distance_to_resistance': features.distance_to_resistance,
                'nearest_volume_node': features.nearest_volume_node,
                'volume_node_strength': features.volume_node_strength
            }
            
            # Create feature vector based on feature list
            if not feature_list:
                # Use all available features
                feature_vector = [feature_dict.get(f, 0.0) for f in feature_dict.keys()]
            else:
                # Use only specified features
                feature_vector = [feature_dict.get(f, 0.0) for f in feature_list]
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting feature vector: {e}")
            return None
    
    def _calculate_feature_contributions(self, model: Any, feature_vector: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature contributions (simplified SHAP-like values)"""
        try:
            # For LightGBM, we can use feature importance as a proxy for contributions
            if hasattr(model, 'feature_importance'):
                importance = model.feature_importance()
                contributions = {}
                
                for i, feature_name in enumerate(feature_names):
                    if i < len(importance):
                        # Normalize by feature value
                        contributions[feature_name] = float(importance[i] * feature_vector[i])
                    else:
                        contributions[feature_name] = 0.0
                
                return contributions
            else:
                # Fallback: equal contributions
                return {name: 0.1 for name in feature_names}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating feature contributions: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _determine_prediction_type(self, model_type: str) -> PredictionType:
        """Determine prediction type based on model type"""
        if 'breakout' in model_type.lower():
            return PredictionType.BREAKOUT
        elif 'return' in model_type.lower():
            return PredictionType.RETURN
        elif 'direction' in model_type.lower():
            return PredictionType.DIRECTION
        elif 'anomaly' in model_type.lower():
            return PredictionType.ANOMALY
        else:
            return PredictionType.BREAKOUT  # Default
    
    async def _store_prediction(self, prediction: PredictionResult) -> bool:
        """Store prediction in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                INSERT INTO model_predictions 
                (symbol, timeframe, timestamp, model_version, prediction_type, prediction_value, 
                 confidence_score, feature_contributions, shap_values, model_metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                await conn.execute(query,
                    prediction.symbol,
                    prediction.timeframe,
                    prediction.timestamp,
                    prediction.model_version,
                    prediction.prediction_type.value,
                    prediction.prediction_value,
                    prediction.confidence_score,
                    prediction.feature_contributions,
                    prediction.shap_values,
                    prediction.model_metadata
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing prediction: {e}")
            return False
    
    async def get_recent_predictions(self, symbol: str = None, timeframe: str = None, limit: int = 100) -> List[Dict]:
        """Get recent predictions from database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT symbol, timeframe, timestamp, model_version, prediction_type, 
                       prediction_value, confidence_score, feature_contributions, model_metadata
                FROM model_predictions
                """
                
                params = []
                if symbol:
                    query += " WHERE symbol = $1"
                    params.append(symbol)
                    if timeframe:
                        query += " AND timeframe = $2"
                        params.append(timeframe)
                elif timeframe:
                    query += " WHERE timeframe = $1"
                    params.append(timeframe)
                
                query += " ORDER BY timestamp DESC LIMIT $3"
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting recent predictions: {e}")
            return []
    
    async def get_prediction_statistics(self, symbol: str = None, timeframe: str = None, hours: int = 24) -> Dict:
        """Get prediction statistics for the last N hours"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT 
                    prediction_type,
                    COUNT(*) as prediction_count,
                    AVG(prediction_value) as avg_prediction,
                    AVG(confidence_score) as avg_confidence,
                    STDDEV(prediction_value) as std_prediction
                FROM model_predictions
                WHERE timestamp >= NOW() - INTERVAL '$1 hours'
                """
                
                params = [hours]
                if symbol:
                    query += " AND symbol = $2"
                    params.append(symbol)
                    if timeframe:
                        query += " AND timeframe = $3"
                        params.append(timeframe)
                elif timeframe:
                    query += " AND timeframe = $2"
                    params.append(timeframe)
                
                query += " GROUP BY prediction_type"
                
                rows = await conn.fetch(query, *params)
                
                stats = {}
                for row in rows:
                    stats[row['prediction_type']] = {
                        'count': row['prediction_count'],
                        'avg_prediction': float(row['avg_prediction']) if row['avg_prediction'] else 0.0,
                        'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.0,
                        'std_prediction': float(row['std_prediction']) if row['std_prediction'] else 0.0
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Error getting prediction statistics: {e}")
            return {}
    
    async def clear_model_cache(self):
        """Clear the model cache"""
        self.model_cache.clear()
        self.model_metadata_cache.clear()
        self.logger.info("🧹 Model cache cleared")
    
    async def get_model_performance(self, model_version: str) -> Optional[Dict]:
        """Get performance metrics for a specific model"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT 
                    model_name, version, model_type, validation_auc, validation_precision, 
                    validation_recall, training_samples, created_at
                FROM model_versions
                WHERE version = $1
                """
                
                row = await conn.fetchrow(query, model_version)
                if row:
                    return dict(row)
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting model performance: {e}")
            return None

    # ==================== DEEP LEARNING PREDICTIONS ====================
    
    async def predict_deep_learning(self, symbol: str, timeframe: str, ohlcv_data: List[Dict], 
                                   model_type: str = "lstm") -> Optional[PredictionResult]:
        """Make deep learning prediction for the given symbol and timeframe"""
        try:
            # Get active deep learning model
            active_model = await self._get_active_deep_learning_model(symbol, timeframe, model_type)
            if not active_model:
                self.logger.warning(f"⚠️ No active deep learning model found for {symbol} {timeframe}")
                return None
            
            # Generate features for deep learning
            features = await self.feature_service.generate_comprehensive_features(symbol, timeframe, ohlcv_data)
            if not features:
                self.logger.error(f"❌ Failed to generate features for {symbol}")
                return None
            
            # Load deep learning model
            model = await self._load_deep_learning_model(active_model['version'])
            if not model:
                self.logger.error(f"❌ Failed to load deep learning model {active_model['version']}")
                return None
            
            # Make deep learning prediction
            prediction = await self._make_deep_learning_prediction(model, features, active_model, model_type)
            if not prediction:
                self.logger.error(f"❌ Failed to make deep learning prediction for {symbol}")
                return None
            
            # Store deep learning prediction
            await self._store_deep_learning_prediction(prediction)
            
            self.logger.info(f"🧠 Deep learning prediction made for {symbol}: {prediction.prediction_type.value} = {prediction.prediction_value:.4f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error making deep learning prediction: {e}")
            return None
    
    async def _get_active_deep_learning_model(self, symbol: str, timeframe: str, model_type: str) -> Optional[Dict]:
        """Get the active deep learning model for the given symbol and timeframe"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT model_name, version, model_type, validation_auc, validation_precision, 
                       validation_recall, model_path, hyperparameters
                FROM model_versions
                WHERE model_name LIKE $1 AND model_type = $2 AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
                """
                
                model_name_pattern = f"{symbol}_{timeframe}_dl"
                row = await conn.fetchrow(query, model_name_pattern, model_type)
                
                if row:
                    return dict(row)
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting active deep learning model: {e}")
            return None
    
    async def _load_deep_learning_model(self, model_version: str) -> Optional[nn.Module]:
        """Load a deep learning model from storage"""
        if not DL_AVAILABLE:
            self.logger.error("❌ PyTorch not available for deep learning")
            return None
        
        try:
            # Check cache first
            if model_version in self.model_cache:
                return self.model_cache[model_version]
            
            # Get model path from database
            model_path = await self._get_model_path(model_version)
            if not model_path:
                self.logger.error(f"❌ Model path not found for {model_version}")
                return None
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Recreate model architecture
            model_config = checkpoint['model_config']
            model = self._create_deep_learning_model_from_config(model_config)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Cache model
            self.model_cache[model_version] = model
            self.model_metadata_cache[model_version] = checkpoint
            
            self.logger.info(f"✅ Deep learning model loaded: {model_version}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Error loading deep learning model: {e}")
            return None
    
    def _create_deep_learning_model_from_config(self, config) -> nn.Module:
        """Create deep learning model from saved configuration"""
        if not DL_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Import the model creation functions from training service
        from .ml_model_training_service import MLModelTrainingService
        
        training_service = MLModelTrainingService(self.db_pool)
        
        # Create model based on type
        if config.model_type.value == "lstm":
            return training_service._create_lstm_model(
                input_size=len(config.features),
                hidden_size=config.hyperparameters['hidden_size'],
                num_layers=config.hyperparameters['num_layers'],
                dropout=config.hyperparameters['dropout']
            )
        elif config.model_type.value == "transformer":
            return training_service._create_transformer_model(
                input_size=len(config.features),
                d_model=config.hyperparameters['d_model'],
                nhead=config.hyperparameters['nhead'],
                num_layers=config.hyperparameters['num_layers'],
                dropout=config.hyperparameters['dropout']
            )
        elif config.model_type.value == "cnn":
            return training_service._create_cnn_model(
                input_size=len(config.features),
                num_filters=config.hyperparameters['num_filters'],
                kernel_sizes=config.hyperparameters['kernel_sizes'],
                dropout=config.hyperparameters['dropout']
            )
        else:
            raise ValueError(f"Unsupported deep learning model type: {config.model_type.value}")
    
    async def _make_deep_learning_prediction(self, model: nn.Module, features: Dict, 
                                           active_model: Dict, model_type: str) -> Optional[PredictionResult]:
        """Make prediction using deep learning model"""
        try:
            # Prepare input data
            X = self._prepare_deep_learning_input(features, active_model)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            X_tensor = X_tensor.to(device)
            
            start_time = datetime.now()
            
            with torch.no_grad():
                prediction = model(X_tensor)
                prediction_value = prediction.squeeze().cpu().numpy()
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms
            
            # Calculate confidence score (simplified)
            confidence_score = 0.8  # This could be improved with uncertainty quantification
            
            # Create prediction result
            result = PredictionResult(
                symbol=features.get('symbol', ''),
                timeframe=features.get('timeframe', ''),
                timestamp=datetime.now(),
                model_version=active_model['version'],
                prediction_type=PredictionType.PRICE,
                prediction_value=float(prediction_value),
                confidence_score=confidence_score,
                feature_contributions={},  # Could be calculated with SHAP for deep learning
                shap_values={},
                model_metadata={
                    'model_type': model_type,
                    'inference_latency_ms': inference_time,
                    'gpu_used': torch.cuda.is_available(),
                    'batch_size': 1,
                    'input_sequence_length': X.shape[0]
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error making deep learning prediction: {e}")
            return None
    
    def _prepare_deep_learning_input(self, features: Dict, active_model: Dict) -> np.ndarray:
        """Prepare input data for deep learning model"""
        # Extract feature values
        feature_values = []
        for feature_name in features.keys():
            if feature_name not in ['symbol', 'timeframe', 'timestamp']:
                feature_values.append(features[feature_name])
        
        # Convert to numpy array
        X = np.array(feature_values).reshape(1, -1)
        
        # Normalize features (this should use the same scaler as training)
        # For now, we'll use a simple normalization
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)
        
        return X
    
    async def _store_deep_learning_prediction(self, prediction: PredictionResult):
        """Store deep learning prediction in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store in model_predictions table with deep learning metadata
                query = """
                INSERT INTO model_predictions 
                (symbol, timeframe, timestamp, model_version, prediction_type, prediction_value, 
                 confidence_score, feature_contributions, shap_values, model_metadata,
                 deep_learning_model_version, model_architecture, training_parameters, 
                 inference_latency_ms, gpu_used, batch_size)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """
                
                await conn.execute(query,
                    prediction.symbol,
                    prediction.timeframe,
                    prediction.timestamp,
                    prediction.model_version,
                    prediction.prediction_type.value,
                    prediction.prediction_value,
                    prediction.confidence_score,
                    json.dumps(prediction.feature_contributions),
                    json.dumps(prediction.shap_values),
                    json.dumps(prediction.model_metadata),
                    prediction.model_version,  # deep_learning_model_version
                    json.dumps(prediction.model_metadata.get('model_architecture', {})),
                    json.dumps(prediction.model_metadata.get('training_parameters', {})),
                    prediction.model_metadata.get('inference_latency_ms', 0),
                    prediction.model_metadata.get('gpu_used', False),
                    prediction.model_metadata.get('batch_size', 1)
                )
                
                # Also store in deep_learning_predictions table
                dl_query = """
                INSERT INTO deep_learning_predictions 
                (timestamp, symbol, timeframe, model_type, model_version, prediction_type,
                 prediction_value, confidence_score, input_sequence_length, output_horizon,
                 model_architecture, training_parameters, inference_metadata, attention_weights,
                 feature_contributions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """
                
                await conn.execute(dl_query,
                    prediction.timestamp,
                    prediction.symbol,
                    prediction.timeframe,
                    prediction.model_metadata.get('model_type', 'lstm'),
                    prediction.model_version,
                    prediction.prediction_type.value,
                    prediction.prediction_value,
                    prediction.confidence_score,
                    prediction.model_metadata.get('input_sequence_length', 1),
                    1,  # output_horizon
                    json.dumps(prediction.model_metadata.get('model_architecture', {})),
                    json.dumps(prediction.model_metadata.get('training_parameters', {})),
                    json.dumps(prediction.model_metadata),
                    json.dumps({}),  # attention_weights (for transformer models)
                    json.dumps(prediction.feature_contributions)
                )
                
                self.logger.info(f"💾 Deep learning prediction stored for {prediction.symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error storing deep learning prediction: {e}")
    
    async def _get_model_path(self, model_version: str) -> Optional[str]:
        """Get model file path from database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT model_path FROM model_versions WHERE version = $1
                """
                row = await conn.fetchrow(query, model_version)
                return row['model_path'] if row else None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting model path: {e}")
            return None
