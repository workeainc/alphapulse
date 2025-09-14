#!/usr/bin/env python3
"""
Transformer Service for AlphaPulse
Advanced Transformer models for cross-timeframe dependencies with attention mechanisms
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

# Deep Learning imports with conditional handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
    load_model = None
    Dense = None
    Dropout = None
    LayerNormalization = None
    MultiHeadAttention = None
    GlobalAveragePooling1D = None
    Adam = None
    EarlyStopping = None
    ModelCheckpoint = None
    logging.warning("TensorFlow not available - using mock models")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score

logger = logging.getLogger(__name__)

@dataclass
class TransformerPrediction:
    """Transformer prediction result"""
    symbol: str
    timestamp: datetime
    prediction_horizon: int  # minutes
    cross_timeframe_signal: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    confidence_score: float
    attention_weights: Dict[str, List[float]]  # Attention weights for each timeframe
    context_embedding: List[float]
    market_regime: str  # 'trending', 'ranging', 'volatile'
    metadata: Dict[str, Any] = None

class TransformerService:
    """Advanced Transformer service for cross-timeframe analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.models_dir = self.config.get('models_dir', 'models/transformer')
        self.timeframes = self.config.get('timeframes', ['15m', '1h', '4h'])  # Multi-timeframe analysis
        self.sequence_length = self.config.get('sequence_length', 48)  # 48 time steps per timeframe
        self.prediction_horizons = self.config.get('prediction_horizons', [30, 60, 240])  # minutes
        self.update_frequency = self.config.get('update_frequency', 600)  # 10 minutes
        
        # Model storage
        self.transformer_models = {}
        self.scalers = {}
        self.feature_scalers = {}
        
        # Database connection
        self.database_url = self.config.get('database_url', os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"))
        self.engine = create_engine(self.database_url)
        
        # Performance metrics
        self.performance_metrics = {
            'predictions_made': 0,
            'accuracy_scores': [],
            'model_retraining_count': 0,
            'last_retraining': None,
            'average_prediction_time': 0.0
        }
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # TensorFlow configuration
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True) if tf.config.experimental.list_physical_devices('GPU') else None
        
    async def initialize(self):
        """Initialize the Transformer service"""
        self.logger.info("Initializing Transformer Service...")
        
        try:
            # Load existing models
            await self._load_existing_models()
            
            # Initialize performance tracking
            self.performance_metrics['last_retraining'] = datetime.now()
            
            self.logger.info("Transformer Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Transformer service: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing Transformer models from disk"""
        try:
            for horizon in self.prediction_horizons:
                model_path = os.path.join(self.models_dir, f'transformer_model_{horizon}.h5')
                scaler_path = os.path.join(self.models_dir, f'transformer_scaler_{horizon}.pkl')
                feature_scaler_path = os.path.join(self.models_dir, f'transformer_feature_scaler_{horizon}.pkl')
                
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    self.transformer_models[horizon] = model
                    self.logger.info(f"Loaded Transformer model for {horizon}min horizon")
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.scalers[horizon] = scaler
                
                if os.path.exists(feature_scaler_path):
                    feature_scaler = joblib.load(feature_scaler_path)
                    self.feature_scalers[horizon] = feature_scaler
                    
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def predict_cross_timeframe_signal(self, symbol: str, market_data: Dict[str, Any]) -> TransformerPrediction:
        """Predict cross-timeframe signals using Transformer models"""
        try:
            start_time = datetime.now()
            
            # Engineer multi-timeframe features
            multi_timeframe_features = await self._engineer_multi_timeframe_features(symbol, market_data)
            
            # Get predictions for different horizons
            predictions = {}
            attention_weights = {}
            
            for horizon in self.prediction_horizons:
                if horizon in self.transformer_models:
                    prediction, attention = await self._make_transformer_prediction(symbol, multi_timeframe_features, horizon)
                    predictions[horizon] = prediction
                    attention_weights[horizon] = attention
                else:
                    # Train new model if not available
                    await self._train_transformer_model(symbol, horizon)
                    prediction, attention = await self._make_transformer_prediction(symbol, multi_timeframe_features, horizon)
                    predictions[horizon] = prediction
                    attention_weights[horizon] = attention
            
            # Aggregate predictions
            if predictions:
                # Weighted average (shorter horizons get higher weight)
                weights = [1.0 / h for h in predictions.keys()]
                total_weight = sum(weights)
                
                avg_prediction = sum(pred * weight for pred, weight in zip(predictions.values(), weights)) / total_weight
                confidence_score = float(1 - np.std(list(predictions.values()))) if len(predictions) > 1 else 0.7
                
                # Determine cross-timeframe signal
                if avg_prediction > 0.8:
                    cross_timeframe_signal = 'strong_buy'
                elif avg_prediction > 0.6:
                    cross_timeframe_signal = 'buy'
                elif avg_prediction > 0.4:
                    cross_timeframe_signal = 'hold'
                elif avg_prediction > 0.2:
                    cross_timeframe_signal = 'sell'
                else:
                    cross_timeframe_signal = 'strong_sell'
                
                # Determine market regime
                volatility = np.std(list(predictions.values()))
                if volatility > 0.3:
                    market_regime = 'volatile'
                elif avg_prediction > 0.7 or avg_prediction < 0.3:
                    market_regime = 'trending'
                else:
                    market_regime = 'ranging'
                
            else:
                # Fallback values
                avg_prediction = 0.5
                confidence_score = 0.3
                cross_timeframe_signal = 'hold'
                market_regime = 'ranging'
            
            # Create context embedding from multi-timeframe features
            context_embedding = self._create_context_embedding(multi_timeframe_features)
            
            # Update performance metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['predictions_made'] += 1
            self.performance_metrics['average_prediction_time'] = (
                (self.performance_metrics['average_prediction_time'] * (self.performance_metrics['predictions_made'] - 1) + prediction_time) 
                / self.performance_metrics['predictions_made']
            )
            
            # Store prediction in database
            await self._store_transformer_prediction(symbol, cross_timeframe_signal, avg_prediction, confidence_score, multi_timeframe_features)
            
            return TransformerPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_horizon=max(self.prediction_horizons),
                cross_timeframe_signal=cross_timeframe_signal,
                confidence_score=confidence_score,
                attention_weights=attention_weights,
                context_embedding=context_embedding,
                market_regime=market_regime,
                metadata={
                    'horizon_predictions': predictions,
                    'prediction_time': prediction_time,
                    'model_versions': {h: self._get_model_version(h) for h in self.prediction_horizons},
                    'timeframes_analyzed': self.timeframes
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting cross-timeframe signal for {symbol}: {e}")
            return self._get_default_transformer_prediction(symbol)
    
    async def _engineer_multi_timeframe_features(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Engineer multi-timeframe features for Transformer input"""
        try:
            multi_timeframe_features = {}
            
            for timeframe in self.timeframes:
                # Get historical data for this timeframe
                timeframe_data = await self._get_timeframe_data(symbol, timeframe)
                
                # Extract features from timeframe data
                features = self._extract_timeframe_features(timeframe_data, market_data)
                multi_timeframe_features[timeframe] = features
            
            return multi_timeframe_features
            
        except Exception as e:
            self.logger.error(f"Error engineering multi-timeframe features for {symbol}: {e}")
            return {tf: [0.0] * 50 for tf in self.timeframes}
    
    async def _get_timeframe_data(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get historical data for a specific timeframe"""
        try:
            # Convert timeframe to minutes
            timeframe_minutes = {
                '15m': 15,
                '1h': 60,
                '4h': 240
            }.get(timeframe, 60)
            
            with self.engine.connect() as conn:
                # Get aggregated data for the timeframe
                result = conn.execute(text("""
                    SELECT 
                        DATE_TRUNC('hour', timestamp) + 
                        INTERVAL ':timeframe_minutes minutes' * 
                        (EXTRACT(MINUTE FROM timestamp) / :timeframe_minutes) as timeframe_timestamp,
                        AVG(bid_price) as avg_bid_price,
                        AVG(ask_price) as avg_ask_price,
                        AVG(spread) as avg_spread,
                        AVG(liquidity_imbalance) as avg_liquidity_imbalance,
                        AVG(depth_pressure) as avg_depth_pressure,
                        AVG(order_flow_toxicity) as avg_order_flow_toxicity,
                        COUNT(*) as data_points
                    FROM order_book_snapshots 
                    WHERE symbol = :symbol 
                    AND timestamp >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE_TRUNC('hour', timestamp) + 
                             INTERVAL ':timeframe_minutes minutes' * 
                             (EXTRACT(MINUTE FROM timestamp) / :timeframe_minutes)
                    ORDER BY timeframe_timestamp DESC 
                    LIMIT :sequence_length
                """), {
                    "symbol": symbol,
                    "timeframe_minutes": timeframe_minutes,
                    "sequence_length": self.sequence_length
                })
                
                timeframe_data = []
                for row in result:
                    timeframe_data.append({
                        'timestamp': row[0],
                        'avg_bid_price': row[1],
                        'avg_ask_price': row[2],
                        'avg_spread': row[3],
                        'avg_liquidity_imbalance': row[4],
                        'avg_depth_pressure': row[5],
                        'avg_order_flow_toxicity': row[6],
                        'data_points': row[7]
                    })
                
                return timeframe_data
                
        except Exception as e:
            self.logger.error(f"Error getting timeframe data for {symbol} {timeframe}: {e}")
            return []
    
    def _extract_timeframe_features(self, timeframe_data: List[Dict[str, Any]], market_data: Dict[str, Any]) -> List[float]:
        """Extract features from timeframe data"""
        try:
            features = []
            
            if len(timeframe_data) >= 10:  # Need minimum data points
                # Price-based features
                prices = [row['avg_bid_price'] for row in timeframe_data]
                spreads = [row['avg_spread'] for row in timeframe_data]
                
                # Technical indicators
                features.extend([
                    np.mean(prices),  # Average price
                    np.std(prices),   # Price volatility
                    (prices[0] - prices[-1]) / prices[-1] if prices[-1] > 0 else 0,  # Price change
                    np.mean(spreads),  # Average spread
                    np.std(spreads),   # Spread volatility
                ])
                
                # Momentum features
                for i in range(1, min(len(prices), 10)):
                    features.append((prices[i-1] - prices[i]) / prices[i] if prices[i] > 0 else 0)
                
                # Liquidity features
                liquidity_imbalances = [row['avg_liquidity_imbalance'] for row in timeframe_data]
                depth_pressures = [row['avg_depth_pressure'] for row in timeframe_data]
                
                features.extend([
                    np.mean(liquidity_imbalances),
                    np.std(liquidity_imbalances),
                    np.mean(depth_pressures),
                    np.std(depth_pressures)
                ])
                
                # Market microstructure features
                order_flow_toxicities = [row['avg_order_flow_toxicity'] for row in timeframe_data]
                features.extend([
                    np.mean(order_flow_toxicities),
                    np.std(order_flow_toxicities)
                ])
                
                # Market data features
                if market_data:
                    features.extend([
                        market_data.get('historical_volatility', 0.0),
                        market_data.get('price_change_24h', 0.0) / 100.0,
                        market_data.get('volume_24h', 0.0) / 1000000.0,
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                # Normalize features
                features = [max(-10.0, min(10.0, f)) if np.isfinite(f) else 0.0 for f in features]
                
            else:
                # Fallback features if insufficient data
                features = [0.0] * 50
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting timeframe features: {e}")
            return [0.0] * 50
    
    async def _make_transformer_prediction(self, symbol: str, multi_timeframe_features: Dict[str, List[float]], horizon: int) -> Tuple[float, Dict[str, List[float]]]:
        """Make Transformer prediction for a specific horizon"""
        try:
            if horizon not in self.transformer_models:
                return 0.5, {}  # Default prediction
            
            model = self.transformer_models[horizon]
            scaler = self.feature_scalers.get(horizon)
            
            # Prepare multi-timeframe input
            combined_features = []
            for timeframe in self.timeframes:
                if timeframe in multi_timeframe_features:
                    combined_features.extend(multi_timeframe_features[timeframe])
                else:
                    combined_features.extend([0.0] * 50)  # Default features
            
            # Prepare input
            features_array = np.array(combined_features).reshape(1, -1)
            
            if scaler:
                features_scaled = scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Reshape for Transformer (batch_size, timesteps, features)
            # For multi-timeframe, we'll use the timeframes as timesteps
            features_reshaped = features_scaled.reshape(1, len(self.timeframes), -1)
            
            # Make prediction
            prediction = model.predict(features_reshaped, verbose=0)[0][0]
            
            # Get attention weights for each timeframe
            attention_weights = {}
            try:
                # Extract attention weights from the model
                attention_layers = [layer for layer in model.layers if 'multi_head_attention' in layer.name.lower()]
                if attention_layers:
                    attention_layer = attention_layers[0]
                    attention_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.output)
                    attention_outputs = attention_model.predict(features_reshaped, verbose=0)
                    
                    # Process attention weights for each timeframe
                    for i, timeframe in enumerate(self.timeframes):
                        if len(attention_outputs) > 0:
                            attention_weights[timeframe] = attention_outputs[0][i].flatten().tolist()
                        else:
                            attention_weights[timeframe] = [1.0 / len(self.timeframes)] * len(self.timeframes)
            except:
                # Fallback attention weights
                attention_weights = {tf: [1.0 / len(self.timeframes)] * len(self.timeframes) for tf in self.timeframes}
            
            return float(prediction), attention_weights
            
        except Exception as e:
            self.logger.error(f"Error making Transformer prediction for {symbol} horizon {horizon}: {e}")
            return 0.5, {}
    
    async def _train_transformer_model(self, symbol: str, horizon: int):
        """Train Transformer model for a specific horizon"""
        try:
            self.logger.info(f"Training Transformer model for {symbol} horizon {horizon}min...")
            
            # Get training data
            training_data = await self._get_training_data(symbol, horizon)
            
            if len(training_data) < 100:  # Need sufficient data
                self.logger.warning(f"Insufficient training data for {symbol} horizon {horizon}min")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) < 50:  # Need sufficient sequences
                self.logger.warning(f"Insufficient sequences for {symbol} horizon {horizon}min")
                return
            
            # Create and train model
            model = self._create_transformer_model(X.shape[2])
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(self.models_dir, f'transformer_model_{horizon}.h5'),
                    save_best_only=True
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=16,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model and scalers
            model.save(os.path.join(self.models_dir, f'transformer_model_{horizon}.h5'))
            
            # Create and save scalers
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
            joblib.dump(feature_scaler, os.path.join(self.models_dir, f'transformer_feature_scaler_{horizon}.pkl'))
            
            # Store model version
            await self._store_model_version(symbol, horizon, 'transformer', len(training_data))
            
            # Update performance metrics
            self.performance_metrics['model_retraining_count'] += 1
            self.performance_metrics['last_retraining'] = datetime.now()
            
            # Store model in memory
            self.transformer_models[horizon] = model
            self.feature_scalers[horizon] = feature_scaler
            
            self.logger.info(f"✅ Transformer model trained for {symbol} horizon {horizon}min")
            
        except Exception as e:
            self.logger.error(f"Error training Transformer model for {symbol} horizon {horizon}min: {e}")
    
    def _create_transformer_model(self, input_features: int) -> Sequential:
        """Create Transformer model architecture"""
        model = Sequential([
            # Input layer
            Dense(128, input_shape=(len(self.timeframes), input_features)),
            LayerNormalization(),
            
            # Multi-head attention layer
            MultiHeadAttention(num_heads=8, key_dim=16),
            Dropout(0.1),
            LayerNormalization(),
            
            # Feed-forward network
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            
            # Global average pooling
            GlobalAveragePooling1D(),
            
            # Output layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def _get_training_data(self, symbol: str, horizon: int) -> List[Dict[str, Any]]:
        """Get training data for Transformer model"""
        try:
            with self.engine.connect() as conn:
                # Get historical data with future price movements
                result = conn.execute(text("""
                    SELECT 
                        obs.timestamp,
                        obs.bid_price,
                        obs.ask_price,
                        obs.spread,
                        obs.liquidity_imbalance,
                        obs.depth_pressure,
                        obs.order_flow_toxicity,
                        -- Future price movement (target)
                        CASE 
                            WHEN future_obs.bid_price > obs.bid_price * 1.002 THEN 1
                            WHEN future_obs.bid_price < obs.bid_price * 0.998 THEN 0
                            ELSE 0.5
                        END as price_movement
                    FROM order_book_snapshots obs
                    LEFT JOIN order_book_snapshots future_obs ON
                        future_obs.symbol = obs.symbol
                        AND future_obs.timestamp = obs.timestamp + INTERVAL ':horizon minutes'
                    WHERE obs.symbol = :symbol
                    AND obs.timestamp >= NOW() - INTERVAL '14 days'
                    ORDER BY obs.timestamp
                """), {
                    "symbol": symbol,
                    "horizon": horizon
                })
                
                training_data = []
                for row in result:
                    training_data.append({
                        'timestamp': row[0],
                        'bid_price': row[1],
                        'ask_price': row[2],
                        'spread': row[3],
                        'liquidity_imbalance': row[4],
                        'depth_pressure': row[5],
                        'order_flow_toxicity': row[6],
                        'price_movement': row[7]
                    })
                
                return training_data
                
        except Exception as e:
            self.logger.error(f"Error getting training data for {symbol}: {e}")
            return []
    
    def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for Transformer"""
        try:
            sequences = []
            targets = []
            
            for i in range(len(data) - self.sequence_length):
                sequence = data[i:i + self.sequence_length]
                target = data[i + self.sequence_length]['price_movement']
                
                # Extract features from sequence for each timeframe
                timeframe_features = {}
                for timeframe in self.timeframes:
                    timeframe_data = self._aggregate_for_timeframe(sequence, timeframe)
                    timeframe_features[timeframe] = self._extract_timeframe_features(timeframe_data, {})
                
                # Combine features from all timeframes
                combined_features = []
                for timeframe in self.timeframes:
                    combined_features.extend(timeframe_features.get(timeframe, [0.0] * 50))
                
                sequences.append(combined_features)
                targets.append(target)
            
            X = np.array(sequences).reshape(-1, len(self.timeframes), -1)
            y = np.array(targets)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def _aggregate_for_timeframe(self, sequence: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
        """Aggregate sequence data for a specific timeframe"""
        try:
            # Simple aggregation - in practice, you'd want more sophisticated aggregation
            aggregated = []
            chunk_size = max(1, len(sequence) // 10)  # Divide into chunks
            
            for i in range(0, len(sequence), chunk_size):
                chunk = sequence[i:i + chunk_size]
                if chunk:
                    aggregated.append({
                        'avg_bid_price': np.mean([p['bid_price'] for p in chunk]),
                        'avg_ask_price': np.mean([p['ask_price'] for p in chunk]),
                        'avg_spread': np.mean([p['spread'] for p in chunk]),
                        'avg_liquidity_imbalance': np.mean([p['liquidity_imbalance'] for p in chunk]),
                        'avg_depth_pressure': np.mean([p['depth_pressure'] for p in chunk]),
                        'avg_order_flow_toxicity': np.mean([p['order_flow_toxicity'] for p in chunk])
                    })
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating for timeframe {timeframe}: {e}")
            return []
    
    def _create_context_embedding(self, multi_timeframe_features: Dict[str, List[float]]) -> List[float]:
        """Create context embedding from multi-timeframe features"""
        try:
            # Combine features from all timeframes with attention weights
            context_embedding = []
            
            for timeframe in self.timeframes:
                if timeframe in multi_timeframe_features:
                    # Weight features by timeframe importance
                    weight = 1.0 / (self.timeframes.index(timeframe) + 1)  # Shorter timeframes get higher weight
                    weighted_features = [f * weight for f in multi_timeframe_features[timeframe]]
                    context_embedding.extend(weighted_features)
                else:
                    context_embedding.extend([0.0] * 50)
            
            # Normalize embedding
            if context_embedding:
                max_val = max(abs(f) for f in context_embedding)
                if max_val > 0:
                    context_embedding = [f / max_val for f in context_embedding]
            
            return context_embedding
            
        except Exception as e:
            self.logger.error(f"Error creating context embedding: {e}")
            return [0.0] * (len(self.timeframes) * 50)
    
    async def _store_transformer_prediction(self, symbol: str, signal: str, probability: float, confidence: float, features: Dict[str, List[float]]):
        """Store Transformer prediction in database"""
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
                    "model_type": "transformer",
                    "prediction_type": "price_direction",
                    "prediction_value": probability,
                    "confidence_score": confidence,
                    "prediction_metadata": json.dumps({
                        "cross_timeframe_signal": signal,
                        "model_version": "transformer_v1",
                        "timeframes_analyzed": self.timeframes
                    }),
                    "feature_vector": json.dumps(features)
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing Transformer prediction: {e}")
    
    async def _store_model_version(self, symbol: str, horizon: int, model_type: str, training_samples: int):
        """Store model version information"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO ml_model_versions (
                        symbol, model_type, version, horizon_minutes, training_samples,
                        accuracy_score, created_at, is_active
                    ) VALUES (
                        :symbol, :model_type, :version, :horizon_minutes, :training_samples,
                        :accuracy_score, :created_at, :is_active
                    )
                """), {
                    "symbol": symbol,
                    "model_type": model_type,
                    "version": f"{model_type}_v1",
                    "horizon_minutes": horizon,
                    "training_samples": training_samples,
                    "accuracy_score": 0.7,  # Placeholder
                    "created_at": datetime.now(),
                    "is_active": True
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing model version: {e}")
    
    def _get_model_version(self, horizon: int) -> str:
        """Get model version for a horizon"""
        return f"transformer_v1_h{horizon}"
    
    def _get_default_transformer_prediction(self, symbol: str) -> TransformerPrediction:
        """Get default Transformer prediction"""
        return TransformerPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction_horizon=max(self.prediction_horizons),
            cross_timeframe_signal='hold',
            confidence_score=0.3,
            attention_weights={tf: [1.0 / len(self.timeframes)] * len(self.timeframes) for tf in self.timeframes},
            context_embedding=[0.0] * (len(self.timeframes) * 50),
            market_regime='ranging',
            metadata={'error': 'fallback_prediction'}
        )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'models_loaded': len(self.transformer_models),
            'prediction_horizons': self.prediction_horizons,
            'timeframes_analyzed': self.timeframes,
            'sequence_length': self.sequence_length
        }
