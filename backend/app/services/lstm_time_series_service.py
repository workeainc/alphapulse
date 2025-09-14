#!/usr/bin/env python3
"""
LSTM Time-Series Service for AlphaPulse
Advanced LSTM models for time-series sequence modeling and directional bias prediction
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
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Sequential = None
    load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    BatchNormalization = None
    Adam = None
    EarlyStopping = None
    ModelCheckpoint = None
    logging.warning("TensorFlow not available - using mock models")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score

logger = logging.getLogger(__name__)

@dataclass
class LSTMPrediction:
    """LSTM prediction result"""
    symbol: str
    timestamp: datetime
    prediction_horizon: int  # minutes
    directional_bias: str  # 'bullish', 'bearish', 'neutral'
    confidence_score: float
    price_movement_probability: float
    volatility_forecast: float
    sequence_features: List[float]
    attention_weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

class LSTMTimeSeriesService:
    """Advanced LSTM service for time-series sequence modeling"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.models_dir = self.config.get('models_dir', 'models/lstm')
        self.sequence_length = self.config.get('sequence_length', 60)  # 60 time steps
        self.prediction_horizons = self.config.get('prediction_horizons', [15, 30, 60])  # minutes
        self.update_frequency = self.config.get('update_frequency', 300)  # 5 minutes
        
        # Model storage
        self.lstm_models = {}
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
        """Initialize the LSTM service"""
        self.logger.info("Initializing LSTM Time-Series Service...")
        
        try:
            # Load existing models
            await self._load_existing_models()
            
            # Initialize performance tracking
            self.performance_metrics['last_retraining'] = datetime.now()
            
            self.logger.info("LSTM Time-Series Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing LSTM service: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing LSTM models from disk"""
        try:
            for horizon in self.prediction_horizons:
                model_path = os.path.join(self.models_dir, f'lstm_model_{horizon}.h5')
                scaler_path = os.path.join(self.models_dir, f'lstm_scaler_{horizon}.pkl')
                feature_scaler_path = os.path.join(self.models_dir, f'lstm_feature_scaler_{horizon}.pkl')
                
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    self.lstm_models[horizon] = model
                    self.logger.info(f"Loaded LSTM model for {horizon}min horizon")
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.scalers[horizon] = scaler
                
                if os.path.exists(feature_scaler_path):
                    feature_scaler = joblib.load(feature_scaler_path)
                    self.feature_scalers[horizon] = feature_scaler
                    
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def predict_directional_bias(self, symbol: str, market_data: Dict[str, Any]) -> LSTMPrediction:
        """Predict directional bias using LSTM models"""
        try:
            start_time = datetime.now()
            
            # Engineer sequence features
            sequence_features = await self._engineer_sequence_features(symbol, market_data)
            
            # Get predictions for different horizons
            predictions = {}
            attention_weights = {}
            
            for horizon in self.prediction_horizons:
                if horizon in self.lstm_models:
                    prediction, attention = await self._make_lstm_prediction(symbol, sequence_features, horizon)
                    predictions[horizon] = prediction
                    attention_weights[horizon] = attention
                else:
                    # Train new model if not available
                    await self._train_lstm_model(symbol, horizon)
                    prediction, attention = await self._make_lstm_prediction(symbol, sequence_features, horizon)
                    predictions[horizon] = prediction
                    attention_weights[horizon] = attention
            
            # Aggregate predictions
            if predictions:
                # Weighted average (shorter horizons get higher weight)
                weights = [1.0 / h for h in predictions.keys()]
                total_weight = sum(weights)
                
                avg_prediction = sum(pred * weight for pred, weight in zip(predictions.values(), weights)) / total_weight
                confidence_score = float(1 - np.std(list(predictions.values()))) if len(predictions) > 1 else 0.7
                
                # Determine directional bias
                if avg_prediction > 0.6:
                    directional_bias = 'bullish'
                elif avg_prediction < 0.4:
                    directional_bias = 'bearish'
                else:
                    directional_bias = 'neutral'
                
                # Calculate volatility forecast
                volatility_forecast = np.std(list(predictions.values()))
                
            else:
                # Fallback values
                avg_prediction = 0.5
                confidence_score = 0.3
                directional_bias = 'neutral'
                volatility_forecast = 0.2
            
            # Update performance metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['predictions_made'] += 1
            self.performance_metrics['average_prediction_time'] = (
                (self.performance_metrics['average_prediction_time'] * (self.performance_metrics['predictions_made'] - 1) + prediction_time) 
                / self.performance_metrics['predictions_made']
            )
            
            # Store prediction in database
            await self._store_lstm_prediction(symbol, directional_bias, avg_prediction, confidence_score, sequence_features)
            
            return LSTMPrediction(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction_horizon=max(self.prediction_horizons),
                directional_bias=directional_bias,
                confidence_score=confidence_score,
                price_movement_probability=avg_prediction,
                volatility_forecast=volatility_forecast,
                sequence_features=sequence_features,
                attention_weights=list(attention_weights.values()) if attention_weights else None,
                metadata={
                    'horizon_predictions': predictions,
                    'prediction_time': prediction_time,
                    'model_versions': {h: self._get_model_version(h) for h in self.prediction_horizons}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting directional bias for {symbol}: {e}")
            return self._get_default_lstm_prediction(symbol)
    
    async def _engineer_sequence_features(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Engineer sequence features for LSTM input"""
        try:
            # Get historical sequence data
            sequence_data = await self._get_historical_sequence(symbol)
            
            # Extract features from sequence
            features = []
            
            if len(sequence_data) >= self.sequence_length:
                # Price-based features
                prices = [row['price'] for row in sequence_data[-self.sequence_length:]]
                volumes = [row['volume'] for row in sequence_data[-self.sequence_length:]]
                
                # Technical indicators
                features.extend([
                    np.mean(prices),  # Average price
                    np.std(prices),   # Price volatility
                    (prices[-1] - prices[0]) / prices[0],  # Price change
                    np.mean(volumes),  # Average volume
                    np.std(volumes),   # Volume volatility
                ])
                
                # Price momentum features
                for i in range(1, len(prices)):
                    features.append((prices[i] - prices[i-1]) / prices[i-1])
                
                # Volume momentum features
                for i in range(1, len(volumes)):
                    features.append((volumes[i] - volumes[i-1]) / volumes[i-1])
                
                # Market microstructure features
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
                features = [0.0] * 50  # Default feature vector
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering sequence features for {symbol}: {e}")
            return [0.0] * 50
    
    async def _get_historical_sequence(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical sequence data for LSTM input"""
        try:
            with self.engine.connect() as conn:
                # Get recent order book snapshots
                result = conn.execute(text("""
                    SELECT timestamp, bid_price, ask_price, spread, liquidity_imbalance, 
                           depth_pressure, order_flow_toxicity
                    FROM order_book_snapshots 
                    WHERE symbol = :symbol 
                    ORDER BY timestamp DESC 
                    LIMIT :sequence_length
                """), {
                    "symbol": symbol,
                    "sequence_length": self.sequence_length
                })
                
                sequence_data = []
                for row in result:
                    sequence_data.append({
                        'timestamp': row[0],
                        'price': (row[1] + row[2]) / 2,  # Mid price
                        'volume': row[3] * 1000,  # Spread-based volume proxy
                        'liquidity_imbalance': row[4],
                        'depth_pressure': row[5],
                        'order_flow_toxicity': row[6]
                    })
                
                return sequence_data
                
        except Exception as e:
            self.logger.error(f"Error getting historical sequence for {symbol}: {e}")
            return []
    
    async def _make_lstm_prediction(self, symbol: str, features: List[float], horizon: int) -> Tuple[float, List[float]]:
        """Make LSTM prediction for a specific horizon"""
        try:
            if horizon not in self.lstm_models:
                return 0.5, []  # Default prediction
            
            model = self.lstm_models[horizon]
            scaler = self.feature_scalers.get(horizon)
            
            # Prepare input
            features_array = np.array(features).reshape(1, -1)
            
            if scaler:
                features_scaled = scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Reshape for LSTM (batch_size, timesteps, features)
            # For now, we'll use a simple approach with the features as a single timestep
            features_reshaped = features_scaled.reshape(1, 1, -1)
            
            # Make prediction
            prediction = model.predict(features_reshaped, verbose=0)[0][0]
            
            # Get attention weights (if available)
            attention_weights = []
            try:
                # Try to extract attention weights from intermediate layers
                layer_outputs = [layer.output for layer in model.layers if 'lstm' in layer.name.lower()]
                if layer_outputs:
                    attention_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                    attention_outputs = attention_model.predict(features_reshaped, verbose=0)
                    attention_weights = attention_outputs[0].flatten().tolist() if len(attention_outputs) > 0 else []
            except:
                attention_weights = []
            
            return float(prediction), attention_weights
            
        except Exception as e:
            self.logger.error(f"Error making LSTM prediction for {symbol} horizon {horizon}: {e}")
            return 0.5, []
    
    async def _train_lstm_model(self, symbol: str, horizon: int):
        """Train LSTM model for a specific horizon"""
        try:
            self.logger.info(f"Training LSTM model for {symbol} horizon {horizon}min...")
            
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
            model = self._create_lstm_model(X.shape[2])
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(self.models_dir, f'lstm_model_{horizon}.h5'),
                    save_best_only=True
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Save model and scalers
            model.save(os.path.join(self.models_dir, f'lstm_model_{horizon}.h5'))
            
            # Create and save scalers
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
            joblib.dump(feature_scaler, os.path.join(self.models_dir, f'lstm_feature_scaler_{horizon}.pkl'))
            
            # Store model version
            await self._store_model_version(symbol, horizon, 'lstm', len(training_data))
            
            # Update performance metrics
            self.performance_metrics['model_retraining_count'] += 1
            self.performance_metrics['last_retraining'] = datetime.now()
            
            # Store model in memory
            self.lstm_models[horizon] = model
            self.feature_scalers[horizon] = feature_scaler
            
            self.logger.info(f"✅ LSTM model trained for {symbol} horizon {horizon}min")
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model for {symbol} horizon {horizon}min: {e}")
    
    def _create_lstm_model(self, input_features: int) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(1, input_features)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def _get_training_data(self, symbol: str, horizon: int) -> List[Dict[str, Any]]:
        """Get training data for LSTM model"""
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
                            WHEN future_obs.bid_price > obs.bid_price * 1.001 THEN 1
                            WHEN future_obs.bid_price < obs.bid_price * 0.999 THEN 0
                            ELSE 0.5
                        END as price_movement
                    FROM order_book_snapshots obs
                    LEFT JOIN order_book_snapshots future_obs ON
                        future_obs.symbol = obs.symbol
                        AND future_obs.timestamp = obs.timestamp + INTERVAL ':horizon minutes'
                    WHERE obs.symbol = :symbol
                    AND obs.timestamp >= NOW() - INTERVAL '7 days'
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
        """Prepare training data for LSTM"""
        try:
            sequences = []
            targets = []
            
            for i in range(len(data) - self.sequence_length):
                sequence = data[i:i + self.sequence_length]
                target = data[i + self.sequence_length]['price_movement']
                
                # Extract features from sequence
                sequence_features = []
                for point in sequence:
                    features = [
                        point['bid_price'],
                        point['ask_price'],
                        point['spread'],
                        point['liquidity_imbalance'],
                        point['depth_pressure'],
                        point['order_flow_toxicity']
                    ]
                    sequence_features.extend(features)
                
                sequences.append(sequence_features)
                targets.append(target)
            
            X = np.array(sequences).reshape(-1, 1, len(sequences[0]))
            y = np.array(targets)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    async def _store_lstm_prediction(self, symbol: str, directional_bias: str, probability: float, confidence: float, features: List[float]):
        """Store LSTM prediction in database"""
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
                    "model_type": "lstm",
                    "prediction_type": "price_direction",
                    "prediction_value": probability,
                    "confidence_score": confidence,
                    "prediction_metadata": json.dumps({
                        "directional_bias": directional_bias,
                        "model_version": "lstm_v1"
                    }),
                    "feature_vector": json.dumps(features)
                })
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing LSTM prediction: {e}")
    
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
        return f"lstm_v1_h{horizon}"
    
    def _get_default_lstm_prediction(self, symbol: str) -> LSTMPrediction:
        """Get default LSTM prediction"""
        return LSTMPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction_horizon=max(self.prediction_horizons),
            directional_bias='neutral',
            confidence_score=0.3,
            price_movement_probability=0.5,
            volatility_forecast=0.2,
            sequence_features=[0.0] * 50,
            metadata={'error': 'fallback_prediction'}
        )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'models_loaded': len(self.lstm_models),
            'prediction_horizons': self.prediction_horizons,
            'sequence_length': self.sequence_length
        }
