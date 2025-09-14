"""
Deep Learning Engine for AlphaPulse
Comprehensive implementation of LSTM and CNN models for pattern recognition and signal enhancement
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    Model = None
    LSTM = None
    Dense = None
    Dropout = None
    Conv1D = None
    MaxPooling1D = None
    Flatten = None
    Input = None
    Concatenate = None
    Adam = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    to_categorical = None
    logging.warning("TensorFlow not available - using mock models")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available - using TensorFlow only")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Deep Learning model types"""
    LSTM = "lstm"
    CNN = "cnn"
    LSTM_CNN = "lstm_cnn"
    TRANSFORMER = "transformer"

class PredictionType(Enum):
    """Prediction types"""
    PRICE_DIRECTION = "price_direction"
    PATTERN_RECOGNITION = "pattern_recognition"
    SIGNAL_STRENGTH = "signal_strength"
    VOLATILITY = "volatility"

@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    model_type: ModelType
    sequence_length: int = 60
    features: int = 10
    lstm_units: int = 50
    cnn_filters: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2

@dataclass
class PredictionResult:
    """Result of deep learning prediction"""
    symbol: str
    timestamp: datetime
    model_type: ModelType
    prediction_type: PredictionType
    prediction: float
    confidence: float
    features_used: List[str]
    model_metadata: Dict[str, Any]

class LSTMModel:
    """LSTM model for time series prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - returning mock model")
            return None
            
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.lstm_units, return_sequences=False),
            Dropout(self.config.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.config.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.config.sequence_length:i])
            y.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)  # Price direction
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model"""
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) < 100:
                logger.warning("Insufficient data for LSTM training")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'history': history.history,
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> float:
        """Make prediction with LSTM model"""
        if not self.is_trained or self.model is None:
            return 0.5
        
        try:
            # Prepare data
            scaled_data = self.scaler.transform(data)
            X = scaled_data[-self.config.sequence_length:].reshape(1, -1, scaled_data.shape[1])
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return 0.5

class CNNModel:
    """CNN model for pattern recognition"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build CNN model"""
        model = Sequential([
            Conv1D(self.config.cnn_filters, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(self.config.cnn_filters * 2, 3, activation='relu'),
            MaxPooling1D(2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CNN model"""
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.config.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.config.sequence_length:i])
            y.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)  # Price direction
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train CNN model"""
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) < 100:
                logger.warning("Insufficient data for CNN training")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'history': history.history,
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> float:
        """Make prediction with CNN model"""
        if not self.is_trained or self.model is None:
            return 0.5
        
        try:
            # Prepare data
            scaled_data = self.scaler.transform(data)
            X = scaled_data[-self.config.sequence_length:].reshape(1, -1, scaled_data.shape[1])
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making CNN prediction: {e}")
            return 0.5

class LSTMCNNModel:
    """Hybrid LSTM-CNN model for advanced pattern recognition"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build hybrid LSTM-CNN model"""
        input_layer = Input(shape=input_shape)
        
        # LSTM branch
        lstm_branch = LSTM(self.config.lstm_units, return_sequences=True)(input_layer)
        lstm_branch = Dropout(self.config.dropout_rate)(lstm_branch)
        lstm_branch = LSTM(self.config.lstm_units, return_sequences=False)(lstm_branch)
        lstm_branch = Dropout(self.config.dropout_rate)(lstm_branch)
        
        # CNN branch
        cnn_branch = Conv1D(self.config.cnn_filters, 3, activation='relu')(input_layer)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Conv1D(self.config.cnn_filters * 2, 3, activation='relu')(cnn_branch)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        
        # Combine branches
        combined = Concatenate()([lstm_branch, cnn_branch])
        combined = Dense(50, activation='relu')(combined)
        combined = Dropout(self.config.dropout_rate)(combined)
        output = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for hybrid model"""
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.config.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.config.sequence_length:i])
            y.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)  # Price direction
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train hybrid model"""
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if len(X) < 100:
                logger.warning("Insufficient data for hybrid model training")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'success': True,
                'history': history.history,
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training hybrid model: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: pd.DataFrame) -> float:
        """Make prediction with hybrid model"""
        if not self.is_trained or self.model is None:
            return 0.5
        
        try:
            # Prepare data
            scaled_data = self.scaler.transform(data)
            X = scaled_data[-self.config.sequence_length:].reshape(1, -1, scaled_data.shape[1])
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making hybrid model prediction: {e}")
            return 0.5

class DeepLearningEngine:
    """Deep Learning engine for pattern recognition and signal enhancement"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Check if deep learning is available
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("⚠️ TensorFlow not available - Deep Learning features disabled")
            self.deep_learning_available = False
        else:
            self.deep_learning_available = True
            self.logger.info("✅ Deep Learning engine initialized with TensorFlow")
        
        # Model configurations
        self.model_configs = {
            ModelType.LSTM: ModelConfig(
                model_type=ModelType.LSTM,
                sequence_length=60,
                features=10,
                lstm_units=50,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=50
            ),
            ModelType.CNN: ModelConfig(
                model_type=ModelType.CNN,
                sequence_length=60,
                features=10,
                cnn_filters=32,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=50
            ),
            ModelType.LSTM_CNN: ModelConfig(
                model_type=ModelType.LSTM_CNN,
                sequence_length=60,
                features=10,
                lstm_units=50,
                cnn_filters=32,
                dropout_rate=0.2,
                learning_rate=0.001,
                batch_size=32,
                epochs=50
            )
        }
        
        # Initialize models
        self.models = {}
        self.training_history = {}
        
        # Performance tracking
        self.stats = {
            'models_trained': 0,
            'predictions_made': 0,
            'avg_accuracy': 0.0,
            'last_update': datetime.now()
        }
        
        logger.info("🚀 Deep Learning Engine initialized")
    
    async def train_models(self, df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """Train deep learning models for a symbol"""
        if not self.deep_learning_available:
            return {'success': False, 'error': 'Deep Learning not available'}
        
        try:
            results = {}
            
            # Prepare features for deep learning
            features_df = self._prepare_features(df)
            
            if len(features_df) < 200:
                logger.warning(f"Insufficient data for training: {len(features_df)} < 200")
                return {'success': False, 'error': 'Insufficient data for training'}
            
            # Train different model types
            for model_type in [ModelType.LSTM, ModelType.CNN, ModelType.LSTM_CNN]:
                model_key = f"{symbol}_{timeframe}_{model_type.value}"
                
                try:
                    # Create and train model
                    if model_type == ModelType.LSTM:
                        model = LSTMModel(self.model_configs[model_type])
                    elif model_type == ModelType.CNN:
                        model = CNNModel(self.model_configs[model_type])
                    elif model_type == ModelType.LSTM_CNN:
                        model = LSTMCNNModel(self.model_configs[model_type])
                    
                    # Train model
                    training_result = model.train(features_df)
                    
                    if training_result['success']:
                        self.models[model_key] = model
                        self.training_history[model_key] = training_result
                        results[model_type.value] = training_result
                        
                        self.stats['models_trained'] += 1
                        self.stats['avg_accuracy'] = (
                            (self.stats['avg_accuracy'] * (self.stats['models_trained'] - 1) + 
                             training_result['final_accuracy']) / self.stats['models_trained']
                        )
                        
                        logger.info(f"✅ Trained {model_type.value} model for {symbol}: "
                                  f"Accuracy={training_result['final_accuracy']:.3f}")
                    else:
                        logger.warning(f"❌ Failed to train {model_type.value} model: {training_result['error']}")
                        results[model_type.value] = training_result
                
                except Exception as e:
                    logger.error(f"❌ Error training {model_type.value} model: {e}")
                    results[model_type.value] = {'success': False, 'error': str(e)}
            
            self.stats['last_update'] = datetime.now()
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in train_models: {e}")
            return {'success': False, 'error': str(e)}
    
    async def predict_with_models(self, df: pd.DataFrame, symbol: str, 
                                timeframe: str = '1h') -> Dict[str, Any]:
        """Make predictions using trained models"""
        if not self.deep_learning_available:
            return {'predictions': {}, 'ensemble_prediction': 0.5, 'confidence': 0.0}
        
        try:
            predictions = {}
            valid_predictions = []
            
            # Prepare features
            features_df = self._prepare_features(df)
            
            if len(features_df) < 60:  # Need minimum sequence length
                return {'predictions': {}, 'ensemble_prediction': 0.5, 'confidence': 0.0}
            
            # Make predictions with each model
            for model_type in [ModelType.LSTM, ModelType.CNN, ModelType.LSTM_CNN]:
                model_key = f"{symbol}_{timeframe}_{model_type.value}"
                
                if model_key in self.models and self.models[model_key].is_trained:
                    try:
                        prediction = self.models[model_key].predict(features_df)
                        predictions[model_type.value] = prediction
                        valid_predictions.append(prediction)
                        
                        self.stats['predictions_made'] += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error making prediction with {model_type.value}: {e}")
                        predictions[model_type.value] = 0.5
            
            # Calculate ensemble prediction
            if valid_predictions:
                ensemble_prediction = np.mean(valid_predictions)
                confidence = 1.0 - np.std(valid_predictions)  # Lower std = higher confidence
            else:
                ensemble_prediction = 0.5
                confidence = 0.0
            
            result = {
                'predictions': predictions,
                'ensemble_prediction': ensemble_prediction,
                'confidence': confidence,
                'models_used': len(valid_predictions),
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in predict_with_models: {e}")
            return {'predictions': {}, 'ensemble_prediction': 0.5, 'confidence': 0.0}
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for deep learning models"""
        try:
            # Create technical indicators
            features_df = df.copy()
            
            # Price-based features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['close_open_ratio'] = features_df['close'] / features_df['open']
            
            # Moving averages
            features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
            features_df['ema_12'] = features_df['close'].ewm(span=12).mean()
            features_df['ema_26'] = features_df['close'].ewm(span=26).mean()
            
            # RSI
            delta = features_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            features_df['bb_middle'] = features_df['close'].rolling(window=20).mean()
            bb_std = features_df['close'].rolling(window=20).std()
            features_df['bb_upper'] = features_df['bb_middle'] + (bb_std * 2)
            features_df['bb_lower'] = features_df['bb_middle'] - (bb_std * 2)
            
            # Volume features
            features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # Volatility
            features_df['volatility'] = features_df['close'].rolling(window=20).std()
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            # Select relevant features
            feature_columns = [
                'close', 'volume', 'price_change', 'high_low_ratio', 'close_open_ratio',
                'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'volume_ratio', 'volatility'
            ]
            
            # Ensure all columns exist
            available_columns = [col for col in feature_columns if col in features_df.columns]
            features_df = features_df[available_columns]
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return df
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance statistics for trained models"""
        return {
            'trained_models': list(self.models.keys()),
            'models_trained': len(self.models),
            'deep_learning_available': self.deep_learning_available,
            'training_history': self.training_history,
            'last_training_time': self.training_history[-1]['timestamp'] if self.training_history else None,
            'average_accuracy': np.mean([h['accuracy'] for h in self.training_history]) if self.training_history else 0.0,
            'last_update': datetime.now().isoformat()
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        return list(self.models.keys())
