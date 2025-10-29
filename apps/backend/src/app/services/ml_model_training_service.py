#!/usr/bin/env python3
"""
ML Model Training Service
Handles model training, evaluation, and management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import os

# ML imports (will be installed later)
try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Install with: pip install lightgbm scikit-learn")

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import tensorflow as tf
    from tensorflow import keras
    from transformers import AutoTokenizer, AutoModel
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("Deep Learning libraries not available. Install with: pip install torch tensorflow transformers")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    GRU = "gru"
    BILSTM = "bilstm"
    ATTENTION_LSTM = "attention_lstm"

class LabelType(Enum):
    BINARY_BREAKOUT = "binary_breakout"
    REGRESSION_RETURN = "regression_return"
    MULTICLASS_DIRECTION = "multiclass_direction"

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: ModelType
    label_type: LabelType
    symbol: str
    timeframe: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    training_window_days: int = 30
    validation_window_days: int = 7
    min_samples: int = 1000
    max_samples: int = 100000

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_version: str
    auc: float
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    training_samples: int
    validation_samples: int
    training_time_seconds: float
    feature_importance: Dict[str, float]

class MLModelTrainingService:
    """Service for training and managing ML models"""
    
    def __init__(self, db_pool, model_storage_path: str = "./models"):
        self.db_pool = db_pool
        self.model_storage_path = model_storage_path
        self.logger = logging.getLogger(__name__)
        
        # Create model storage directory
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Default hyperparameters
        self.default_hyperparameters = {
            ModelType.LIGHTGBM: {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 127,
                "max_depth": 8,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": 42
            },
            ModelType.LSTM: {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "sequence_length": 60,
                "num_epochs": 100,
                "early_stopping_patience": 10
            },
            ModelType.TRANSFORMER: {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 6,
                "dropout": 0.1,
                "learning_rate": 0.0001,
                "batch_size": 16,
                "sequence_length": 100,
                "num_epochs": 50,
                "warmup_steps": 1000
            },
            ModelType.CNN: {
                "num_filters": [64, 128, 256],
                "kernel_sizes": [3, 5, 7],
                "dropout": 0.3,
                "learning_rate": 0.001,
                "batch_size": 64,
                "sequence_length": 60,
                "num_epochs": 100
            },
            ModelType.GRU: {
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "sequence_length": 60,
                "num_epochs": 100
            }
        }
        
        self.logger.info("🤖 ML Model Training Service initialized")
    
    async def train_model(self, config: ModelConfig) -> Optional[ModelPerformance]:
        """Train a new model with the given configuration"""
        if not ML_AVAILABLE:
            self.logger.error("❌ ML libraries not available")
            return None
        
        try:
            self.logger.info(f"🚀 Starting model training for {config.symbol} {config.timeframe}")
            
            # Load training data
            training_data = await self._load_training_data(config)
            if training_data is None or len(training_data) < config.min_samples:
                self.logger.error(f"❌ Insufficient training data: {len(training_data) if training_data else 0} samples")
                return None
            
            # Generate labels
            labeled_data = await self._generate_labels(training_data, config.label_type)
            if labeled_data is None:
                self.logger.error("❌ Failed to generate labels")
                return None
            
            # Prepare features and labels
            X, y = self._prepare_features_and_labels(labeled_data, config.features)
            
            # Train model
            model, performance = await self._train_model_internal(X, y, config)
            if model is None:
                self.logger.error("❌ Model training failed")
                return None
            
            # Save model
            model_version = await self._save_model(model, config, performance)
            if model_version is None:
                self.logger.error("❌ Failed to save model")
                return None
            
            # Register model in database
            await self._register_model_in_database(model_version, config, performance)
            
            self.logger.info(f"✅ Model training completed: {model_version}")
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            return None
    
    async def _load_training_data(self, config: ModelConfig) -> Optional[pd.DataFrame]:
        """Load training data from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config.training_window_days)
                
                query = """
                SELECT 
                    timestamp,
                    features,
                    technical_features,
                    order_book_features,
                    time_features,
                    multi_timeframe_features,
                    market_regime,
                    volatility_regime
                FROM volume_analysis_ml_dataset
                WHERE symbol = $1 
                AND timeframe = $2 
                AND timestamp >= $3 
                AND timestamp <= $4
                ORDER BY timestamp ASC
                """
                
                rows = await conn.fetch(query, config.symbol, config.timeframe, start_date, end_date)
                
                if not rows:
                    return None
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    record = {
                        'timestamp': row['timestamp'],
                        'market_regime': row['market_regime'],
                        'volatility_regime': row['volatility_regime']
                    }
                    
                    # Extract features from JSONB
                    if row['features']:
                        record.update(row['features'])
                    if row['technical_features']:
                        record.update(row['technical_features'])
                    if row['order_book_features']:
                        record.update(row['order_book_features'])
                    if row['time_features']:
                        record.update(row['time_features'])
                    if row['multi_timeframe_features']:
                        record.update(row['multi_timeframe_features'])
                    
                    data.append(record)
                
                df = pd.DataFrame(data)
                self.logger.info(f"📊 Loaded {len(df)} training samples")
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Error loading training data: {e}")
            return None
    
    async def _generate_labels(self, data: pd.DataFrame, label_type: LabelType) -> Optional[pd.DataFrame]:
        """Generate labels for supervised learning"""
        try:
            df = data.copy()
            
            if label_type == LabelType.BINARY_BREAKOUT:
                # Binary classification: Will there be a volume breakout in the next N minutes?
                df['label'] = (df['volume_breakout'] == True).astype(int)
                
            elif label_type == LabelType.REGRESSION_RETURN:
                # Regression: Future return prediction
                # For now, use a simple forward-looking return
                df['future_return'] = df['volume_ratio'].shift(-1)  # Simplified
                df['label'] = df['future_return'].fillna(0)
                
            elif label_type == LabelType.MULTICLASS_DIRECTION:
                # Multi-class: Price direction prediction
                # 0: sideways, 1: up, 2: down
                df['price_change'] = df['volume_ratio'].pct_change()
                df['label'] = 0  # sideways
                df.loc[df['price_change'] > 0.01, 'label'] = 1  # up
                df.loc[df['price_change'] < -0.01, 'label'] = 2  # down
            
            # Remove rows with NaN labels
            df = df.dropna(subset=['label'])
            
            self.logger.info(f"🏷️ Generated {len(df)} labeled samples")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error generating labels: {e}")
            return None
    
    def _prepare_features_and_labels(self, data: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        try:
            # Select features
            available_features = [f for f in feature_names if f in data.columns]
            missing_features = [f for f in feature_names if f not in data.columns]
            
            if missing_features:
                self.logger.warning(f"⚠️ Missing features: {missing_features}")
            
            # Fill missing values
            for feature in available_features:
                if data[feature].dtype in ['float64', 'int64']:
                    data[feature] = data[feature].fillna(data[feature].median())
                else:
                    data[feature] = data[feature].fillna('unknown')
            
            # Convert categorical features
            categorical_features = []
            for feature in available_features:
                if data[feature].dtype == 'object':
                    data[feature] = data[feature].astype('category').cat.codes
                    categorical_features.append(feature)
            
            # Prepare X and y
            X = data[available_features].values
            y = data['label'].values
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            self.logger.info(f"🔧 Prepared {X.shape[0]} samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing features and labels: {e}")
            raise
    
    async def _train_model_internal(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Tuple[Any, ModelPerformance]:
        """Internal model training logic"""
        try:
            start_time = datetime.now()
            
            if config.model_type == ModelType.LIGHTGBM:
                return await self._train_lightgbm(X, y, config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Error in model training: {e}")
            raise
    
    async def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, config: ModelConfig) -> Tuple[lgb.Booster, ModelPerformance]:
        """Train LightGBM model"""
        try:
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=5)
            models = []
            aucs = []
            precisions = []
            recalls = []
            accuracies = []
            
            # Get hyperparameters
            params = config.hyperparameters or self.default_hyperparameters[ModelType.LIGHTGBM]
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                self.logger.info(f"🔄 Training fold {fold + 1}/5")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
                
                # Predictions
                y_pred = model.predict(X_val)
                y_pred_binary = (y_pred > 0.5).astype(int) if config.label_type == LabelType.BINARY_BREAKOUT else y_pred
                
                # Metrics
                auc = roc_auc_score(y_val, y_pred) if config.label_type == LabelType.BINARY_BREAKOUT else 0.0
                precision = precision_score(y_val, y_pred_binary, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred_binary, average='weighted', zero_division=0)
                accuracy = accuracy_score(y_val, y_pred_binary)
                
                aucs.append(auc)
                precisions.append(precision)
                recalls.append(recall)
                accuracies.append(accuracy)
                models.append(model)
                
                self.logger.info(f"   Fold {fold + 1}: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            # Select best model
            best_idx = np.argmax(aucs)
            best_model = models[best_idx]
            
            # Calculate feature importance
            feature_importance = dict(zip(config.features, best_model.feature_importance()))
            
            # Performance summary
            performance = ModelPerformance(
                model_version=f"lgb_{config.symbol}_{config.timeframe}_{int(datetime.now().timestamp())}",
                auc=np.mean(aucs),
                precision=np.mean(precisions),
                recall=np.mean(recalls),
                accuracy=np.mean(accuracies),
                f1_score=2 * (np.mean(precisions) * np.mean(recalls)) / (np.mean(precisions) + np.mean(recalls)),
                training_samples=len(X),
                validation_samples=len(X) // 5,
                training_time_seconds=(datetime.now() - start_time).total_seconds(),
                feature_importance=feature_importance
            )
            
            self.logger.info(f"✅ LightGBM training completed: AUC={performance.auc:.4f}")
            return best_model, performance
            
        except Exception as e:
            self.logger.error(f"❌ LightGBM training failed: {e}")
            raise
    
    async def _save_model(self, model: Any, config: ModelConfig, performance: ModelPerformance) -> Optional[str]:
        """Save trained model to disk"""
        try:
            model_version = performance.model_version
            model_path = os.path.join(self.model_storage_path, f"{model_version}.pkl")
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'model_version': model_version,
                'model_type': config.model_type.value,
                'label_type': config.label_type.value,
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'features': config.features,
                'hyperparameters': config.hyperparameters,
                'performance': {
                    'auc': performance.auc,
                    'precision': performance.precision,
                    'recall': performance.recall,
                    'accuracy': performance.accuracy,
                    'f1_score': performance.f1_score,
                    'training_samples': performance.training_samples,
                    'validation_samples': performance.validation_samples,
                    'training_time_seconds': performance.training_time_seconds
                },
                'feature_importance': performance.feature_importance,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.model_storage_path, f"{model_version}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"💾 Model saved: {model_path}")
            return model_version
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            return None
    
    async def _register_model_in_database(self, model_version: str, config: ModelConfig, performance: ModelPerformance) -> bool:
        """Register model in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Insert model version
                query = """
                INSERT INTO model_versions 
                (model_name, version, model_type, training_start, training_end, training_samples,
                 validation_auc, validation_precision, validation_recall, model_path, feature_list, hyperparameters)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (model_name, version) DO UPDATE SET
                training_end = EXCLUDED.training_end,
                validation_auc = EXCLUDED.validation_auc,
                validation_precision = EXCLUDED.validation_precision,
                validation_recall = EXCLUDED.validation_recall
                """
                
                model_path = os.path.join(self.model_storage_path, f"{model_version}.pkl")
                
                await conn.execute(query, 
                    f"{config.symbol}_{config.timeframe}",  # model_name
                    model_version,  # version
                    config.model_type.value,  # model_type
                    datetime.now() - timedelta(seconds=performance.training_time_seconds),  # training_start
                    datetime.now(),  # training_end
                    performance.training_samples,  # training_samples
                    performance.auc,  # validation_auc
                    performance.precision,  # validation_precision
                    performance.recall,  # validation_recall
                    model_path,  # model_path
                    config.features,  # feature_list
                    config.hyperparameters  # hyperparameters
                )
                
                # Insert feature importance
                for feature, importance in performance.feature_importance.items():
                    feature_query = """
                    INSERT INTO feature_importance 
                    (model_version, symbol, feature_name, importance_score, feature_category)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (model_version, feature_name) DO UPDATE SET
                    importance_score = EXCLUDED.importance_score
                    """
                    
                    # Determine feature category
                    category = self._determine_feature_category(feature)
                    
                    await conn.execute(feature_query, 
                        model_version, config.symbol, feature, importance, category
                    )
                
                self.logger.info(f"📝 Model registered in database: {model_version}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error registering model in database: {e}")
            return False
    
    def _determine_feature_category(self, feature: str) -> str:
        """Determine feature category based on feature name"""
        if 'volume' in feature.lower():
            return 'volume'
        elif any(x in feature.lower() for x in ['price', 'close', 'high', 'low', 'open']):
            return 'price'
        elif any(x in feature.lower() for x in ['ema', 'rsi', 'macd', 'atr', 'obv']):
            return 'technical'
        elif any(x in feature.lower() for x in ['bid', 'ask', 'depth', 'spread', 'liquidity']):
            return 'orderbook'
        elif any(x in feature.lower() for x in ['time', 'hour', 'day', 'session']):
            return 'time'
        elif any(x in feature.lower() for x in ['h1', 'h4', 'd1', 'return']):
            return 'multi_timeframe'
        else:
            return 'other'
    
    async def get_active_models(self, symbol: str = None, timeframe: str = None) -> List[Dict]:
        """Get active models from database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT model_name, version, model_type, validation_auc, validation_precision, 
                       validation_recall, is_active, is_production, created_at
                FROM model_versions
                WHERE is_active = TRUE
                """
                
                params = []
                if symbol:
                    query += " AND model_name LIKE $1"
                    params.append(f"{symbol}%")
                if timeframe:
                    query += " AND model_name LIKE $2"
                    params.append(f"%{timeframe}%")
                
                query += " ORDER BY created_at DESC"
                
                rows = await conn.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting active models: {e}")
            return []
    
    async def activate_model(self, model_version: str) -> bool:
        """Activate a model version"""
        try:
            async with self.db_pool.acquire() as conn:
                # Deactivate all other models of the same type
                query = """
                UPDATE model_versions 
                SET is_active = FALSE 
                WHERE model_name = (SELECT model_name FROM model_versions WHERE version = $1)
                """
                await conn.execute(query, model_version)
                
                # Activate the specified model
                query = """
                UPDATE model_versions 
                SET is_active = TRUE 
                WHERE version = $1
                """
                await conn.execute(query, model_version)
                
                self.logger.info(f"✅ Activated model: {model_version}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error activating model: {e}")
            return False

    # ==================== DEEP LEARNING MODELS ====================
    
    def _create_lstm_model(self, input_size: int, hidden_size: int, num_layers: int, 
                          dropout: float, output_size: int = 1) -> nn.Module:
        """Create LSTM model for time series prediction"""
        if not DL_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return LSTMModel(input_size, hidden_size, num_layers, dropout, output_size)
    
    def _create_transformer_model(self, input_size: int, d_model: int, nhead: int, 
                                num_layers: int, dropout: float, output_size: int = 1) -> nn.Module:
        """Create Transformer model for time series prediction"""
        if not DL_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        class TransformerModel(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dropout, output_size):
                super(TransformerModel, self).__init__()
                self.d_model = d_model
                
                self.input_projection = nn.Linear(input_size, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(d_model, output_size)
                
            def forward(self, x):
                # Project input to d_model dimensions
                x = self.input_projection(x)
                
                # Add positional encoding
                seq_len = x.size(1)
                pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0)
                x = x + pos_encoding
                
                # Apply transformer
                x = self.transformer(x)
                x = self.dropout(x[:, -1, :])  # Take last sequence element
                x = self.fc(x)
                return x
        
        return TransformerModel(input_size, d_model, nhead, num_layers, dropout, output_size)
    
    def _create_cnn_model(self, input_size: int, num_filters: List[int], 
                         kernel_sizes: List[int], dropout: float, output_size: int = 1) -> nn.Module:
        """Create CNN model for time series prediction"""
        if not DL_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        class CNNModel(nn.Module):
            def __init__(self, input_size, num_filters, kernel_sizes, dropout, output_size):
                super(CNNModel, self).__init__()
                
                self.conv_layers = nn.ModuleList()
                in_channels = 1
                
                for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
                    self.conv_layers.append(
                        nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
                    )
                    in_channels = filters
                
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(num_filters[-1], output_size)
                
            def forward(self, x):
                # Reshape for CNN: (batch, channels, sequence_length, features)
                x = x.unsqueeze(1)  # Add channel dimension
                
                for conv in self.conv_layers:
                    x = torch.relu(conv(x))
                    x = torch.max_pool1d(x, 2)
                
                # Global average pooling
                x = torch.mean(x, dim=2)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        return CNNModel(input_size, num_filters, kernel_sizes, dropout, output_size)
    
    async def train_deep_learning_model(self, config: ModelConfig, training_data: pd.DataFrame) -> ModelPerformance:
        """Train a deep learning model"""
        if not DL_AVAILABLE:
            raise ImportError("Deep learning libraries not available")
        
        try:
            self.logger.info(f"🧠 Training deep learning model: {config.model_type.value}")
            
            # Prepare data
            X, y = self._prepare_deep_learning_data(training_data, config)
            
            # Create model
            model = self._create_deep_learning_model(config, X.shape[-1])
            
            # Train model
            performance = await self._train_deep_learning_model(model, X, y, config)
            
            # Save model
            model_path = await self._save_deep_learning_model(model, config)
            
            # Update performance with model path
            performance.model_path = model_path
            
            self.logger.info(f"✅ Deep learning model trained successfully: {performance.auc:.4f}")
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Error training deep learning model: {e}")
            raise
    
    def _prepare_deep_learning_data(self, data: pd.DataFrame, config: ModelConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for deep learning models"""
        # Extract features
        feature_cols = [col for col in data.columns if col in config.features]
        X = data[feature_cols].values
        
        # Create sequences for time series models
        sequence_length = config.hyperparameters.get('sequence_length', 60)
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(data.iloc[i]['target'])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Normalize features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        return X, y
    
    def _create_deep_learning_model(self, config: ModelConfig, input_size: int) -> nn.Module:
        """Create deep learning model based on configuration"""
        if config.model_type == ModelType.LSTM:
            return self._create_lstm_model(
                input_size=input_size,
                hidden_size=config.hyperparameters['hidden_size'],
                num_layers=config.hyperparameters['num_layers'],
                dropout=config.hyperparameters['dropout']
            )
        elif config.model_type == ModelType.TRANSFORMER:
            return self._create_transformer_model(
                input_size=input_size,
                d_model=config.hyperparameters['d_model'],
                nhead=config.hyperparameters['nhead'],
                num_layers=config.hyperparameters['num_layers'],
                dropout=config.hyperparameters['dropout']
            )
        elif config.model_type == ModelType.CNN:
            return self._create_cnn_model(
                input_size=input_size,
                num_filters=config.hyperparameters['num_filters'],
                kernel_sizes=config.hyperparameters['kernel_sizes'],
                dropout=config.hyperparameters['dropout']
            )
        else:
            raise ValueError(f"Unsupported deep learning model type: {config.model_type}")
    
    async def _train_deep_learning_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray, 
                                       config: ModelConfig) -> ModelPerformance:
        """Train deep learning model"""
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters['batch_size'], shuffle=True)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.hyperparameters['learning_rate'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = datetime.now()
        
        for epoch in range(config.hyperparameters['num_epochs']):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_tensor = X_val.to(device)
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val)
                
                # Calculate metrics
                val_predictions = val_outputs.squeeze().cpu().numpy()
                val_actuals = y_val.numpy()
                
                # Convert to binary classification for metrics
                val_pred_binary = (val_predictions > 0.5).astype(int)
                val_actual_binary = (val_actuals > 0.5).astype(int)
                
                auc = roc_auc_score(val_actual_binary, val_predictions)
                precision = precision_score(val_actual_binary, val_pred_binary, zero_division=0)
                recall = recall_score(val_actual_binary, val_pred_binary, zero_division=0)
                accuracy = accuracy_score(val_actual_binary, val_pred_binary)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.hyperparameters.get('early_stopping_patience', 10):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                               f"Val Loss: {val_loss:.4f}, AUC: {auc:.4f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create performance object
        performance = ModelPerformance(
            model_version=f"{config.model_type.value}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            auc=auc,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            training_time_seconds=training_time,
            feature_importance={}  # Will be calculated separately for deep learning
        )
        
        return performance
    
    async def _save_deep_learning_model(self, model: nn.Module, config: ModelConfig) -> str:
        """Save deep learning model"""
        model_version = f"{config.model_type.value}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(self.model_storage_path, f"{model_version}.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'model_type': config.model_type.value,
            'hyperparameters': config.hyperparameters
        }, model_path)
        
        self.logger.info(f"💾 Deep learning model saved: {model_path}")
        return model_path
