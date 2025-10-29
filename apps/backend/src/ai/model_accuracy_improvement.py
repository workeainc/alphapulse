import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
# Deep Learning imports with conditional handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    Conv1D = None
    MaxPooling1D = None
    Adam = None
    logging.warning("TensorFlow not available - using mock models")
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Pattern classification types"""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class ModelAccuracyImprovement:
    """
    Advanced model accuracy improvement system for AlphaPulse.
    
    Implements:
    1. Separate models for pattern groups (reversal vs continuation)
    2. Ensemble methods (LightGBM, XGBoost, CNN/LSTM)
    3. Cross-validation by market conditions
    4. Probability calibration (Platt Scaling, Isotonic Regression)
    5. Self-learning loop with continuous retraining
    """
    
    def __init__(self, models_dir: str = "models", 
                 calibration_method: str = "platt",
                 ensemble_size: int = 3):
        """
        Initialize the model accuracy improvement system.
        
        Args:
            models_dir: Directory to store trained models
            calibration_method: "platt" or "isotonic"
            ensemble_size: Number of base models in ensemble
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibration_method = calibration_method
        self.ensemble_size = ensemble_size
        
        # Model storage
        self.pattern_models = {}  # Separate models for reversal/continuation
        self.regime_models = {}   # Models for different market regimes
        self.ensemble_models = {} # Ensemble models
        self.calibrators = {}     # Probability calibrators
        
        # Performance tracking
        self.model_performance = {}
        self.retraining_history = []
        
        # Self-learning parameters
        self.retrain_interval = timedelta(hours=24)  # Retrain every 24 hours
        self.min_performance_threshold = 0.6  # Minimum performance to keep model
        self.last_retrain = None
        
    def train_separate_pattern_models(self, data: pd.DataFrame, 
                                    pattern_labels: pd.Series) -> Dict[str, Any]:
        """
        Train separate models for reversal and continuation patterns.
        
        Args:
            data: Feature DataFrame
            pattern_labels: Pattern type labels (reversal/continuation)
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        logger.info("Training separate pattern models...")
        
        results = {}
        
        for pattern_type in PatternType:
            # Filter data for this pattern type
            pattern_mask = pattern_labels == pattern_type.value
            pattern_data = data[pattern_mask]
            
            if len(pattern_data) < 100:  # Need minimum data
                logger.warning(f"Insufficient data for {pattern_type.value} patterns")
                continue
            
            # Create features for this pattern type
            features = self._create_pattern_features(pattern_data, pattern_type)
            
            # Train multiple model types
            models = {}
            
            # 1. LightGBM
            lgb_model = self._train_lightgbm_model(features, pattern_data['target'])
            models['lightgbm'] = lgb_model
            
            # 2. XGBoost
            xgb_model = self._train_xgboost_model(features, pattern_data['target'])
            models['xgboost'] = xgb_model
            
            # 3. Random Forest
            rf_model = self._train_random_forest_model(features, pattern_data['target'])
            models['random_forest'] = rf_model
            
            # 4. Neural Network (for sequence data)
            if 'sequence_features' in pattern_data.columns:
                nn_model = self._train_neural_network_model(
                    pattern_data['sequence_features'], pattern_data['target']
                )
                models['neural_network'] = nn_model
            
            # Evaluate models
            performance = self._evaluate_models(models, features, pattern_data['target'])
            
            # Select best model
            best_model_name = max(performance, key=lambda k: performance[k]['roc_auc'])
            best_model = models[best_model_name]
            
            # Calibrate probabilities
            calibrated_model = self._calibrate_probabilities(best_model, features, pattern_data['target'])
            
            # Store results
            results[pattern_type.value] = {
                'model': calibrated_model,
                'performance': performance,
                'best_model_name': best_model_name,
                'feature_importance': self._get_feature_importance(best_model, features.columns)
            }
            
            # Save model
            self._save_model(calibrated_model, f"{pattern_type.value}_pattern_model.pkl")
            
            logger.info(f"Trained {pattern_type.value} pattern model with ROC AUC: {performance[best_model_name]['roc_auc']:.3f}")
        
        self.pattern_models.update(results)
        return results
    
    def train_regime_specific_models(self, data: pd.DataFrame, 
                                   regime_labels: pd.Series) -> Dict[str, Any]:
        """
        Train separate models for different market regimes.
        
        Args:
            data: Feature DataFrame
            regime_labels: Market regime labels
            
        Returns:
            Dictionary with regime-specific models
        """
        logger.info("Training regime-specific models...")
        
        results = {}
        
        for regime in MarketRegime:
            # Filter data for this regime
            regime_mask = regime_labels == regime.value
            regime_data = data[regime_mask]
            
            if len(regime_data) < 200:  # Need more data for regime models
                logger.warning(f"Insufficient data for {regime.value} regime")
                continue
            
            # Create regime-specific features
            features = self._create_regime_features(regime_data, regime)
            
            # Train ensemble model for this regime
            ensemble_model = self._train_regime_ensemble(features, regime_data['target'])
            
            # Cross-validate by time periods
            cv_scores = self._cross_validate_by_time(ensemble_model, features, regime_data['target'])
            
            # Calibrate probabilities
            calibrated_model = self._calibrate_probabilities(ensemble_model, features, regime_data['target'])
            
            # Store results
            results[regime.value] = {
                'model': calibrated_model,
                'cv_scores': cv_scores,
                'avg_cv_score': np.mean(cv_scores),
                'feature_importance': self._get_feature_importance(ensemble_model, features.columns)
            }
            
            # Save model
            self._save_model(calibrated_model, f"{regime.value}_regime_model.pkl")
            
            logger.info(f"Trained {regime.value} regime model with CV score: {np.mean(cv_scores):.3f}")
        
        self.regime_models.update(results)
        return results
    
    def create_ensemble_model(self, data: pd.DataFrame, 
                            pattern_models: Dict, 
                            regime_models: Dict) -> Dict[str, Any]:
        """
        Create ensemble model combining pattern and regime models.
        
        Args:
            data: Feature DataFrame
            pattern_models: Trained pattern models
            regime_models: Trained regime models
            
        Returns:
            Ensemble model with meta-learner
        """
        logger.info("Creating ensemble model...")
        
        # Generate predictions from base models
        base_predictions = {}
        
        # Pattern model predictions
        for pattern_type, model_info in pattern_models.items():
            pattern_features = self._create_pattern_features(data, PatternType(pattern_type))
            pred_proba = model_info['model'].predict_proba(pattern_features)[:, 1]
            base_predictions[f'pattern_{pattern_type}'] = pred_proba
        
        # Regime model predictions
        for regime_type, model_info in regime_models.items():
            regime_features = self._create_regime_features(data, MarketRegime(regime_type))
            pred_proba = model_info['model'].predict_proba(regime_features)[:, 1]
            base_predictions[f'regime_{regime_type}'] = pred_proba
        
        # Create meta-features
        meta_features = pd.DataFrame(base_predictions)
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=42)
        meta_learner.fit(meta_features, data['target'])
        
        # Calibrate meta-learner
        calibrated_meta = self._calibrate_probabilities(meta_learner, meta_features, data['target'])
        
        ensemble_model = {
            'base_models': {
                'pattern_models': pattern_models,
                'regime_models': regime_models
            },
            'meta_learner': calibrated_meta,
            'feature_names': list(meta_features.columns)
        }
        
        # Save ensemble model
        self._save_model(ensemble_model, "ensemble_model.pkl")
        
        self.ensemble_models['main'] = ensemble_model
        logger.info("Ensemble model created successfully")
        
        return ensemble_model
    
    def predict_with_ensemble(self, data: pd.DataFrame, 
                            current_regime: MarketRegime) -> Tuple[np.ndarray, float]:
        """
        Make predictions using the ensemble model.
        
        Args:
            data: Feature DataFrame
            current_regime: Current market regime
            
        Returns:
            Tuple of (predictions, confidence_score)
        """
        if 'main' not in self.ensemble_models:
            raise ValueError("Ensemble model not trained")
        
        ensemble = self.ensemble_models['main']
        
        # Get base model predictions
        base_predictions = {}
        
        # Pattern model predictions
        for pattern_type, model_info in ensemble['base_models']['pattern_models'].items():
            pattern_features = self._create_pattern_features(data, PatternType(pattern_type))
            pred_proba = model_info['model'].predict_proba(pattern_features)[:, 1]
            base_predictions[f'pattern_{pattern_type}'] = pred_proba
        
        # Regime model predictions (generate predictions for all regimes)
        for regime_type, model_info in ensemble['base_models']['regime_models'].items():
            regime_features = self._create_regime_features(data, MarketRegime(regime_type))
            pred_proba = model_info['model'].predict_proba(regime_features)[:, 1]
            base_predictions[f'regime_{regime_type}'] = pred_proba
        
        # Create meta-features with all expected features
        meta_features = pd.DataFrame(base_predictions)
        
        # Ensure all expected features are present
        expected_features = ensemble['feature_names']
        for feature in expected_features:
            if feature not in meta_features.columns:
                meta_features[feature] = 0.0  # Default value for missing features
        
        # Reorder columns to match training order
        meta_features = meta_features[expected_features]
        
        # Get ensemble prediction
        ensemble_pred = ensemble['meta_learner'].predict_proba(meta_features)[:, 1]
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(ensemble_pred, meta_features)
        
        return ensemble_pred, confidence_score
    
    def start_self_learning_loop(self, data_stream_func, 
                               retrain_interval: timedelta = None):
        """
        Start the self-learning loop for continuous model improvement.
        
        Args:
            data_stream_func: Function that provides new data
            retrain_interval: How often to retrain models
        """
        if retrain_interval:
            self.retrain_interval = retrain_interval
        
        logger.info("Starting self-learning loop...")
        
        async def learning_loop():
            while True:
                try:
                    # Get new data
                    new_data = await data_stream_func()
                    
                    if new_data is not None and len(new_data) > 1000:
                        # Check if retraining is needed
                        if (self.last_retrain is None or 
                            datetime.now() - self.last_retrain > self.retrain_interval):
                            
                            logger.info("Starting model retraining...")
                            
                            # Retrain models
                            await self._retrain_models(new_data)
                            
                            # Update retrain timestamp
                            self.last_retrain = datetime.now()
                            
                            # Store retraining history
                            self.retraining_history.append({
                                'timestamp': self.last_retrain,
                                'data_size': len(new_data),
                                'performance': self.model_performance.copy()
                            })
                    
                    # Wait before next iteration
                    await asyncio.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Error in self-learning loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
        
        # Start the learning loop
        asyncio.create_task(learning_loop())
    
    async def _retrain_models(self, new_data: pd.DataFrame):
        """Retrain models with new data."""
        try:
            # Prepare new data
            pattern_labels = self._classify_patterns(new_data)
            regime_labels = self._classify_market_regimes(new_data)
            
            # Retrain pattern models
            pattern_results = self.train_separate_pattern_models(new_data, pattern_labels)
            
            # Retrain regime models
            regime_results = self.train_regime_specific_models(new_data, regime_labels)
            
            # Recreate ensemble
            ensemble_results = self.create_ensemble_model(new_data, pattern_results, regime_results)
            
            # Evaluate new models
            performance = self._evaluate_retrained_models(new_data)
            
            # Check if new models are better
            if self._should_keep_new_models(performance):
                logger.info("New models perform better - keeping them")
                self.model_performance.update(performance)
            else:
                logger.info("New models perform worse - keeping old models")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _train_lightgbm_model(self, features: pd.DataFrame, targets: pd.Series) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        model.fit(features, targets)
        return model
    
    def _train_xgboost_model(self, features: pd.DataFrame, targets: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(features, targets)
        return model
    
    def _train_random_forest_model(self, features: pd.DataFrame, targets: pd.Series) -> RandomForestClassifier:
        """Train Random Forest model."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(features, targets)
        return model
    
    def _train_neural_network_model(self, sequences: pd.Series, targets: pd.Series) -> Sequential:
        """Train LSTM/CNN model for sequence data."""
        # Convert sequences to numpy arrays
        X = np.array([seq for seq in sequences])
        y = np.array(targets)
        
        # Reshape for LSTM (samples, timesteps, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        return model
    
    def _train_regime_ensemble(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train ensemble model for specific regime."""
        # Create ensemble of different model types
        models = []
        
        # LightGBM
        lgb_model = self._train_lightgbm_model(features, targets)
        models.append(lgb_model)
        
        # XGBoost
        xgb_model = self._train_xgboost_model(features, targets)
        models.append(xgb_model)
        
        # Random Forest
        rf_model = self._train_random_forest_model(features, targets)
        models.append(rf_model)
        
        # Return the best performing model
        best_model = self._select_best_model(models, features, targets)
        return best_model
    
    def _cross_validate_by_time(self, model: Any, features: pd.DataFrame, 
                               targets: pd.Series, n_splits: int = 5) -> List[float]:
        """Cross-validate model by time periods."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = targets.iloc[train_idx], targets.iloc[val_idx]
            
            # Train model on this fold
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred_proba = model_copy.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
        
        return scores
    
    def _calibrate_probabilities(self, model: Any, features: pd.DataFrame, 
                                targets: pd.Series) -> Any:
        """Calibrate model probabilities."""
        if self.calibration_method == "platt":
            calibrated_model = CalibratedClassifierCV(
                model, method='sigmoid', cv='prefit'
            )
        else:  # isotonic
            calibrated_model = CalibratedClassifierCV(
                model, method='isotonic', cv='prefit'
            )
        
        calibrated_model.fit(features, targets)
        return calibrated_model
    
    def _create_pattern_features(self, data: pd.DataFrame, 
                               pattern_type: PatternType) -> pd.DataFrame:
        """Create features specific to pattern type."""
        features = data.copy()
        
        if pattern_type == PatternType.REVERSAL:
            # Add reversal-specific features using engineered features
            if 'price_change_abs' in features.columns:
                features['price_reversal_strength'] = features['price_change_abs']
            else:
                features['price_reversal_strength'] = 0.0
                
            if 'volume_ratio' in features.columns:
                features['volume_reversal_confirmation'] = features['volume_ratio']
            else:
                features['volume_reversal_confirmation'] = 1.0
                
            features['rsi_divergence'] = self._calculate_rsi_divergence(features)
            
        elif pattern_type == PatternType.CONTINUATION:
            # Add continuation-specific features using engineered features
            if 'ema_9' in features.columns and 'ema_21' in features.columns:
                features['trend_strength'] = abs(features['ema_9'] - features['ema_21']) / features['ema_21']
            else:
                features['trend_strength'] = 0.0
                
            features['consolidation_breakout'] = self._calculate_breakout_strength(features)
            
            if 'volume_ratio' in features.columns:
                features['volume_trend_confirmation'] = features['volume_ratio']
            else:
                features['volume_trend_confirmation'] = 1.0
        
        return features
    
    def _create_regime_features(self, data: pd.DataFrame, 
                              regime: MarketRegime) -> pd.DataFrame:
        """Create features specific to market regime."""
        features = data.copy()
        
        if regime == MarketRegime.BULL:
            # Bull market features using engineered features
            if 'momentum_20' in features.columns:
                features['bull_momentum'] = features['momentum_20']
            else:
                features['bull_momentum'] = 0.0
                
            if 'volume_ratio' in features.columns:
                features['bull_volume_trend'] = features['volume_ratio']
            else:
                features['bull_volume_trend'] = 1.0
            
        elif regime == MarketRegime.BEAR:
            # Bear market features using engineered features
            if 'momentum_20' in features.columns:
                features['bear_momentum'] = -features['momentum_20']
            else:
                features['bear_momentum'] = 0.0
                
            if 'volume_ratio' in features.columns:
                features['bear_volume_trend'] = features['volume_ratio']
            else:
                features['bear_volume_trend'] = 1.0
            
        elif regime == MarketRegime.SIDEWAYS:
            # Sideways market features using engineered features
            if 'volatility_20' in features.columns:
                features['range_bound'] = features['volatility_20']
            else:
                features['range_bound'] = 0.0
                
            if 'sma_20' in features.columns:
                features['mean_reversion_strength'] = abs(features['price_change'] if 'price_change' in features.columns else 0.0)
            else:
                features['mean_reversion_strength'] = 0.0
            
        elif regime == MarketRegime.VOLATILE:
            # Volatile market features using engineered features
            if 'atr' in features.columns:
                features['volatility_ratio'] = features['atr'] / features['atr'].rolling(20).mean()
            else:
                features['volatility_ratio'] = 1.0
                
            if 'price_change_abs' in features.columns:
                features['price_gaps'] = features['price_change_abs']
            else:
                features['price_gaps'] = 0.0
        
        return features
    
    def _classify_patterns(self, data: pd.DataFrame) -> pd.Series:
        """Classify patterns as reversal or continuation."""
        # Simple rule-based classification using engineered features
        pattern_labels = []
        
        for i in range(len(data)):
            if i < 20:  # Need enough history
                pattern_labels.append(PatternType.CONTINUATION.value)
                continue
            
            # Use engineered features for classification
            price_change = 0.0
            volume_ratio = 1.0
            
            # Try to get price change from engineered features
            if 'price_change' in data.columns:
                price_change = abs(data.iloc[i]['price_change'])
            elif 'price_change_abs' in data.columns:
                price_change = data.iloc[i]['price_change_abs']
            
            # Try to get volume ratio from engineered features
            if 'volume_ratio' in data.columns:
                volume_ratio = data.iloc[i]['volume_ratio']
            elif 'volume_ratio_5' in data.columns:
                volume_ratio = data.iloc[i]['volume_ratio_5']
            
            # Simple classification logic
            if price_change > 0.02 and volume_ratio > 1.5:  # Large move with volume
                pattern_labels.append(PatternType.REVERSAL.value)
            else:
                pattern_labels.append(PatternType.CONTINUATION.value)
        
        return pd.Series(pattern_labels, index=data.index)
    
    def _classify_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Classify market regimes."""
        regime_labels = []
        
        for i in range(len(data)):
            if i < 50:  # Need enough history
                regime_labels.append(MarketRegime.SIDEWAYS.value)
                continue
            
            # Calculate regime indicators using engineered features
            price_trend = 0.0
            volatility = 0.0
            
            # Try to get price trend from engineered features
            if 'momentum_20' in data.columns:
                price_trend = data.iloc[i]['momentum_20']
            elif 'price_change' in data.columns:
                # Use cumulative price change over 20 periods
                price_trend = data.iloc[i]['price_change'] * 20
            
            # Try to get volatility from engineered features
            if 'volatility_20' in data.columns:
                volatility = data.iloc[i]['volatility_20']
            elif 'atr' in data.columns:
                # Normalize ATR by price (approximate)
                volatility = data.iloc[i]['atr'] / 100  # Rough normalization
            
            # Classification logic
            if volatility > 0.03:  # High volatility
                regime_labels.append(MarketRegime.VOLATILE.value)
            elif price_trend > 0.05:  # Strong uptrend
                regime_labels.append(MarketRegime.BULL.value)
            elif price_trend < -0.05:  # Strong downtrend
                regime_labels.append(MarketRegime.BEAR.value)
            else:
                regime_labels.append(MarketRegime.SIDEWAYS.value)
        
        return pd.Series(regime_labels, index=data.index)
    
    def _calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI divergence using engineered features."""
        # Simplified RSI divergence calculation
        divergence = pd.Series(0.0, index=data.index)
        
        if 'rsi' not in data.columns:
            return divergence
        
        rsi = data['rsi']
        price_change = data.get('price_change', pd.Series(0.0, index=data.index))
        
        for i in range(20, len(data)):
            # Check for bullish divergence
            if (price_change.iloc[i] < 0 and 
                rsi.iloc[i] > rsi.iloc[i-10]):
                divergence.iloc[i] = 1.0
            # Check for bearish divergence
            elif (price_change.iloc[i] > 0 and 
                  rsi.iloc[i] < rsi.iloc[i-10]):
                divergence.iloc[i] = -1.0
        
        return divergence
    
    def _calculate_breakout_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate breakout strength using engineered features."""
        # Simplified breakout calculation
        breakout_strength = pd.Series(0.0, index=data.index)
        
        # Use volatility and price change as proxy for breakout strength
        if 'volatility_20' in data.columns:
            volatility = data['volatility_20']
            breakout_strength = volatility * 10  # Scale up for visibility
        
        if 'price_change_abs' in data.columns:
            price_change = data['price_change_abs']
            breakout_strength = breakout_strength + price_change
        
        return breakout_strength
    
    def _evaluate_models(self, models: Dict, features: pd.DataFrame, 
                        targets: pd.Series) -> Dict[str, Dict]:
        """Evaluate multiple models."""
        results = {}
        
        for name, model in models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(features)[:, 1]
                else:
                    pred_proba = model.predict(features)
                
                # Calculate metrics
                roc_auc = roc_auc_score(targets, pred_proba)
                
                results[name] = {
                    'roc_auc': roc_auc,
                    'model': model
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                results[name] = {'roc_auc': 0.0, 'model': model}
        
        return results
    
    def _select_best_model(self, models: List, features: pd.DataFrame, 
                          targets: pd.Series) -> Any:
        """Select the best performing model."""
        best_score = 0.0
        best_model = None
        
        for model in models:
            try:
                pred_proba = model.predict_proba(features)[:, 1]
                score = roc_auc_score(targets, pred_proba)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                logger.error(f"Error evaluating model: {e}")
        
        return best_model
    
    def _clone_model(self, model: Any) -> Any:
        """Create a copy of a model."""
        if hasattr(model, 'clone'):
            return model.clone()
        else:
            # For models without clone method, create new instance
            return type(model)()
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return {}
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _calculate_confidence_score(self, predictions: np.ndarray, 
                                  meta_features: pd.DataFrame) -> float:
        """Calculate confidence score for ensemble predictions."""
        # Use prediction variance as confidence measure
        prediction_std = np.std(predictions)
        confidence = 1.0 - prediction_std  # Higher variance = lower confidence
        
        return max(0.0, min(1.0, confidence))
    
    def _should_keep_new_models(self, new_performance: Dict) -> bool:
        """Determine if new models should replace old ones."""
        if not self.model_performance:
            return True
        
        # Compare average performance
        old_avg = np.mean([perf.get('roc_auc', 0) for perf in self.model_performance.values()])
        new_avg = np.mean([perf.get('roc_auc', 0) for perf in new_performance.values()])
        
        # Keep new models if they're significantly better
        return new_avg > old_avg + 0.05  # 5% improvement threshold
    
    def _evaluate_retrained_models(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate retrained models."""
        # This would evaluate the current models on new data
        # Implementation depends on specific evaluation needs
        return {}
    
    def _save_model(self, model: Any, filename: str):
        """Save model to disk."""
        try:
            filepath = self.models_dir / filename
            joblib.dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filename: str) -> Any:
        """Load model from disk."""
        try:
            filepath = self.models_dir / filename
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        return {
            'pattern_models': {name: info.get('performance', {}) 
                             for name, info in self.pattern_models.items()},
            'regime_models': {name: info.get('cv_scores', []) 
                            for name, info in self.regime_models.items()},
            'retraining_history': self.retraining_history,
            'last_retrain': self.last_retrain
        }
