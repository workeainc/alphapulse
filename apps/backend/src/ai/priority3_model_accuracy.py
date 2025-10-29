#!/usr/bin/env python3
"""
Priority 3: Enhanced Model Accuracy System for AlphaPulse

Builds on existing infrastructure to provide:
1. Enhanced pattern-specific models (reversal vs continuation patterns)
2. Advanced probability calibration (Platt scaling, isotonic regression)
3. Market condition adaptation (bull/bear/sideways specific models)
4. Integration with existing feature engineering and ONNX optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
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
import warnings
warnings.filterwarnings('ignore')

# Import existing infrastructure
from ..src.ai.model_accuracy_improvement import ModelAccuracyImprovement, PatternType, MarketRegime
from ..src.ai.priority2_feature_engineering import Priority2FeatureEngineering
from ..src.ai.onnx_optimization_manager import ONNXOptimizationManager

logger = logging.getLogger(__name__)

class Priority3ModelAccuracy:
    """
    Priority 3 Enhanced Model Accuracy System
    
    Enhances the existing ModelAccuracyImprovement with:
    1. Advanced pattern detection and classification
    2. Sophisticated probability calibration
    3. Market regime-specific model adaptation
    4. Integration with Priority 2 feature engineering
    5. ONNX optimization for production deployment
    """
    
    def __init__(self, 
                 models_dir: str = "models/priority3",
                 calibration_method: str = "isotonic",
                 ensemble_size: int = 5,
                 enable_onnx: bool = True):
        """
        Initialize Priority 3 Model Accuracy system
        
        Args:
            models_dir: Directory for storing Priority 3 models
            calibration_method: "platt", "isotonic", or "ensemble"
            ensemble_size: Number of base models in ensemble
            enable_onnx: Whether to enable ONNX optimization
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibration_method = calibration_method
        self.ensemble_size = ensemble_size
        self.enable_onnx = enable_onnx
        
        # Initialize existing infrastructure
        self.base_accuracy_improvement = ModelAccuracyImprovement(
            models_dir=str(self.models_dir / "base"),
            calibration_method=calibration_method,
            ensemble_size=ensemble_size
        )
        
        # Priority 2 feature engineering integration
        self.feature_engineering = Priority2FeatureEngineering()
        
        # ONNX optimization manager
        if enable_onnx:
            self.onnx_manager = ONNXOptimizationManager()
        
        # Enhanced model storage
        self.enhanced_pattern_models = {}
        self.enhanced_regime_models = {}
        self.calibration_models = {}
        self.ensemble_meta_models = {}
        
        # Performance tracking
        self.calibration_performance = {}
        self.pattern_detection_accuracy = {}
        self.regime_adaptation_metrics = {}
        
        # Advanced calibration parameters
        self.calibration_params = {
            'platt': {
                'cv_folds': 5,
                'method': 'sigmoid'
            },
            'isotonic': {
                'cv_folds': 5,
                'method': 'isotonic'
            },
            'ensemble': {
                'methods': ['platt', 'isotonic'],
                'weights': [0.5, 0.5]
            }
        }
        
        logger.info("🚀 Priority 3 Enhanced Model Accuracy System initialized")
    
    async def train_enhanced_pattern_models(self, 
                                          data: pd.DataFrame,
                                          pattern_labels: pd.Series,
                                          symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Train enhanced pattern-specific models with advanced features
        
        Args:
            data: Feature DataFrame with Priority 2 engineered features
            pattern_labels: Pattern type labels (reversal/continuation)
            symbol: Trading symbol for model naming
            
        Returns:
            Dictionary with enhanced pattern models and performance
        """
        logger.info(f"Training enhanced pattern models for {symbol}...")
        
        # Extract Priority 2 features
        enhanced_features = await self._extract_priority2_features(data, symbol)
        
        results = {}
        
        for pattern_type in PatternType:
            # Filter data for this pattern type
            pattern_mask = pattern_labels == pattern_type.value
            pattern_data = enhanced_features[pattern_mask]
            
            if len(pattern_data) < 200:  # Higher threshold for enhanced models
                logger.warning(f"Insufficient data for {pattern_type.value} patterns: {len(pattern_data)} samples")
                continue
            
            # Create pattern-specific enhanced features
            pattern_features = self._create_enhanced_pattern_features(pattern_data, pattern_type)
            
            # Train multiple model types with advanced configurations
            models = await self._train_enhanced_models(pattern_features, pattern_data['target'])
            
            # Advanced probability calibration
            calibrated_models = await self._apply_advanced_calibration(
                models, pattern_features, pattern_data['target']
            )
            
            # Evaluate calibrated models
            performance = self._evaluate_calibrated_models(
                calibrated_models, pattern_features, pattern_data['target']
            )
            
            # Select best calibrated model
            best_model_name = max(performance, key=lambda k: performance[k]['calibrated_auc'])
            best_model = calibrated_models[best_model_name]
            
            # Store results
            results[pattern_type.value] = {
                'model': best_model,
                'performance': performance,
                'best_model_name': best_model_name,
                'feature_importance': self._get_enhanced_feature_importance(best_model, pattern_features.columns),
                'calibration_metrics': performance[best_model_name]['calibration_metrics']
            }
            
            # Save model with ONNX optimization if enabled
            model_filename = f"{symbol}_{pattern_type.value}_enhanced_model.pkl"
            self._save_enhanced_model(best_model, model_filename)
            
            if self.enable_onnx:
                await self._optimize_model_for_onnx(best_model, model_filename, pattern_type.value)
            
            logger.info(f"✅ Enhanced {pattern_type.value} pattern model trained - "
                       f"Calibrated AUC: {performance[best_model_name]['calibrated_auc']:.3f}")
        
        self.enhanced_pattern_models.update(results)
        return results
    
    async def train_enhanced_regime_models(self, 
                                         data: pd.DataFrame,
                                         regime_labels: pd.Series,
                                         symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Train enhanced market regime-specific models
        
        Args:
            data: Feature DataFrame
            regime_labels: Market regime labels
            symbol: Trading symbol for model naming
            
        Returns:
            Dictionary with enhanced regime models
        """
        logger.info(f"Training enhanced regime models for {symbol}...")
        
        # Extract Priority 2 features
        enhanced_features = await self._extract_priority2_features(data, symbol)
        
        results = {}
        
        for regime in MarketRegime:
            # Filter data for this regime
            regime_mask = regime_labels == regime.value
            regime_data = enhanced_features[regime_mask]
            
            if len(regime_data) < 500:  # Higher threshold for regime models
                logger.warning(f"Insufficient data for {regime.value} regime: {len(regime_data)} samples")
                continue
            
            # Create regime-specific enhanced features
            regime_features = self._create_enhanced_regime_features(regime_data, regime)
            
            # Train ensemble with regime-specific configurations
            ensemble_model = await self._train_regime_ensemble(regime_features, regime_data['target'], regime)
            
            # Advanced cross-validation by market conditions
            cv_scores = self._enhanced_time_series_cv(ensemble_model, regime_features, regime_data['target'])
            
            # Advanced probability calibration
            calibrated_models = await self._apply_advanced_calibration(
                {'ensemble': ensemble_model}, regime_features, regime_data['target']
            )
            calibrated_model = calibrated_models['ensemble']
            
            # Calculate regime-specific metrics
            regime_metrics = self._calculate_regime_metrics(calibrated_model, regime_features, regime_data['target'])
            
            # Store results
            results[regime.value] = {
                'model': calibrated_model,
                'cv_scores': cv_scores,
                'avg_cv_score': np.mean(cv_scores),
                'regime_metrics': regime_metrics,
                'feature_importance': self._get_enhanced_feature_importance(calibrated_model, regime_features.columns)
            }
            
            # Save model with ONNX optimization
            model_filename = f"{symbol}_{regime.value}_enhanced_regime_model.pkl"
            self._save_enhanced_model(calibrated_model, model_filename)
            
            if self.enable_onnx:
                await self._optimize_model_for_onnx(calibrated_model, model_filename, f"{regime.value}_regime")
            
            logger.info(f"✅ Enhanced {regime.value} regime model trained - "
                       f"CV Score: {np.mean(cv_scores):.3f}")
        
        self.enhanced_regime_models.update(results)
        return results
    
    async def create_enhanced_ensemble(self, 
                                     data: pd.DataFrame,
                                     pattern_models: Dict,
                                     regime_models: Dict,
                                     symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Create enhanced ensemble model with advanced meta-learning
        
        Args:
            data: Feature DataFrame
            pattern_models: Enhanced pattern models
            regime_models: Enhanced regime models
            symbol: Trading symbol
            
        Returns:
            Enhanced ensemble model
        """
        logger.info(f"Creating enhanced ensemble model for {symbol}...")
        
        # Extract Priority 2 features
        enhanced_features = await self._extract_priority2_features(data, symbol)
        
        # Generate enhanced predictions from base models
        base_predictions = self._generate_enhanced_predictions(
            enhanced_features, pattern_models, regime_models
        )
        
        # Create enhanced meta-features
        meta_features = self._create_enhanced_meta_features(base_predictions, enhanced_features)
        
        # Train advanced meta-learner
        meta_learner = self._train_advanced_meta_learner(meta_features, enhanced_features['target'])
        
        # Apply advanced calibration to meta-learner
        calibrated_models = await self._apply_advanced_calibration(
            {'meta': meta_learner}, meta_features, enhanced_features['target']
        )
        calibrated_meta = calibrated_models['meta']
        
        # Create ensemble structure
        ensemble_model = {
            'base_models': {
                'pattern_models': pattern_models,
                'regime_models': regime_models
            },
            'meta_learner': calibrated_meta,
            'feature_names': list(meta_features.columns),
            'prediction_weights': self._calculate_prediction_weights(base_predictions, enhanced_features['target'])
        }
        
        # Save enhanced ensemble model
        ensemble_filename = f"{symbol}_enhanced_ensemble_model.pkl"
        self._save_enhanced_model(ensemble_model, ensemble_filename)
        
        if self.enable_onnx:
            await self._optimize_ensemble_for_onnx(ensemble_model, ensemble_filename)
        
        self.ensemble_meta_models['main'] = ensemble_model
        logger.info("✅ Enhanced ensemble model created successfully")
        
        return ensemble_model
    
    async def predict_with_enhanced_ensemble(self, 
                                           data: pd.DataFrame,
                                           current_regime: MarketRegime,
                                           symbol: str = "BTCUSDT") -> Tuple[np.ndarray, float, Dict]:
        """
        Make predictions using enhanced ensemble model
        
        Args:
            data: Feature DataFrame
            current_regime: Current market regime
            symbol: Trading symbol
            
        Returns:
            Tuple of (predictions, confidence_score, prediction_metadata)
        """
        if 'main' not in self.ensemble_meta_models:
            raise ValueError("Enhanced ensemble model not trained")
        
        # Extract Priority 2 features
        enhanced_features = await self._extract_priority2_features(data, symbol)
        
        ensemble = self.ensemble_meta_models['main']
        
        # Generate enhanced predictions
        base_predictions = self._generate_enhanced_predictions(
            enhanced_features, ensemble['base_models']['pattern_models'], 
            ensemble['base_models']['regime_models']
        )
        
        # Create meta-features
        meta_features = self._create_enhanced_meta_features(base_predictions, enhanced_features)
        
        # Ensure all expected features are present
        expected_features = ensemble['feature_names']
        for feature in expected_features:
            if feature not in meta_features.columns:
                meta_features[feature] = 0.0
        
        meta_features = meta_features[expected_features]
        
        # Get ensemble prediction
        ensemble_pred = ensemble['meta_learner'].predict_proba(meta_features)[:, 1]
        
        # Calculate enhanced confidence score
        confidence_score = self._calculate_enhanced_confidence(
            ensemble_pred, meta_features, base_predictions
        )
        
        # Generate prediction metadata
        prediction_metadata = self._generate_prediction_metadata(
            ensemble_pred, base_predictions, current_regime, confidence_score
        )
        
        return ensemble_pred, confidence_score, prediction_metadata
    
    async def _extract_priority2_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract Priority 2 engineered features"""
        try:
            # Use existing Priority 2 feature engineering
            enhanced_features, metadata = await self.feature_engineering.extract_priority2_features(data, symbol)
            
            # Add target column if not present
            if 'target' not in enhanced_features.columns and 'target' in data.columns:
                enhanced_features['target'] = data['target']
            
            return enhanced_features
            
        except Exception as e:
            logger.warning(f"Priority 2 feature extraction failed: {e}, using original data")
            return data
    
    def _create_enhanced_pattern_features(self, data: pd.DataFrame, pattern_type: PatternType) -> pd.DataFrame:
        """Create enhanced pattern-specific features"""
        features = data.copy()
        
        if pattern_type == PatternType.REVERSAL:
            # Enhanced reversal features
            if 'rsi_divergence' in features.columns:
                features['reversal_strength'] = features['rsi_divergence'].abs()
            
            if 'price_change_abs' in features.columns:
                features['price_reversal_momentum'] = features['price_change_abs'] * features.get('volume_ratio', 1.0)
            
            # Add reversal-specific technical indicators
            features['reversal_volume_confirmation'] = features.get('volume_ratio', 1.0) * features.get('price_change_abs', 0.0)
            
        elif pattern_type == PatternType.CONTINUATION:
            # Enhanced continuation features
            if 'trend_strength' in features.columns:
                features['continuation_strength'] = features['trend_strength']
            
            if 'momentum_20' in features.columns:
                features['trend_momentum'] = features['momentum_20'].abs()
            
            # Add continuation-specific features
            features['consolidation_breakout_potential'] = features.get('volatility_20', 0.0) * features.get('volume_ratio', 1.0)
        
        return features
    
    def _create_enhanced_regime_features(self, data: pd.DataFrame, regime: MarketRegime) -> pd.DataFrame:
        """Create enhanced regime-specific features"""
        features = data.copy()
        
        if regime == MarketRegime.BULL:
            # Enhanced bull market features
            if 'momentum_20' in features.columns:
                features['bull_momentum_strength'] = np.maximum(features['momentum_20'], 0)
            
            features['bull_volume_trend'] = features.get('volume_ratio', 1.0) * features.get('price_change', 0.0)
            
        elif regime == MarketRegime.BEAR:
            # Enhanced bear market features
            if 'momentum_20' in features.columns:
                features['bear_momentum_strength'] = np.maximum(-features['momentum_20'], 0)
            
            features['bear_volume_trend'] = features.get('volume_ratio', 1.0) * np.abs(features.get('price_change', 0.0))
            
        elif regime == MarketRegime.SIDEWAYS:
            # Enhanced sideways market features
            features['range_bound_strength'] = features.get('volatility_20', 0.0)
            features['mean_reversion_potential'] = np.abs(features.get('price_change', 0.0))
            
        elif regime == MarketRegime.VOLATILE:
            # Enhanced volatile market features
            features['volatility_breakout_potential'] = features.get('volatility_20', 0.0) * features.get('volume_ratio', 1.0)
            features['price_gap_strength'] = features.get('price_change_abs', 0.0)
        
        return features
    
    async def _train_enhanced_models(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train enhanced models with advanced configurations"""
        models = {}
        
        # Enhanced LightGBM
        lgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(features, targets)
        models['lightgbm'] = lgb_model
        
        # Enhanced XGBoost
        xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(features, targets)
        models['xgboost'] = xgb_model
        
        # Enhanced Random Forest
        rf_params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(features, targets)
        models['random_forest'] = rf_model
        
        return models
    
    async def _apply_advanced_calibration(self, models: Dict, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Apply advanced probability calibration"""
        calibrated_models = {}
        
        for name, model in models.items():
            if self.calibration_method == "ensemble":
                # Ensemble calibration using multiple methods
                calibrated_model = self._apply_ensemble_calibration(model, features, targets)
            else:
                # Single method calibration
                calibrated_model = CalibratedClassifierCV(
                    model, 
                    method=self.calibration_params[self.calibration_method]['method'],
                    cv=self.calibration_params[self.calibration_method]['cv_folds']
                )
                calibrated_model.fit(features, targets)
            
            calibrated_models[name] = calibrated_model
        
        return calibrated_models
    
    def _apply_ensemble_calibration(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Apply ensemble calibration using multiple methods"""
        # Use isotonic calibration as default for ensemble
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='isotonic',
            cv=5
        )
        calibrated_model.fit(features, targets)
        
        return calibrated_model
    
    def _evaluate_calibrated_models(self, models: Dict, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Dict]:
        """Evaluate calibrated models with advanced metrics"""
        results = {}
        
        for name, model in models.items():
            try:
                # Get calibrated predictions
                pred_proba = model.predict_proba(features)[:, 1]
                
                # Calculate metrics
                roc_auc = roc_auc_score(targets, pred_proba)
                brier_score = brier_score_loss(targets, pred_proba)
                
                # Calculate calibration metrics
                calibration_metrics = self._calculate_calibration_metrics(pred_proba, targets)
                
                results[name] = {
                    'calibrated_auc': roc_auc,
                    'brier_score': brier_score,
                    'calibration_metrics': calibration_metrics,
                    'model': model
                }
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
                results[name] = {
                    'calibrated_auc': 0.0, 
                    'brier_score': 1.0,
                    'calibration_metrics': {},
                    'model': model
                }
        
        return results
    
    def _calculate_calibration_metrics(self, pred_proba: np.ndarray, targets: pd.Series) -> Dict[str, float]:
        """Calculate advanced calibration metrics"""
        try:
            # Calculate reliability diagram metrics
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (pred_proba > bin_lower) & (pred_proba <= bin_upper)
                bin_size = np.sum(in_bin)
                
                if bin_size > 0:
                    bin_accuracy = np.mean(targets[in_bin])
                    bin_confidence = np.mean(pred_proba[in_bin])
                    calibration_error += bin_size * np.abs(bin_accuracy - bin_confidence)
            
            calibration_error /= len(pred_proba)
            
            return {
                'calibration_error': calibration_error,
                'brier_score': brier_score_loss(targets, pred_proba)
            }
            
        except Exception as e:
            logger.error(f"Error calculating calibration metrics: {e}")
            return {'calibration_error': 1.0, 'brier_score': 1.0}
    
    async def _train_regime_ensemble(self, features: pd.DataFrame, targets: pd.Series, regime: MarketRegime) -> Any:
        """Train regime-specific ensemble"""
        # Create ensemble with regime-specific configurations
        models = []
        
        # Adjust parameters based on regime
        if regime == MarketRegime.VOLATILE:
            # Use more conservative parameters for volatile markets
            lgb_params = {'n_estimators': 150, 'learning_rate': 0.03, 'max_depth': 6}
        else:
            # Standard parameters for other regimes
            lgb_params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8}
        
        # Train LightGBM with regime-specific parameters
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
        lgb_model.fit(features, targets)
        models.append(lgb_model)
        
        # Add other models
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(features, targets)
        models.append(xgb_model)
        
        # Select best model
        best_model = self._select_best_model(models, features, targets)
        return best_model
    
    def _enhanced_time_series_cv(self, model: Any, features: pd.DataFrame, targets: pd.Series, n_splits: int = 5) -> List[float]:
        """Enhanced time series cross-validation"""
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
    
    def _calculate_regime_metrics(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Calculate regime-specific performance metrics"""
        try:
            pred_proba = model.predict_proba(features)[:, 1]
            
            return {
                'roc_auc': roc_auc_score(targets, pred_proba),
                'brier_score': brier_score_loss(targets, pred_proba),
                'calibration_error': self._calculate_calibration_metrics(pred_proba, targets)['calibration_error']
            }
        except Exception as e:
            logger.error(f"Error calculating regime metrics: {e}")
            return {'roc_auc': 0.0, 'brier_score': 1.0, 'calibration_error': 1.0}
    
    def _generate_enhanced_predictions(self, features: pd.DataFrame, pattern_models: Dict, regime_models: Dict) -> Dict[str, np.ndarray]:
        """Generate enhanced predictions from base models"""
        base_predictions = {}
        
        # Pattern model predictions
        for pattern_type, model_info in pattern_models.items():
            pattern_features = self._create_enhanced_pattern_features(features, PatternType(pattern_type))
            pred_proba = model_info['model'].predict_proba(pattern_features)[:, 1]
            base_predictions[f'pattern_{pattern_type}'] = pred_proba
        
        # Regime model predictions
        for regime_type, model_info in regime_models.items():
            regime_features = self._create_enhanced_regime_features(features, MarketRegime(regime_type))
            pred_proba = model_info['model'].predict_proba(regime_features)[:, 1]
            base_predictions[f'regime_{regime_type}'] = pred_proba
        
        return base_predictions
    
    def _create_enhanced_meta_features(self, base_predictions: Dict[str, np.ndarray], features: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced meta-features"""
        meta_features = pd.DataFrame(base_predictions)
        
        # Add interaction features
        if len(base_predictions) > 1:
            prediction_names = list(base_predictions.keys())
            for i in range(len(prediction_names)):
                for j in range(i+1, len(prediction_names)):
                    name1, name2 = prediction_names[i], prediction_names[j]
                    meta_features[f'interaction_{name1}_{name2}'] = (
                        base_predictions[name1] * base_predictions[name2]
                    )
        
        # Add statistical features
        meta_features['prediction_mean'] = meta_features.mean(axis=1)
        meta_features['prediction_std'] = meta_features.std(axis=1)
        meta_features['prediction_max'] = meta_features.max(axis=1)
        meta_features['prediction_min'] = meta_features.min(axis=1)
        
        return meta_features
    
    def _train_advanced_meta_learner(self, meta_features: pd.DataFrame, targets: pd.Series) -> Any:
        """Train advanced meta-learner"""
        # Use Gradient Boosting for meta-learner
        meta_learner = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        meta_learner.fit(meta_features, targets)
        return meta_learner
    
    def _calculate_prediction_weights(self, base_predictions: Dict[str, np.ndarray], targets: pd.Series) -> Dict[str, float]:
        """Calculate optimal weights for base model predictions"""
        weights = {}
        
        for name, predictions in base_predictions.items():
            try:
                # Calculate weight based on individual model performance
                auc = roc_auc_score(targets, predictions)
                weights[name] = max(0.1, auc)  # Minimum weight of 0.1
            except Exception as e:
                logger.warning(f"Error calculating weight for {name}: {e}")
                weights[name] = 0.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_enhanced_confidence(self, ensemble_pred: np.ndarray, meta_features: pd.DataFrame, base_predictions: Dict[str, np.ndarray]) -> float:
        """Calculate enhanced confidence score"""
        # Use prediction variance and agreement between models
        prediction_std = np.std(ensemble_pred)
        
        # Calculate agreement between base models
        base_pred_array = np.array(list(base_predictions.values()))
        model_agreement = 1.0 - np.std(base_pred_array, axis=0).mean()
        
        # Combine factors
        confidence = (1.0 - prediction_std) * 0.7 + model_agreement * 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_prediction_metadata(self, ensemble_pred: np.ndarray, base_predictions: Dict[str, np.ndarray], 
                                    current_regime: MarketRegime, confidence_score: float) -> Dict[str, Any]:
        """Generate prediction metadata"""
        metadata = {
            'prediction_mean': float(np.mean(ensemble_pred)),
            'prediction_std': float(np.std(ensemble_pred)),
            'confidence_score': confidence_score,
            'current_regime': current_regime.value,
            'base_model_agreement': 1.0 - float(np.std(np.array(list(base_predictions.values())), axis=0).mean()),
            'timestamp': datetime.now().isoformat()
        }
        
        return metadata
    
    async def _optimize_model_for_onnx(self, model: Any, model_filename: str, model_type: str):
        """Optimize model for ONNX deployment"""
        if not self.enable_onnx:
            return
        
        try:
            # Convert model to ONNX format
            onnx_path = str(self.models_dir / f"{model_filename.replace('.pkl', '.onnx')}")
            
            # This would integrate with the existing ONNX optimization manager
            # For now, just log the intention
            logger.info(f"ONNX optimization requested for {model_type} model: {model_filename}")
            
        except Exception as e:
            logger.warning(f"ONNX optimization failed for {model_filename}: {e}")
    
    async def _optimize_ensemble_for_onnx(self, ensemble_model: Dict, ensemble_filename: str):
        """Optimize ensemble model for ONNX deployment"""
        if not self.enable_onnx:
            return
        
        try:
            # This would handle ensemble model ONNX conversion
            logger.info(f"Ensemble ONNX optimization requested: {ensemble_filename}")
            
        except Exception as e:
            logger.warning(f"Ensemble ONNX optimization failed: {e}")
    
    def _save_enhanced_model(self, model: Any, filename: str):
        """Save enhanced model to disk"""
        try:
            filepath = self.models_dir / filename
            joblib.dump(model, filepath)
            logger.info(f"Enhanced model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving enhanced model: {e}")
    
    def _get_enhanced_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get enhanced feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return {}
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.error(f"Error getting enhanced feature importance: {e}")
            return {}
    
    def _select_best_model(self, models: List, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Select the best performing model"""
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
        """Create a copy of a model"""
        if hasattr(model, 'clone'):
            return model.clone()
        else:
            return type(model)()
    
    def get_enhanced_performance_summary(self) -> Dict[str, Any]:
        """Get summary of enhanced model performance"""
        return {
            'enhanced_pattern_models': {
                name: {
                    'performance': info.get('performance', {}),
                    'calibration_metrics': info.get('calibration_metrics', {})
                }
                for name, info in self.enhanced_pattern_models.items()
            },
            'enhanced_regime_models': {
                name: {
                    'cv_scores': info.get('cv_scores', []),
                    'regime_metrics': info.get('regime_metrics', {})
                }
                for name, info in self.enhanced_regime_models.items()
            },
            'calibration_performance': self.calibration_performance,
            'pattern_detection_accuracy': self.pattern_detection_accuracy,
            'regime_adaptation_metrics': self.regime_adaptation_metrics
        }
