"""
Model Registry for AlphaPulse
Pre-loads models in memory for fast inference access
"""

import asyncio
import logging
import os
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import existing ML components
from .feature_engineering import FeatureExtractor
from .model_accuracy_improvement import ModelAccuracyImprovement, PatternType, MarketRegime

# Import ONNX components
try:
    from .onnx_converter import onnx_converter
    from .onnx_inference import onnx_inference_engine
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX components not available. Install with: pip install skl2onnx")

# Import Phase 3 optimization components
try:
    from .mixed_precision_engine import mixed_precision_engine
    from .advanced_batching import advanced_batching_system
    from .gpu_memory_manager import gpu_memory_manager
    from .model_quantization import model_quantization_system
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False
    logging.warning("Phase 3 optimization components not available")

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Global model registry that pre-loads all models at startup
    for instant inference access without load delays.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Pre-loaded models storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_extractors: Dict[str, Any] = {}
        self.ensembles: Dict[str, Any] = {}
        
        # Model metadata
        self.model_metadata: Dict[str, Dict] = {}
        self.last_updated: Dict[str, datetime] = {}
        
        # Performance tracking
        self.inference_times: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.model_improvement = ModelAccuracyImprovement()
        
        # ONNX support
        self.onnx_enabled = ONNX_AVAILABLE
        self.onnx_models = {}  # Store ONNX model paths
        
        # Phase 3 optimization support
        self.phase3_enabled = PHASE3_AVAILABLE
        self.mixed_precision_enabled = False
        self.advanced_batching_enabled = False
        self.gpu_memory_optimization_enabled = False
        self.quantization_enabled = False
        
        # Initialize Phase 3 components if available
        if self.phase3_enabled:
            self._initialize_phase3_components()
        
        logger.info(f"ModelRegistry initialized (ONNX: {'enabled' if self.onnx_enabled else 'disabled'}, Phase3: {'enabled' if self.phase3_enabled else 'disabled'})")
    
    async def load_all_models(self) -> bool:
        """
        Load all models at startup to eliminate load delays during inference.
        Returns True if all models loaded successfully.
        """
        logger.info("🔄 Loading all models into memory...")
        
        try:
            # Load feature extractors
            await self._load_feature_extractors()
            
            # Load pattern-specific models
            await self._load_pattern_models()
            
            # Load market regime models
            await self._load_regime_models()
            
            # Load ensemble models
            await self._load_ensemble_models()
            
            # Load scalers
            await self._load_scalers()
            
            # Convert models to ONNX if enabled
            if self.onnx_enabled:
                await self._convert_models_to_onnx()
                await self._load_onnx_models()
            
            # Apply Phase 3 optimizations if enabled
            if self.phase3_enabled:
                await self._apply_phase3_optimizations()
            
            logger.info(f"✅ Successfully loaded {len(self.models)} models into memory")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    async def _load_feature_extractors(self):
        """Load feature extractors for different timeframes"""
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for timeframe in timeframes:
            model_name = f"feature_extractor_{timeframe}"
            self.feature_extractors[model_name] = self.feature_extractor
            self.model_metadata[model_name] = {
                'type': 'feature_extractor',
                'timeframe': timeframe,
                'created_at': datetime.now()
            }
    
    async def _load_pattern_models(self):
        """Load pattern-specific models for reversal and continuation patterns"""
        pattern_types = [PatternType.REVERSAL, PatternType.CONTINUATION]
        
        for pattern_type in pattern_types:
            model_name = f"pattern_model_{pattern_type.value}"
            
            # Create and train a simple model for testing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Generate sample training data
            sample_data = self._generate_sample_training_data()
            features, _ = self.feature_extractor.extract_features(sample_data, symbol="BTCUSDT")
            
            # Create sample labels (alternating for testing)
            labels = np.array([0, 1] * (len(features) // 2 + 1))[:len(features)]
            
            # Train the model
            model.fit(features, labels)
            self.models[model_name] = model
            
            self.model_metadata[model_name] = {
                'type': 'pattern_model',
                'pattern_type': pattern_type.value,
                'created_at': datetime.now()
            }
    
    async def _load_regime_models(self):
        """Load market regime-specific models"""
        regimes = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]
        
        for regime in regimes:
            model_name = f"regime_model_{regime.value}"
            
            # Create and train a simple model for testing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Generate sample training data
            sample_data = self._generate_sample_training_data()
            features, _ = self.feature_extractor.extract_features(sample_data, symbol="BTCUSDT")
            
            # Create sample labels (alternating for testing)
            labels = np.array([0, 1] * (len(features) // 2 + 1))[:len(features)]
            
            # Train the model
            model.fit(features, labels)
            self.models[model_name] = model
            
            self.model_metadata[model_name] = {
                'type': 'regime_model',
                'market_regime': regime.value,
                'created_at': datetime.now()
            }
    
    async def _load_ensemble_models(self):
        """Load ensemble models for final prediction"""
        ensemble_name = "ensemble_meta_learner"
        
        # Create and train a simple ensemble model for testing
        from sklearn.ensemble import RandomForestClassifier
        ensemble = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Generate sample training data for ensemble
        sample_data = self._generate_sample_training_data()
        features, _ = self.feature_extractor.extract_features(sample_data, symbol="BTCUSDT")
        
        # Create sample meta-features (pattern and regime predictions)
        meta_features = []
        for i in range(len(features)):
            meta_features.append({
                'pattern_reversal': np.random.random(),
                'pattern_continuation': np.random.random(),
                'regime_bull': np.random.random(),
                'regime_bear': np.random.random(),
                'regime_sideways': np.random.random()
            })
        
        meta_features_df = pd.DataFrame(meta_features)
        labels = np.array([0, 1] * (len(meta_features_df) // 2 + 1))[:len(meta_features_df)]
        
        # Train the ensemble
        ensemble.fit(meta_features_df, labels)
        self.ensembles[ensemble_name] = ensemble
        
        self.model_metadata[ensemble_name] = {
            'type': 'ensemble',
            'created_at': datetime.now()
        }
    
    async def _load_scalers(self):
        """Load feature scalers for different symbols"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        
        for symbol in symbols:
            scaler_name = f"scaler_{symbol}"
            
            # Create scaler for this symbol
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.scalers[scaler_name] = scaler
            
            self.model_metadata[scaler_name] = {
                'type': 'scaler',
                'symbol': symbol,
                'created_at': datetime.now()
            }
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a pre-loaded model by name"""
        if model_name in self.models:
            self.cache_hits += 1
            return self.models[model_name]
        
        self.cache_misses += 1
        logger.warning(f"Model {model_name} not found in registry")
        return None
    
    def get_scaler(self, scaler_name: str) -> Optional[Any]:
        """Get a pre-loaded scaler by name"""
        if scaler_name in self.scalers:
            return self.scalers[scaler_name]
        
        logger.warning(f"Scaler {scaler_name} not found in registry")
        return None
    
    def get_feature_extractor(self, timeframe: str) -> Optional[Any]:
        """Get feature extractor for specific timeframe"""
        model_name = f"feature_extractor_{timeframe}"
        if model_name in self.feature_extractors:
            return self.feature_extractors[model_name]
        
        logger.warning(f"Feature extractor for {timeframe} not found in registry")
        return None
    
    def get_ensemble(self, ensemble_name: str = "ensemble_meta_learner") -> Optional[Any]:
        """Get ensemble model"""
        if ensemble_name in self.ensembles:
            return self.ensembles[ensemble_name]
        
        logger.warning(f"Ensemble {ensemble_name} not found in registry")
        return None
    
    async def predict_pattern(self, data: pd.DataFrame, pattern_type: PatternType) -> Dict[str, Any]:
        """Make pattern-specific prediction using pre-loaded model"""
        start_time = datetime.now()
        
        try:
            model_name = f"pattern_model_{pattern_type.value}"
            
            # Extract features
            features_df, _ = self.feature_extractor.extract_features(data, symbol="BTCUSDT")
            # Convert DataFrame to numpy array and take the last row
            features = features_df.values[-1] if len(features_df.values) > 0 else np.zeros(features_df.shape[1])
            features = features.reshape(1, -1)  # Ensure 2D array
            
            # Try ONNX inference first if available
            if self.onnx_enabled and model_name in onnx_inference_engine.sessions:
                try:
                    prediction = onnx_inference_engine.predict_proba(model_name, features)[0]
                    inference_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    # Track performance
                    if model_name not in self.inference_times:
                        self.inference_times[model_name] = []
                    self.inference_times[model_name].append(inference_time)
                    
                    return {
                        'prediction': prediction,
                        'confidence': prediction,
                        'inference_time_ms': inference_time,
                        'model_name': model_name,
                        'inference_engine': 'onnx'
                    }
                except Exception as onnx_error:
                    logger.warning(f"ONNX inference failed for {model_name}, falling back to scikit-learn: {onnx_error}")
            
            # Fallback to scikit-learn model
            model = self.get_model(model_name)
            if model is None:
                return {'prediction': 0.0, 'confidence': 0.0, 'error': 'Model not found'}
            
            # Make prediction
            prediction = model.predict_proba(features)[0][1]  # Probability of positive class
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            # Track performance
            if model_name not in self.inference_times:
                self.inference_times[model_name] = []
            self.inference_times[model_name].append(inference_time)
            
            return {
                'prediction': prediction,
                'confidence': prediction,
                'inference_time_ms': inference_time,
                'model_name': model_name,
                'inference_engine': 'scikit-learn'
            }
            
        except Exception as e:
            logger.error(f"Error in pattern prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    async def predict_regime(self, data: pd.DataFrame, regime: MarketRegime) -> Dict[str, Any]:
        """Make regime-specific prediction using pre-loaded model"""
        start_time = datetime.now()
        
        try:
            model_name = f"regime_model_{regime.value}"
            
            # Extract features
            features_df, _ = self.feature_extractor.extract_features(data, symbol="BTCUSDT")
            # Convert DataFrame to numpy array and take the last row
            features = features_df.values[-1] if len(features_df.values) > 0 else np.zeros(features_df.shape[1])
            features = features.reshape(1, -1)  # Ensure 2D array
            
            # Try ONNX inference first if available
            if self.onnx_enabled and model_name in onnx_inference_engine.sessions:
                try:
                    prediction = onnx_inference_engine.predict_proba(model_name, features)[0]
                    inference_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    # Track performance
                    if model_name not in self.inference_times:
                        self.inference_times[model_name] = []
                    self.inference_times[model_name].append(inference_time)
                    
                    return {
                        'prediction': prediction,
                        'confidence': prediction,
                        'inference_time_ms': inference_time,
                        'model_name': model_name,
                        'inference_engine': 'onnx'
                    }
                except Exception as onnx_error:
                    logger.warning(f"ONNX inference failed for {model_name}, falling back to scikit-learn: {onnx_error}")
            
            # Fallback to scikit-learn model
            model = self.get_model(model_name)
            if model is None:
                return {'prediction': 0.0, 'confidence': 0.0, 'error': 'Model not found'}
            
            # Make prediction
            prediction = model.predict_proba(features)[0][1]  # Probability of positive class
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            # Track performance
            if model_name not in self.inference_times:
                self.inference_times[model_name] = []
            self.inference_times[model_name].append(inference_time)
            
            return {
                'prediction': prediction,
                'confidence': prediction,
                'inference_time_ms': inference_time,
                'model_name': model_name,
                'inference_engine': 'scikit-learn'
            }
            
        except Exception as e:
            logger.error(f"Error in regime prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    async def predict_ensemble(self, data: pd.DataFrame, current_regime: MarketRegime) -> Dict[str, Any]:
        """Make ensemble prediction using pre-loaded models"""
        start_time = datetime.now()
        
        try:
            ensemble = self.get_ensemble()
            
            if ensemble is None:
                return {'prediction': 0.0, 'confidence': 0.0, 'error': 'Ensemble not found'}
            
            # Get predictions from all pattern models
            pattern_predictions = {}
            for pattern_type in [PatternType.REVERSAL, PatternType.CONTINUATION]:
                result = await self.predict_pattern(data, pattern_type)
                pattern_predictions[f"pattern_{pattern_type.value}"] = result['prediction']
            
            # Get predictions from all regime models
            regime_predictions = {}
            for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]:
                result = await self.predict_regime(data, regime)
                regime_predictions[f"regime_{regime.value}"] = result['prediction']
            
            # Combine predictions for ensemble
            meta_features = {**pattern_predictions, **regime_predictions}
            
            # Make ensemble prediction
            meta_features_df = pd.DataFrame([meta_features])
            final_prediction = ensemble.predict_proba(meta_features_df)[0][1]
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            return {
                'prediction': final_prediction,
                'confidence': final_prediction,
                'inference_time_ms': inference_time,
                'pattern_predictions': pattern_predictions,
                'regime_predictions': regime_predictions,
                'model_name': 'ensemble_meta_learner'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {
            'total_models': len(self.models),
            'total_scalers': len(self.scalers),
            'total_feature_extractors': len(self.feature_extractors),
            'total_ensembles': len(self.ensembles),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'model_performance': {}
        }
        
        # Calculate average inference times
        for model_name, times in self.inference_times.items():
            if times:
                stats['model_performance'][model_name] = {
                    'avg_inference_time_ms': np.mean(times),
                    'min_inference_time_ms': np.min(times),
                    'max_inference_time_ms': np.max(times),
                    'total_predictions': len(times)
                }
        
        return stats
    
    async def update_model(self, model_name: str, new_model: Any) -> bool:
        """Update a model in the registry"""
        try:
            self.models[model_name] = new_model
            self.last_updated[model_name] = datetime.now()
            logger.info(f"Updated model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
            return False
    
    async def save_models(self) -> bool:
        """Save all models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"Saved {len(self.models)} models to disk")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def _generate_sample_training_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data for training"""
        np.random.seed(42)
        
        # Generate 200 rows of sample data
        rows = 200
        base_price = 50000
        
        # Generate price movements
        returns = np.random.normal(0, 0.02, rows)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.1))
        
        # Generate OHLCV data
        data = []
        for i in range(rows):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=rows-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _convert_models_to_onnx(self):
        """Convert all models to ONNX format"""
        try:
            logger.info("🔄 Converting models to ONNX format...")
            
            # Convert pattern and regime models
            for model_name, model in self.models.items():
                if 'pattern_model' in model_name or 'regime_model' in model_name:
                    onnx_path = onnx_converter.convert_model(model, model_name)
                    if onnx_path:
                        self.onnx_models[model_name] = onnx_path
            
            # Convert ensemble models
            for ensemble_name, ensemble in self.ensembles.items():
                onnx_path = onnx_converter.convert_model(ensemble, ensemble_name)
                if onnx_path:
                    self.onnx_models[ensemble_name] = onnx_path
            
            logger.info(f"✅ Converted {len(self.onnx_models)} models to ONNX")
            
        except Exception as e:
            logger.error(f"❌ Failed to convert models to ONNX: {e}")
    
    async def _load_onnx_models(self):
        """Load ONNX models into inference engine"""
        try:
            if self.onnx_models:
                logger.info("🔄 Loading ONNX models into inference engine...")
                results = onnx_inference_engine.load_all_models(self.onnx_models)
                loaded_count = sum(results.values())
                logger.info(f"✅ Loaded {loaded_count}/{len(self.onnx_models)} ONNX models")
            else:
                logger.warning("No ONNX models available to load")
                
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX models: {e}")
    
    async def load_models_from_disk(self) -> bool:
        """Load models from disk if they exist"""
        try:
            for model_path in self.models_dir.glob("*.pkl"):
                model_name = model_path.stem
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
            
            logger.info(f"Loaded {len(self.models)} models from disk")
            return True
        except Exception as e:
            logger.error(f"Error loading models from disk: {e}")
            return False
    
    def _initialize_phase3_components(self):
        """Initialize Phase 3 optimization components"""
        try:
            # Enable mixed precision if GPU is available
            if mixed_precision_engine.providers and any('CUDA' in provider[0] for provider in mixed_precision_engine.providers):
                self.mixed_precision_enabled = True
                logger.info("✅ Mixed precision (FP16) enabled")
            
            # Enable advanced batching
            self.advanced_batching_enabled = True
            logger.info("✅ Advanced batching enabled")
            
            # Enable GPU memory optimization if NVML is available
            if gpu_memory_manager.nvml_available:
                self.gpu_memory_optimization_enabled = True
                logger.info("✅ GPU memory optimization enabled")
            
            # Enable quantization
            self.quantization_enabled = True
            logger.info("✅ Model quantization enabled")
            
        except Exception as e:
            logger.error(f"Error initializing Phase 3 components: {e}")
    
    async def _apply_phase3_optimizations(self):
        """Apply Phase 3 optimizations to all models"""
        try:
            logger.info("🔄 Applying Phase 3 optimizations...")
            
            # Apply mixed precision optimization
            if self.mixed_precision_enabled:
                await self._apply_mixed_precision_optimization()
            
            # Apply advanced batching optimization
            if self.advanced_batching_enabled:
                await self._apply_advanced_batching_optimization()
            
            # Apply GPU memory optimization
            if self.gpu_memory_optimization_enabled:
                await self._apply_gpu_memory_optimization()
            
            # Apply quantization optimization
            if self.quantization_enabled:
                await self._apply_quantization_optimization()
            
            logger.info("✅ Phase 3 optimizations applied successfully")
            
        except Exception as e:
            logger.error(f"❌ Error applying Phase 3 optimizations: {e}")
    
    async def _apply_mixed_precision_optimization(self):
        """Apply mixed precision optimization"""
        try:
            logger.info("🔄 Applying mixed precision optimization...")
            
            # Convert ONNX models to use mixed precision
            for model_name, onnx_path in self.onnx_models.items():
                try:
                    # Load with mixed precision support
                    session = mixed_precision_engine.load_model_with_mixed_precision(onnx_path, model_name)
                    if session:
                        # Store mixed precision session
                        if not hasattr(self, 'mixed_precision_sessions'):
                            self.mixed_precision_sessions = {}
                        self.mixed_precision_sessions[model_name] = session
                        
                except Exception as e:
                    logger.warning(f"Failed to apply mixed precision to {model_name}: {e}")
            
            logger.info(f"✅ Mixed precision optimization applied to {len(getattr(self, 'mixed_precision_sessions', {}))} models")
            
        except Exception as e:
            logger.error(f"Error applying mixed precision optimization: {e}")
    
    async def _apply_advanced_batching_optimization(self):
        """Apply advanced batching optimization"""
        try:
            logger.info("🔄 Applying advanced batching optimization...")
            
            # Configure advanced batching for different model types
            advanced_batching_system.set_target_latency(50.0)  # 50ms target latency
            advanced_batching_system.set_batch_size_bounds(10, 500)  # Batch size bounds
            
            logger.info("✅ Advanced batching optimization configured")
            
        except Exception as e:
            logger.error(f"Error applying advanced batching optimization: {e}")
    
    async def _apply_gpu_memory_optimization(self):
        """Apply GPU memory optimization"""
        try:
            logger.info("🔄 Applying GPU memory optimization...")
            
            # Start GPU memory monitoring
            gpu_memory_manager.start_monitoring()
            
            # Optimize memory usage
            optimization_result = gpu_memory_manager.optimize_memory_usage()
            if optimization_result.get('optimization_applied'):
                logger.info(f"✅ GPU memory optimization applied: {optimization_result}")
            
            logger.info("✅ GPU memory optimization applied")
            
        except Exception as e:
            logger.error(f"Error applying GPU memory optimization: {e}")
    
    async def _apply_quantization_optimization(self):
        """Apply quantization optimization"""
        try:
            logger.info("🔄 Applying quantization optimization...")
            
            # Quantize models to INT8
            quantized_models = model_quantization_system.quantize_model_registry(self)
            
            if quantized_models:
                # Store quantized model paths
                if not hasattr(self, 'quantized_models'):
                    self.quantized_models = {}
                self.quantized_models.update(quantized_models)
                
                logger.info(f"✅ Quantization optimization applied to {len(quantized_models)} models")
            else:
                logger.warning("No models were successfully quantized")
            
        except Exception as e:
            logger.error(f"Error applying quantization optimization: {e}")
    
    async def predict_with_phase3_optimizations(self, data: pd.DataFrame, 
                                              prediction_type: str = "pattern",
                                              model_type: str = "reversal") -> Dict[str, Any]:
        """
        Make prediction using Phase 3 optimizations.
        
        Args:
            data: Input data
            prediction_type: Type of prediction ("pattern", "regime", "ensemble")
            model_type: Specific model type
            
        Returns:
            Prediction result with optimization details
        """
        start_time = datetime.now()
        
        try:
            # Extract features
            features_df, _ = self.feature_extractor.extract_features(data, symbol="BTCUSDT")
            features = features_df.values[-1] if len(features_df.values) > 0 else np.zeros(features_df.shape[1])
            features = features.reshape(1, -1)
            
            result = {
                'prediction': 0.0,
                'confidence': 0.0,
                'inference_time_ms': 0.0,
                'optimizations_used': [],
                'error': None
            }
            
            # Try mixed precision inference first
            if self.mixed_precision_enabled and hasattr(self, 'mixed_precision_sessions'):
                model_name = f"{prediction_type}_model_{model_type}"
                if model_name in self.mixed_precision_sessions:
                    try:
                        session = self.mixed_precision_sessions[model_name]
                        prediction = mixed_precision_engine.predict_with_mixed_precision(session, features)
                        
                        if prediction is not None:
                            inference_time = (datetime.now() - start_time).total_seconds() * 1000
                            result.update({
                                'prediction': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                                'confidence': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                                'inference_time_ms': inference_time,
                                'optimizations_used': ['mixed_precision', 'fp16']
                            })
                            return result
                            
                    except Exception as e:
                        logger.warning(f"Mixed precision inference failed: {e}")
            
            # Try quantized model inference
            if self.quantization_enabled and hasattr(self, 'quantized_models'):
                model_name = f"{prediction_type}_model_{model_type}"
                if model_name in self.quantized_models:
                    try:
                        quantized_path = self.quantized_models[model_name]
                        prediction = model_quantization_system._predict_with_quantized_model(quantized_path, features)
                        
                        if prediction is not None:
                            inference_time = (datetime.now() - start_time).total_seconds() * 1000
                            result.update({
                                'prediction': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                                'confidence': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                                'inference_time_ms': inference_time,
                                'optimizations_used': ['quantization', 'int8']
                            })
                            return result
                            
                    except Exception as e:
                        logger.warning(f"Quantized inference failed: {e}")
            
            # Fallback to ONNX inference
            if self.onnx_enabled:
                if prediction_type == "pattern":
                    pattern_type = PatternType(model_type.upper())
                    fallback_result = await self.predict_pattern(data, pattern_type)
                elif prediction_type == "regime":
                    regime = MarketRegime(model_type.upper())
                    fallback_result = await self.predict_regime(data, regime)
                else:
                    regime = MarketRegime.BULL  # Default
                    fallback_result = await self.predict_ensemble(data, regime)
                
                result.update(fallback_result)
                result['optimizations_used'].append('onnx')
                return result
            
            # Final fallback to scikit-learn
            if prediction_type == "pattern":
                pattern_type = PatternType(model_type.upper())
                fallback_result = await self.predict_pattern(data, pattern_type)
            elif prediction_type == "regime":
                regime = MarketRegime(model_type.upper())
                fallback_result = await self.predict_regime(data, regime)
            else:
                regime = MarketRegime.BULL  # Default
                fallback_result = await self.predict_ensemble(data, regime)
            
            result.update(fallback_result)
            result['optimizations_used'].append('scikit-learn')
            return result
            
        except Exception as e:
            logger.error(f"Error in Phase 3 prediction: {e}")
            result['error'] = str(e)
            return result
    
    def get_phase3_stats(self) -> Dict[str, Any]:
        """Get Phase 3 optimization statistics"""
        stats = {
            'phase3_enabled': self.phase3_enabled,
            'mixed_precision_enabled': self.mixed_precision_enabled,
            'advanced_batching_enabled': self.advanced_batching_enabled,
            'gpu_memory_optimization_enabled': self.gpu_memory_optimization_enabled,
            'quantization_enabled': self.quantization_enabled
        }
        
        if self.phase3_enabled:
            # Mixed precision stats
            if self.mixed_precision_enabled:
                stats['mixed_precision'] = mixed_precision_engine.get_performance_stats()
            
            # Advanced batching stats
            if self.advanced_batching_enabled:
                stats['advanced_batching'] = advanced_batching_system.get_performance_stats()
            
            # GPU memory stats
            if self.gpu_memory_optimization_enabled:
                stats['gpu_memory'] = gpu_memory_manager.get_memory_stats()
            
            # Quantization stats
            if self.quantization_enabled:
                stats['quantization'] = model_quantization_system.get_quantization_stats()
        
        return stats


# Global model registry instance
model_registry = ModelRegistry()
