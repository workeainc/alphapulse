#!/usr/bin/env python3
"""
Test script for Priority 3: Enhanced Model Accuracy

Tests:
1. Enhanced pattern-specific models (reversal vs continuation)
2. Advanced probability calibration (Platt scaling, isotonic regression)
3. Market condition adaptation (bull/bear/sideways specific models)
4. Integration with Priority 2 feature engineering
5. ONNX optimization integration
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import Priority 3 system
from ..ai.priority3_model_accuracy import Priority3ModelAccuracy
from ..ai.model_accuracy_improvement import PatternType, MarketRegime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Priority3ModelAccuracyTester:
    """Test suite for Priority 3 Enhanced Model Accuracy"""
    
    def __init__(self):
        self.priority3_system = Priority3ModelAccuracy(
            models_dir="test_models/priority3",
            calibration_method="isotonic",
            ensemble_size=3,
            enable_onnx=True
        )
        
        # Test data storage
        self.test_data = {}
        self.test_results = {}
        
    def generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data with realistic market patterns"""
        logger.info(f"Generating {n_samples} synthetic test samples...")
        
        # Create time index
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        
        # Generate realistic price data
        np.random.seed(42)
        
        # Base price trend
        base_price = 50000 + np.cumsum(np.random.normal(0, 100, n_samples))
        
        # Add market cycles
        cycle = 50 * np.sin(np.arange(n_samples) * 2 * np.pi / 200)
        base_price += cycle
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': base_price + np.random.normal(0, 50, n_samples),
            'high': base_price + np.random.normal(50, 100, n_samples),
            'low': base_price + np.random.normal(-50, 100, n_samples),
            'close': base_price + np.random.normal(0, 50, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Calculate basic features
        data['price_change'] = data['close'].pct_change()
        data['price_change_abs'] = data['price_change'].abs()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Add technical indicators
        data['rsi'] = self._calculate_rsi(data['close'])
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['atr'] = self._calculate_atr(data)
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        data['volatility_20'] = data['price_change'].rolling(20).std()
        
        # Add RSI divergence
        data['rsi_divergence'] = self._calculate_rsi_divergence(data)
        
        # Generate target labels (simplified)
        data['target'] = (data['price_change'].shift(-1) > 0).astype(int)
        
        # Clean up NaN values
        data = data.dropna()
        
        # Remove timestamp column to avoid dtype issues
        if 'timestamp' in data.columns:
            data = data.drop('timestamp', axis=1)
        
        logger.info(f"✅ Generated test data with {len(data)} samples")
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate simplified RSI divergence"""
        rsi = data['rsi']
        price = data['close']
        
        divergence = pd.Series(0.0, index=data.index)
        
        for i in range(20, len(data)):
            # Simple divergence detection
            if (price.iloc[i] < price.iloc[i-10] and rsi.iloc[i] > rsi.iloc[i-10]):
                divergence.iloc[i] = 1.0  # Bullish divergence
            elif (price.iloc[i] > price.iloc[i-10] and rsi.iloc[i] < rsi.iloc[i-10]):
                divergence.iloc[i] = -1.0  # Bearish divergence
        
        return divergence
    
    def classify_patterns(self, data: pd.DataFrame) -> pd.Series:
        """Classify patterns as reversal or continuation"""
        pattern_labels = []
        
        for i in range(len(data)):
            if i < 20:
                pattern_labels.append(PatternType.CONTINUATION.value)
                continue
            
            # Simple pattern classification with more lenient thresholds
            price_change = abs(data.iloc[i]['price_change'])
            volume_ratio = data.iloc[i]['volume_ratio']
            
            # More lenient thresholds to ensure we get both pattern types
            if price_change > 0.01 and volume_ratio > 1.2:  # Lowered thresholds
                pattern_labels.append(PatternType.REVERSAL.value)
            else:
                pattern_labels.append(PatternType.CONTINUATION.value)
        
        return pd.Series(pattern_labels, index=data.index)
    
    def classify_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Classify market regimes"""
        regime_labels = []
        
        for i in range(len(data)):
            if i < 50:
                regime_labels.append(MarketRegime.SIDEWAYS.value)
                continue
            
            # Simple regime classification with more lenient thresholds
            momentum = data.iloc[i]['momentum_20']
            volatility = data.iloc[i]['volatility_20']
            
            # More lenient thresholds to ensure we get different regimes
            if volatility > 0.02:  # Lowered threshold
                regime_labels.append(MarketRegime.VOLATILE.value)
            elif momentum > 0.02:  # Lowered threshold
                regime_labels.append(MarketRegime.BULL.value)
            elif momentum < -0.02:  # Lowered threshold
                regime_labels.append(MarketRegime.BEAR.value)
            else:
                regime_labels.append(MarketRegime.SIDEWAYS.value)
        
        return pd.Series(regime_labels, index=data.index)
    
    async def test_enhanced_pattern_models(self) -> bool:
        """Test enhanced pattern-specific models"""
        logger.info("🧪 Testing Enhanced Pattern Models...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(1000)
            pattern_labels = self.classify_patterns(test_data)
            
            # Train enhanced pattern models
            pattern_results = await self.priority3_system.train_enhanced_pattern_models(
                test_data, pattern_labels, "BTCUSDT"
            )
            
            # Validate results
            success = True
            pattern_count = 0
            
            for pattern_type, result in pattern_results.items():
                if result and 'performance' in result:
                    pattern_count += 1
                    best_model_name = result.get('best_model_name', 'unknown')
                    calibrated_auc = result['performance'].get(best_model_name, {}).get('calibrated_auc', 0.0)
                    
                    logger.info(f"  {pattern_type} pattern model: {best_model_name} - AUC: {calibrated_auc:.3f}")
                    
                    if calibrated_auc < 0.5:
                        logger.warning(f"  ⚠️ Low AUC for {pattern_type} pattern model: {calibrated_auc:.3f}")
                        success = False
            
            logger.info(f"✅ Enhanced Pattern Models: {pattern_count} models trained")
            return success and pattern_count > 0
            
        except Exception as e:
            logger.error(f"❌ Enhanced Pattern Models test failed: {e}")
            return False
    
    async def test_enhanced_regime_models(self) -> bool:
        """Test enhanced market regime-specific models"""
        logger.info("🧪 Testing Enhanced Regime Models...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(1000)
            regime_labels = self.classify_market_regimes(test_data)
            
            # Train enhanced regime models
            regime_results = await self.priority3_system.train_enhanced_regime_models(
                test_data, regime_labels, "BTCUSDT"
            )
            
            # Validate results
            success = True
            regime_count = 0
            
            for regime_type, result in regime_results.items():
                if result and 'cv_scores' in result:
                    regime_count += 1
                    avg_cv_score = result.get('avg_cv_score', 0.0)
                    regime_metrics = result.get('regime_metrics', {})
                    
                    logger.info(f"  {regime_type} regime model: CV Score: {avg_cv_score:.3f}")
                    logger.info(f"    ROC AUC: {regime_metrics.get('roc_auc', 0.0):.3f}")
                    logger.info(f"    Brier Score: {regime_metrics.get('brier_score', 1.0):.3f}")
                    
                    if avg_cv_score < 0.5:
                        logger.warning(f"  ⚠️ Low CV score for {regime_type} regime model: {avg_cv_score:.3f}")
                        success = False
            
            logger.info(f"✅ Enhanced Regime Models: {regime_count} models trained")
            return success and regime_count > 0
            
        except Exception as e:
            logger.error(f"❌ Enhanced Regime Models test failed: {e}")
            return False
    
    async def test_enhanced_ensemble(self) -> bool:
        """Test enhanced ensemble model creation"""
        logger.info("🧪 Testing Enhanced Ensemble Model...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(1000)
            pattern_labels = self.classify_patterns(test_data)
            regime_labels = self.classify_market_regimes(test_data)
            
            # Train pattern and regime models first
            pattern_results = await self.priority3_system.train_enhanced_pattern_models(
                test_data, pattern_labels, "BTCUSDT"
            )
            
            regime_results = await self.priority3_system.train_enhanced_regime_models(
                test_data, regime_labels, "BTCUSDT"
            )
            
            # Create enhanced ensemble
            ensemble_result = await self.priority3_system.create_enhanced_ensemble(
                test_data, pattern_results, regime_results, "BTCUSDT"
            )
            
            # Validate ensemble
            if ensemble_result and 'meta_learner' in ensemble_result:
                logger.info("✅ Enhanced Ensemble Model created successfully")
                logger.info(f"  Base models: {len(ensemble_result.get('base_models', {}).get('pattern_models', {}))} pattern + {len(ensemble_result.get('base_models', {}).get('regime_models', {}))} regime")
                logger.info(f"  Meta-learner: {type(ensemble_result['meta_learner']).__name__}")
                return True
            else:
                logger.error("❌ Enhanced Ensemble Model creation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced Ensemble test failed: {e}")
            return False
    
    async def test_prediction_with_enhanced_ensemble(self) -> bool:
        """Test predictions with enhanced ensemble"""
        logger.info("🧪 Testing Enhanced Ensemble Predictions...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(100)
            current_regime = MarketRegime.BULL  # Test with bull market regime
            
            # Make predictions
            predictions, confidence_score, metadata = await self.priority3_system.predict_with_enhanced_ensemble(
                test_data, current_regime, "BTCUSDT"
            )
            
            # Validate predictions
            if predictions is not None and len(predictions) > 0:
                logger.info(f"✅ Enhanced Ensemble Predictions: {len(predictions)} predictions generated")
                logger.info(f"  Confidence Score: {confidence_score:.3f}")
                logger.info(f"  Prediction Mean: {metadata.get('prediction_mean', 0.0):.3f}")
                logger.info(f"  Prediction Std: {metadata.get('prediction_std', 0.0):.3f}")
                logger.info(f"  Current Regime: {metadata.get('current_regime', 'unknown')}")
                logger.info(f"  Model Agreement: {metadata.get('base_model_agreement', 0.0):.3f}")
                
                return True
            else:
                logger.error("❌ Enhanced Ensemble Predictions failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced Ensemble Predictions test failed: {e}")
            return False
    
    async def test_probability_calibration(self) -> bool:
        """Test probability calibration methods"""
        logger.info("🧪 Testing Probability Calibration...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(500)
            
            # Test different calibration methods
            calibration_methods = ['platt', 'isotonic', 'ensemble']
            success_count = 0
            
            for method in calibration_methods:
                try:
                    # Create system with specific calibration method
                    calibration_system = Priority3ModelAccuracy(
                        models_dir=f"test_models/priority3_{method}",
                        calibration_method=method,
                        ensemble_size=2,
                        enable_onnx=False
                    )
                    
                    # Train a simple model to test calibration
                    pattern_labels = self.classify_patterns(test_data)
                    pattern_results = await calibration_system.train_enhanced_pattern_models(
                        test_data, pattern_labels, "BTCUSDT"
                    )
                    
                    if pattern_results:
                        success_count += 1
                        logger.info(f"  ✅ {method.upper()} calibration: Success")
                    else:
                        logger.warning(f"  ⚠️ {method.upper()} calibration: No models trained")
                        
                except Exception as e:
                    logger.error(f"  ❌ {method.upper()} calibration failed: {e}")
            
            logger.info(f"✅ Probability Calibration: {success_count}/{len(calibration_methods)} methods successful")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Probability Calibration test failed: {e}")
            return False
    
    async def test_integration_with_priority2(self) -> bool:
        """Test integration with Priority 2 feature engineering"""
        logger.info("🧪 Testing Priority 2 Integration...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(500)
            
            # Test feature extraction integration
            enhanced_features = await self.priority3_system._extract_priority2_features(test_data, "BTCUSDT")
            
            if enhanced_features is not None and len(enhanced_features) > 0:
                logger.info(f"✅ Priority 2 Integration: {len(enhanced_features)} enhanced features extracted")
                logger.info(f"  Feature columns: {list(enhanced_features.columns)}")
                
                # Check for Priority 2 specific features (PCA features are present)
                priority2_features = ['standard_pca_', 'sliding_window_', 'advanced_']
                found_features = [f for f in priority2_features if any(f in col for col in enhanced_features.columns)]
                
                if found_features:
                    logger.info(f"  Found Priority 2 features: {found_features}")
                    return True
                else:
                    logger.warning("  ⚠️ No Priority 2 specific features found")
                    return False
            else:
                logger.error("❌ Priority 2 Integration: Feature extraction failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Priority 2 Integration test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all Priority 3 tests"""
        logger.info("🚀 Starting Priority 3 Enhanced Model Accuracy Tests")
        logger.info("=" * 60)
        
        test_results = {}
        
        # Test 1: Enhanced Pattern Models
        test_results['enhanced_pattern_models'] = await self.test_enhanced_pattern_models()
        
        # Test 2: Enhanced Regime Models
        test_results['enhanced_regime_models'] = await self.test_enhanced_regime_models()
        
        # Test 3: Enhanced Ensemble
        test_results['enhanced_ensemble'] = await self.test_enhanced_ensemble()
        
        # Test 4: Enhanced Ensemble Predictions
        test_results['enhanced_ensemble_predictions'] = await self.test_prediction_with_enhanced_ensemble()
        
        # Test 5: Probability Calibration
        test_results['probability_calibration'] = await self.test_probability_calibration()
        
        # Test 6: Priority 2 Integration
        test_results['priority2_integration'] = await self.test_integration_with_priority2()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 Priority 3 Test Results Summary:")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        logger.info(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All Priority 3 tests passed successfully!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
        
        return test_results

async def main():
    """Main test execution"""
    tester = Priority3ModelAccuracyTester()
    results = await tester.run_all_tests()
    
    # Return results for potential CI/CD integration
    return results

if __name__ == "__main__":
    asyncio.run(main())
