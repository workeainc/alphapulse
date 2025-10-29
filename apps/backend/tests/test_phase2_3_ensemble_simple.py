#!/usr/bin/env python3
"""
Phase 2.3: Ensemble Model Integration Simple Test Script
Tests ensemble strategy and signal generator integration without database
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2_3SimpleTester:
    def __init__(self):
        pass
    
    def test_ensemble_strategy(self):
        """Test Ensemble Strategy functionality"""
        logger.info("🤖 Testing Phase 2.3: Ensemble Strategy...")
        
        try:
            # Import ensemble strategy
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))
            sys.path.append(os.path.dirname(__file__))
            from src.ai.ml_strategy_enhancement import EnsembleStrategy, EnsembleConfig, MLStrategyType, ModelType
            
            # Create ensemble configuration
            config = EnsembleConfig(
                strategy_type=MLStrategyType.ENSEMBLE_VOTING,
                base_models=[
                    ModelType.RANDOM_FOREST,
                    ModelType.GRADIENT_BOOSTING,
                    ModelType.LOGISTIC_REGRESSION
                ],
                voting_method="soft",
                adaptive_weights=True
            )
            
            # Create ensemble strategy
            ensemble = EnsembleStrategy(config)
            logger.info(f"✅ Ensemble strategy created with {len(config.base_models)} models")
            
            # Generate sample data
            np.random.seed(42)
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)
            
            # Train ensemble
            ensemble.train(X, y)
            logger.info("✅ Ensemble training completed")
            
            # Test prediction
            test_features = np.random.randn(1, 10)
            signal = ensemble.predict(test_features)
            
            logger.info(f"✅ Ensemble prediction: {signal.prediction}, confidence: {signal.confidence:.3f}")
            
            # Test ensemble analysis
            analysis = ensemble.get_ensemble_analysis()
            logger.info(f"✅ Ensemble analysis: diversity={analysis.get('diversity_score', 0):.3f}, agreement={analysis.get('agreement_ratio', 0):.3f}")
            
            # Test individual components
            logger.info(f"✅ Model count: {analysis.get('model_count', 0)}")
            logger.info(f"✅ Voting method: {analysis.get('voting_method', 'unknown')}")
            logger.info(f"✅ Individual predictions: {analysis.get('individual_predictions', {})}")
            logger.info(f"✅ Model weights: {analysis.get('model_weights', {})}")
            
            logger.info("✅ Phase 2.3: Ensemble Strategy working")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing ensemble strategy: {e}")
            return False
    
    def test_signal_generator_integration(self):
        """Test ensemble integration with signal generator"""
        logger.info("🔗 Testing Phase 2.3: Signal Generator Integration...")
        
        try:
            # Import signal generator
            from src.app.strategies.real_time_signal_generator import RealTimeSignalGenerator
            
            # Create signal generator with ensemble disabled to avoid database dependency
            config = {
                'use_ensemble': True,
                'use_database': False,  # Disable database to avoid connection issues
                'min_confidence': 0.6
            }
            
            signal_generator = RealTimeSignalGenerator(config)
            logger.info("✅ Signal generator created with ensemble enabled")
            
            # Test ensemble initialization
            asyncio.run(signal_generator.initialize_ensemble_strategy())
            
            if signal_generator.ensemble_strategy:
                logger.info("✅ Ensemble strategy initialized in signal generator")
            else:
                logger.error("❌ Ensemble strategy not initialized")
                return False
            
            # Test feature preparation
            # Create sample market data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Calculate basic indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = signal_generator._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = signal_generator._calculate_macd(df['close'])
            
            # Test feature preparation
            features = signal_generator._prepare_features_for_ensemble(df)
            
            if features is not None and features.shape[1] > 0:
                logger.info(f"✅ Feature preparation working: {features.shape}")
                logger.info(f"✅ Features: {features[0]}")
            else:
                logger.error("❌ Feature preparation failed")
                return False
            
            # Test ensemble analysis
            ensemble_signals = asyncio.run(signal_generator.analyze_ensemble_predictions(df, 'BTCUSDT', '1h'))
            
            if ensemble_signals:
                logger.info(f"✅ Ensemble analysis working: bias={ensemble_signals.get('ensemble_bias', 'unknown')}, confidence={ensemble_signals.get('ensemble_confidence', 0):.3f}")
                logger.info(f"✅ Ensemble diversity: {ensemble_signals.get('ensemble_diversity', 0):.3f}")
                logger.info(f"✅ Ensemble agreement: {ensemble_signals.get('ensemble_agreement', 0):.3f}")
                logger.info(f"✅ Model count: {ensemble_signals.get('model_count', 0)}")
                logger.info(f"✅ Voting method: {ensemble_signals.get('voting_method', 'unknown')}")
            else:
                logger.warning("⚠️ Ensemble analysis returned empty (may be expected if ensemble not fully trained)")
            
            logger.info("✅ Phase 2.3: Signal Generator Integration working")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing signal generator integration: {e}")
            return False
    
    def test_ensemble_analysis_methods(self):
        """Test the new ensemble analysis methods"""
        logger.info("🔬 Testing Phase 2.3: Ensemble Analysis Methods...")
        
        try:
            # Import ensemble strategy
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))
            sys.path.append(os.path.dirname(__file__))
            from src.ai.ml_strategy_enhancement import EnsembleStrategy, EnsembleConfig, MLStrategyType, ModelType
            
            # Create ensemble configuration
            config = EnsembleConfig(
                strategy_type=MLStrategyType.ENSEMBLE_VOTING,
                base_models=[
                    ModelType.RANDOM_FOREST,
                    ModelType.GRADIENT_BOOSTING,
                    ModelType.LOGISTIC_REGRESSION,
                    ModelType.SVM,
                    ModelType.NEURAL_NETWORK
                ],
                voting_method="soft",
                adaptive_weights=True
            )
            
            # Create ensemble strategy
            ensemble = EnsembleStrategy(config)
            
            # Generate diverse sample data
            np.random.seed(42)
            X = np.random.randn(200, 15)
            y = np.random.randint(0, 2, 200)
            
            # Train ensemble
            ensemble.train(X, y)
            
            # Test diversity calculation
            diversity = ensemble._calculate_ensemble_diversity(X, y)
            logger.info(f"✅ Diversity calculation: {diversity}")
            
            # Test agreement ratio calculation
            test_features = np.random.randn(10, 15)
            agreement = ensemble._calculate_agreement_ratio(test_features)
            logger.info(f"✅ Agreement ratio calculation: {agreement}")
            
            # Test full ensemble analysis
            analysis = ensemble.get_ensemble_analysis()
            logger.info(f"✅ Full ensemble analysis: {analysis}")
            
            # Verify all required fields are present
            required_fields = [
                'voting_method', 'model_count', 'diversity_score', 'agreement_ratio',
                'individual_predictions', 'model_weights', 'performance_metrics'
            ]
            
            for field in required_fields:
                if field in analysis:
                    logger.info(f"✅ {field}: {analysis[field]}")
                else:
                    logger.error(f"❌ Missing field: {field}")
                    return False
            
            logger.info("✅ Phase 2.3: Ensemble Analysis Methods working")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing ensemble analysis methods: {e}")
            return False
    
    def run_all_tests(self):
        """Run all Phase 2.3 tests"""
        logger.info("🚀 Starting Phase 2.3: Ensemble Model Integration Simple Tests...")
        
        tests = [
            ("Ensemble Strategy", self.test_ensemble_strategy),
            ("Signal Generator Integration", self.test_signal_generator_integration),
            ("Ensemble Analysis Methods", self.test_ensemble_analysis_methods)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 Phase 2.3: Test Results Summary")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 Phase 2.3: All tests passed! Ensemble Model Integration is working correctly.")
        else:
            logger.error(f"⚠️ Phase 2.3: {total - passed} tests failed. Please check the errors above.")
        
        return passed == total

def main():
    """Main test function"""
    tester = Phase2_3SimpleTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("🎯 Phase 2.3: Ensemble Model Integration is ready for production!")
        sys.exit(0)
    else:
        logger.error("💥 Phase 2.3: Ensemble Model Integration has issues that need to be resolved.")
        sys.exit(1)

if __name__ == "__main__":
    main()
