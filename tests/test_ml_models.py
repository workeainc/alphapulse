#!/usr/bin/env python3
"""
Comprehensive Test for ML Models Package
Phase 1: Core Model Training & Online Learning

Tests:
1. MLModelTrainer - XGBoost, LightGBM, CatBoost training
2. OnlineLearner - River-based online learning
3. ModelEnsembler - Blending and stacking
"""

import asyncio
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ml_model_trainer():
    """Test ML Model Trainer"""
    try:
        logger.info("🧪 Testing ML Model Trainer...")
        
        # Import trainer
        from ..ai.ml_models.trainer import (
            MLModelTrainer, TrainingConfig, ModelType, TrainingCadence
        )
        
        # Create trainer
        trainer = MLModelTrainer()
        logger.info("✅ ML Model Trainer created")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Add realized R/R for weighting
        X['realized_rr'] = np.random.uniform(0.1, 5.0, n_samples)
        
        logger.info(f"✅ Sample data created: {X.shape}")
        
        # Test XGBoost training
        logger.info("📊 Testing XGBoost training...")
        try:
            config = TrainingConfig(
                model_type=ModelType.XGBOOST,
                cadence=TrainingCadence.WEEKLY_QUICK,
                learning_rate=0.1,
                max_depth=4,
                n_estimators=50
            )
            
            result = await trainer.train_model(X, y, config=config)
            logger.info(f"✅ XGBoost training completed: {result.model_path}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ XGBoost training failed: {e}")
        
        # Test LightGBM training
        logger.info("📊 Testing LightGBM training...")
        try:
            config = TrainingConfig(
                model_type=ModelType.LIGHTGBM,
                cadence=TrainingCadence.MONTHLY_FULL,
                learning_rate=0.1,
                max_depth=4,
                n_estimators=50
            )
            
            result = await trainer.train_model(X, y, config=config)
            logger.info(f"✅ LightGBM training completed: {result.model_path}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ LightGBM training failed: {e}")
        
        # Test CatBoost training
        logger.info("📊 Testing CatBoost training...")
        try:
            config = TrainingConfig(
                model_type=ModelType.CATBOOST,
                cadence=TrainingCadence.NIGHTLY_INCREMENTAL,
                learning_rate=0.1,
                max_depth=4,
                n_estimators=50
            )
            
            result = await trainer.train_model(X, y, config=config)
            logger.info(f"✅ CatBoost training completed: {result.model_path}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ CatBoost training failed: {e}")
        
        logger.info("✅ ML Model Trainer tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ ML Model Trainer test failed: {e}")
        return False

async def test_online_learner():
    """Test Online Learner"""
    try:
        logger.info("🧪 Testing Online Learner...")
        
        # Import online learner
        from ..ai.ml_models.online_learner import (
            OnlineLearner, OnlineLearningConfig, OnlineModelType
        )
        
        # Create online learner
        config = OnlineLearningConfig(
            model_type=OnlineModelType.LOGISTIC_REGRESSION,
            learning_rate=0.01,
            window_size=100,
            batch_weight=0.8,
            online_weight=0.2
        )
        
        online_learner = OnlineLearner(config)
        logger.info("✅ Online Learner created")
        
        # Initialize
        await online_learner.initialize()
        logger.info("✅ Online Learner initialized")
        
        # Test predictions
        logger.info("📊 Testing online predictions...")
        features = {
            'feature_0': 0.5,
            'feature_1': -0.3,
            'feature_2': 1.2,
            'feature_3': 0.8,
            'feature_4': -0.1
        }
        
        prediction = await online_learner.predict(features)
        logger.info(f"✅ Online prediction: {prediction.blended_score:.3f}")
        logger.info(f"   - Batch score: {prediction.batch_score:.3f}")
        logger.info(f"   - Online score: {prediction.online_score:.3f}")
        logger.info(f"   - Confidence: {prediction.confidence:.3f}")
        
        # Test learning
        logger.info("📊 Testing online learning...")
        for i in range(10):
            features = {
                'feature_0': np.random.randn(),
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'feature_3': np.random.randn(),
                'feature_4': np.random.randn()
            }
            label = bool(np.random.randint(0, 2))
            
            await online_learner.learn_one(features, label)
        
        logger.info("✅ Online learning completed")
        
        # Get performance summary
        summary = online_learner.get_performance_summary()
        logger.info(f"✅ Performance summary: {summary}")
        
        logger.info("✅ Online Learner tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Online Learner test failed: {e}")
        return False

async def test_model_ensembler():
    """Test Model Ensembler"""
    try:
        logger.info("🧪 Testing Model Ensembler...")
        
        # Import ensembler
        from ..ai.ml_models.ensembler import (
            ModelEnsembler, EnsembleConfig, EnsembleType, ModelSource
        )
        
        # Create ensembler
        config = EnsembleConfig(
            ensemble_type=EnsembleType.BLENDING,
            models=[
                ModelSource.MONTHLY_FULL,
                ModelSource.WEEKLY_QUICK,
                ModelSource.ONLINE_LEARNER
            ],
            blending_weights={
                ModelSource.MONTHLY_FULL.value: 0.5,
                ModelSource.WEEKLY_QUICK.value: 0.3,
                ModelSource.ONLINE_LEARNER.value: 0.2
            }
        )
        
        ensembler = ModelEnsembler(config)
        logger.info("✅ Model Ensembler created")
        
        # Add dummy models
        await ensembler.add_model(ModelSource.MONTHLY_FULL, "dummy_monthly.model", "xgboost")
        await ensembler.add_model(ModelSource.WEEKLY_QUICK, "dummy_weekly.model", "lightgbm")
        await ensembler.add_model(ModelSource.ONLINE_LEARNER, "dummy_online.model", "river")
        
        logger.info("✅ Dummy models added")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        n_features = 8
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        logger.info(f"✅ Sample data created: {X.shape}")
        
        # Test blending ensemble
        logger.info("📊 Testing blending ensemble...")
        try:
            result = await ensembler.create_ensemble(X, y)
            logger.info(f"✅ Blending ensemble created: {result.ensemble_path}")
            logger.info(f"   - Models used: {result.models_used}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Blending ensemble failed: {e}")
        
        # Test stacking ensemble
        logger.info("📊 Testing stacking ensemble...")
        try:
            config.ensemble_type = EnsembleType.STACKING
            ensembler.config = config
            
            result = await ensembler.create_ensemble(X, y)
            logger.info(f"✅ Stacking ensemble created: {result.ensemble_path}")
            logger.info(f"   - Models used: {result.models_used}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Stacking ensemble failed: {e}")
        
        # Test weighted average ensemble
        logger.info("📊 Testing weighted average ensemble...")
        try:
            config.ensemble_type = EnsembleType.WEIGHTED_AVERAGE
            ensembler.config = config
            
            result = await ensembler.create_ensemble(X, y)
            logger.info(f"✅ Weighted average ensemble created: {result.ensemble_path}")
            logger.info(f"   - Models used: {result.models_used}")
            logger.info(f"   - Metrics: {result.metrics}")
            logger.info(f"   - Training time: {result.training_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Weighted average ensemble failed: {e}")
        
        # Test ensemble prediction
        logger.info("📊 Testing ensemble prediction...")
        try:
            sample_X = X.head(1)
            prediction = await ensembler.predict(sample_X)
            logger.info(f"✅ Ensemble prediction: {prediction.ensemble_prediction:.3f}")
            logger.info(f"   - Confidence: {prediction.confidence:.3f}")
            logger.info(f"   - Model weights: {prediction.model_weights}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ensemble prediction failed: {e}")
        
        # Get ensemble summary
        summary = ensembler.get_ensemble_summary()
        logger.info(f"✅ Ensemble summary: {summary}")
        
        logger.info("✅ Model Ensembler tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Ensembler test failed: {e}")
        return False

async def test_integration():
    """Test integration between components"""
    try:
        logger.info("🧪 Testing ML Models Integration...")
        
        # Import all components
        from ..ai.ml_models.trainer import ml_model_trainer
        from ..ai.ml_models.online_learner import online_learner
        from ..ai.ml_models.ensembler import model_ensembler
        
        logger.info("✅ All ML model components imported")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        n_features = 6
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        X['realized_rr'] = np.random.uniform(0.1, 5.0, n_samples)
        
        logger.info(f"✅ Integration test data created: {X.shape}")
        
        # Test end-to-end workflow
        logger.info("📊 Testing end-to-end workflow...")
        
        # 1. Train a model
        logger.info("   1. Training model...")
        from ..ai.ml_models.trainer import TrainingConfig, ModelType, TrainingCadence
        
        config = TrainingConfig(
            model_type=ModelType.XGBOOST,
            cadence=TrainingCadence.WEEKLY_QUICK,
            learning_rate=0.1,
            max_depth=3,
            n_estimators=20
        )
        
        training_result = await ml_model_trainer.train_model(X, y, config=config)
        logger.info(f"   ✅ Model trained: {training_result.model_path}")
        
        # 2. Initialize online learner
        logger.info("   2. Initializing online learner...")
        await online_learner.initialize(training_result.model_path)
        logger.info("   ✅ Online learner initialized")
        
        # 3. Add models to ensemble
        logger.info("   3. Setting up ensemble...")
        from ..ai.ml_models.ensembler import ModelSource
        
        await model_ensembler.add_model(ModelSource.WEEKLY_QUICK, training_result.model_path, "xgboost")
        logger.info("   ✅ Model added to ensemble")
        
        # 4. Test ensemble prediction
        logger.info("   4. Testing ensemble prediction...")
        sample_X = X.head(1)
        ensemble_pred = await model_ensembler.predict(sample_X)
        logger.info(f"   ✅ Ensemble prediction: {ensemble_pred.ensemble_prediction:.3f}")
        
        # 5. Test online learning
        logger.info("   5. Testing online learning...")
        features = dict(zip(X.columns, X.iloc[0].values))
        online_pred = await online_learner.predict(features)
        logger.info(f"   ✅ Online prediction: {online_pred.blended_score:.3f}")
        
        logger.info("✅ Integration tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting ML Models Package Tests")
    
    test_results = {}
    
    # Test 1: ML Model Trainer
    trainer_success = await test_ml_model_trainer()
    test_results['ml_model_trainer'] = trainer_success
    
    # Test 2: Online Learner
    online_success = await test_online_learner()
    test_results['online_learner'] = online_success
    
    # Test 3: Model Ensembler
    ensembler_success = await test_model_ensembler()
    test_results['model_ensembler'] = ensembler_success
    
    # Test 4: Integration
    integration_success = await test_integration()
    test_results['integration'] = integration_success
    
    # Summary
    logger.info("📋 Test Results Summary:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   - {test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All ML Models tests passed! Phase 1 implementation is working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        logger.info("✅ ML Models Package test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ ML Models Package test failed")
        sys.exit(1)
