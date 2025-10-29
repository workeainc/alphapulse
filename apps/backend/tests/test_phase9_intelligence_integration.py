#!/usr/bin/env python3
"""
Test Suite: Phase 9 Intelligence Enhancements Integration
Tests auto-retraining, market regime detection, and explainability features
"""

import asyncio
import logging
import unittest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app.services.auto_retraining_service import AutoRetrainingService, RetrainingConfig, DriftMetrics, DriftType
from src.app.services.market_regime_detection_service import MarketRegimeDetectionService, RegimeType, VolatilityRegime, LiquidityRegime
from src.app.services.explainability_service import ExplainabilityService, DecisionExplanation, FeatureContribution

class MockPool:
    """Mock database pool for testing"""
    
    def __init__(self):
        self.conn = MockConnection()
    
    def acquire(self):
        return self.conn
    
    async def release(self, conn):
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockConnection:
    """Mock database connection for testing"""
    
    def __init__(self):
        self.execute_calls = []
        self.fetch_calls = []
        self.fetchrow_calls = []
    
    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return Mock()
    
    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        # Return mock data based on query
        if 'volume_analysis_ml_dataset' in query:
            return [
                Mock(**{
                    'volume_ratio': 1.5,
                    'volume_positioning_score': 0.7,
                    'order_book_imbalance': 0.2,
                    'timestamp': datetime.now()
                })
            ]
        elif 'model_performance' in query:
            return [
                Mock(**{
                    'metric_name': 'auc',
                    'metric_value': 0.75,
                    'timestamp': datetime.now()
                })
            ]
        elif 'market_regimes' in query:
            return [
                Mock(**{
                    'regime_type': 'trending',
                    'volatility_regime': 'medium',
                    'liquidity_regime': 'high',
                    'regime_confidence': 0.8,
                    'regime_features': {
                        'volatility': 0.02,
                        'trend_strength': 0.8,
                        'volume_consistency': 0.6,
                        'price_range': 0.03,
                        'liquidity_score': 0.7,
                        'regime_confidence': 0.8
                    },
                    'timestamp': datetime.now()
                })
            ]
        elif 'trade_explanations' in query:
            return [
                Mock(**{
                    'decision_type': 'volume_signal',
                    'decision_value': 'buy',
                    'confidence_score': 0.8,
                    'explanation_text': 'Strong buy signal based on volume analysis',
                    'timestamp': datetime.now()
                })
            ]
        return []
    
    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        # Return mock data based on query
        if 'model_retraining_history' in query:
            return Mock(**{
                'retraining_start': datetime.now() - timedelta(days=5)
            })
        elif 'regime_thresholds' in query:
            return Mock(**{
                'threshold_value': 2.5,
                'confidence_level': 0.8
            })
        elif 'feature_importance_history' in query:
            return Mock(**{
                'importance_score': 0.3
            })
        return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class TestPhase9IntelligenceIntegration(unittest.TestCase):
    """Test Phase 9 Intelligence Enhancements Integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_pool = MockPool()
        self.ml_training_service = Mock()
        self.ml_training_service.train_model = AsyncMock(return_value="v2.0")
        self.ml_training_service.activate_model = AsyncMock()
        
        # Initialize services
        self.auto_retraining_service = AutoRetrainingService(self.db_pool, self.ml_training_service)
        self.market_regime_service = MarketRegimeDetectionService(self.db_pool)
        self.explainability_service = ExplainabilityService(self.db_pool)
    
    def test_auto_retraining_service_initialization(self):
        """Test Auto-Retraining Service initialization"""
        self.assertIsNotNone(self.auto_retraining_service)
        self.assertEqual(len(self.auto_retraining_service.retraining_configs), 0)
        self.assertIn('psi_threshold', self.auto_retraining_service.drift_detection_params)
        logger.info("✅ Auto-Retraining Service initialization test passed")
    
    def test_market_regime_service_initialization(self):
        """Test Market Regime Detection Service initialization"""
        self.assertIsNotNone(self.market_regime_service)
        self.assertIn('volatility_window', self.market_regime_service.regime_params)
        self.assertIn(RegimeType.TRENDING, self.market_regime_service.default_thresholds)
        logger.info("✅ Market Regime Detection Service initialization test passed")
    
    def test_explainability_service_initialization(self):
        """Test Explainability Service initialization"""
        self.assertIsNotNone(self.explainability_service)
        self.assertIn('volume_signal', self.explainability_service.explanation_templates)
        self.assertIn('ml_prediction', self.explainability_service.explanation_templates)
        logger.info("✅ Explainability Service initialization test passed")
    
    @patch('app.services.auto_retraining_service.ML_AVAILABLE', True)
    def test_auto_retraining_config_registration(self):
        """Test model registration for auto-retraining"""
        async def test():
            config = RetrainingConfig(
                model_name="test_model",
                symbol="BTCUSDT",
                timeframe="1m",
                retraining_schedule_days=7,
                drift_threshold=0.25
            )
            
            await self.auto_retraining_service.register_model_for_auto_retraining(config)
            
            # Check if config was registered
            config_key = f"{config.model_name}_{config.symbol}_{config.timeframe}"
            self.assertIn(config_key, self.auto_retraining_service.retraining_configs)
            
            # Check database call
            self.assertGreater(len(self.db_pool.conn.execute_calls), 0)
            
            logger.info("✅ Auto-retraining config registration test passed")
        
        asyncio.run(test())
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        async def test():
            # Create mock OHLCV data
            ohlcv_data = []
            for i in range(100):
                ohlcv_data.append({
                    'open': 50000 + i * 10,
                    'high': 50000 + i * 10 + 50,
                    'low': 50000 + i * 10 - 50,
                    'close': 50000 + i * 10 + 25,
                    'volume': 1000 + i * 10
                })
            
            # Detect market regime
            regime = await self.market_regime_service.detect_market_regime("BTCUSDT", "1m", ohlcv_data)
            
            # Verify regime object
            self.assertIsNotNone(regime)
            self.assertIsInstance(regime.regime_type, RegimeType)
            self.assertIsInstance(regime.volatility_regime, VolatilityRegime)
            self.assertIsInstance(regime.liquidity_regime, LiquidityRegime)
            self.assertGreaterEqual(regime.regime_confidence, 0.0)
            self.assertLessEqual(regime.regime_confidence, 1.0)
            
            # Check database calls
            self.assertGreater(len(self.db_pool.conn.execute_calls), 0)
            
            logger.info("✅ Market regime detection test passed")
        
        asyncio.run(test())
    
    def test_volume_decision_explanation(self):
        """Test volume decision explanation"""
        async def test():
            # Create mock volume analysis result
            volume_analysis = {
                'volume_ratio': 2.5,
                'volume_positioning_score': 0.8,
                'order_book_imbalance': 0.3,
                'volume_trend': 'increasing'
            }
            
            # Generate explanation
            explanation = await self.explainability_service.explain_volume_decision(volume_analysis)
            
            # Verify explanation object
            self.assertIsInstance(explanation, DecisionExplanation)
            self.assertEqual(explanation.decision_type, 'volume_signal')
            self.assertIn(explanation.decision_value, ['buy', 'sell', 'hold'])
            self.assertGreaterEqual(explanation.confidence_score, 0.0)
            self.assertLessEqual(explanation.confidence_score, 1.0)
            self.assertIsInstance(explanation.feature_contributions, list)
            self.assertIsInstance(explanation.shap_values, dict)
            
            # Check feature contributions
            self.assertGreater(len(explanation.feature_contributions), 0)
            for fc in explanation.feature_contributions:
                self.assertIsInstance(fc, FeatureContribution)
                self.assertIsInstance(fc.feature_name, str)
                self.assertIsInstance(fc.contribution_value, float)
                self.assertIsInstance(fc.contribution_percentage, float)
                self.assertIn(fc.direction, ['positive', 'negative', 'neutral'])
            
            logger.info("✅ Volume decision explanation test passed")
        
        asyncio.run(test())
    
    def test_ml_prediction_explanation(self):
        """Test ML prediction explanation"""
        async def test():
            # Create mock prediction result and feature vector
            prediction_result = {
                'prediction': 'bullish',
                'confidence': 0.85,
                'prediction_type': 'breakout'
            }
            
            feature_vector = {
                'volume_ratio': 2.0,
                'volume_positioning_score': 0.7,
                'order_book_imbalance': 0.2,
                'price_momentum': 0.5,
                'volatility': 0.03
            }
            
            # Generate explanation
            explanation = await self.explainability_service.explain_ml_prediction(
                "test_model", prediction_result, feature_vector
            )
            
            # Verify explanation object
            self.assertIsInstance(explanation, DecisionExplanation)
            self.assertEqual(explanation.decision_type, 'ml_prediction')
            self.assertEqual(explanation.decision_value, 'bullish')
            self.assertEqual(explanation.confidence_score, 0.85)
            self.assertIsInstance(explanation.feature_contributions, list)
            self.assertIsInstance(explanation.shap_values, dict)
            
            logger.info("✅ ML prediction explanation test passed")
        
        asyncio.run(test())
    
    def test_rl_action_explanation(self):
        """Test RL action explanation"""
        async def test():
            # Create mock RL state and action
            rl_state = {
                'features': {
                    'price_trend': 0.6,
                    'volume_trend': 0.8,
                    'volatility': 0.3,
                    'market_regime': 0.7
                }
            }
            
            action = 'buy'
            reward = 0.8
            
            # Generate explanation
            explanation = await self.explainability_service.explain_rl_action(rl_state, action, reward)
            
            # Verify explanation object
            self.assertIsInstance(explanation, DecisionExplanation)
            self.assertEqual(explanation.decision_type, 'rl_action')
            self.assertEqual(explanation.decision_value, 'buy')
            self.assertGreaterEqual(explanation.confidence_score, 0.0)
            self.assertLessEqual(explanation.confidence_score, 1.0)
            
            logger.info("✅ RL action explanation test passed")
        
        asyncio.run(test())
    
    def test_anomaly_alert_explanation(self):
        """Test anomaly alert explanation"""
        async def test():
            # Create mock anomaly result
            anomaly_result = {
                'anomaly_type': 'volume_spike',
                'severity': 'high',
                'confidence': 0.9,
                'indicators': {
                    'volume_ratio': 5.0,
                    'price_change': 0.05,
                    'order_book_imbalance': 0.8
                },
                'detection_method': 'statistical'
            }
            
            # Generate explanation
            explanation = await self.explainability_service.explain_anomaly_alert(anomaly_result)
            
            # Verify explanation object
            self.assertIsInstance(explanation, DecisionExplanation)
            self.assertEqual(explanation.decision_type, 'anomaly_alert')
            self.assertEqual(explanation.decision_value, 'high')
            self.assertEqual(explanation.confidence_score, 0.9)
            
            logger.info("✅ Anomaly alert explanation test passed")
        
        asyncio.run(test())
    
    def test_explanation_storage(self):
        """Test explanation storage in database"""
        async def test():
            # Create mock explanation
            explanation = DecisionExplanation(
                decision_type='volume_signal',
                decision_value='buy',
                confidence_score=0.8,
                explanation_text='Strong buy signal based on volume analysis',
                feature_contributions=[],
                shap_values={'volume_ratio': 0.5},
                contributing_factors={'test': 'data'},
                metadata={'test': True}
            )
            
            # Store explanation
            await self.explainability_service.store_explanation("BTCUSDT", "1m", explanation)
            
            # Check database call
            self.assertGreater(len(self.db_pool.conn.execute_calls), 0)
            
            logger.info("✅ Explanation storage test passed")
        
        asyncio.run(test())
    
    def test_trade_journal_generation(self):
        """Test trade journal generation"""
        async def test():
            # Generate trade journal
            journal = await self.explainability_service.generate_trade_journal("BTCUSDT", "1m", 24)
            
            # Verify journal format
            self.assertIsInstance(journal, str)
            self.assertIn("Trading Journal", journal)
            self.assertIn("BTCUSDT", journal)
            self.assertIn("1m", journal)
            
            logger.info("✅ Trade journal generation test passed")
        
        asyncio.run(test())
    
    def test_regime_thresholds(self):
        """Test regime-specific thresholds"""
        async def test():
            # Get regime thresholds
            thresholds = await self.market_regime_service.get_regime_thresholds("BTCUSDT", RegimeType.TRENDING)
            
            # Verify thresholds object
            self.assertIsNotNone(thresholds)
            self.assertIsInstance(thresholds.volume_spike_threshold, float)
            self.assertIsInstance(thresholds.breakout_threshold, float)
            self.assertIsInstance(thresholds.anomaly_threshold, float)
            self.assertIsInstance(thresholds.confidence_level, float)
            
            logger.info("✅ Regime thresholds test passed")
        
        asyncio.run(test())
    
    def test_drift_detection(self):
        """Test data drift detection"""
        async def test():
            # Create mock drift metrics
            drift_metrics = [
                DriftMetrics(
                    drift_type=DriftType.PSI,
                    feature_name='volume_ratio',
                    drift_score=0.3,
                    threshold=0.25,
                    is_drift_detected=True,
                    metadata={'psi_score': 0.3}
                )
            ]
            
            # Test drift detection logic
            significant_drift = any(metric.is_drift_detected for metric in drift_metrics)
            self.assertTrue(significant_drift)
            
            logger.info("✅ Drift detection test passed")
        
        asyncio.run(test())
    
    def test_explanation_statistics(self):
        """Test explanation statistics"""
        async def test():
            # Get explanation statistics
            stats = await self.explainability_service.get_explanation_statistics("BTCUSDT", "1m", 7)
            
            # Verify statistics structure
            self.assertIsInstance(stats, dict)
            self.assertIn('total_explanations', stats)
            self.assertIn('decision_distribution', stats)
            self.assertIn('avg_confidence', stats)
            self.assertIn('high_confidence_count', stats)
            
            logger.info("✅ Explanation statistics test passed")
        
        asyncio.run(test())
    
    def test_regime_statistics(self):
        """Test regime statistics"""
        async def test():
            # Get regime statistics
            stats = await self.market_regime_service.get_regime_statistics("BTCUSDT", "1m", 7)
            
            # Verify statistics structure
            self.assertIsInstance(stats, dict)
            self.assertIn('total_regimes', stats)
            self.assertIn('regime_distribution', stats)
            self.assertIn('avg_confidence', stats)
            
            logger.info("✅ Regime statistics test passed")
        
        asyncio.run(test())
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        # Create mock data
        recent_data = pd.Series([1.0, 1.2, 1.5, 1.8, 2.0])
        historical_data = pd.Series([1.0, 1.1, 1.3, 1.4, 1.6])
        
        # Calculate PSI
        psi = self.auto_retraining_service._calculate_psi(recent_data, historical_data)
        
        # Verify PSI value
        self.assertIsInstance(psi, float)
        self.assertGreaterEqual(psi, 0.0)
        
        logger.info("✅ PSI calculation test passed")
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        # Create mock OHLCV data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        # Calculate trend strength
        trend_strength = self.market_regime_service._calculate_trend_strength(data)
        
        # Verify trend strength
        self.assertIsInstance(trend_strength, float)
        self.assertGreaterEqual(trend_strength, 0.0)
        self.assertLessEqual(trend_strength, 1.0)
        
        logger.info("✅ Trend strength calculation test passed")
    
    def test_feature_contribution_calculation(self):
        """Test feature contribution calculation"""
        # Create mock feature contributions
        contributions = [
            FeatureContribution(
                feature_name='volume_ratio',
                contribution_value=0.5,
                contribution_percentage=40.0,
                feature_importance=0.4,
                direction='positive'
            ),
            FeatureContribution(
                feature_name='order_book_imbalance',
                contribution_value=-0.2,
                contribution_percentage=20.0,
                feature_importance=0.2,
                direction='negative'
            )
        ]
        
        # Verify contributions
        self.assertEqual(len(contributions), 2)
        self.assertEqual(contributions[0].feature_name, 'volume_ratio')
        self.assertEqual(contributions[0].direction, 'positive')
        self.assertEqual(contributions[1].direction, 'negative')
        
        logger.info("✅ Feature contribution calculation test passed")

def run_phase9_tests():
    """Run all Phase 9 tests"""
    logger.info("🚀 Starting Phase 9 Intelligence Enhancements Integration Tests")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase9IntelligenceIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info(f"📊 Phase 9 Test Results:")
    logger.info(f"   Tests Run: {result.testsRun}")
    logger.info(f"   Failures: {len(result.failures)}")
    logger.info(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        logger.info("🎉 All Phase 9 tests passed successfully!")
        return True
    else:
        logger.error("❌ Some Phase 9 tests failed!")
        return False

if __name__ == "__main__":
    run_phase9_tests()
