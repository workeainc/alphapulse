#!/usr/bin/env python3
"""
Test script for model comparison implementation
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model comparison components
from app.core.model_comparison_manager import model_comparison_manager, ModelVersion, ModelComparisonResult

async def test_model_comparison_manager():
    """Test the model comparison manager functionality"""
    
    logger.info("🧪 Testing Model Comparison Implementation")
    
    # Test 1: Model directory scanning
    logger.info("Test 1: Model directory scanning")
    
    try:
        # Scan model directory
        model_registry = await model_comparison_manager.scan_model_directory()
        logger.info(f"✅ Model registry built: {len(model_registry)} models found")
        
        # Display found models
        for model_id, model_version in model_registry.items():
            logger.info(f"   📦 {model_id} v{model_version.version} ({model_version.model_type})")
        
    except Exception as e:
        logger.error(f"❌ Model directory scanning failed: {e}")
    
    # Test 2: Model comparison (simulated)
    logger.info("Test 2: Model comparison")
    
    try:
        # Create mock model versions
        baseline_model = ModelVersion(
            model_id="catboost_nightly_incremental",
            version="20250814_140809",
            model_type="baseline",
            file_path="models/catboost_nightly_incremental_20250814_140809.model",
            created_at=datetime.now() - timedelta(days=30),
            metadata={"model_type": "catboost", "training_data_size": 10000}
        )
        
        optimized_model = ModelVersion(
            model_id="catboost_nightly_incremental",
            version="20250814_151525",
            model_type="optimized",
            file_path="models/catboost_nightly_incremental_20250814_151525.model",
            created_at=datetime.now() - timedelta(days=7),
            metadata={"model_type": "catboost", "training_data_size": 12000, "optimization": "hyperparameter_tuning"}
        )
        
        # Add to registry
        model_comparison_manager.model_registry[baseline_model.model_id] = baseline_model
        model_comparison_manager.model_registry[f"{optimized_model.model_id}_optimized"] = optimized_model
        
        logger.info("✅ Mock models created and added to registry")
        
    except Exception as e:
        logger.error(f"❌ Model comparison setup failed: {e}")
    
    # Test 3: Database integration
    logger.info("Test 3: Database integration")
    
    try:
        from ..database.connection import TimescaleDBConnection
        from ..database.queries import TimescaleQueries
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            # Test model comparison summary query
            summary = await TimescaleQueries.get_model_comparison_summary(
                session, "baseline_model", "optimized_model", days=30
            )
            logger.info(f"Database comparison summary: {len(summary.get('improvements', {}))} improvements calculated")
            
            # Test promotion candidates query
            candidates = await TimescaleQueries.get_promotion_candidates(session, days=30)
            logger.info(f"Promotion candidates: {len(candidates)} models found")
            
            # Test model evolution trends query
            trends = await TimescaleQueries.get_model_evolution_trends(session, "test_model", days=90)
            logger.info(f"Model evolution trends: {len(trends)} data points")
            
            # Test rollback analysis query
            rollback_analysis = await TimescaleQueries.get_model_rollback_analysis(session, "test_model", days=30)
            logger.info(f"Rollback analysis: Should rollback = {rollback_analysis.get('should_rollback', False)}")
            
        logger.info("✅ Database integration tests completed successfully!")
        
    except ImportError as e:
        logger.warning(f"Database not available: {e}")
    except Exception as e:
        logger.error(f"Database integration test failed: {e}")
    
    # Test 4: Promotion criteria evaluation
    logger.info("Test 4: Promotion criteria evaluation")
    
    try:
        # Test different scenarios
        test_scenarios = [
            {
                'name': 'High improvement scenario',
                'f1_improvement': 0.08,
                'win_rate_improvement': 0.15,
                'latency_improvement': 75.0,
                'test_period': 45
            },
            {
                'name': 'Low improvement scenario',
                'f1_improvement': 0.02,
                'win_rate_improvement': 0.05,
                'latency_improvement': 20.0,
                'test_period': 15
            },
            {
                'name': 'Mixed improvement scenario',
                'f1_improvement': 0.06,
                'win_rate_improvement': 0.08,
                'latency_improvement': 60.0,
                'test_period': 60
            }
        ]
        
        for scenario in test_scenarios:
            # Create mock comparison
            from app.core.accuracy_evaluator import BaselineComparison, ModelAccuracyMetrics
            
            baseline_metrics = ModelAccuracyMetrics(
                precision=0.70, recall=0.65, f1_score=0.67, accuracy=0.68, roc_auc=0.72,
                win_rate=0.60, profit_factor=1.60, avg_win=120.0, avg_loss=-75.0,
                total_return=900.0, sharpe_ratio=1.20, max_drawdown=400.0,
                total_trades=50, winning_trades=30, losing_trades=20,
                avg_holding_period=5.0, risk_reward_ratio=1.60,
                model_id="baseline_model", evaluation_date=datetime.now(),
                test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
            )
            
            optimized_metrics = ModelAccuracyMetrics(
                precision=0.70 + scenario['f1_improvement'], 
                recall=0.65 + scenario['f1_improvement'], 
                f1_score=0.67 + scenario['f1_improvement'], 
                accuracy=0.68 + scenario['f1_improvement'], 
                roc_auc=0.72 + scenario['f1_improvement'],
                win_rate=0.60 + scenario['win_rate_improvement'], 
                profit_factor=1.60 + scenario['f1_improvement'], 
                avg_win=120.0, avg_loss=-75.0,
                total_return=900.0, sharpe_ratio=1.20, max_drawdown=400.0,
                total_trades=50, winning_trades=30, losing_trades=20,
                avg_holding_period=5.0, risk_reward_ratio=1.60,
                model_id="optimized_model", evaluation_date=datetime.now(),
                test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
            )
            
            comparison = BaselineComparison(
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                precision_improvement=scenario['f1_improvement'],
                recall_improvement=scenario['f1_improvement'],
                f1_improvement=scenario['f1_improvement'],
                win_rate_improvement=scenario['win_rate_improvement'],
                profit_factor_improvement=scenario['f1_improvement'],
                latency_improvement_ms=scenario['latency_improvement'],
                is_significant=True,
                p_value=0.01
            )
            
            # Evaluate promotion criteria
            should_promote, confidence_score, promotion_reason = model_comparison_manager._evaluate_promotion_criteria(
                comparison, scenario['latency_improvement'], scenario['test_period']
            )
            
            logger.info(f"   {scenario['name']}: Should promote = {should_promote}, Confidence = {confidence_score:.3f}")
            logger.info(f"      Reason: {promotion_reason}")
        
        logger.info("✅ Promotion criteria evaluation completed")
        
    except Exception as e:
        logger.error(f"❌ Promotion criteria evaluation failed: {e}")
    
    # Test 5: Business impact calculations
    logger.info("Test 5: Business impact calculations")
    
    try:
        # Create mock metrics
        baseline_metrics = ModelAccuracyMetrics(
            precision=0.70, recall=0.65, f1_score=0.67, accuracy=0.68, roc_auc=0.72,
            win_rate=0.60, profit_factor=1.60, avg_win=120.0, avg_loss=-75.0,
            total_return=900.0, sharpe_ratio=1.20, max_drawdown=400.0,
            total_trades=50, winning_trades=30, losing_trades=20,
            avg_holding_period=5.0, risk_reward_ratio=1.60,
            model_id="baseline_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        optimized_metrics = ModelAccuracyMetrics(
            precision=0.75, recall=0.68, f1_score=0.71, accuracy=0.72, roc_auc=0.78,
            win_rate=0.65, profit_factor=1.85, avg_win=150.0, avg_loss=-80.0,
            total_return=1250.0, sharpe_ratio=1.45, max_drawdown=350.0,
            total_trades=50, winning_trades=32, losing_trades=18,
            avg_holding_period=4.5, risk_reward_ratio=1.88,
            model_id="optimized_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        # Calculate business impact
        pnl_improvement = model_comparison_manager._calculate_pnl_improvement(baseline_metrics, optimized_metrics)
        risk_adjustment = model_comparison_manager._calculate_risk_adjustment(baseline_metrics, optimized_metrics)
        
        logger.info(f"✅ Business impact calculated:")
        logger.info(f"   Expected PnL improvement: {pnl_improvement:+.2f}")
        logger.info(f"   Risk adjustment factor: {risk_adjustment:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Business impact calculation failed: {e}")
    
    # Test 6: Report generation
    logger.info("Test 6: Report generation")
    
    try:
        # Create a mock comparison result
        from app.core.accuracy_evaluator import BaselineComparison
        
        baseline_model = ModelVersion(
            model_id="catboost_baseline",
            version="20250814_140809",
            model_type="baseline",
            file_path="models/catboost_baseline_20250814_140809.model",
            created_at=datetime.now() - timedelta(days=30),
            metadata={"model_type": "catboost"}
        )
        
        optimized_model = ModelVersion(
            model_id="catboost_optimized",
            version="20250814_151525",
            model_type="optimized",
            file_path="models/catboost_optimized_20250814_151525.model",
            created_at=datetime.now() - timedelta(days=7),
            metadata={"model_type": "catboost", "optimization": "hyperparameter_tuning"}
        )
        
        baseline_metrics = ModelAccuracyMetrics(
            precision=0.70, recall=0.65, f1_score=0.67, accuracy=0.68, roc_auc=0.72,
            win_rate=0.60, profit_factor=1.60, avg_win=120.0, avg_loss=-75.0,
            total_return=900.0, sharpe_ratio=1.20, max_drawdown=400.0,
            total_trades=50, winning_trades=30, losing_trades=20,
            avg_holding_period=5.0, risk_reward_ratio=1.60,
            model_id="baseline_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        optimized_metrics = ModelAccuracyMetrics(
            precision=0.75, recall=0.68, f1_score=0.71, accuracy=0.72, roc_auc=0.78,
            win_rate=0.65, profit_factor=1.85, avg_win=150.0, avg_loss=-80.0,
            total_return=1250.0, sharpe_ratio=1.45, max_drawdown=350.0,
            total_trades=50, winning_trades=32, losing_trades=18,
            avg_holding_period=4.5, risk_reward_ratio=1.88,
            model_id="optimized_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        accuracy_comparison = BaselineComparison(
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            precision_improvement=0.05,
            recall_improvement=0.03,
            f1_improvement=0.04,
            win_rate_improvement=0.05,
            profit_factor_improvement=0.25,
            latency_improvement_ms=50.0,
            is_significant=True,
            p_value=0.02
        )
        
        comparison_result = ModelComparisonResult(
            baseline_model=baseline_model,
            optimized_model=optimized_model,
            accuracy_comparison=accuracy_comparison,
            latency_improvement_ms=50.0,
            throughput_improvement_percent=15.0,
            expected_pnl_improvement=350.0,
            risk_adjustment_factor=0.125,
            should_promote=True,
            confidence_score=0.85,
            promotion_reason="Significant improvements in F1-score and win rate",
            comparison_date=datetime.now(),
            test_period_days=90,
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        
        # Generate report
        await model_comparison_manager._save_comparison_report(comparison_result)
        logger.info("✅ Comparison report generated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
    
    logger.info("🎉 All model comparison tests completed!")

async def test_model_registry_operations():
    """Test model registry operations"""
    
    logger.info("🧪 Testing Model Registry Operations")
    
    try:
        # Test model type determination
        test_cases = [
            ("catboost_baseline_20250814_140809.model", "baseline"),
            ("xgboost_optimized_20250814_151525.model", "optimized"),
            ("lightgbm_candidate_20250814_160000.model", "candidate"),
            ("catboost_nightly_incremental_20250814_170000.model", "optimized")
        ]
        
        for filename, expected_type in test_cases:
            model_info = model_comparison_manager._parse_model_filename(filename)
            if model_info:
                actual_type = model_comparison_manager._determine_model_type(model_info)
                logger.info(f"   {filename} -> {actual_type} (expected: {expected_type})")
        
        logger.info("✅ Model registry operations completed")
        
    except Exception as e:
        logger.error(f"❌ Model registry operations failed: {e}")

async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Model Comparison Tests")
    
    # Test model comparison manager
    await test_model_comparison_manager()
    
    # Test model registry operations
    await test_model_registry_operations()
    
    logger.info("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
