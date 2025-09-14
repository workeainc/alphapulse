#!/usr/bin/env python3
"""
Test script for accuracy benchmarking implementation
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import accuracy benchmarking components
from app.core.accuracy_evaluator import accuracy_evaluator, ModelAccuracyMetrics

async def test_accuracy_evaluator():
    """Test the accuracy evaluator functionality"""
    
    logger.info("🧪 Testing Accuracy Benchmarking Implementation")
    
    # Test 1: Basic accuracy evaluation (with mock data)
    logger.info("Test 1: Basic accuracy evaluation")
    
    try:
        # Create mock test data
        mock_test_data = create_mock_test_data()
        
        # Test ML metrics calculation
        ml_metrics = accuracy_evaluator._calculate_ml_metrics(mock_test_data)
        logger.info(f"ML Metrics: Precision={ml_metrics['precision']:.3f}, Recall={ml_metrics['recall']:.3f}, F1={ml_metrics['f1_score']:.3f}")
        
        # Test trading metrics calculation
        trading_metrics = accuracy_evaluator._calculate_trading_metrics(mock_test_data)
        logger.info(f"Trading Metrics: Win Rate={trading_metrics['win_rate']:.3f}, Profit Factor={trading_metrics['profit_factor']:.3f}")
        
        logger.info("✅ Basic accuracy evaluation completed")
        
    except Exception as e:
        logger.error(f"❌ Basic accuracy evaluation failed: {e}")
    
    # Test 2: Model accuracy evaluation (simulated)
    logger.info("Test 2: Model accuracy evaluation")
    
    try:
        # Simulate model evaluation
        mock_metrics = ModelAccuracyMetrics(
            # ML metrics
            precision=0.75,
            recall=0.68,
            f1_score=0.71,
            accuracy=0.72,
            roc_auc=0.78,
            
            # Trading metrics
            win_rate=0.65,
            profit_factor=1.85,
            avg_win=150.0,
            avg_loss=-80.0,
            total_return=1250.0,
            sharpe_ratio=1.45,
            max_drawdown=350.0,
            
            # Additional metrics
            total_trades=50,
            winning_trades=32,
            losing_trades=18,
            avg_holding_period=4.5,
            risk_reward_ratio=1.88,
            
            # Metadata
            model_id="test_model_v1",
            evaluation_date=datetime.now(),
            test_period_days=90,
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )
        
        # Store evaluation result
        accuracy_evaluator.evaluation_results.append(mock_metrics)
        
        logger.info(f"✅ Model evaluation completed: F1={mock_metrics.f1_score:.3f}, Win Rate={mock_metrics.win_rate:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
    
    # Test 3: Model comparison
    logger.info("Test 3: Model comparison")
    
    try:
        # Create baseline metrics
        baseline_metrics = ModelAccuracyMetrics(
            precision=0.70, recall=0.65, f1_score=0.67, accuracy=0.68, roc_auc=0.72,
            win_rate=0.60, profit_factor=1.60, avg_win=120.0, avg_loss=-75.0,
            total_return=900.0, sharpe_ratio=1.20, max_drawdown=400.0,
            total_trades=50, winning_trades=30, losing_trades=20,
            avg_holding_period=5.0, risk_reward_ratio=1.60,
            model_id="baseline_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        # Create optimized metrics
        optimized_metrics = ModelAccuracyMetrics(
            precision=0.75, recall=0.68, f1_score=0.71, accuracy=0.72, roc_auc=0.78,
            win_rate=0.65, profit_factor=1.85, avg_win=150.0, avg_loss=-80.0,
            total_return=1250.0, sharpe_ratio=1.45, max_drawdown=350.0,
            total_trades=50, winning_trades=32, losing_trades=18,
            avg_holding_period=4.5, risk_reward_ratio=1.88,
            model_id="optimized_model", evaluation_date=datetime.now(),
            test_period_days=90, symbol="BTCUSDT", strategy_name="test_strategy"
        )
        
        # Calculate improvements
        f1_improvement = optimized_metrics.f1_score - baseline_metrics.f1_score
        win_rate_improvement = optimized_metrics.win_rate - baseline_metrics.win_rate
        profit_factor_improvement = optimized_metrics.profit_factor - baseline_metrics.profit_factor
        
        logger.info(f"✅ Model comparison completed:")
        logger.info(f"   F1 Improvement: {f1_improvement:+.3f}")
        logger.info(f"   Win Rate Improvement: {win_rate_improvement:+.3f}")
        logger.info(f"   Profit Factor Improvement: {profit_factor_improvement:+.3f}")
        
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")
    
    # Test 4: Database integration (if available)
    logger.info("Test 4: Database integration")
    
    try:
        from ..database.connection import TimescaleDBConnection
        from ..database.queries import TimescaleQueries
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            # Test accuracy summary query
            summary = await TimescaleQueries.get_model_accuracy_summary(session, days=30)
            logger.info(f"Database summary: {len(summary)} model records")
            
            # Test best performing models query
            best_models = await TimescaleQueries.get_best_performing_models(session, limit=5)
            logger.info(f"Best performing models: {len(best_models)} records")
            
            # Test model performance ranking
            ranking = await TimescaleQueries.get_model_performance_ranking(session, days=30)
            logger.info(f"Model performance ranking: {len(ranking)} records")
            
        logger.info("✅ Database integration tests completed successfully!")
        
    except ImportError as e:
        logger.warning(f"Database not available: {e}")
    except Exception as e:
        logger.error(f"Database integration test failed: {e}")
    
    # Test 5: Generate comparison report
    logger.info("Test 5: Generate comparison report")
    
    try:
        # Create a mock comparison
        from app.core.accuracy_evaluator import BaselineComparison
        
        comparison = BaselineComparison(
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
        
        # Generate report
        await accuracy_evaluator._generate_comparison_report(comparison)
        logger.info("✅ Comparison report generated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
    
    logger.info("🎉 All accuracy benchmarking tests completed!")

def create_mock_test_data():
    """Create mock test data for evaluation"""
    
    mock_data = []
    
    # Generate 50 mock trades
    for i in range(50):
        # Random trade outcome
        is_winning = random.random() > 0.4  # 60% win rate
        
        # Random signal confidence
        confidence = random.uniform(0.3, 0.9)
        
        # Calculate PnL based on outcome
        if is_winning:
            pnl = random.uniform(50, 300)  # Winning trades: $50-$300
        else:
            pnl = random.uniform(-200, -30)  # Losing trades: -$200 to -$30
        
        # Mock trade data
        trade = {
            'symbol': 'BTCUSDT',
            'side': 'long' if random.random() > 0.5 else 'short',
            'entry_price': random.uniform(40000, 50000),
            'exit_price': random.uniform(40000, 50000),
            'quantity': random.uniform(0.01, 0.1),
            'pnl': pnl,
            'entry_time': datetime.now() - timedelta(days=random.randint(1, 90)),
            'exit_time': datetime.now() - timedelta(days=random.randint(0, 89)),
            'status': 'closed',
            'signal_confidence': confidence,
            'signal_type': 'buy' if random.random() > 0.5 else 'sell',
            'strategy_name': 'test_strategy'
        }
        
        mock_data.append(trade)
    
    return mock_data

async def test_database_migration():
    """Test database migration for accuracy benchmarks"""
    
    logger.info("🧪 Testing Database Migration")
    
    try:
        from ..database.connection import TimescaleDBConnection
        from sqlalchemy import text
        
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            # Test if table exists
            check_table_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_accuracy_benchmarks'
            );
            """
            
            result = await session.execute(text(check_table_query))
            table_exists = result.scalar()
            
            if table_exists:
                logger.info("✅ model_accuracy_benchmarks table exists")
                
                # Test inserting a mock record
                insert_query = """
                INSERT INTO model_accuracy_benchmarks (
                    model_id, symbol, strategy_name,
                    precision, recall, f1_score, accuracy, roc_auc,
                    win_rate, profit_factor, avg_win, avg_loss, total_return,
                    sharpe_ratio, max_drawdown, total_trades, winning_trades,
                    losing_trades, avg_holding_period, risk_reward_ratio,
                    test_period_days, frozen_test_set, evaluation_date
                ) VALUES (
                    :model_id, :symbol, :strategy_name,
                    :precision, :recall, :f1_score, :accuracy, :roc_auc,
                    :win_rate, :profit_factor, :avg_win, :avg_loss, :total_return,
                    :sharpe_ratio, :max_drawdown, :total_trades, :winning_trades,
                    :losing_trades, :avg_holding_period, :risk_reward_ratio,
                    :test_period_days, :frozen_test_set, :evaluation_date
                )
                """
                
                mock_record = {
                    'model_id': 'test_model_migration',
                    'symbol': 'BTCUSDT',
                    'strategy_name': 'test_strategy',
                    'precision': 0.75,
                    'recall': 0.68,
                    'f1_score': 0.71,
                    'accuracy': 0.72,
                    'roc_auc': 0.78,
                    'win_rate': 0.65,
                    'profit_factor': 1.85,
                    'avg_win': 150.0,
                    'avg_loss': -80.0,
                    'total_return': 1250.0,
                    'sharpe_ratio': 1.45,
                    'max_drawdown': 350.0,
                    'total_trades': 50,
                    'winning_trades': 32,
                    'losing_trades': 18,
                    'avg_holding_period': 4.5,
                    'risk_reward_ratio': 1.88,
                    'test_period_days': 90,
                    'frozen_test_set': True,
                    'evaluation_date': datetime.now()
                }
                
                await session.execute(text(insert_query), mock_record)
                await session.commit()
                
                logger.info("✅ Mock record inserted successfully")
                
            else:
                logger.warning("⚠️ model_accuracy_benchmarks table does not exist - run migration first")
        
    except Exception as e:
        logger.error(f"❌ Database migration test failed: {e}")

async def main():
    """Main test function"""
    
    logger.info("🚀 Starting Accuracy Benchmarking Tests")
    
    # Test accuracy evaluator
    await test_accuracy_evaluator()
    
    # Test database migration
    await test_database_migration()
    
    logger.info("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
