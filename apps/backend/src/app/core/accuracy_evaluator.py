"""
Accuracy Benchmarking Framework
Comprehensive model evaluation and trading performance analysis
Builds on existing infrastructure to provide ML metrics and business KPIs
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available - plotting disabled")

from pathlib import Path

# Import existing infrastructure
from ..src.database.connection import TimescaleDBConnection
from ..src.database.queries import TimescaleQueries

@dataclass
class ModelAccuracyMetrics:
    """Comprehensive model accuracy metrics"""
    # ML Classification Metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    roc_auc: float
    
    # Trading Performance Metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Additional Metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_period: float
    risk_reward_ratio: float
    
    # Metadata
    model_id: str
    evaluation_date: datetime
    test_period_days: int
    symbol: str
    strategy_name: str

@dataclass
class BaselineComparison:
    """Comparison between baseline and optimized models"""
    baseline_metrics: ModelAccuracyMetrics
    optimized_metrics: ModelAccuracyMetrics
    
    # Improvement metrics
    precision_improvement: float
    recall_improvement: float
    f1_improvement: float
    win_rate_improvement: float
    profit_factor_improvement: float
    latency_improvement_ms: float
    
    # Statistical significance
    is_significant: bool
    p_value: float

class AccuracyEvaluator:
    """Comprehensive accuracy evaluation framework"""
    
    def __init__(self, output_dir: str = "results/accuracy_benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_connection = None  # Initialize lazily
        
        # Evaluation results storage
        self.evaluation_results: List[ModelAccuracyMetrics] = []
        self.baseline_comparisons: List[BaselineComparison] = []
        
        # Configuration
        self.min_trades_for_evaluation = 10
        self.confidence_interval = 0.95
        self.significance_threshold = 0.05
    
    async def _get_db_connection(self):
        """Get database connection (initialize if needed)"""
        if self.db_connection is None:
            self.db_connection = TimescaleDBConnection()
            self.db_connection.initialize()
        return self.db_connection
    
    async def evaluate_model_accuracy(
        self,
        model_id: str,
        symbol: str,
        strategy_name: str,
        test_period_days: int = 90,
        frozen_test_set: bool = True
    ) -> ModelAccuracyMetrics:
        """
        Evaluate model accuracy using frozen test set
        
        Args:
            model_id: ML model identifier
            symbol: Trading symbol
            strategy_name: Strategy name
            test_period_days: Number of days for test set
            frozen_test_set: Use frozen test set (not used in training)
        """
        
        self.logger.info(f"🔍 Evaluating model accuracy: {model_id} ({symbol})")
        
        try:
            # Get test data
            test_data = await self._get_test_data(
                symbol, strategy_name, test_period_days, frozen_test_set
            )
            
            if len(test_data) < self.min_trades_for_evaluation:
                raise ValueError(f"Insufficient test data: {len(test_data)} trades")
            
            # Calculate ML metrics
            ml_metrics = self._calculate_ml_metrics(test_data)
            
            # Calculate trading performance metrics
            trading_metrics = self._calculate_trading_metrics(test_data)
            
            # Create comprehensive metrics
            accuracy_metrics = ModelAccuracyMetrics(
                # ML metrics
                precision=ml_metrics['precision'],
                recall=ml_metrics['recall'],
                f1_score=ml_metrics['f1_score'],
                accuracy=ml_metrics['accuracy'],
                roc_auc=ml_metrics['roc_auc'],
                
                # Trading metrics
                win_rate=trading_metrics['win_rate'],
                profit_factor=trading_metrics['profit_factor'],
                avg_win=trading_metrics['avg_win'],
                avg_loss=trading_metrics['avg_loss'],
                total_return=trading_metrics['total_return'],
                sharpe_ratio=trading_metrics['sharpe_ratio'],
                max_drawdown=trading_metrics['max_drawdown'],
                
                # Additional metrics
                total_trades=trading_metrics['total_trades'],
                winning_trades=trading_metrics['winning_trades'],
                losing_trades=trading_metrics['losing_trades'],
                avg_holding_period=trading_metrics['avg_holding_period'],
                risk_reward_ratio=trading_metrics['risk_reward_ratio'],
                
                # Metadata
                model_id=model_id,
                evaluation_date=datetime.now(),
                test_period_days=test_period_days,
                symbol=symbol,
                strategy_name=strategy_name
            )
            
            # Store result
            self.evaluation_results.append(accuracy_metrics)
            
            # Save detailed results
            await self._save_evaluation_results(accuracy_metrics, test_data)
            
            self.logger.info(f"✅ Model evaluation completed: F1={accuracy_metrics.f1_score:.3f}, Win Rate={accuracy_metrics.win_rate:.3f}")
            
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating model {model_id}: {e}")
            raise
    
    async def compare_models(
        self,
        baseline_model_id: str,
        optimized_model_id: str,
        symbol: str,
        strategy_name: str,
        test_period_days: int = 90
    ) -> BaselineComparison:
        """
        Compare baseline vs optimized model performance
        
        Args:
            baseline_model_id: Baseline model identifier
            optimized_model_id: Optimized model identifier
            symbol: Trading symbol
            strategy_name: Strategy name
            test_period_days: Test period length
        """
        
        self.logger.info(f"🔄 Comparing models: {baseline_model_id} vs {optimized_model_id}")
        
        try:
            # Evaluate both models
            baseline_metrics = await self.evaluate_model_accuracy(
                baseline_model_id, symbol, strategy_name, test_period_days
            )
            
            optimized_metrics = await self.evaluate_model_accuracy(
                optimized_model_id, symbol, strategy_name, test_period_days
            )
            
            # Calculate improvements
            comparison = BaselineComparison(
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                
                # Calculate improvements
                precision_improvement=optimized_metrics.precision - baseline_metrics.precision,
                recall_improvement=optimized_metrics.recall - baseline_metrics.recall,
                f1_improvement=optimized_metrics.f1_score - baseline_metrics.f1_score,
                win_rate_improvement=optimized_metrics.win_rate - baseline_metrics.win_rate,
                profit_factor_improvement=optimized_metrics.profit_factor - baseline_metrics.profit_factor,
                latency_improvement_ms=0.0,  # Will be filled from latency tracker
                
                # Statistical significance (placeholder)
                is_significant=False,
                p_value=0.0
            )
            
            # Store comparison
            self.baseline_comparisons.append(comparison)
            
            # Generate comparison report
            await self._generate_comparison_report(comparison)
            
            self.logger.info(f"✅ Model comparison completed: F1 improvement = {comparison.f1_improvement:.3f}")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing models: {e}")
            raise
    
    async def _get_test_data(
        self,
        symbol: str,
        strategy_name: str,
        test_period_days: int,
        frozen_test_set: bool
    ) -> List[Dict]:
        """Get test data for evaluation"""
        
        try:
            async with (await self._get_db_connection()).get_async_session() as session:
                # Get trades from the test period
                query = """
                SELECT 
                    t.symbol,
                    t.side,
                    t.entry_price,
                    t.exit_price,
                    t.quantity,
                    t.pnl,
                    t.entry_time,
                    t.exit_time,
                    t.status,
                    s.confidence as signal_confidence,
                    s.signal_type,
                    s.strategy_name
                FROM trades t
                LEFT JOIN trading_signals s ON t.signal_id = s.id
                WHERE t.symbol = :symbol
                    AND s.strategy_name = :strategy_name
                    AND t.entry_time >= NOW() - INTERVAL ':days days'
                    AND t.status = 'closed'
                ORDER BY t.entry_time
                """
                
                result = await session.execute(
                    query,
                    {
                        "symbol": symbol,
                        "strategy_name": strategy_name,
                        "days": test_period_days
                    }
                )
                
                trades = [dict(row) for row in result.fetchall()]
                
                # If frozen test set, filter to specific period
                if frozen_test_set:
                    # Use last 30 days of the test period as frozen set
                    cutoff_date = datetime.now() - timedelta(days=test_period_days - 30)
                    trades = [t for t in trades if t['entry_time'] >= cutoff_date]
                
                return trades
                
        except Exception as e:
            self.logger.error(f"Error getting test data: {e}")
            return []
    
    def _calculate_ml_metrics(self, test_data: List[Dict]) -> Dict[str, float]:
        """Calculate ML classification metrics"""
        
        if not test_data:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'accuracy': 0.0, 'roc_auc': 0.0
            }
        
        # Prepare data for ML metrics
        y_true = []  # Actual outcomes (1 for profitable, 0 for loss)
        y_pred = []  # Predicted outcomes (based on signal confidence)
        y_scores = []  # Prediction probabilities
        
        for trade in test_data:
            # Actual outcome
            actual_outcome = 1 if trade.get('pnl', 0) > 0 else 0
            y_true.append(actual_outcome)
            
            # Predicted outcome (based on signal confidence)
            confidence = trade.get('signal_confidence', 0.5)
            predicted_outcome = 1 if confidence > 0.5 else 0
            y_pred.append(predicted_outcome)
            
            # Prediction probability
            y_scores.append(confidence)
        
        # Calculate metrics
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            # ROC AUC (handle edge cases)
            if len(set(y_true)) > 1 and len(set(y_scores)) > 1:
                roc_auc = roc_auc_score(y_true, y_scores)
            else:
                roc_auc = 0.5  # Random classifier
            
        except Exception as e:
            self.logger.warning(f"Error calculating ML metrics: {e}")
            precision = recall = f1 = accuracy = roc_auc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
    
    def _calculate_trading_metrics(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        
        if not test_data:
            return {
                'win_rate': 0.0, 'profit_factor': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'total_return': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown': 0.0, 'total_trades': 0, 'winning_trades': 0,
                'losing_trades': 0, 'avg_holding_period': 0.0, 'risk_reward_ratio': 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(test_data)
        winning_trades = [t for t in test_data if t.get('pnl', 0) > 0]
        losing_trades = [t for t in test_data if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.get('pnl', 0) for t in test_data)
        avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        if profit_factor == float('inf'):
            profit_factor = 10.0  # Cap at reasonable value
        
        # Risk metrics
        returns = [t.get('pnl', 0) for t in test_data]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Holding period
        holding_periods = []
        for trade in test_data:
            if trade.get('entry_time') and trade.get('exit_time'):
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                holding_periods.append(duration)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Risk-reward ratio
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_holding_period': avg_holding_period,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    async def _save_evaluation_results(
        self,
        metrics: ModelAccuracyMetrics,
        test_data: List[Dict]
    ):
        """Save detailed evaluation results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_evaluation_{metrics.model_id}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare results
        results = {
            'metrics': {
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'accuracy': metrics.accuracy,
                'roc_auc': metrics.roc_auc,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            },
            'metadata': {
                'model_id': metrics.model_id,
                'symbol': metrics.symbol,
                'strategy_name': metrics.strategy_name,
                'evaluation_date': metrics.evaluation_date.isoformat(),
                'test_period_days': metrics.test_period_days,
                'total_trades': metrics.total_trades
            },
            'test_data_summary': {
                'total_trades': len(test_data),
                'winning_trades': len([t for t in test_data if t.get('pnl', 0) > 0]),
                'losing_trades': len([t for t in test_data if t.get('pnl', 0) < 0]),
                'total_pnl': sum(t.get('pnl', 0) for t in test_data)
            }
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📊 Evaluation results saved: {filepath}")
    
    async def _generate_comparison_report(self, comparison: BaselineComparison):
        """Generate detailed comparison report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.md"
        filepath = self.output_dir / filename
        
        report = f"""# Model Comparison Report

## Overview
- **Baseline Model**: {comparison.baseline_metrics.model_id}
- **Optimized Model**: {comparison.optimized_metrics.model_id}
- **Symbol**: {comparison.baseline_metrics.symbol}
- **Strategy**: {comparison.baseline_metrics.strategy_name}
- **Test Period**: {comparison.baseline_metrics.test_period_days} days

## ML Metrics Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Precision | {comparison.baseline_metrics.precision:.3f} | {comparison.optimized_metrics.precision:.3f} | {comparison.precision_improvement:+.3f} |
| Recall | {comparison.baseline_metrics.recall:.3f} | {comparison.optimized_metrics.recall:.3f} | {comparison.recall_improvement:+.3f} |
| F1-Score | {comparison.baseline_metrics.f1_score:.3f} | {comparison.optimized_metrics.f1_score:.3f} | {comparison.f1_improvement:+.3f} |
| Accuracy | {comparison.baseline_metrics.accuracy:.3f} | {comparison.optimized_metrics.accuracy:.3f} | {comparison.optimized_metrics.accuracy - comparison.baseline_metrics.accuracy:+.3f} |
| ROC AUC | {comparison.baseline_metrics.roc_auc:.3f} | {comparison.optimized_metrics.roc_auc:.3f} | {comparison.optimized_metrics.roc_auc - comparison.baseline_metrics.roc_auc:+.3f} |

## Trading Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Win Rate | {comparison.baseline_metrics.win_rate:.3f} | {comparison.optimized_metrics.win_rate:.3f} | {comparison.win_rate_improvement:+.3f} |
| Profit Factor | {comparison.baseline_metrics.profit_factor:.3f} | {comparison.optimized_metrics.profit_factor:.3f} | {comparison.profit_factor_improvement:+.3f} |
| Total Return | {comparison.baseline_metrics.total_return:.2f} | {comparison.optimized_metrics.total_return:.2f} | {comparison.optimized_metrics.total_return - comparison.baseline_metrics.total_return:+.2f} |
| Sharpe Ratio | {comparison.baseline_metrics.sharpe_ratio:.3f} | {comparison.optimized_metrics.sharpe_ratio:.3f} | {comparison.optimized_metrics.sharpe_ratio - comparison.baseline_metrics.sharpe_ratio:+.3f} |
| Max Drawdown | {comparison.baseline_metrics.max_drawdown:.2f} | {comparison.optimized_metrics.max_drawdown:.2f} | {comparison.optimized_metrics.max_drawdown - comparison.baseline_metrics.max_drawdown:+.2f} |

## Summary

- **F1-Score Improvement**: {comparison.f1_improvement:+.3f}
- **Win Rate Improvement**: {comparison.win_rate_improvement:+.3f}
- **Profit Factor Improvement**: {comparison.profit_factor_improvement:+.3f}
- **Statistical Significance**: {'Yes' if comparison.is_significant else 'No'}

## Recommendations

"""
        
        # Add recommendations based on improvements
        if comparison.f1_improvement > 0.05:
            report += "- ✅ **Significant ML performance improvement** - Consider promoting optimized model\n"
        if comparison.win_rate_improvement > 0.1:
            report += "- ✅ **Strong trading performance improvement** - Model shows better trade selection\n"
        if comparison.profit_factor_improvement > 0.5:
            report += "- ✅ **Improved risk-adjusted returns** - Better risk management\n"
        
        if comparison.f1_improvement < 0:
            report += "- ⚠️ **ML performance degraded** - Investigate optimization approach\n"
        if comparison.win_rate_improvement < 0:
            report += "- ⚠️ **Trading performance degraded** - Consider reverting to baseline\n"
        
        # Save report
        with open(filepath, 'w') as f:
            f.write(report)
        
        self.logger.info(f"📋 Comparison report saved: {filepath}")

# Global instance
accuracy_evaluator = AccuracyEvaluator()
