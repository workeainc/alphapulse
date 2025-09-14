"""
Model Comparison Manager
Comprehensive baseline vs optimized model comparison system
Builds on existing accuracy evaluator and model storage infrastructure
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

# Import existing infrastructure
from app.core.accuracy_evaluator import accuracy_evaluator, ModelAccuracyMetrics, BaselineComparison
from app.core.latency_tracker import latency_tracker
from ..database.connection import TimescaleDBConnection
from ..database.queries import TimescaleQueries

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    model_type: str  # 'baseline', 'optimized', 'candidate'
    file_path: str
    created_at: datetime
    metadata: Dict[str, Any]
    performance_metrics: Optional[ModelAccuracyMetrics] = None
    latency_metrics: Optional[Dict[str, float]] = None

@dataclass
class ModelComparisonResult:
    """Comprehensive model comparison result"""
    baseline_model: ModelVersion
    optimized_model: ModelVersion
    
    # Accuracy improvements
    accuracy_comparison: BaselineComparison
    
    # Latency improvements
    latency_improvement_ms: float
    throughput_improvement_percent: float
    
    # Business impact
    expected_pnl_improvement: float
    risk_adjustment_factor: float
    
    # Deployment recommendation
    should_promote: bool
    confidence_score: float
    promotion_reason: str
    
    # Metadata
    comparison_date: datetime
    test_period_days: int
    symbol: str
    strategy_name: str

class ModelComparisonManager:
    """Comprehensive model comparison and promotion manager"""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "results/model_comparisons"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_connection = None  # Initialize lazily
        
        # Configuration
        self.promotion_thresholds = {
            'min_f1_improvement': 0.05,
            'min_win_rate_improvement': 0.10,
            'min_latency_improvement_ms': 50.0,
            'min_confidence_score': 0.7,
            'min_test_period_days': 30
        }
        
        # Model registry
        self.model_registry: Dict[str, ModelVersion] = {}
        self.comparison_history: List[ModelComparisonResult] = []
    
    async def _get_db_connection(self):
        """Get database connection (initialize if needed)"""
        if self.db_connection is None:
            self.db_connection = TimescaleDBConnection()
            self.db_connection.initialize()
        return self.db_connection
    
    async def scan_model_directory(self) -> Dict[str, ModelVersion]:
        """Scan models directory and build model registry"""
        
        self.logger.info(f"🔍 Scanning model directory: {self.models_dir}")
        
        model_versions = {}
        
        try:
            for model_file in self.models_dir.rglob("*.model"):
                # Parse model filename to extract metadata
                model_info = self._parse_model_filename(model_file.name)
                
                if model_info:
                    # Get metadata file if it exists
                    metadata_file = model_file.with_suffix('.json')
                    metadata = {}
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                        except Exception as e:
                            self.logger.warning(f"Could not load metadata for {model_file.name}: {e}")
                    
                    # Create model version
                    model_version = ModelVersion(
                        model_id=model_info['model_id'],
                        version=model_info['version'],
                        model_type=self._determine_model_type(model_info),
                        file_path=str(model_file),
                        created_at=model_info['created_at'],
                        metadata=metadata
                    )
                    
                    model_versions[model_version.model_id] = model_version
                    self.logger.info(f"📦 Found model: {model_version.model_id} v{model_version.version}")
            
            self.model_registry = model_versions
            self.logger.info(f"✅ Model registry built: {len(model_versions)} models found")
            
            return model_versions
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning model directory: {e}")
            return {}
    
    def _parse_model_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse model filename to extract metadata"""
        
        try:
            # Example: catboost_nightly_incremental_20250814_151525.model
            parts = filename.replace('.model', '').split('_')
            
            if len(parts) >= 4:
                # Extract model type and timestamp
                model_type = '_'.join(parts[:-2])  # catboost_nightly_incremental
                date_str = parts[-2]  # 20250814
                time_str = parts[-1]  # 151525
                
                # Combine date and time
                timestamp_str = f"{date_str}_{time_str}"  # 20250814_151525
                
                # Parse timestamp
                created_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                return {
                    'model_id': model_type,
                    'version': timestamp_str,
                    'created_at': created_at,
                    'model_type': model_type
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not parse filename {filename}: {e}")
            return None
    
    def _determine_model_type(self, model_info: Dict[str, Any]) -> str:
        """Determine if model is baseline, optimized, or candidate"""
        
        model_id = model_info['model_id']
        
        # Check if it's a baseline model (first production model)
        if 'baseline' in model_id.lower():
            return 'baseline'
        
        # Check if it's an optimized model (improved version)
        if any(keyword in model_id.lower() for keyword in ['optimized', 'improved', 'enhanced']):
            return 'optimized'
        
        # Check if it's a candidate model (new version being tested)
        if any(keyword in model_id.lower() for keyword in ['candidate', 'test', 'experimental']):
            return 'candidate'
        
        # Default to optimized for newer models
        return 'optimized'
    
    async def compare_models(
        self,
        baseline_model_id: str,
        optimized_model_id: str,
        symbol: str,
        strategy_name: str,
        test_period_days: int = 90
    ) -> ModelComparisonResult:
        """
        Comprehensive comparison between baseline and optimized models
        
        Args:
            baseline_model_id: Baseline model identifier
            optimized_model_id: Optimized model identifier
            symbol: Trading symbol
            strategy_name: Strategy name
            test_period_days: Test period length
        """
        
        self.logger.info(f"🔄 Comparing models: {baseline_model_id} vs {optimized_model_id}")
        
        try:
            # Get model versions
            baseline_model = self.model_registry.get(baseline_model_id)
            optimized_model = self.model_registry.get(optimized_model_id)
            
            if not baseline_model or not optimized_model:
                raise ValueError(f"Model not found in registry: {baseline_model_id} or {optimized_model_id}")
            
            # Evaluate both models
            baseline_metrics = await accuracy_evaluator.evaluate_model_accuracy(
                baseline_model_id, symbol, strategy_name, test_period_days
            )
            
            optimized_metrics = await accuracy_evaluator.evaluate_model_accuracy(
                optimized_model_id, symbol, strategy_name, test_period_days
            )
            
            # Get latency metrics
            baseline_latency = await self._get_model_latency_metrics(baseline_model_id)
            optimized_latency = await self._get_model_latency_metrics(optimized_model_id)
            
            # Calculate latency improvements
            latency_improvement = baseline_latency.get('avg_total_latency_ms', 0) - optimized_latency.get('avg_total_latency_ms', 0)
            throughput_improvement = ((baseline_latency.get('avg_total_latency_ms', 1) - optimized_latency.get('avg_total_latency_ms', 1)) / 
                                   baseline_latency.get('avg_total_latency_ms', 1)) * 100
            
            # Create accuracy comparison
            accuracy_comparison = BaselineComparison(
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                precision_improvement=optimized_metrics.precision - baseline_metrics.precision,
                recall_improvement=optimized_metrics.recall - baseline_metrics.recall,
                f1_improvement=optimized_metrics.f1_score - baseline_metrics.f1_score,
                win_rate_improvement=optimized_metrics.win_rate - baseline_metrics.win_rate,
                profit_factor_improvement=optimized_metrics.profit_factor - baseline_metrics.profit_factor,
                latency_improvement_ms=latency_improvement,
                is_significant=False,  # Will be calculated
                p_value=0.0
            )
            
            # Calculate business impact
            expected_pnl_improvement = self._calculate_pnl_improvement(baseline_metrics, optimized_metrics)
            risk_adjustment_factor = self._calculate_risk_adjustment(baseline_metrics, optimized_metrics)
            
            # Determine promotion recommendation
            should_promote, confidence_score, promotion_reason = self._evaluate_promotion_criteria(
                accuracy_comparison, latency_improvement, test_period_days
            )
            
            # Create comparison result
            comparison_result = ModelComparisonResult(
                baseline_model=baseline_model,
                optimized_model=optimized_model,
                accuracy_comparison=accuracy_comparison,
                latency_improvement_ms=latency_improvement,
                throughput_improvement_percent=throughput_improvement,
                expected_pnl_improvement=expected_pnl_improvement,
                risk_adjustment_factor=risk_adjustment_factor,
                should_promote=should_promote,
                confidence_score=confidence_score,
                promotion_reason=promotion_reason,
                comparison_date=datetime.now(),
                test_period_days=test_period_days,
                symbol=symbol,
                strategy_name=strategy_name
            )
            
            # Store comparison
            self.comparison_history.append(comparison_result)
            
            # Save detailed comparison report
            await self._save_comparison_report(comparison_result)
            
            self.logger.info(f"✅ Model comparison completed: Should promote = {should_promote}")
            
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing models: {e}")
            raise
    
    async def _get_model_latency_metrics(self, model_id: str) -> Dict[str, float]:
        """Get latency metrics for a specific model"""
        
        try:
            db_connection = await self._get_db_connection()
            
            async with db_connection.get_async_session() as session:
                # Get latency metrics for the model
                query = """
                SELECT 
                    avg(total_latency_ms) as avg_total_latency_ms,
                    avg(fetch_time_ms) as avg_fetch_time_ms,
                    avg(preprocess_time_ms) as avg_preprocess_time_ms,
                    avg(inference_time_ms) as avg_inference_time_ms,
                    avg(postprocess_time_ms) as avg_postprocess_time_ms,
                    count(*) as operation_count
                FROM latency_metrics
                WHERE model_id = :model_id
                    AND timestamp >= NOW() - INTERVAL '7 days'
                """
                
                result = await session.execute(query, {"model_id": model_id})
                row = result.fetchone()
                
                if row:
                    return dict(row)
                else:
                    return {
                        'avg_total_latency_ms': 0.0,
                        'avg_fetch_time_ms': 0.0,
                        'avg_preprocess_time_ms': 0.0,
                        'avg_inference_time_ms': 0.0,
                        'avg_postprocess_time_ms': 0.0,
                        'operation_count': 0
                    }
                    
        except Exception as e:
            self.logger.warning(f"Could not get latency metrics for {model_id}: {e}")
            return {
                'avg_total_latency_ms': 0.0,
                'avg_fetch_time_ms': 0.0,
                'avg_preprocess_time_ms': 0.0,
                'avg_inference_time_ms': 0.0,
                'avg_postprocess_time_ms': 0.0,
                'operation_count': 0
            }
    
    def _calculate_pnl_improvement(self, baseline: ModelAccuracyMetrics, optimized: ModelAccuracyMetrics) -> float:
        """Calculate expected PnL improvement"""
        
        # Calculate expected return improvement
        baseline_expected_return = baseline.total_return * baseline.win_rate
        optimized_expected_return = optimized.total_return * optimized.win_rate
        
        return optimized_expected_return - baseline_expected_return
    
    def _calculate_risk_adjustment(self, baseline: ModelAccuracyMetrics, optimized: ModelAccuracyMetrics) -> float:
        """Calculate risk adjustment factor"""
        
        # Lower max drawdown is better (lower risk)
        baseline_risk = baseline.max_drawdown
        optimized_risk = optimized.max_drawdown
        
        if baseline_risk > 0:
            risk_improvement = (baseline_risk - optimized_risk) / baseline_risk
            return max(0.0, min(1.0, risk_improvement))  # Clamp between 0 and 1
        else:
            return 0.0
    
    def _evaluate_promotion_criteria(
        self,
        comparison: BaselineComparison,
        latency_improvement: float,
        test_period_days: int
    ) -> Tuple[bool, float, str]:
        """Evaluate if model should be promoted"""
        
        reasons = []
        confidence_factors = []
        
        # Check F1 improvement
        if comparison.f1_improvement >= self.promotion_thresholds['min_f1_improvement']:
            reasons.append(f"F1 improvement: {comparison.f1_improvement:+.3f}")
            confidence_factors.append(min(1.0, comparison.f1_improvement / 0.1))  # Normalize to 0-1
        
        # Check win rate improvement
        if comparison.win_rate_improvement >= self.promotion_thresholds['min_win_rate_improvement']:
            reasons.append(f"Win rate improvement: {comparison.win_rate_improvement:+.3f}")
            confidence_factors.append(min(1.0, comparison.win_rate_improvement / 0.2))
        
        # Check latency improvement
        if latency_improvement >= self.promotion_thresholds['min_latency_improvement_ms']:
            reasons.append(f"Latency improvement: {latency_improvement:.1f}ms")
            confidence_factors.append(min(1.0, latency_improvement / 100.0))
        
        # Check test period
        if test_period_days >= self.promotion_thresholds['min_test_period_days']:
            reasons.append(f"Sufficient test period: {test_period_days} days")
            confidence_factors.append(min(1.0, test_period_days / 90.0))
        
        # Calculate overall confidence
        confidence_score = np.mean(confidence_factors) if confidence_factors else 0.0
        
        # Determine promotion
        should_promote = (
            len(reasons) >= 2 and  # At least 2 criteria met
            confidence_score >= self.promotion_thresholds['min_confidence_score']
        )
        
        promotion_reason = "; ".join(reasons) if reasons else "No significant improvements"
        
        return should_promote, confidence_score, promotion_reason
    
    async def _save_comparison_report(self, comparison: ModelComparisonResult):
        """Save detailed comparison report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{comparison.baseline_model.model_id}_vs_{comparison.optimized_model.model_id}_{timestamp}.md"
        filepath = self.output_dir / filename
        
        report = f"""# Model Comparison Report

## Overview
- **Baseline Model**: {comparison.baseline_model.model_id} v{comparison.baseline_model.version}
- **Optimized Model**: {comparison.optimized_model.model_id} v{comparison.optimized_model.version}
- **Symbol**: {comparison.symbol}
- **Strategy**: {comparison.strategy_name}
- **Test Period**: {comparison.test_period_days} days
- **Comparison Date**: {comparison.comparison_date.strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

### Baseline Model
- **File**: {comparison.baseline_model.file_path}
- **Created**: {comparison.baseline_model.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Type**: {comparison.baseline_model.model_type}

### Optimized Model
- **File**: {comparison.optimized_model.file_path}
- **Created**: {comparison.optimized_model.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Type**: {comparison.optimized_model.model_type}

## Accuracy Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Precision | {comparison.accuracy_comparison.baseline_metrics.precision:.3f} | {comparison.accuracy_comparison.optimized_metrics.precision:.3f} | {comparison.accuracy_comparison.precision_improvement:+.3f} |
| Recall | {comparison.accuracy_comparison.baseline_metrics.recall:.3f} | {comparison.accuracy_comparison.optimized_metrics.recall:.3f} | {comparison.accuracy_comparison.recall_improvement:+.3f} |
| F1-Score | {comparison.accuracy_comparison.baseline_metrics.f1_score:.3f} | {comparison.accuracy_comparison.optimized_metrics.f1_score:.3f} | {comparison.accuracy_comparison.f1_improvement:+.3f} |
| Win Rate | {comparison.accuracy_comparison.baseline_metrics.win_rate:.3f} | {comparison.accuracy_comparison.optimized_metrics.win_rate:.3f} | {comparison.accuracy_comparison.win_rate_improvement:+.3f} |
| Profit Factor | {comparison.accuracy_comparison.baseline_metrics.profit_factor:.3f} | {comparison.accuracy_comparison.optimized_metrics.profit_factor:.3f} | {comparison.accuracy_comparison.profit_factor_improvement:+.3f} |

## Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Return | {comparison.accuracy_comparison.baseline_metrics.total_return:.2f} | {comparison.accuracy_comparison.optimized_metrics.total_return:.2f} | {comparison.accuracy_comparison.optimized_metrics.total_return - comparison.accuracy_comparison.baseline_metrics.total_return:+.2f} |
| Sharpe Ratio | {comparison.accuracy_comparison.baseline_metrics.sharpe_ratio:.3f} | {comparison.accuracy_comparison.optimized_metrics.sharpe_ratio:.3f} | {comparison.accuracy_comparison.optimized_metrics.sharpe_ratio - comparison.accuracy_comparison.baseline_metrics.sharpe_ratio:+.3f} |
| Max Drawdown | {comparison.accuracy_comparison.baseline_metrics.max_drawdown:.2f} | {comparison.accuracy_comparison.optimized_metrics.max_drawdown:.2f} | {comparison.accuracy_comparison.optimized_metrics.max_drawdown - comparison.accuracy_comparison.baseline_metrics.max_drawdown:+.2f} |

## Latency Comparison

- **Latency Improvement**: {comparison.latency_improvement_ms:+.1f}ms
- **Throughput Improvement**: {comparison.throughput_improvement_percent:+.1f}%

## Business Impact

- **Expected PnL Improvement**: {comparison.expected_pnl_improvement:+.2f}
- **Risk Adjustment Factor**: {comparison.risk_adjustment_factor:.3f}

## Promotion Decision

- **Should Promote**: {'✅ YES' if comparison.should_promote else '❌ NO'}
- **Confidence Score**: {comparison.confidence_score:.3f}
- **Reason**: {comparison.promotion_reason}

## Recommendations

"""
        
        if comparison.should_promote:
            report += f"""
### ✅ Promotion Recommended

The optimized model shows significant improvements and should be promoted to production:

1. **Accuracy Improvements**: {comparison.accuracy_comparison.f1_improvement:+.3f} F1-score improvement
2. **Performance Gains**: {comparison.accuracy_comparison.win_rate_improvement:+.3f} win rate improvement  
3. **Latency Reduction**: {comparison.latency_improvement_ms:+.1f}ms faster inference
4. **Business Impact**: Expected {comparison.expected_pnl_improvement:+.2f} PnL improvement

**Next Steps**:
- Deploy optimized model to staging environment
- Monitor performance for 24-48 hours
- If stable, promote to production
- Update baseline model reference
"""
        else:
            report += f"""
### ⚠️ Promotion Not Recommended

The optimized model does not meet promotion criteria:

**Issues**:
- Insufficient improvements in key metrics
- Confidence score below threshold ({comparison.confidence_score:.3f} < {self.promotion_thresholds['min_confidence_score']})
- {comparison.promotion_reason}

**Next Steps**:
- Continue testing with longer evaluation period
- Investigate model optimization approach
- Consider feature engineering improvements
- Re-evaluate after collecting more data
"""
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"📋 Comparison report saved: {filepath}")
    
    async def get_promotion_candidates(self, symbol: str, strategy_name: str) -> List[ModelComparisonResult]:
        """Get models that are candidates for promotion"""
        
        candidates = []
        
        for comparison in self.comparison_history:
            if (comparison.symbol == symbol and 
                comparison.strategy_name == strategy_name and
                comparison.should_promote):
                candidates.append(comparison)
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return candidates
    
    async def get_comparison_history(self, symbol: str = None, strategy_name: str = None) -> List[ModelComparisonResult]:
        """Get comparison history with optional filtering"""
        
        history = self.comparison_history
        
        if symbol:
            history = [c for c in history if c.symbol == symbol]
        
        if strategy_name:
            history = [c for c in history if c.strategy_name == strategy_name]
        
        # Sort by comparison date
        history.sort(key=lambda x: x.comparison_date, reverse=True)
        
        return history

# Global instance
model_comparison_manager = ModelComparisonManager()
