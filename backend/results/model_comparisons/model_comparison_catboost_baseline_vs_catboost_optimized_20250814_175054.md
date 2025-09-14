# Model Comparison Report

## Overview
- **Baseline Model**: catboost_baseline v20250814_140809
- **Optimized Model**: catboost_optimized v20250814_151525
- **Symbol**: BTCUSDT
- **Strategy**: test_strategy
- **Test Period**: 90 days
- **Comparison Date**: 2025-08-14 17:50:54

## Model Information

### Baseline Model
- **File**: models/catboost_baseline_20250814_140809.model
- **Created**: 2025-07-15 17:50:54
- **Type**: baseline

### Optimized Model
- **File**: models/catboost_optimized_20250814_151525.model
- **Created**: 2025-08-07 17:50:54
- **Type**: optimized

## Accuracy Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Precision | 0.700 | 0.750 | +0.050 |
| Recall | 0.650 | 0.680 | +0.030 |
| F1-Score | 0.670 | 0.710 | +0.040 |
| Win Rate | 0.600 | 0.650 | +0.050 |
| Profit Factor | 1.600 | 1.850 | +0.250 |

## Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Return | 900.00 | 1250.00 | +350.00 |
| Sharpe Ratio | 1.200 | 1.450 | +0.250 |
| Max Drawdown | 400.00 | 350.00 | -50.00 |

## Latency Comparison

- **Latency Improvement**: +50.0ms
- **Throughput Improvement**: +15.0%

## Business Impact

- **Expected PnL Improvement**: +350.00
- **Risk Adjustment Factor**: 0.125

## Promotion Decision

- **Should Promote**: ✅ YES
- **Confidence Score**: 0.850
- **Reason**: Significant improvements in F1-score and win rate

## Recommendations


### ✅ Promotion Recommended

The optimized model shows significant improvements and should be promoted to production:

1. **Accuracy Improvements**: +0.040 F1-score improvement
2. **Performance Gains**: +0.050 win rate improvement  
3. **Latency Reduction**: +50.0ms faster inference
4. **Business Impact**: Expected +350.00 PnL improvement

**Next Steps**:
- Deploy optimized model to staging environment
- Monitor performance for 24-48 hours
- If stable, promote to production
- Update baseline model reference
