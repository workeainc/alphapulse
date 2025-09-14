# Phase 6: Database Migration Implementation Guide

## 🎯 Overview
This guide provides the complete implementation for Phase 6 database migrations that need to be executed when the database connection is available.

## 📋 Database Migration Requirements

### 1. New Tables to Create

#### A. ML Model Performance Tracking Table
```sql
CREATE TABLE IF NOT EXISTS ml_model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_score FLOAT,
    latency_ms FLOAT,
    throughput_per_sec FLOAT,
    memory_usage_mb FLOAT,
    gpu_usage_percent FLOAT,
    prediction_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    drift_score FLOAT,
    health_score FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### B. Model Health Monitoring Table
```sql
CREATE TABLE IF NOT EXISTS model_health_monitoring (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    health_status VARCHAR(20) DEFAULT 'healthy',
    overall_health_score FLOAT,
    feature_drift_score FLOAT,
    concept_drift_score FLOAT,
    performance_drift_score FLOAT,
    data_quality_score FLOAT,
    model_stability_score FLOAT,
    alert_level VARCHAR(20) DEFAULT 'none',
    alert_message TEXT,
    recommendations JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### C. Advanced ML Integration Results Table
```sql
CREATE TABLE IF NOT EXISTS advanced_ml_integration_results (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    catboost_prediction FLOAT,
    catboost_confidence FLOAT,
    drift_detection_score FLOAT,
    chart_pattern_score FLOAT,
    candlestick_pattern_score FLOAT,
    volume_analysis_score FLOAT,
    ensemble_prediction FLOAT,
    ensemble_confidence FLOAT,
    ml_health_score FLOAT,
    processing_time_ms FLOAT,
    model_versions JSONB,
    feature_importance JSONB,
    prediction_explanations JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### D. ML Model Registry Table
```sql
CREATE TABLE IF NOT EXISTS ml_model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    model_path VARCHAR(500) NOT NULL,
    onnx_path VARCHAR(500),
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    accuracy FLOAT,
    training_date TIMESTAMPTZ,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    model_size_mb FLOAT,
    input_features JSONB,
    output_classes JSONB,
    hyperparameters JSONB,
    performance_metrics JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### E. Model Training History Table
```sql
CREATE TABLE IF NOT EXISTS model_training_history (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    training_run_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    training_status VARCHAR(20) DEFAULT 'completed',
    training_duration_seconds FLOAT,
    training_samples INTEGER,
    validation_samples INTEGER,
    test_samples INTEGER,
    initial_accuracy FLOAT,
    final_accuracy FLOAT,
    accuracy_improvement FLOAT,
    loss_history JSONB,
    metrics_history JSONB,
    hyperparameters JSONB,
    feature_importance JSONB,
    training_logs TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. Enhanced Existing Tables

#### A. Add ML Columns to Signals Table
```sql
ALTER TABLE signals 
ADD COLUMN IF NOT EXISTS ml_model_confidence FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS ml_health_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS catboost_prediction FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS drift_detection_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS chart_pattern_ml_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS candlestick_ml_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS volume_ml_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS ml_processing_time_ms FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS ml_model_versions JSONB,
ADD COLUMN IF NOT EXISTS ml_prediction_explanations JSONB;
```

### 3. Performance Indexes

```sql
-- ML Model Performance indexes
CREATE INDEX IF NOT EXISTS idx_ml_model_performance_model_timestamp 
ON ml_model_performance(model_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ml_model_performance_symbol_timeframe 
ON ml_model_performance(symbol, timeframe, timestamp DESC);

-- Model Health Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_model_health_monitoring_model_timestamp 
ON model_health_monitoring(model_name, timestamp DESC);

-- Advanced ML Integration Results indexes
CREATE INDEX IF NOT EXISTS idx_advanced_ml_integration_results_signal_id 
ON advanced_ml_integration_results(signal_id);

CREATE INDEX IF NOT EXISTS idx_advanced_ml_integration_results_symbol_timeframe 
ON advanced_ml_integration_results(symbol, timeframe, timestamp DESC);

-- ML Model Registry indexes
CREATE INDEX IF NOT EXISTS idx_ml_model_registry_model_type_status 
ON ml_model_registry(model_type, status);

-- Model Training History indexes
CREATE INDEX IF NOT EXISTS idx_model_training_history_model_timestamp 
ON model_training_history(model_name, timestamp DESC);
```

### 4. Default Data Insertion

#### A. Insert Default ML Model Registry Entries
```sql
INSERT INTO ml_model_registry (model_name, model_type, model_path, version, status, metadata)
VALUES 
('catboost_nightly_incremental', 'catboost', 'models/catboost_nightly_incremental_20250814_151525.model', '1.0.0', 'active', '{"description": "Nightly incremental CatBoost model", "training_frequency": "daily"}'),
('xgboost_weekly_quick', 'xgboost', 'models/xgboost_weekly_quick_20250814_151525.model', '1.0.0', 'active', '{"description": "Weekly quick XGBoost model", "training_frequency": "weekly"}'),
('lightgbm_monthly_full', 'lightgbm', 'models/lightgbm_monthly_full_20250814_151525.model', '1.0.0', 'active', '{"description": "Monthly full LightGBM model", "training_frequency": "monthly"}')
ON CONFLICT (model_name) DO NOTHING;
```

## 🚀 Implementation Steps

### Step 1: Run the Migration Script
When database connection is available, run:
```bash
python run_phase6_migration_fixed.py
```

### Step 2: Verify Migration Success
Run the verification script:
```bash
python test_phase6_advanced_ml_integration.py
```

### Step 3: Check Database Tables
Verify all tables were created successfully:
```sql
-- Check if all Phase 6 tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'ml_model_performance',
    'model_health_monitoring', 
    'advanced_ml_integration_results',
    'ml_model_registry',
    'model_training_history'
)
ORDER BY table_name;

-- Check if ML columns were added to signals table
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'signals' 
AND column_name IN (
    'ml_model_confidence',
    'ml_health_score', 
    'catboost_prediction',
    'drift_detection_score',
    'chart_pattern_ml_score',
    'candlestick_ml_score',
    'volume_ml_score'
)
ORDER BY column_name;
```

## 📊 Expected Results

### After Successful Migration:
- ✅ 5 new tables created
- ✅ 10 new columns added to signals table
- ✅ 7 performance indexes created
- ✅ 3 default ML model registry entries inserted
- ✅ All ML integration features ready for use

### Database Schema Summary:
```
Phase 6 Tables:
├── ml_model_performance (21 columns)
├── model_health_monitoring (15 columns)
├── advanced_ml_integration_results (20 columns)
├── ml_model_registry (17 columns)
└── model_training_history (19 columns)

Enhanced Tables:
└── signals (+10 ML columns)
```

## 🔧 Integration with Signal Generator

The enhanced signal generator (`intelligent_signal_generator.py`) is already updated to use these new database tables:

1. **ML Model Performance Tracking**: Automatically logs model performance metrics
2. **Model Health Monitoring**: Tracks model health and drift detection
3. **Advanced ML Integration Results**: Stores comprehensive ML analysis results
4. **ML Model Registry**: Manages available ML models
5. **Model Training History**: Tracks training runs and improvements

## 📈 Benefits After Migration

1. **Comprehensive ML Tracking**: Full visibility into ML model performance
2. **Health Monitoring**: Real-time model health assessment
3. **Performance Optimization**: Data-driven model improvement
4. **Scalable Architecture**: Easy addition of new ML models
5. **Historical Analysis**: Complete training and performance history

## ⚠️ Important Notes

1. **Database Connection Required**: Migration can only be run when database is accessible
2. **Backup Recommended**: Always backup database before running migrations
3. **Rollback Available**: Migration script includes rollback functionality
4. **Verification Required**: Always verify migration success after completion

## 🎯 Next Steps

1. **Run Migration**: Execute `run_phase6_migration_fixed.py` when database is available
2. **Verify Success**: Run `test_phase6_advanced_ml_integration.py` to confirm
3. **Test Integration**: Verify ML components work with new database schema
4. **Monitor Performance**: Use new tables to track ML model performance
5. **Proceed to Phase 7**: Continue with real-time processing enhancements

The Phase 6 database migration is ready for implementation and will provide the foundation for advanced ML model tracking and monitoring in the AlphaPlus trading system.
