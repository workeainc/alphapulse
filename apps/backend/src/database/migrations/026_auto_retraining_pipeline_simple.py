#!/usr/bin/env python3
"""
Migration 026: Enhanced Auto-Retraining Pipeline (Simplified)
Creates comprehensive tables for automated model retraining, drift detection, and performance monitoring
"""

import os
import sys
from sqlalchemy import create_engine, text

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_migration():
    """Run the auto-retraining pipeline migration"""
    
    # Database configuration
    DATABASE_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    engine = create_engine(DATABASE_URL)
    
    print("🔄 Creating Enhanced Auto-Retraining Pipeline tables...")
    
    # 1. Auto-Retraining Jobs Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS auto_retraining_jobs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                job_id VARCHAR(255) UNIQUE NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                timeframe VARCHAR(20) NOT NULL,
                schedule_cron VARCHAR(100) DEFAULT '0 2 * * *',
                priority INTEGER DEFAULT 1,
                auto_deploy BOOLEAN DEFAULT TRUE,
                performance_threshold DECIMAL(5,4) DEFAULT 0.8,
                drift_threshold DECIMAL(5,4) DEFAULT 0.25,
                min_training_samples INTEGER DEFAULT 1000,
                max_training_age_days INTEGER DEFAULT 30,
                last_run TIMESTAMPTZ,
                next_run TIMESTAMPTZ,
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
    
    # 2. Retraining Job History Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS retraining_job_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                job_id VARCHAR(255) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                model_version VARCHAR(100),
                trigger_type VARCHAR(50) NOT NULL,
                trigger_details JSONB,
                training_start TIMESTAMPTZ,
                training_end TIMESTAMPTZ,
                status VARCHAR(50) NOT NULL,
                performance_metrics JSONB,
                training_samples INTEGER,
                validation_samples INTEGER,
                training_time_seconds DECIMAL(10,2),
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
    
    # 3. Model Drift Monitoring Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_drift_monitoring (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(255) NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                drift_type VARCHAR(50) NOT NULL,
                feature_name VARCHAR(255),
                drift_score DECIMAL(10,6) NOT NULL,
                threshold DECIMAL(10,6) NOT NULL,
                is_drift_detected BOOLEAN NOT NULL,
                monitoring_window_hours INTEGER DEFAULT 24,
                samples_analyzed INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
    
    # 4. Model Performance Tracking Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_performance_tracking (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(255) NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                timeframe VARCHAR(20) NOT NULL,
                auc_score DECIMAL(5,4),
                precision_score DECIMAL(5,4),
                recall_score DECIMAL(5,4),
                accuracy_score DECIMAL(5,4),
                f1_score DECIMAL(5,4),
                sharpe_ratio DECIMAL(8,4),
                max_drawdown DECIMAL(8,4),
                win_rate DECIMAL(5,4),
                profit_factor DECIMAL(8,4),
                total_trades INTEGER,
                profitable_trades INTEGER,
                evaluation_window_hours INTEGER DEFAULT 24,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
    
    # 5. Model Version Management Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_version_management (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(255) NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                file_path VARCHAR(500),
                file_size_bytes BIGINT,
                model_hash VARCHAR(64),
                training_config JSONB,
                hyperparameters JSONB,
                feature_importance JSONB,
                training_metrics JSONB,
                deployment_status VARCHAR(50) DEFAULT 'trained',
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                deployed_at TIMESTAMPTZ,
                UNIQUE(model_name, model_version, symbol)
            )
        """))
        conn.commit()
    
    # 6. Auto-Retraining Configuration Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS auto_retraining_config (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                config_name VARCHAR(255) UNIQUE NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                default_schedule_cron VARCHAR(100) DEFAULT '0 2 * * *',
                default_drift_threshold DECIMAL(5,4) DEFAULT 0.25,
                default_performance_threshold DECIMAL(5,4) DEFAULT 0.8,
                default_min_samples INTEGER DEFAULT 1000,
                default_max_age_days INTEGER DEFAULT 30,
                retraining_strategy VARCHAR(50) DEFAULT 'incremental',
                drift_detection_methods JSONB DEFAULT '["psi", "kl_divergence", "statistical"]',
                performance_metrics JSONB DEFAULT '["auc", "precision", "recall", "f1"]',
                auto_deployment_rules JSONB,
                notification_settings JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
    
    # Create indexes for performance
    with engine.connect() as conn:
        # Auto-retraining jobs indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_auto_retraining_jobs_model_name ON auto_retraining_jobs(model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_auto_retraining_jobs_status ON auto_retraining_jobs(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_auto_retraining_jobs_next_run ON auto_retraining_jobs(next_run)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_auto_retraining_jobs_symbol ON auto_retraining_jobs(symbol)"))
        
        # Retraining job history indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_retraining_job_history_job_id ON retraining_job_history(job_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_retraining_job_history_model_name ON retraining_job_history(model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_retraining_job_history_status ON retraining_job_history(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_retraining_job_history_training_start ON retraining_job_history(training_start)"))
        
        # Model drift monitoring indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_drift_monitoring_model_name ON model_drift_monitoring(model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_drift_monitoring_drift_detected ON model_drift_monitoring(is_drift_detected)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_drift_monitoring_drift_type ON model_drift_monitoring(drift_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_drift_monitoring_symbol ON model_drift_monitoring(symbol)"))
        
        # Model performance tracking indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_performance_tracking_model_name ON model_performance_tracking(model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_performance_tracking_symbol ON model_performance_tracking(symbol)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_performance_tracking_auc_score ON model_performance_tracking(auc_score)"))
        
        # Model version management indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_version_management_model_name ON model_version_management(model_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_version_management_deployment_status ON model_version_management(deployment_status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_version_management_is_active ON model_version_management(is_active)"))
        
        conn.commit()
    
    # Insert default configurations
    with engine.connect() as conn:
        # Default auto-retraining configurations
        default_configs = [
            {
                'config_name': 'lightgbm_default',
                'model_type': 'lightgbm',
                'default_schedule_cron': '0 2 * * *',
                'default_drift_threshold': 0.25,
                'default_performance_threshold': 0.8,
                'retraining_strategy': 'incremental'
            },
            {
                'config_name': 'lstm_default',
                'model_type': 'lstm',
                'default_schedule_cron': '0 3 * * *',
                'default_drift_threshold': 0.3,
                'default_performance_threshold': 0.75,
                'retraining_strategy': 'full'
            },
            {
                'config_name': 'transformer_default',
                'model_type': 'transformer',
                'default_schedule_cron': '0 4 * * *',
                'default_drift_threshold': 0.3,
                'default_performance_threshold': 0.75,
                'retraining_strategy': 'full'
            },
            {
                'config_name': 'ensemble_default',
                'model_type': 'ensemble',
                'default_schedule_cron': '0 5 * * *',
                'default_drift_threshold': 0.2,
                'default_performance_threshold': 0.85,
                'retraining_strategy': 'incremental'
            }
        ]
        
        for config in default_configs:
            conn.execute(text("""
                INSERT INTO auto_retraining_config (
                    config_name, model_type, default_schedule_cron, default_drift_threshold,
                    default_performance_threshold, retraining_strategy, drift_detection_methods,
                    performance_metrics, auto_deployment_rules, notification_settings
                ) VALUES (
                    :config_name, :model_type, :default_schedule_cron, :default_drift_threshold,
                    :default_performance_threshold, :retraining_strategy,
                    '["psi", "kl_divergence", "statistical"]',
                    '["auc", "precision", "recall", "f1", "sharpe_ratio"]',
                    '{"min_auc": 0.7, "max_drawdown": 0.2, "min_trades": 10}',
                    '{"email": true, "slack": false, "webhook": false}'
                ) ON CONFLICT (config_name) DO NOTHING
            """), config)
        
        conn.commit()
    
    print("✅ Enhanced Auto-Retraining Pipeline tables created successfully!")
    print("📊 Created tables:")
    print("   - auto_retraining_jobs")
    print("   - retraining_job_history")
    print("   - model_drift_monitoring")
    print("   - model_performance_tracking")
    print("   - model_version_management")
    print("   - auto_retraining_config")
    print("🔗 Created performance indexes")
    print("⚙️  Inserted default configurations for LightGBM, LSTM, Transformer, and Ensemble models")

if __name__ == "__main__":
    run_migration()
