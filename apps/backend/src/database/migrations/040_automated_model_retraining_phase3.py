"""
Phase 3: Automated Model Retraining and Deployment Pipeline
Builds upon surgical upgrades to create a comprehensive ML pipeline
"""

import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

class AutomatedModelRetrainingMigration:
    """Migration for automated model retraining and deployment pipeline"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'alpha_emon',
            'password': 'Emon_@17711',
            'database': 'alphapulse'
        }
    
    async def create_connection_pool(self) -> asyncpg.Pool:
        """Create database connection pool"""
        try:
            pool = await asyncpg.create_pool(**self.db_config)
            logger.info("✅ Database connection pool created successfully")
            return pool
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            raise
    
    async def run_migration(self):
        """Run the complete migration"""
        pool = None
        try:
            pool = await self.create_connection_pool()
            
            async with pool.acquire() as conn:
                # Create core tables first
                await self.create_model_training_tables(conn)
                await self.create_deployment_pipeline_tables(conn)
                await self.create_performance_monitoring_tables(conn)
                
                # Wait a moment for tables to be fully created
                await asyncio.sleep(1)
                
                # Create basic indexes for core tables
                await self.create_basic_indexes(conn)
                
                # Insert default configurations
                await self.insert_default_configurations(conn)
                
            logger.info("✅ Phase 3: Automated Model Retraining migration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            if pool:
                await pool.close()
    
    async def create_model_training_tables(self, conn: asyncpg.Connection):
        """Create model training related tables"""
        
        try:
            # Model Training Jobs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_training_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                    model_name VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20) NOT NULL,
                    training_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    training_config JSONB NOT NULL,
                    hyperparameters JSONB,
                    feature_config JSONB,
                    data_sources JSONB NOT NULL,
                    current_epoch INTEGER DEFAULT 0,
                    total_epochs INTEGER,
                    current_metric FLOAT,
                    best_metric FLOAT,
                    training_loss FLOAT,
                    validation_loss FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    estimated_completion TIMESTAMP WITH TIME ZONE,
                    gpu_utilization FLOAT,
                    memory_usage_mb FLOAT,
                    cpu_utilization FLOAT,
                    training_metrics JSONB,
                    validation_metrics JSONB,
                    model_artifacts JSONB,
                    error_message TEXT,
                    parent_model_id INTEGER REFERENCES model_training_jobs(id),
                    trigger_type VARCHAR(50),
                    trigger_metadata JSONB
                )
            """)
            logger.info("✅ model_training_jobs table created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create model_training_jobs table: {e}")
            raise
        
        # Training Data Management
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data_management (
                id SERIAL PRIMARY KEY,
                dataset_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                dataset_name VARCHAR(100) NOT NULL,
                dataset_version VARCHAR(20) NOT NULL,
                data_type VARCHAR(50) NOT NULL,
                data_sources JSONB NOT NULL,
                preprocessing_config JSONB,
                feature_engineering_config JSONB,
                validation_split FLOAT DEFAULT 0.2,
                test_split FLOAT DEFAULT 0.1,
                total_samples INTEGER,
                feature_count INTEGER,
                class_distribution JSONB,
                data_quality_metrics JSONB,
                storage_path TEXT,
                storage_size_mb FLOAT,
                compression_ratio FLOAT,
                parent_dataset_id INTEGER REFERENCES training_data_management(id),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                description TEXT,
                tags JSONB,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        try:
            # Model Performance Tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_tracking (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES model_training_jobs(id),
                    evaluation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    accuracy FLOAT,
                    precision FLOAT,
                    recall FLOAT,
                    f1_score FLOAT,
                    auc_roc FLOAT,
                    log_loss FLOAT,
                    profit_factor FLOAT,
                    sharpe_ratio FLOAT,
                    max_drawdown FLOAT,
                    win_rate FLOAT,
                    avg_win FLOAT,
                    avg_loss FLOAT,
                    feature_drift_score FLOAT,
                    concept_drift_score FLOAT,
                    data_drift_score FLOAT,
                    calibration_error FLOAT,
                    reliability_score FLOAT,
                    uncertainty_estimate FLOAT,
                    test_dataset_id INTEGER REFERENCES training_data_management(id),
                    evaluation_config JSONB,
                    performance_metadata JSONB
                )
            """)
            logger.info("✅ model_performance_tracking table created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create model_performance_tracking table: {e}")
            raise
    
    async def create_deployment_pipeline_tables(self, conn: asyncpg.Connection):
        """Create deployment pipeline tables"""
        
        # Model Deployment Pipeline
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_deployment_pipeline (
                id SERIAL PRIMARY KEY,
                deployment_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                model_job_id INTEGER REFERENCES model_training_jobs(id),
                
                -- Deployment Configuration
                deployment_type VARCHAR(50) NOT NULL, -- 'canary', 'blue_green', 'rolling', 'shadow'
                target_environment VARCHAR(50) NOT NULL, -- 'staging', 'production', 'testing'
                deployment_config JSONB NOT NULL,
                
                -- Pipeline Stages
                current_stage VARCHAR(50) DEFAULT 'pending', -- 'pending', 'building', 'testing', 'deploying', 'monitoring', 'completed'
                stage_progress FLOAT DEFAULT 0.0,
                stage_started_at TIMESTAMP WITH TIME ZONE,
                stage_completed_at TIMESTAMP WITH TIME ZONE,
                
                -- Validation Results
                validation_passed BOOLEAN,
                validation_metrics JSONB,
                validation_errors JSONB,
                
                -- Deployment Status
                deployment_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'in_progress', 'success', 'failed', 'rolled_back'
                deployment_started_at TIMESTAMP WITH TIME ZONE,
                deployment_completed_at TIMESTAMP WITH TIME ZONE,
                
                -- Rollback Information
                rollback_reason TEXT,
                rollback_timestamp TIMESTAMP WITH TIME ZONE,
                previous_deployment_id INTEGER REFERENCES model_deployment_pipeline(id),
                
                -- Resource Allocation
                resource_requirements JSONB,
                actual_resource_usage JSONB,
                
                -- Monitoring
                health_checks JSONB,
                performance_monitoring JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # A/B Testing Framework
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_testing_framework (
                id SERIAL PRIMARY KEY,
                test_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                test_name VARCHAR(100) NOT NULL,
                description TEXT,
                
                -- Test Configuration
                control_model_id INTEGER REFERENCES model_training_jobs(id),
                treatment_model_id INTEGER REFERENCES model_training_jobs(id),
                traffic_split JSONB NOT NULL, -- {'control': 0.5, 'treatment': 0.5}
                
                -- Test Parameters
                test_duration_days INTEGER,
                min_sample_size INTEGER,
                confidence_level FLOAT DEFAULT 0.95,
                statistical_power FLOAT DEFAULT 0.8,
                
                -- Test Status
                status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'running', 'paused', 'completed', 'stopped'
                start_date TIMESTAMP WITH TIME ZONE,
                end_date TIMESTAMP WITH TIME ZONE,
                
                -- Results
                control_metrics JSONB,
                treatment_metrics JSONB,
                statistical_significance BOOLEAN,
                p_value FLOAT,
                effect_size FLOAT,
                winner VARCHAR(20), -- 'control', 'treatment', 'none'
                
                -- Traffic Management
                current_traffic_split JSONB,
                traffic_rules JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Model Versioning and Rollback
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versioning (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                version_tag VARCHAR(20) NOT NULL,
                deployment_id INTEGER REFERENCES model_deployment_pipeline(id),
                
                -- Version Information
                version_type VARCHAR(20) NOT NULL, -- 'major', 'minor', 'patch', 'hotfix'
                release_notes TEXT,
                breaking_changes BOOLEAN DEFAULT FALSE,
                
                -- Model Artifacts
                model_path TEXT,
                model_size_mb FLOAT,
                model_hash VARCHAR(64),
                dependencies JSONB,
                
                -- Performance Baseline
                baseline_metrics JSONB,
                performance_thresholds JSONB,
                
                -- Deployment History
                deployed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                deployed_by VARCHAR(100),
                deployment_environment VARCHAR(50),
                
                -- Status
                is_active BOOLEAN DEFAULT FALSE,
                is_deprecated BOOLEAN DEFAULT FALSE,
                deprecation_date TIMESTAMP WITH TIME ZONE,
                
                UNIQUE(model_name, version_tag)
            )
        """)
    
    async def create_performance_monitoring_tables(self, conn: asyncpg.Connection):
        """Create performance monitoring tables"""
        
        # Real-time Model Performance
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS real_time_model_performance (
                id SERIAL PRIMARY KEY,
                model_id INTEGER REFERENCES model_training_jobs(id),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Inference Metrics
                inference_latency_ms FLOAT,
                throughput_rps FLOAT,
                error_rate FLOAT,
                success_rate FLOAT,
                
                -- Prediction Quality
                prediction_confidence FLOAT,
                prediction_uncertainty FLOAT,
                calibration_error FLOAT,
                
                -- Resource Usage
                cpu_usage_percent FLOAT,
                memory_usage_mb FLOAT,
                gpu_usage_percent FLOAT,
                gpu_memory_mb FLOAT,
                
                -- Business Impact
                signal_accuracy FLOAT,
                profit_loss FLOAT,
                risk_adjusted_return FLOAT,
                
                -- Alerting
                alert_level VARCHAR(20), -- 'normal', 'warning', 'critical'
                alert_message TEXT,
                alert_metadata JSONB
            )
        """)
        
        # Model Drift Detection
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_drift_detection (
                id SERIAL PRIMARY KEY,
                model_id INTEGER REFERENCES model_training_jobs(id),
                detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Drift Metrics
                feature_drift_score FLOAT,
                concept_drift_score FLOAT,
                data_drift_score FLOAT,
                prediction_drift_score FLOAT,
                
                -- Statistical Tests
                kolmogorov_smirnov_p_value FLOAT,
                population_stability_index FLOAT,
                chi_square_p_value FLOAT,
                
                -- Feature-level Drift
                feature_drift_details JSONB,
                top_drifted_features JSONB,
                
                -- Alerting
                drift_severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
                alert_triggered BOOLEAN DEFAULT FALSE,
                alert_message TEXT,
                
                -- Action Taken
                action_taken VARCHAR(50), -- 'none', 'retrain_scheduled', 'model_rolled_back', 'investigation_started'
                action_metadata JSONB
            )
        """)
        
        # Automated Retraining Triggers
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automated_retraining_triggers (
                id SERIAL PRIMARY KEY,
                trigger_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                model_name VARCHAR(100) NOT NULL,
                
                -- Trigger Configuration
                trigger_type VARCHAR(50) NOT NULL, -- 'drift_threshold', 'performance_degradation', 'time_based', 'manual'
                trigger_conditions JSONB NOT NULL,
                trigger_thresholds JSONB,
                
                -- Trigger Status
                is_active BOOLEAN DEFAULT TRUE,
                last_triggered_at TIMESTAMP WITH TIME ZONE,
                trigger_count INTEGER DEFAULT 0,
                
                -- Retraining Configuration
                retraining_config JSONB,
                priority INTEGER DEFAULT 5,
                
                -- Monitoring
                monitoring_window_hours INTEGER DEFAULT 24,
                evaluation_metrics JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def create_automation_control_tables(self, conn: asyncpg.Connection):
        """Create automation control tables"""
        
        # ML Pipeline Orchestration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_pipeline_orchestration (
                id SERIAL PRIMARY KEY,
                pipeline_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                pipeline_name VARCHAR(100) NOT NULL,
                
                -- Pipeline Configuration
                pipeline_type VARCHAR(50) NOT NULL, -- 'training', 'deployment', 'monitoring', 'retraining'
                pipeline_config JSONB NOT NULL,
                dependencies JSONB,
                
                -- Execution Control
                status VARCHAR(20) DEFAULT 'idle', -- 'idle', 'running', 'paused', 'completed', 'failed'
                current_step VARCHAR(100),
                step_progress FLOAT DEFAULT 0.0,
                
                -- Scheduling
                schedule_config JSONB,
                next_run_at TIMESTAMP WITH TIME ZONE,
                last_run_at TIMESTAMP WITH TIME ZONE,
                
                -- Resource Management
                resource_limits JSONB,
                resource_usage JSONB,
                
                -- Error Handling
                max_retries INTEGER DEFAULT 3,
                retry_delay_seconds INTEGER DEFAULT 300,
                error_handling_config JSONB,
                
                -- Monitoring
                health_status VARCHAR(20) DEFAULT 'healthy',
                performance_metrics JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Automated Decision Making
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS automated_decision_making (
                id SERIAL PRIMARY KEY,
                decision_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                decision_type VARCHAR(50) NOT NULL, -- 'retrain', 'deploy', 'rollback', 'alert'
                
                -- Decision Context
                trigger_event JSONB,
                context_data JSONB,
                decision_criteria JSONB,
                
                -- Decision Process
                decision_algorithm VARCHAR(100),
                confidence_score FLOAT,
                reasoning TEXT,
                
                -- Decision Outcome
                decision VARCHAR(20), -- 'approve', 'reject', 'escalate', 'auto_approve'
                action_taken VARCHAR(100),
                action_metadata JSONB,
                
                -- Approval Process
                requires_approval BOOLEAN DEFAULT FALSE,
                approved_by VARCHAR(100),
                approved_at TIMESTAMP WITH TIME ZONE,
                approval_notes TEXT,
                
                -- Execution
                executed_at TIMESTAMP WITH TIME ZONE,
                execution_status VARCHAR(20), -- 'pending', 'executing', 'completed', 'failed'
                execution_result JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Quality Gates
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_gates (
                id SERIAL PRIMARY KEY,
                gate_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                gate_name VARCHAR(100) NOT NULL,
                gate_type VARCHAR(50) NOT NULL, -- 'performance', 'drift', 'business_metrics', 'technical'
                
                -- Gate Configuration
                gate_conditions JSONB NOT NULL,
                thresholds JSONB NOT NULL,
                evaluation_window_hours INTEGER DEFAULT 24,
                
                -- Gate Status
                is_active BOOLEAN DEFAULT TRUE,
                current_status VARCHAR(20) DEFAULT 'pass', -- 'pass', 'fail', 'warning'
                last_evaluated_at TIMESTAMP WITH TIME ZONE,
                
                -- Evaluation Results
                evaluation_metrics JSONB,
                failure_reasons JSONB,
                recommendations JSONB,
                
                -- Actions
                actions_on_failure JSONB,
                notification_config JSONB,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def create_ml_ops_tables(self, conn: asyncpg.Connection):
        """Create ML Ops tables"""
        
        # ML Ops Dashboard
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_ops_dashboard (
                id SERIAL PRIMARY KEY,
                dashboard_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                dashboard_name VARCHAR(100) NOT NULL,
                
                -- Dashboard Configuration
                dashboard_type VARCHAR(50) NOT NULL, -- 'overview', 'model_performance', 'pipeline_status', 'alerts'
                layout_config JSONB,
                refresh_interval_seconds INTEGER DEFAULT 300,
                
                -- Widgets
                widgets JSONB,
                custom_metrics JSONB,
                
                -- Access Control
                access_level VARCHAR(20) DEFAULT 'read', -- 'read', 'write', 'admin'
                user_permissions JSONB,
                
                -- Status
                is_active BOOLEAN DEFAULT TRUE,
                last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # ML Ops Alerts
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_ops_alerts (
                id SERIAL PRIMARY KEY,
                alert_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                alert_type VARCHAR(50) NOT NULL, -- 'performance', 'drift', 'pipeline', 'system'
                
                -- Alert Configuration
                alert_name VARCHAR(100) NOT NULL,
                alert_description TEXT,
                severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
                
                -- Alert Conditions
                trigger_conditions JSONB,
                evaluation_expression TEXT,
                
                -- Alert Status
                status VARCHAR(20) DEFAULT 'active', -- 'active', 'acknowledged', 'resolved', 'suppressed'
                is_enabled BOOLEAN DEFAULT TRUE,
                
                -- Notification
                notification_channels JSONB, -- ['email', 'slack', 'webhook']
                notification_template TEXT,
                escalation_config JSONB,
                
                -- Alert History
                last_triggered_at TIMESTAMP WITH TIME ZONE,
                trigger_count INTEGER DEFAULT 0,
                resolution_time_minutes INTEGER,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # ML Ops Reports
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_ops_reports (
                id SERIAL PRIMARY KEY,
                report_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
                report_name VARCHAR(100) NOT NULL,
                
                -- Report Configuration
                report_type VARCHAR(50) NOT NULL, -- 'daily', 'weekly', 'monthly', 'custom'
                report_template JSONB,
                data_sources JSONB,
                
                -- Report Content
                report_data JSONB,
                summary_metrics JSONB,
                recommendations JSONB,
                
                -- Generation
                generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                generated_by VARCHAR(100),
                generation_duration_seconds FLOAT,
                
                -- Distribution
                distribution_list JSONB,
                delivery_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'sent', 'failed'
                delivery_metadata JSONB,
                
                -- Storage
                report_file_path TEXT,
                report_file_size_mb FLOAT,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    async def create_basic_indexes(self, conn: asyncpg.Connection):
        """Create basic performance indexes for core tables"""
        
        try:
            # Model Training Jobs indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON model_training_jobs(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_model_name ON model_training_jobs(model_name)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON model_training_jobs(created_at)")
            logger.info("✅ Model training jobs indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create model training jobs indexes: {e}")
        
        try:
            # Training Data Management indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_data_name ON training_data_management(dataset_name)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_data_active ON training_data_management(is_active)")
            logger.info("✅ Training data management indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create training data management indexes: {e}")
        
        try:
            # Model Performance Tracking indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_model_id ON model_performance_tracking(model_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON model_performance_tracking(evaluation_timestamp)")
            logger.info("✅ Model performance tracking indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create model performance tracking indexes: {e}")
        
        try:
            # Deployment Pipeline indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_deployment_status ON model_deployment_pipeline(deployment_status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_deployment_environment ON model_deployment_pipeline(target_environment)")
            logger.info("✅ Deployment pipeline indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create deployment pipeline indexes: {e}")
        
        try:
            # Real-time Performance indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_realtime_model_id ON real_time_model_performance(model_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_realtime_timestamp ON real_time_model_performance(timestamp)")
            logger.info("✅ Real-time performance indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create real-time performance indexes: {e}")
        
        try:
            # Drift Detection indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_model_id ON model_drift_detection(model_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON model_drift_detection(detection_timestamp)")
            logger.info("✅ Drift detection indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create drift detection indexes: {e}")
        
        try:
            # Automated Retraining Triggers indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_retraining_triggers_model_name ON automated_retraining_triggers(model_name)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_retraining_triggers_active ON automated_retraining_triggers(is_active)")
            logger.info("✅ Automated retraining triggers indexes created")
        except Exception as e:
            logger.error(f"❌ Failed to create automated retraining triggers indexes: {e}")
    
    async def insert_default_configurations(self, conn: asyncpg.Connection):
        """Insert default configurations"""
        
        # Default retraining triggers
        try:
            await conn.execute("""
                INSERT INTO automated_retraining_triggers 
                (model_name, trigger_type, trigger_conditions, trigger_thresholds, retraining_config, priority)
                VALUES 
                ('catboost_signal_predictor', 'drift_threshold', 
                 '{"feature_drift_score": ">0.3", "concept_drift_score": ">0.25"}',
                 '{"feature_drift_threshold": 0.3, "concept_drift_threshold": 0.25, "performance_degradation_threshold": 0.1}',
                 '{"training_type": "retrain", "hyperparameters": {"iterations": 1000, "learning_rate": 0.1}, "validation_split": 0.2}',
                 8),
                ('xgboost_signal_predictor', 'performance_degradation',
                 '{"accuracy": "<0.85", "f1_score": "<0.80"}',
                 '{"accuracy_threshold": 0.85, "f1_threshold": 0.80, "drift_threshold": 0.3}',
                 '{"training_type": "incremental", "hyperparameters": {"n_estimators": 500, "max_depth": 6}, "validation_split": 0.2}',
                 7),
                ('lightgbm_signal_predictor', 'time_based',
                 '{"days_since_last_training": ">7"}',
                 '{"max_days_between_training": 7, "min_performance_threshold": 0.82}',
                 '{"training_type": "retrain", "hyperparameters": {"num_leaves": 31, "learning_rate": 0.05}, "validation_split": 0.2}',
                 6)
                ON CONFLICT DO NOTHING
            """)
            logger.info("✅ Default retraining triggers inserted")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert default retraining triggers: {e}")
        
        # Default quality gates
        try:
            await conn.execute("""
                INSERT INTO quality_gates 
                (gate_name, gate_type, gate_conditions, thresholds, evaluation_window_hours, actions_on_failure)
                VALUES 
                ('Model Performance Gate', 'performance', 
                 '{"accuracy": ">=0.85", "f1_score": ">=0.80", "latency": "<=100"}',
                 '{"min_accuracy": 0.85, "min_f1_score": 0.80, "max_latency_ms": 100}',
                 24,
                 '{"action": "block_deployment", "notification": "slack", "escalation": "manual_review"}'),
                ('Drift Detection Gate', 'drift',
                 '{"feature_drift_score": "<=0.3", "concept_drift_score": "<=0.25"}',
                 '{"max_feature_drift": 0.3, "max_concept_drift": 0.25}',
                 24,
                 '{"action": "trigger_retraining", "notification": "email", "escalation": "auto_retrain"}'),
                ('Business Metrics Gate', 'business_metrics',
                 '{"profit_factor": ">=1.5", "sharpe_ratio": ">=1.0", "max_drawdown": "<=0.1"}',
                 '{"min_profit_factor": 1.5, "min_sharpe_ratio": 1.0, "max_drawdown_threshold": 0.1}',
                 168,
                 '{"action": "alert_only", "notification": "dashboard", "escalation": "none"}')
                ON CONFLICT DO NOTHING
            """)
            logger.info("✅ Default quality gates inserted")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert default quality gates: {e}")
        
        # Default ML Ops alerts
        try:
            await conn.execute("""
                INSERT INTO ml_ops_alerts 
                (alert_name, alert_type, alert_description, severity, trigger_conditions, notification_channels)
                VALUES 
                ('Model Performance Degradation', 'performance',
                 'Model performance has dropped below acceptable thresholds',
                 'warning',
                 '{"accuracy": "<0.85", "f1_score": "<0.80"}',
                 '["slack", "email"]'),
                ('High Drift Detection', 'drift',
                 'Feature or concept drift detected above threshold',
                 'error',
                 '{"feature_drift_score": ">0.3", "concept_drift_score": ">0.25"}',
                 '["slack", "email", "webhook"]'),
                ('Pipeline Failure', 'pipeline',
                 'ML pipeline has failed or is stuck',
                 'critical',
                 '{"pipeline_status": "failed", "retry_count": ">3"}',
                 '["slack", "email", "webhook", "sms"]'),
                ('System Resource Alert', 'system',
                 'System resources are running low',
                 'warning',
                 '{"cpu_usage": ">80", "memory_usage": ">85", "gpu_usage": ">90"}',
                 '["slack", "email"]')
                ON CONFLICT DO NOTHING
            """)
            logger.info("✅ Default ML Ops alerts inserted")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert default ML Ops alerts: {e}")

async def main():
    """Main migration function"""
    migration = AutomatedModelRetrainingMigration()
    await migration.run_migration()

if __name__ == "__main__":
    asyncio.run(main())
