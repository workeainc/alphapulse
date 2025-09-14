#!/usr/bin/env python3
"""
Migration 027: Closed-Loop Monitoring Integration
Creates tables for automated feedback loop between monitoring and retraining
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_migration():
    """Run the closed-loop monitoring migration"""
    
    # Database configuration
    DATABASE_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    engine = create_engine(DATABASE_URL)
    
    print("🔄 Creating Closed-Loop Monitoring tables...")
    
    # 1. Monitoring Alert Triggers Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monitoring_alert_triggers (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                alert_id VARCHAR(255) UNIQUE NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                alert_type VARCHAR(100) NOT NULL,
                severity_level VARCHAR(50) NOT NULL,
                trigger_condition JSONB NOT NULL,
                current_value DECIMAL(10,6),
                threshold_value DECIMAL(10,6),
                is_triggered BOOLEAN DEFAULT FALSE,
                triggered_at TIMESTAMPTZ,
                retraining_job_id VARCHAR(255),
                retraining_status VARCHAR(50) DEFAULT 'pending',
                alert_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
        
        # Create TimescaleDB hypertable
        try:
            conn.execute(text("SELECT create_hypertable('monitoring_alert_triggers', 'timestamp', if_not_exists => TRUE)"))
            conn.commit()
        except Exception as e:
            print(f"⚠️  TimescaleDB hypertable creation failed: {e}, using regular table")
    
    # 2. Closed-Loop Actions Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS closed_loop_actions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                action_id VARCHAR(255) UNIQUE NOT NULL,
                alert_id VARCHAR(255) NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                action_type VARCHAR(100) NOT NULL,
                action_status VARCHAR(50) DEFAULT 'pending',
                trigger_source VARCHAR(100) NOT NULL,
                action_config JSONB NOT NULL,
                execution_start TIMESTAMPTZ,
                execution_end TIMESTAMPTZ,
                success BOOLEAN,
                error_message TEXT,
                action_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
        
        try:
            conn.execute(text("SELECT create_hypertable('closed_loop_actions', 'timestamp', if_not_exists => TRUE)"))
            conn.commit()
        except Exception as e:
            print(f"⚠️  TimescaleDB hypertable creation failed: {e}, using regular table")
    
    # 3. Monitoring-Retraining Integration Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monitoring_retraining_integration (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                integration_id VARCHAR(255) UNIQUE NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                monitoring_rule_id VARCHAR(255) NOT NULL,
                retraining_job_id VARCHAR(255) NOT NULL,
                integration_type VARCHAR(100) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                trigger_conditions JSONB NOT NULL,
                action_sequence JSONB NOT NULL,
                cooldown_period_minutes INTEGER DEFAULT 60,
                last_triggered TIMESTAMPTZ,
                trigger_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                integration_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
        
        try:
            conn.execute(text("SELECT create_hypertable('monitoring_retraining_integration', 'timestamp', if_not_exists => TRUE)"))
            conn.commit()
        except Exception as e:
            print(f"⚠️  TimescaleDB hypertable creation failed: {e}, using regular table")
    
    # 4. Feedback Loop Metrics Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback_loop_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                metric_id VARCHAR(255) UNIQUE NOT NULL,
                model_id VARCHAR(255) NOT NULL,
                loop_type VARCHAR(100) NOT NULL,
                trigger_to_action_latency_seconds DECIMAL(10,2),
                action_success_rate DECIMAL(5,4),
                performance_improvement DECIMAL(5,4),
                drift_reduction DECIMAL(5,4),
                false_positive_rate DECIMAL(5,4),
                false_negative_rate DECIMAL(5,4),
                total_triggers INTEGER DEFAULT 0,
                successful_actions INTEGER DEFAULT 0,
                failed_actions INTEGER DEFAULT 0,
                metrics_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
        
        try:
            conn.execute(text("SELECT create_hypertable('feedback_loop_metrics', 'timestamp', if_not_exists => TRUE)"))
            conn.commit()
        except Exception as e:
            print(f"⚠️  TimescaleDB hypertable creation failed: {e}, using regular table")
    
    # 5. Automated Response Rules Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS automated_response_rules (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                rule_id VARCHAR(255) UNIQUE NOT NULL,
                rule_name VARCHAR(255) NOT NULL,
                rule_type VARCHAR(100) NOT NULL,
                model_id VARCHAR(255),
                trigger_conditions JSONB NOT NULL,
                response_actions JSONB NOT NULL,
                priority INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                cooldown_period_minutes INTEGER DEFAULT 30,
                max_triggers_per_hour INTEGER DEFAULT 10,
                rule_metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.commit()
        
        try:
            conn.execute(text("SELECT create_hypertable('automated_response_rules', 'timestamp', if_not_exists => TRUE)"))
            conn.commit()
        except Exception as e:
            print(f"⚠️  TimescaleDB hypertable creation failed: {e}, using regular table")
    
    # Create indexes for performance
    with engine.connect() as conn:
        # Monitoring alert triggers indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_alert_triggers_model_id ON monitoring_alert_triggers(model_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_alert_triggers_alert_type ON monitoring_alert_triggers(alert_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_alert_triggers_is_triggered ON monitoring_alert_triggers(is_triggered)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_alert_triggers_retraining_status ON monitoring_alert_triggers(retraining_status)"))
        
        # Closed-loop actions indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_closed_loop_actions_alert_id ON closed_loop_actions(alert_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_closed_loop_actions_model_id ON closed_loop_actions(model_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_closed_loop_actions_action_type ON closed_loop_actions(action_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_closed_loop_actions_action_status ON closed_loop_actions(action_status)"))
        
        # Monitoring-retraining integration indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_retraining_integration_model_id ON monitoring_retraining_integration(model_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_retraining_integration_is_active ON monitoring_retraining_integration(is_active)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_monitoring_retraining_integration_integration_type ON monitoring_retraining_integration(integration_type)"))
        
        # Feedback loop metrics indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_loop_metrics_model_id ON feedback_loop_metrics(model_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_feedback_loop_metrics_loop_type ON feedback_loop_metrics(loop_type)"))
        
        # Automated response rules indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_automated_response_rules_rule_type ON automated_response_rules(rule_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_automated_response_rules_model_id ON automated_response_rules(model_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_automated_response_rules_is_active ON automated_response_rules(is_active)"))
        
        conn.commit()
    
    # Insert default automated response rules
    with engine.connect() as conn:
        # Default rules for different scenarios
        default_rules = [
            {
                'rule_id': 'drift_retraining_rule',
                'rule_name': 'Drift Detection Auto-Retraining',
                'rule_type': 'drift_retraining',
                'trigger_conditions': '{"drift_score_threshold": 0.25, "consecutive_detections": 3, "time_window_hours": 24}',
                'response_actions': '{"action": "trigger_retraining", "priority": "high", "strategy": "full"}',
                'priority': 1
            },
            {
                'rule_id': 'performance_degradation_rule',
                'rule_name': 'Performance Degradation Auto-Retraining',
                'rule_type': 'performance_retraining',
                'trigger_conditions': '{"performance_drop_threshold": 0.1, "metrics": ["precision", "recall", "f1"], "time_window_hours": 12}',
                'response_actions': '{"action": "trigger_retraining", "priority": "medium", "strategy": "incremental"}',
                'priority': 2
            },
            {
                'rule_id': 'model_age_retraining_rule',
                'rule_name': 'Model Age Auto-Retraining',
                'rule_type': 'age_retraining',
                'trigger_conditions': '{"max_age_days": 30, "check_frequency_hours": 24}',
                'response_actions': '{"action": "trigger_retraining", "priority": "low", "strategy": "incremental"}',
                'priority': 3
            },
            {
                'rule_id': 'risk_alert_rule',
                'rule_name': 'Risk Alert Auto-Retraining',
                'rule_type': 'risk_retraining',
                'trigger_conditions': '{"risk_score_threshold": 80, "max_drawdown_threshold": 0.2, "time_window_hours": 6}',
                'response_actions': '{"action": "trigger_retraining", "priority": "critical", "strategy": "full"}',
                'priority': 1
            }
        ]
        
        for rule in default_rules:
            conn.execute(text("""
                INSERT INTO automated_response_rules (
                    rule_id, rule_name, rule_type, trigger_conditions, response_actions, priority
                ) VALUES (
                    :rule_id, :rule_name, :rule_type, :trigger_conditions, :response_actions, :priority
                ) ON CONFLICT (rule_id) DO NOTHING
            """), rule)
        
        conn.commit()
    
    print("✅ Closed-Loop Monitoring tables created successfully!")
    print("📊 Created tables:")
    print("   - monitoring_alert_triggers")
    print("   - closed_loop_actions")
    print("   - monitoring_retraining_integration")
    print("   - feedback_loop_metrics")
    print("   - automated_response_rules")
    print("🔗 Created performance indexes")
    print("⚙️  Inserted default automated response rules")

if __name__ == "__main__":
    run_migration()
