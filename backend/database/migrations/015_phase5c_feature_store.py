#!/usr/bin/env python3
"""
Phase 5C: Feature Store + Reproducible Pipelines Migration
Implements:
1. Feature definitions and schema contracts
2. Feature snapshots for time-travel
3. Pipeline runs for reproducible training
4. Feature drift detection and monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
import uuid

# Import TimescaleDB functions
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Revision identifiers
revision = '015_phase5c_feature_store'
down_revision = '014_phase5b_ensemble_meta_learner'
branch_labels = None
depends_on = None

def upgrade():
    """Create Phase 5C feature store tables"""
    
    # 1. Feature Definitions Table
    op.create_table(
        'feature_definitions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('schema', JSONB, nullable=False),  # JSON schema definition
        sa.Column('data_type', sa.String(50), nullable=False),  # numeric, categorical, datetime
        sa.Column('source_table', sa.String(255)),  # Source table for the feature
        sa.Column('computation_logic', sa.Text),  # SQL or Python logic
        sa.Column('owner', sa.String(255)),
        sa.Column('tags', JSONB),  # Feature tags for organization
        sa.Column('validation_rules', JSONB),  # Custom validation rules
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), default=sa.func.now(), onupdate=sa.func.now()),
        
        # Indexes
        sa.Index('idx_feature_definitions_name', 'name'),
        sa.Index('idx_feature_definitions_version', 'version'),
        sa.Index('idx_feature_definitions_active', 'is_active'),
        sa.Index('idx_feature_definitions_owner', 'owner'),
        sa.Index('idx_feature_definitions_tags', 'tags', postgresql_using='gin'),
    )
    
    # 2. Feature Snapshots Table (Time-travelable)
    op.create_table(
        'feature_snapshots',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('feature_definition_id', UUID(as_uuid=True), nullable=False),
        sa.Column('snapshot_timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('data_hash', sa.String(64), nullable=False),  # SHA256 hash of feature data
        sa.Column('feature_values', JSONB, nullable=False),  # Actual feature values
        sa.Column('metadata', JSONB),  # Additional metadata
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        
        # Foreign key
        sa.ForeignKeyConstraint(['feature_definition_id'], ['feature_definitions.id'], ondelete='CASCADE'),
        
        # Indexes
        sa.Index('idx_feature_snapshots_definition', 'feature_definition_id'),
        sa.Index('idx_feature_snapshots_timestamp', 'snapshot_timestamp'),
        sa.Index('idx_feature_snapshots_hash', 'data_hash'),
        sa.Index('idx_feature_snapshots_definition_timestamp', 'feature_definition_id', 'snapshot_timestamp'),
    )
    
    # 3. Feature Contracts Table (Schema validation)
    op.create_table(
        'feature_contracts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('schema_contract', JSONB, nullable=False),  # Expected schema
        sa.Column('validation_rules', JSONB),  # Custom validation rules
        sa.Column('drift_thresholds', JSONB),  # Drift detection thresholds
        sa.Column('owner', sa.String(255)),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), default=sa.func.now(), onupdate=sa.func.now()),
        
        # Indexes
        sa.Index('idx_feature_contracts_name', 'name'),
        sa.Index('idx_feature_contracts_version', 'version'),
        sa.Index('idx_feature_contracts_active', 'is_active'),
    )
    
    # 4. Pipeline Runs Table (Reproducible training)
    op.create_table(
        'pipeline_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('pipeline_name', sa.String(255), nullable=False),
        sa.Column('run_id', sa.String(255), nullable=False, unique=True),
        sa.Column('status', sa.String(50), nullable=False),  # running, completed, failed
        sa.Column('pipeline_config', JSONB, nullable=False),  # Pipeline configuration
        sa.Column('input_features', JSONB),  # Input feature versions
        sa.Column('output_artifacts', JSONB),  # Output model versions
        sa.Column('execution_metadata', JSONB),  # Execution details
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('completed_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('duration_seconds', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        
        # Indexes
        sa.Index('idx_pipeline_runs_name', 'pipeline_name'),
        sa.Index('idx_pipeline_runs_status', 'status'),
        sa.Index('idx_pipeline_runs_started', 'started_at'),
        sa.Index('idx_pipeline_runs_run_id', 'run_id'),
    )
    
    # 5. Feature Drift Detection Table
    op.create_table(
        'feature_drift_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('feature_definition_id', UUID(as_uuid=True), nullable=False),
        sa.Column('detection_timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('drift_type', sa.String(50), nullable=False),  # schema, distribution, missing
        sa.Column('drift_score', sa.Float),  # Drift severity score
        sa.Column('baseline_snapshot_id', UUID(as_uuid=True)),  # Reference snapshot
        sa.Column('current_snapshot_id', UUID(as_uuid=True)),  # Current snapshot
        sa.Column('drift_details', JSONB),  # Detailed drift information
        sa.Column('is_resolved', sa.Boolean, default=False),
        sa.Column('resolution_notes', sa.Text),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['feature_definition_id'], ['feature_definitions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['baseline_snapshot_id'], ['feature_snapshots.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['current_snapshot_id'], ['feature_snapshots.id'], ondelete='SET NULL'),
        
        # Indexes
        sa.Index('idx_feature_drift_feature', 'feature_definition_id'),
        sa.Index('idx_feature_drift_timestamp', 'detection_timestamp'),
        sa.Index('idx_feature_drift_type', 'drift_type'),
        sa.Index('idx_feature_drift_resolved', 'is_resolved'),
    )
    
    # 6. Feature Dependencies Table
    op.create_table(
        'feature_dependencies',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('dependent_feature_id', UUID(as_uuid=True), nullable=False),
        sa.Column('dependency_feature_id', UUID(as_uuid=True), nullable=False),
        sa.Column('dependency_type', sa.String(50), nullable=False),  # required, optional, derived
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), default=sa.func.now()),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['dependent_feature_id'], ['feature_definitions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dependency_feature_id'], ['feature_definitions.id'], ondelete='CASCADE'),
        
        # Indexes
        sa.Index('idx_feature_dependencies_dependent', 'dependent_feature_id'),
        sa.Index('idx_feature_dependencies_dependency', 'dependency_feature_id'),
        sa.Index('idx_feature_dependencies_type', 'dependency_type'),
    )
    
    # Convert to TimescaleDB hypertables
    connection = op.get_bind()
    
    # Convert feature_snapshots to hypertable
    connection.execute(text("""
        SELECT create_hypertable('feature_snapshots', 'snapshot_timestamp', 
                                if_not_exists => TRUE, 
                                chunk_time_interval => INTERVAL '1 day');
    """))
    
    # Convert pipeline_runs to hypertable
    connection.execute(text("""
        SELECT create_hypertable('pipeline_runs', 'started_at', 
                                if_not_exists => TRUE, 
                                chunk_time_interval => INTERVAL '1 hour');
    """))
    
    # Convert feature_drift_logs to hypertable
    connection.execute(text("""
        SELECT create_hypertable('feature_drift_logs', 'detection_timestamp', 
                                if_not_exists => TRUE, 
                                chunk_time_interval => INTERVAL '1 hour');
    """))
    
    # Create continuous aggregates for feature monitoring
    connection.execute(text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS feature_snapshots_hourly
        WITH (timescaledb.continuous) AS
        SELECT 
            feature_definition_id,
            time_bucket('1 hour', snapshot_timestamp) AS bucket,
            COUNT(*) as snapshot_count,
            AVG(EXTRACT(EPOCH FROM (snapshot_timestamp - LAG(snapshot_timestamp) OVER (PARTITION BY feature_definition_id ORDER BY snapshot_timestamp)))) as avg_interval_seconds
        FROM feature_snapshots
        GROUP BY feature_definition_id, bucket
        WITH NO DATA;
    """))
    
    connection.execute(text("""
        SELECT add_continuous_aggregate_policy('feature_snapshots_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """))
    
    # Create continuous aggregate for drift monitoring
    connection.execute(text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS feature_drift_hourly
        WITH (timescaledb.continuous) AS
        SELECT 
            feature_definition_id,
            drift_type,
            time_bucket('1 hour', detection_timestamp) AS bucket,
            COUNT(*) as drift_count,
            AVG(drift_score) as avg_drift_score,
            COUNT(*) FILTER (WHERE is_resolved = false) as unresolved_count
        FROM feature_drift_logs
        GROUP BY feature_definition_id, drift_type, bucket
        WITH NO DATA;
    """))
    
    connection.execute(text("""
        SELECT add_continuous_aggregate_policy('feature_drift_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
    """))
    
    # Insert initial feature definitions for Phase 5B ensemble
    initial_features = [
        {
            'name': 'close_price',
            'version': '1.0.0',
            'description': 'Closing price for candlestick patterns',
            'schema': {'type': 'number', 'minimum': 0, 'required': True},
            'data_type': 'numeric',
            'source_table': 'candlestick_patterns',
            'computation_logic': 'SELECT close FROM candlestick_patterns',
            'owner': 'system',
            'tags': ['price', 'technical', 'phase5b'],
            'validation_rules': {'not_null': True, 'positive': True}
        },
        {
            'name': 'volume',
            'version': '1.0.0',
            'description': 'Trading volume for candlestick patterns',
            'schema': {'type': 'number', 'minimum': 0, 'required': True},
            'data_type': 'numeric',
            'source_table': 'candlestick_patterns',
            'computation_logic': 'SELECT volume FROM candlestick_patterns',
            'owner': 'system',
            'tags': ['volume', 'technical', 'phase5b'],
            'validation_rules': {'not_null': True, 'positive': True}
        },
        {
            'name': 'btc_dominance',
            'version': '1.0.0',
            'description': 'Bitcoin dominance percentage',
            'schema': {'type': 'number', 'minimum': 0, 'maximum': 100, 'required': True},
            'data_type': 'numeric',
            'source_table': 'market_intelligence',
            'computation_logic': 'SELECT btc_dominance FROM market_intelligence',
            'owner': 'system',
            'tags': ['market', 'dominance', 'phase5b'],
            'validation_rules': {'not_null': True, 'range': [0, 100]}
        },
        {
            'name': 'market_correlation',
            'version': '1.0.0',
            'description': 'Market correlation coefficient',
            'schema': {'type': 'number', 'minimum': -1, 'maximum': 1, 'required': True},
            'data_type': 'numeric',
            'source_table': 'market_intelligence',
            'computation_logic': 'SELECT market_correlation FROM market_intelligence',
            'owner': 'system',
            'tags': ['market', 'correlation', 'phase5b'],
            'validation_rules': {'not_null': True, 'range': [-1, 1]}
        },
        {
            'name': 'volume_ratio',
            'version': '1.0.0',
            'description': 'Volume ratio compared to average',
            'schema': {'type': 'number', 'minimum': 0, 'required': True},
            'data_type': 'numeric',
            'source_table': 'candlestick_patterns',
            'computation_logic': 'SELECT volume / AVG(volume) OVER (ORDER BY timestamp ROWS 20 PRECEDING) FROM candlestick_patterns',
            'owner': 'system',
            'tags': ['volume', 'ratio', 'phase5b'],
            'validation_rules': {'not_null': True, 'positive': True}
        },
        {
            'name': 'atr_percentage',
            'version': '1.0.0',
            'description': 'Average True Range as percentage of price',
            'schema': {'type': 'number', 'minimum': 0, 'required': True},
            'data_type': 'numeric',
            'source_table': 'candlestick_patterns',
            'computation_logic': 'SELECT atr / close * 100 FROM candlestick_patterns',
            'owner': 'system',
            'tags': ['volatility', 'atr', 'phase5b'],
            'validation_rules': {'not_null': True, 'positive': True}
        }
    ]
    
    for feature in initial_features:
        connection.execute(text("""
            INSERT INTO feature_definitions 
            (id, name, version, description, schema, data_type, source_table, computation_logic, owner, tags, validation_rules)
            VALUES 
            (:id, :name, :version, :description, :schema, :data_type, :source_table, :computation_logic, :owner, :tags, :validation_rules)
        """), {
            'id': uuid.uuid4(),
            'name': feature['name'],
            'version': feature['version'],
            'description': feature['description'],
            'schema': json.dumps(feature['schema']),
            'data_type': feature['data_type'],
            'source_table': feature['source_table'],
            'computation_logic': feature['computation_logic'],
            'owner': feature['owner'],
            'tags': json.dumps(feature['tags']),
            'validation_rules': json.dumps(feature['validation_rules'])
        })
    
    # Insert initial feature contract
    initial_contract = {
        'name': 'phase5b_ensemble_features',
        'version': '1.0.0',
        'description': 'Feature contract for Phase 5B ensemble models',
        'schema_contract': {
            'required_features': ['close_price', 'volume', 'btc_dominance', 'market_correlation', 'volume_ratio', 'atr_percentage'],
            'feature_types': {
                'close_price': 'numeric',
                'volume': 'numeric',
                'btc_dominance': 'numeric',
                'market_correlation': 'numeric',
                'volume_ratio': 'numeric',
                'atr_percentage': 'numeric'
            },
            'validation_rules': {
                'close_price': {'not_null': True, 'positive': True},
                'volume': {'not_null': True, 'positive': True},
                'btc_dominance': {'not_null': True, 'range': [0, 100]},
                'market_correlation': {'not_null': True, 'range': [-1, 1]},
                'volume_ratio': {'not_null': True, 'positive': True},
                'atr_percentage': {'not_null': True, 'positive': True}
            }
        },
        'validation_rules': {'all_required_present': True, 'no_null_values': True},
        'drift_thresholds': {
            'distribution_drift': 0.1,
            'schema_drift': 0.05,
            'missing_data_threshold': 0.01
        },
        'owner': 'system'
    }
    
    connection.execute(text("""
        INSERT INTO feature_contracts 
        (id, name, version, description, schema_contract, validation_rules, drift_thresholds, owner)
        VALUES 
        (:id, :name, :version, :description, :schema_contract, :validation_rules, :drift_thresholds, :owner)
    """), {
        'id': uuid.uuid4(),
        'name': initial_contract['name'],
        'version': initial_contract['version'],
        'description': initial_contract['description'],
        'schema_contract': json.dumps(initial_contract['schema_contract']),
        'validation_rules': json.dumps(initial_contract['validation_rules']),
        'drift_thresholds': json.dumps(initial_contract['drift_thresholds']),
        'owner': initial_contract['owner']
    })
    
    logger.info("✅ Phase 5C: Feature Store migration completed successfully")

def downgrade():
    """Rollback Phase 5C feature store tables"""
    
    connection = op.get_bind()
    
    # Drop continuous aggregates first
    connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS feature_drift_hourly CASCADE"))
    connection.execute(text("DROP MATERIALIZED VIEW IF EXISTS feature_snapshots_hourly CASCADE"))
    
    # Drop tables
    op.drop_table('feature_dependencies')
    op.drop_table('feature_drift_logs')
    op.drop_table('pipeline_runs')
    op.drop_table('feature_contracts')
    op.drop_table('feature_snapshots')
    op.drop_table('feature_definitions')
    
    logger.info("✅ Phase 5C: Feature Store migration rolled back successfully")

async def seed_data():
    """Seed initial data for Phase 5C"""
    # This will be called after migration to populate initial data
    pass
