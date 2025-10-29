"""
Migration: Create ONNX optimization tracking tables for Priority 1
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006_create_onnx_optimization_tables'
down_revision = '005_create_accuracy_benchmarks_table'
branch_labels = None
depends_on = None


def upgrade():
    """Create ONNX optimization tracking tables"""
    
    # Create onnx_optimization_metrics table
    op.create_table(
        'onnx_optimization_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('optimization_type', sa.String(50), nullable=False),  # 'mixed_precision', 'quantization', 'combined'
        sa.Column('original_model_size_mb', sa.Float(), nullable=True),
        sa.Column('optimized_model_size_mb', sa.Float(), nullable=True),
        sa.Column('size_reduction_percent', sa.Float(), nullable=True),
        
        # Performance metrics
        sa.Column('original_inference_time_ms', sa.Float(), nullable=True),
        sa.Column('optimized_inference_time_ms', sa.Float(), nullable=True),
        sa.Column('speedup_factor', sa.Float(), nullable=True),
        sa.Column('memory_usage_reduction_percent', sa.Float(), nullable=True),
        
        # Optimization details
        sa.Column('optimization_level', sa.String(20), nullable=False, default='balanced'),  # 'speed', 'balanced', 'accuracy'
        sa.Column('optimization_time_seconds', sa.Float(), nullable=True),
        sa.Column('optimization_success', sa.Boolean(), nullable=False, default=True),
        sa.Column('fallback_used', sa.Boolean(), nullable=False, default=False),
        
        # Hardware information
        sa.Column('execution_provider', sa.String(50), nullable=True),  # 'CPUExecutionProvider', 'CUDAExecutionProvider'
        sa.Column('hardware_capabilities', sa.JSON(), nullable=True),
        
        # Metadata
        sa.Column('optimization_date', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('onnx_version', sa.String(20), nullable=True),
        sa.Column('runtime_version', sa.String(20), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create onnx_model_registry table
    op.create_table(
        'onnx_model_registry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False, unique=True),
        sa.Column('model_type', sa.String(50), nullable=False),  # 'sklearn', 'xgboost', 'lightgbm', 'catboost'
        sa.Column('original_model_path', sa.String(500), nullable=True),
        sa.Column('onnx_model_path', sa.String(500), nullable=True),
        sa.Column('quantized_model_path', sa.String(500), nullable=True),
        
        # Model metadata
        sa.Column('input_shape', sa.JSON(), nullable=True),
        sa.Column('output_shape', sa.JSON(), nullable=True),
        sa.Column('model_parameters', sa.JSON(), nullable=True),
        sa.Column('feature_names', sa.JSON(), nullable=True),
        
        # Optimization status
        sa.Column('is_optimized', sa.Boolean(), nullable=False, default=False),
        sa.Column('optimization_date', sa.DateTime(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, default=0),
        
        # Performance tracking
        sa.Column('avg_inference_time_ms', sa.Float(), nullable=True),
        sa.Column('total_inferences', sa.Integer(), nullable=False, default=0),
        sa.Column('error_count', sa.Integer(), nullable=False, default=0),
        
        # Status
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=sa.func.now(), onupdate=sa.func.now()),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create onnx_performance_logs table for detailed performance tracking
    op.create_table(
        'onnx_performance_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('optimization_type', sa.String(50), nullable=True),
        sa.Column('batch_size', sa.Integer(), nullable=True),
        sa.Column('input_size', sa.Integer(), nullable=True),
        
        # Timing metrics
        sa.Column('preprocessing_time_ms', sa.Float(), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=False),
        sa.Column('postprocessing_time_ms', sa.Float(), nullable=True),
        sa.Column('total_time_ms', sa.Float(), nullable=False),
        
        # Resource usage
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('cpu_usage_percent', sa.Float(), nullable=True),
        sa.Column('gpu_usage_percent', sa.Float(), nullable=True),
        
        # Execution context
        sa.Column('execution_provider', sa.String(50), nullable=True),
        sa.Column('session_options', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False, default=True),
        
        # Metadata
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('request_id', sa.String(100), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('ix_onnx_optimization_metrics_model_name', 'onnx_optimization_metrics', ['model_name'])
    op.create_index('ix_onnx_optimization_metrics_optimization_type', 'onnx_optimization_metrics', ['optimization_type'])
    op.create_index('ix_onnx_optimization_metrics_optimization_date', 'onnx_optimization_metrics', ['optimization_date'])
    op.create_index('ix_onnx_optimization_metrics_speedup_factor', 'onnx_optimization_metrics', ['speedup_factor'])
    
    op.create_index('ix_onnx_model_registry_model_name', 'onnx_model_registry', ['model_name'])
    op.create_index('ix_onnx_model_registry_model_type', 'onnx_model_registry', ['model_type'])
    op.create_index('ix_onnx_model_registry_is_optimized', 'onnx_model_registry', ['is_optimized'])
    op.create_index('ix_onnx_model_registry_last_used', 'onnx_model_registry', ['last_used'])
    
    op.create_index('ix_onnx_performance_logs_model_name', 'onnx_performance_logs', ['model_name'])
    op.create_index('ix_onnx_performance_logs_timestamp', 'onnx_performance_logs', ['timestamp'])
    op.create_index('ix_onnx_performance_logs_inference_time', 'onnx_performance_logs', ['inference_time_ms'])
    op.create_index('ix_onnx_performance_logs_success', 'onnx_performance_logs', ['success'])
    
    # Create composite indexes for common queries
    op.create_index('ix_onnx_optimization_metrics_model_type', 'onnx_optimization_metrics', ['model_name', 'optimization_type'])
    op.create_index('ix_onnx_optimization_metrics_date_type', 'onnx_optimization_metrics', ['optimization_date', 'optimization_type'])
    op.create_index('ix_onnx_performance_logs_model_timestamp', 'onnx_performance_logs', ['model_name', 'timestamp'])
    
    # Convert to TimescaleDB hypertables for time-series optimization
    op.execute("SELECT create_hypertable('onnx_optimization_metrics', 'optimization_date', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('onnx_performance_logs', 'timestamp', if_not_exists => TRUE)")
    
    # Set compression policies
    op.execute("""
        ALTER TABLE onnx_optimization_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'model_name,optimization_type',
            timescaledb.compress_orderby = 'optimization_date DESC'
        )
    """)
    
    op.execute("""
        ALTER TABLE onnx_performance_logs SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'model_name,optimization_type',
            timescaledb.compress_orderby = 'timestamp DESC'
        )
    """)
    
    # Add compression policies (compress chunks older than 7 days)
    op.execute("SELECT add_compression_policy('onnx_optimization_metrics', INTERVAL '7 days')")
    op.execute("SELECT add_compression_policy('onnx_performance_logs', INTERVAL '7 days')")
    
    # Add retention policies (keep data for 1 year)
    op.execute("SELECT add_retention_policy('onnx_optimization_metrics', INTERVAL '1 year')")
    op.execute("SELECT add_retention_policy('onnx_performance_logs', INTERVAL '1 year')")


def downgrade():
    """Drop ONNX optimization tracking tables"""
    
    # Drop indexes
    op.drop_index('ix_onnx_performance_logs_model_timestamp', 'onnx_performance_logs')
    op.drop_index('ix_onnx_optimization_metrics_date_type', 'onnx_optimization_metrics')
    op.drop_index('ix_onnx_optimization_metrics_model_type', 'onnx_optimization_metrics')
    op.drop_index('ix_onnx_performance_logs_success', 'onnx_performance_logs')
    op.drop_index('ix_onnx_performance_logs_inference_time', 'onnx_performance_logs')
    op.drop_index('ix_onnx_performance_logs_timestamp', 'onnx_performance_logs')
    op.drop_index('ix_onnx_performance_logs_model_name', 'onnx_performance_logs')
    op.drop_index('ix_onnx_model_registry_last_used', 'onnx_model_registry')
    op.drop_index('ix_onnx_model_registry_is_optimized', 'onnx_model_registry')
    op.drop_index('ix_onnx_model_registry_model_type', 'onnx_model_registry')
    op.drop_index('ix_onnx_model_registry_model_name', 'onnx_model_registry')
    op.drop_index('ix_onnx_optimization_metrics_speedup_factor', 'onnx_optimization_metrics')
    op.drop_index('ix_onnx_optimization_metrics_optimization_date', 'onnx_optimization_metrics')
    op.drop_index('ix_onnx_optimization_metrics_optimization_type', 'onnx_optimization_metrics')
    op.drop_index('ix_onnx_optimization_metrics_model_name', 'onnx_optimization_metrics')
    
    # Drop tables
    op.drop_table('onnx_performance_logs')
    op.drop_table('onnx_model_registry')
    op.drop_table('onnx_optimization_metrics')
