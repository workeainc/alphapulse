#!/usr/bin/env python3
"""
DEPRECATED: Enhanced Database Models for AlphaPulse Testing

⚠️ WARNING: This file is for TESTING ONLY with SQLite.
For production, use backend/database/models.py instead.

This file was created for demo/testing purposes and uses SQLite
instead of PostgreSQL/TimescaleDB. It should not be used in production.

Matches the demo signal output format exactly.
"""

import warnings
warnings.warn(
    "models_enhanced.py is deprecated for production. Use models.py instead. "
    "This file is for testing only with SQLite.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from typing import Optional
import json

# Database URL configuration
def get_database_url():
    """Get database URL from environment or use default"""
    return os.getenv("DATABASE_URL", "sqlite:///test_alphapulse.db")

def get_test_database_url():
    """Get test database URL from environment or use default"""
    return os.getenv("TEST_DB_URL", "sqlite:///test_alphapulse_test.db")

# Create engine and base
engine = create_engine(get_database_url())
test_engine = create_engine(get_test_database_url())
Base = declarative_base()

class Signal(Base):
    """Enhanced trading signals table matching demo output format exactly"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(20), unique=True, nullable=False, index=True)  # e.g., 'ALPHA_000001'
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    direction = Column(String(10), nullable=False)  # 'buy'/'sell'
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    tp1 = Column(Float, nullable=True)
    tp2 = Column(Float, nullable=True)
    tp3 = Column(Float, nullable=True)
    tp4 = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    pattern_type = Column(String(50), nullable=True)  # e.g., 'rsi_divergence'
    volume_confirmation = Column(Boolean, nullable=True)
    trend_alignment = Column(Boolean, nullable=True)
    market_regime = Column(String(20), nullable=True)  # 'trending', 'choppy', 'volatile'
    indicators = Column(JSON, nullable=True)  # {rsi, macd, bb_position, adx, atr}
    validation_metrics = Column(JSON, nullable=True)  # {volume_ratio, price_momentum, volatility_score}
    metadata = Column(JSON, nullable=True)  # {processing_latency_ms, signal_strength, filtered, source}
    outcome = Column(String(20), nullable=True, default='pending')  # 'win'/'loss'/'pending'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    logs = relationship("Log", back_populates="signal")
    feedback = relationship("Feedback", back_populates="signal")
    
    def __repr__(self):
        return f"<Signal(signal_id='{self.signal_id}', symbol='{self.symbol}', direction='{self.direction}', confidence={self.confidence})>"

class Log(Base):
    """False positive logs for signal validation feedback"""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(20), ForeignKey('signals.signal_id'), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    volume_context = Column(JSON, nullable=True)  # {volume_ratio, volume_sma}
    trend_context = Column(JSON, nullable=True)  # {adx, trend_direction}
    outcome = Column(String(20), nullable=True)  # 'false_positive', 'true_positive'
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    signal = relationship("Signal", back_populates="logs")
    
    def __repr__(self):
        return f"<Log(signal_id='{self.signal_id}', pattern_type='{self.pattern_type}', confidence={self.confidence_score})>"

class Feedback(Base):
    """Signal feedback and outcomes"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(20), ForeignKey('signals.signal_id'), nullable=False, index=True)
    market_outcome = Column(Float, nullable=True)  # PnL
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    signal = relationship("Signal", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(signal_id='{self.signal_id}', market_outcome={self.market_outcome})>"

class PerformanceMetrics(Base):
    """Performance metrics for testing and monitoring"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    test_name = Column(String(100), nullable=False, index=True)
    test_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Latency metrics
    avg_latency_ms = Column(Float, nullable=False)
    max_latency_ms = Column(Float, nullable=False)
    min_latency_ms = Column(Float, nullable=False)
    p95_latency_ms = Column(Float, nullable=True)
    p99_latency_ms = Column(Float, nullable=True)
    
    # Accuracy metrics
    win_rate = Column(Float, nullable=True)
    total_signals = Column(Integer, nullable=False)
    winning_signals = Column(Integer, nullable=True)
    losing_signals = Column(Integer, nullable=True)
    filtered_signals = Column(Integer, nullable=True)
    filter_rate = Column(Float, nullable=True)
    
    # Throughput metrics
    throughput_signals_per_sec = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    
    # Test configuration
    test_config = Column(JSON, nullable=True)
    test_results = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<PerformanceMetrics(test='{self.test_name}', avg_latency={self.avg_latency_ms}ms, win_rate={self.win_rate})>"

def create_indexes():
    """Create indexes for better query performance"""
    # Composite indexes for common queries
    Index('idx_signals_symbol_timeframe_timestamp', Signal.symbol, Signal.timeframe, Signal.timestamp)
    Index('idx_signals_confidence_outcome', Signal.confidence, Signal.outcome)
    Index('idx_logs_timestamp', Log.timestamp)
    Index('idx_feedback_signal_id', Feedback.signal_id)
    Index('idx_performance_metrics_test_timestamp', PerformanceMetrics.test_name, PerformanceMetrics.test_timestamp)

def create_tables(engine_instance=None):
    """Create all database tables"""
    target_engine = engine_instance or engine
    Base.metadata.create_all(bind=target_engine)
    create_indexes()
    print("✅ Enhanced database tables created successfully")

def drop_tables(engine_instance=None):
    """Drop all database tables"""
    target_engine = engine_instance or engine
    Base.metadata.drop_all(bind=target_engine)
    print("⚠️  Enhanced database tables dropped")

# Database session management
def get_session(engine_instance=None):
    """Get database session"""
    target_engine = engine_instance or engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=target_engine)
    return SessionLocal()

def get_test_session():
    """Get test database session"""
    return get_session(test_engine)

# Export all models
__all__ = [
    'Base',
    'Signal',
    'Log', 
    'Feedback',
    'PerformanceMetrics',
    'create_tables',
    'drop_tables',
    'get_session',
    'get_test_session',
    'engine',
    'test_engine'
]
