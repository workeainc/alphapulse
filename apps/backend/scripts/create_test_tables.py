#!/usr/bin/env python3
"""
Create AlphaPulse test tables directly using SQLAlchemy
"""

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///test_alphapulse.db")

# Create engine and base
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Signal(Base):
    """Trading signals table for storing generated signals"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    direction = Column(String(10), nullable=False)  # 'buy'/'sell'
    confidence = Column(Float, nullable=False)
    tp1 = Column(Float, nullable=True)
    tp2 = Column(Float, nullable=True)
    tp3 = Column(Float, nullable=True)
    tp4 = Column(Float, nullable=True)
    sl = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    outcome = Column(String(20), nullable=True, default='pending')  # 'win'/'loss'/'pending'

class Log(Base):
    """False positive logs for signal validation feedback"""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    volume_context = Column(JSON, nullable=True)
    trend_context = Column(JSON, nullable=True)
    outcome = Column(String(20), nullable=True)
    timestamp = Column(DateTime, nullable=False)

class Feedback(Base):
    """Signal feedback and outcomes"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=False)
    market_outcome = Column(Float, nullable=True)  # PnL
    notes = Column(Text, nullable=True)

def create_tables():
    """Create all test tables"""
    print("🔧 Creating AlphaPulse test tables...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create indexes
    Index('idx_signals_symbol_timeframe_timestamp', Signal.symbol, Signal.timeframe, Signal.timestamp)
    Index('idx_logs_timestamp', Log.timestamp)
    Index('idx_feedback_signal_id', Feedback.signal_id)
    
    print("✅ Test tables created successfully!")
    
    # Verify tables were created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"📋 Tables in database: {tables}")

def drop_tables():
    """Drop all test tables"""
    print("🗑️  Dropping AlphaPulse test tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ Test tables dropped successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create or drop AlphaPulse test tables")
    parser.add_argument("--drop", action="store_true", help="Drop tables instead of creating them")
    
    args = parser.parse_args()
    
    if args.drop:
        drop_tables()
    else:
        create_tables()
