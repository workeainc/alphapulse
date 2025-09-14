"""
Simple Database Initialization Script for AlphaPlus
Creates tables and adds sample data
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Direct import from models.py
sys.path.append(str(backend_path / 'database'))
from models import Base, Trade, PerformanceMetrics, Signal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

def init_database():
    """Initialize database with tables and sample data"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',  # Connect from host
        'port': 5432,
        'database': 'alphapulse',
        'username': 'alpha_emon',
        'password': 'Emon_@17711'
    }
    
    # Create connection string
    connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    try:
        # Create engine and tables
        engine = create_engine(connection_string)
        Base.metadata.create_all(engine)
        print("✅ Database tables created successfully")
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Add sample data
        add_sample_data(session)
        
        session.close()
        print("✅ Sample data added successfully")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    return True

def add_sample_data(session):
    """Add sample data to the database"""
    
    # Add sample trades
    sample_trades = [
        {
            'signal_id': 'SIG001',
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry_price': 45000.0,
            'exit_price': 46500.0,
            'quantity': 0.1,
            'leverage': 1,
            'pnl': 150.0,
            'pnl_percentage': 3.33,
            'strategy_name': 'mean_reversion',
            'status': 'closed',
            'entry_time': datetime.utcnow() - timedelta(days=2),
            'exit_time': datetime.utcnow() - timedelta(days=1)
        },
        {
            'signal_id': 'SIG002',
            'symbol': 'ETH/USDT',
            'side': 'short',
            'entry_price': 2800.0,
            'exit_price': 2750.0,
            'quantity': 1.0,
            'leverage': 1,
            'pnl': 50.0,
            'pnl_percentage': 1.79,
            'strategy_name': 'momentum',
            'status': 'closed',
            'entry_time': datetime.utcnow() - timedelta(days=3),
            'exit_time': datetime.utcnow() - timedelta(days=2)
        },
        {
            'signal_id': 'SIG003',
            'symbol': 'BTC/USDT',
            'side': 'long',
            'entry_price': 46000.0,
            'exit_price': None,
            'quantity': 0.05,
            'leverage': 1,
            'pnl': None,
            'pnl_percentage': None,
            'strategy_name': 'trend_following',
            'status': 'open',
            'entry_time': datetime.utcnow() - timedelta(hours=6),
            'exit_time': None
        }
    ]
    
    for trade_data in sample_trades:
        trade = Trade(**trade_data)
        session.add(trade)
    
    # Add sample performance metrics
    performance = PerformanceMetrics(
        var_95=2.5,
        max_drawdown=8.2,
        sharpe_ratio=1.85,
        sortino_ratio=2.1,
        timestamp=datetime.utcnow()
    )
    session.add(performance)
    
    # Add sample signals
    sample_signals = [
        {
            'signal_id': 'SIG001',
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'pattern_name': 'double_bottom',
            'confidence': 0.85,
            'direction': 'long',
            'entry_price': 45000.0,
            'stop_loss': 44000.0,
            'take_profit': 47000.0,
            'timestamp': datetime.utcnow() - timedelta(days=2)
        },
        {
            'signal_id': 'SIG002',
            'symbol': 'ETH/USDT',
            'timeframe': '4h',
            'pattern_name': 'head_shoulders',
            'confidence': 0.78,
            'direction': 'short',
            'entry_price': 2800.0,
            'stop_loss': 2850.0,
            'take_profit': 2700.0,
            'timestamp': datetime.utcnow() - timedelta(days=3)
        }
    ]
    
    for signal_data in sample_signals:
        signal = Signal(**signal_data)
        session.add(signal)
    
    # Commit all changes
    session.commit()
    print(f"✅ Added {len(sample_trades)} sample trades, 1 performance metric, and {len(sample_signals)} signals")

if __name__ == "__main__":
    print("🚀 Initializing AlphaPlus Database...")
    success = init_database()
    
    if success:
        print("🎉 Database initialization completed successfully!")
    else:
        print("💥 Database initialization failed!")
        sys.exit(1)
