#!/usr/bin/env python3
"""
Simple database test for AlphaPulse
"""
import sys
import os
import pytest
import sqlite3
from datetime import datetime, timezone

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

class TestDatabaseSimple:
    """Simple database tests"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.db_path = "test_simple.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create simple test tables
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_signals (
                id INTEGER PRIMARY KEY,
                signal_id TEXT UNIQUE,
                symbol TEXT,
                timestamp DATETIME,
                price REAL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_logs (
                id INTEGER PRIMARY KEY,
                signal_id TEXT,
                message TEXT,
                timestamp DATETIME,
                FOREIGN KEY (signal_id) REFERENCES test_signals (signal_id)
            )
        """)
        
        self.conn.commit()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def test_database_connection(self):
        """Test database connection"""
        assert self.conn is not None
        assert self.cursor is not None
    
    def test_table_creation(self):
        """Test table creation"""
        # Check if tables exist
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        
        assert 'test_signals' in tables
        assert 'test_logs' in tables
    
    def test_signal_insertion(self):
        """Test signal insertion"""
        signal_id = "TEST_001"
        symbol = "BTCUSDT"
        price = 50000.0
        timestamp = datetime.now(timezone.utc)
        
        # Insert signal
        self.cursor.execute("""
            INSERT INTO test_signals (signal_id, symbol, timestamp, price)
            VALUES (?, ?, ?, ?)
        """, (signal_id, symbol, timestamp, price))
        
        self.conn.commit()
        
        # Verify insertion
        self.cursor.execute("SELECT * FROM test_signals WHERE signal_id = ?", (signal_id,))
        result = self.cursor.fetchone()
        
        assert result is not None
        assert result[1] == signal_id
        assert result[2] == symbol
        assert result[4] == price

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
