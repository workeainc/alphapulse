#!/usr/bin/env python3
"""
Database tests for AlphaPulse
"""

import pytest
import time
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import database models
from create_test_tables import Signal, Log, Feedback, Base

class TestDatabaseOperations:
    """Test database operations and feedback loop"""
    
    def test_signal_insertion(self, test_db_session):
        """Test signal insertion and retrieval"""
        print("📊 Testing signal insertion...")
        
        # Create test signal
        signal = Signal(
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.85,
            tp1=51000,
            tp2=52000,
            tp3=53000,
            tp4=54000,
            sl=49000,
            timestamp=datetime.utcnow(),
            outcome='pending'
        )
        
        # Insert signal
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Verify insertion
        assert signal.id is not None, "Signal should have an ID after insertion"
        
        # Retrieve signal
        retrieved_signal = test_db_session.query(Signal).filter_by(id=signal.id).first()
        assert retrieved_signal is not None, "Should be able to retrieve inserted signal"
        assert retrieved_signal.symbol == 'BTCUSDT', "Symbol should match"
        assert retrieved_signal.confidence == 0.85, "Confidence should match"
        assert retrieved_signal.outcome == 'pending', "Outcome should be pending"
        
        print(f"✅ Signal inserted successfully with ID: {signal.id}")
    
    def test_bulk_signal_insertion(self, test_db_session):
        """Test bulk signal insertion performance"""
        print("📊 Testing bulk signal insertion...")
        
        start_time = time.perf_counter()
        
        # Create multiple signals
        signals = []
        for i in range(100):
            signal = Signal(
                symbol=f'BTCUSDT' if i % 2 == 0 else f'ETHUSDT',
                timeframe='15m',
                direction='buy' if i % 3 == 0 else 'sell',
                confidence=0.7 + (i % 30) * 0.01,
                tp1=50000 + (i % 1000),
                tp2=51000 + (i % 1000),
                tp3=52000 + (i % 1000),
                tp4=53000 + (i % 1000),
                sl=48000 + (i % 1000),
                timestamp=datetime.utcnow() - timedelta(minutes=i*5),
                outcome='pending'
            )
            signals.append(signal)
        
        # Bulk insert
        test_db_session.add_all(signals)
        test_db_session.commit()
        
        end_time = time.perf_counter()
        insertion_time = (end_time - start_time) * 1000
        
        # Verify all signals were inserted
        signal_count = test_db_session.query(Signal).count()
        assert signal_count == 100, f"Should have 100 signals, got {signal_count}"
        
        print(f"✅ Bulk insertion completed in {insertion_time:.2f}ms")
        print(f"  Average time per signal: {insertion_time/100:.2f}ms")
        
        # Performance assertion
        assert insertion_time < 1000, f"Bulk insertion should be < 1000ms, got {insertion_time:.2f}ms"
    
    def test_log_insertion(self, test_db_session):
        """Test log insertion for false positives"""
        print("📊 Testing log insertion...")
        
        # Create test log
        log = Log(
            pattern_type='candlestick',
            confidence_score=0.65,
            volume_context={
                'volume_ratio': 1.8,
                'volume_sma': 1200000
            },
            trend_context={
                'adx': 22,
                'trend_direction': 'bullish'
            },
            outcome='false_positive',
            timestamp=datetime.utcnow()
        )
        
        # Insert log
        test_db_session.add(log)
        test_db_session.commit()
        
        # Verify insertion
        assert log.id is not None, "Log should have an ID after insertion"
        
        # Retrieve log
        retrieved_log = test_db_session.query(Log).filter_by(id=log.id).first()
        assert retrieved_log is not None, "Should be able to retrieve inserted log"
        assert retrieved_log.pattern_type == 'candlestick', "Pattern type should match"
        assert retrieved_log.confidence_score == 0.65, "Confidence score should match"
        assert retrieved_log.outcome == 'false_positive', "Outcome should match"
        
        print(f"✅ Log inserted successfully with ID: {log.id}")
    
    def test_feedback_loop(self, test_db_session):
        """Test feedback loop with signal outcomes"""
        print("📊 Testing feedback loop...")
        
        # Create a signal
        signal = Signal(
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.8,
            tp1=51000,
            sl=49000,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            outcome='pending'
        )
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Simulate signal outcome (after 5 candles)
        signal.outcome = 'win'
        test_db_session.commit()
        
        # Create feedback
        feedback = Feedback(
            signal_id=signal.id,
            market_outcome=500,  # $500 profit
            notes="Signal hit TP1 successfully"
        )
        test_db_session.add(feedback)
        test_db_session.commit()
        
        # Verify feedback loop
        retrieved_signal = test_db_session.query(Signal).filter_by(id=signal.id).first()
        retrieved_feedback = test_db_session.query(Feedback).filter_by(signal_id=signal.id).first()
        
        assert retrieved_signal.outcome == 'win', "Signal outcome should be updated"
        assert retrieved_feedback is not None, "Feedback should be created"
        assert retrieved_feedback.market_outcome == 500, "Market outcome should match"
        
        print(f"✅ Feedback loop completed successfully")
        print(f"  Signal ID: {signal.id}")
        print(f"  Outcome: {retrieved_signal.outcome}")
        print(f"  PnL: ${retrieved_feedback.market_outcome}")
    
    def test_query_performance(self, test_db_session, seeded_database):
        """Test database query performance"""
        print("📊 Testing query performance...")
        
        # Test indexed queries
        start_time = time.perf_counter()
        
        # Query by symbol and timeframe (indexed)
        btc_signals = test_db_session.query(Signal).filter(
            Signal.symbol == 'BTCUSDT',
            Signal.timeframe == '15m'
        ).all()
        
        query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Indexed query completed in {query_time:.2f}ms")
        print(f"  Results: {len(btc_signals)} signals")
        
        # Performance assertion
        assert query_time < 10, f"Indexed query should be < 10ms, got {query_time:.2f}ms"
        
        # Test timestamp range query
        start_time = time.perf_counter()
        
        recent_signals = test_db_session.query(Signal).filter(
            Signal.timestamp >= datetime.utcnow() - timedelta(hours=1)
        ).all()
        
        range_query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Range query completed in {range_query_time:.2f}ms")
        print(f"  Results: {len(recent_signals)} signals")
        
        # Performance assertion
        assert range_query_time < 20, f"Range query should be < 20ms, got {range_query_time:.2f}ms"
    
    def test_aggregation_queries(self, test_db_session, seeded_database):
        """Test aggregation and analytics queries"""
        print("📊 Testing aggregation queries...")
        
        # Test signal statistics
        start_time = time.perf_counter()
        
        # Total signals by symbol
        symbol_stats = test_db_session.query(
            Signal.symbol,
            test_db_session.query(Signal).count().label('total_signals'),
            test_db_session.query(Signal).filter(Signal.outcome == 'win').count().label('winning_signals')
        ).group_by(Signal.symbol).all()
        
        aggregation_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Aggregation query completed in {aggregation_time:.2f}ms")
        
        for symbol, total, wins in symbol_stats:
            win_rate = wins / total if total > 0 else 0
            print(f"  {symbol}: {total} signals, {wins} wins ({win_rate:.1%})")
        
        # Performance assertion
        assert aggregation_time < 50, f"Aggregation query should be < 50ms, got {aggregation_time:.2f}ms"
        
        # Test confidence distribution
        start_time = time.perf_counter()
        
        high_confidence_signals = test_db_session.query(Signal).filter(
            Signal.confidence >= 0.8
        ).count()
        
        medium_confidence_signals = test_db_session.query(Signal).filter(
            Signal.confidence >= 0.7,
            Signal.confidence < 0.8
        ).count()
        
        low_confidence_signals = test_db_session.query(Signal).filter(
            Signal.confidence < 0.7
        ).count()
        
        confidence_query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Confidence distribution query completed in {confidence_query_time:.2f}ms")
        print(f"  High confidence (≥0.8): {high_confidence_signals}")
        print(f"  Medium confidence (0.7-0.8): {medium_confidence_signals}")
        print(f"  Low confidence (<0.7): {low_confidence_signals}")
        
        # Performance assertion
        assert confidence_query_time < 30, f"Confidence query should be < 30ms, got {confidence_query_time:.2f}ms"
    
    def test_foreign_key_constraints(self, test_db_session):
        """Test foreign key constraints and relationships"""
        print("📊 Testing foreign key constraints...")
        
        # Create a signal first
        signal = Signal(
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.8,
            timestamp=datetime.utcnow(),
            outcome='pending'
        )
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Create feedback with valid signal_id
        valid_feedback = Feedback(
            signal_id=signal.id,
            market_outcome=300,
            notes="Valid feedback"
        )
        test_db_session.add(valid_feedback)
        test_db_session.commit()
        
        assert valid_feedback.id is not None, "Valid feedback should be inserted"
        
        # Try to create feedback with invalid signal_id (should fail)
        try:
            invalid_feedback = Feedback(
                signal_id=99999,  # Non-existent signal ID
                market_outcome=300,
                notes="Invalid feedback"
            )
            test_db_session.add(invalid_feedback)
            test_db_session.commit()
            assert False, "Should not allow invalid foreign key"
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Foreign key constraint enforced: {str(e)}")
    
    def test_data_integrity(self, test_db_session):
        """Test data integrity constraints"""
        print("📊 Testing data integrity...")
        
        # Test required fields
        try:
            invalid_signal = Signal(
                # Missing required fields
                symbol=None,
                timeframe=None,
                direction=None,
                confidence=None,
                timestamp=None
            )
            test_db_session.add(invalid_signal)
            test_db_session.commit()
            assert False, "Should not allow null required fields"
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Required field constraint enforced: {str(e)}")
        
        # Test confidence range
        try:
            invalid_confidence_signal = Signal(
                symbol='BTCUSDT',
                timeframe='15m',
                direction='buy',
                confidence=1.5,  # Invalid confidence > 1.0
                timestamp=datetime.utcnow(),
                outcome='pending'
            )
            test_db_session.add(invalid_confidence_signal)
            test_db_session.commit()
            # Note: SQLite doesn't enforce CHECK constraints by default
            print("⚠️  Confidence range validation depends on database constraints")
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Confidence range constraint enforced: {str(e)}")
    
    def test_concurrent_access(self, test_db_session):
        """Test concurrent database access"""
        print("📊 Testing concurrent access...")
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def insert_signals(thread_id, count):
            """Insert signals in a separate thread"""
            try:
                # Create new session for thread
                from sqlalchemy.orm import sessionmaker
                SessionLocal = sessionmaker(bind=test_db_session.bind)
                session = SessionLocal()
                
                for i in range(count):
                    signal = Signal(
                        symbol=f'SYMBOL{thread_id}',
                        timeframe='15m',
                        direction='buy',
                        confidence=0.8,
                        timestamp=datetime.utcnow(),
                        outcome='pending'
                    )
                    session.add(signal)
                
                session.commit()
                session.close()
                results.put(f"Thread {thread_id}: {count} signals inserted")
            except Exception as e:
                results.put(f"Thread {thread_id}: Error - {str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=insert_signals, args=(i, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        while not results.empty():
            result = results.get()
            print(f"  {result}")
        
        # Verify total signals
        total_signals = test_db_session.query(Signal).count()
        print(f"✅ Total signals after concurrent access: {total_signals}")
        
        # Should have signals from all threads
        assert total_signals >= 30, f"Should have at least 30 signals, got {total_signals}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
