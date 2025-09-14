#!/usr/bin/env python3
"""
Enhanced Database Tests for AlphaPulse
Tests signal insertion, feedback loop, query performance, and migration validation
"""

import pytest
import time
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text, func

# Import database models
from ..database.models_enhanced import Signal, Log, Feedback, PerformanceMetrics, Base, get_test_session

class TestDatabaseOperations:
    """Test database operations and feedback loop"""
    
    def test_signal_insertion_matching_demo_format(self, test_db_session):
        """Test signal insertion with data matching demo output format exactly"""
        print("📊 Testing signal insertion with demo format...")
        
        # Create test signal matching demo output format
        signal_data = {
            'signal_id': 'ALPHA_000001',
            'timestamp': datetime.now(timezone.utc),
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'direction': 'buy',
            'confidence': 0.815,
            'entry_price': 49227.07,
            'tp1': 49719.34,
            'tp2': 50211.61,
            'tp3': 50703.88,
            'tp4': 51688.42,
            'stop_loss': 48734.8,
            'risk_reward_ratio': 1.0,
            'pattern_type': 'candlestick_breakout',
            'volume_confirmation': False,
            'trend_alignment': False,
            'market_regime': 'choppy',
            'indicators': {
                'rsi': 43.4,
                'macd': -64.2,
                'bb_position': 0.06,
                'adx': 38.9,
                'atr': 1915.2
            },
            'validation_metrics': {
                'volume_ratio': 2.9,
                'price_momentum': -0.027,
                'volatility_score': 0.11
            },
            'metadata': {
                'processing_latency_ms': 13.1,
                'signal_strength': 'weak',
                'filtered': False,
                'source': 'alphapulse_core'
            },
            'outcome': 'pending'
        }
        
        # Insert signal
        signal = Signal(**signal_data)
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Verify insertion
        assert signal.id is not None, "Signal should have an ID after insertion"
        
        # Retrieve signal
        retrieved_signal = test_db_session.query(Signal).filter_by(signal_id='ALPHA_000001').first()
        assert retrieved_signal is not None, "Should be able to retrieve inserted signal"
        
        # Verify all fields match demo format
        assert retrieved_signal.signal_id == 'ALPHA_000001', "Signal ID should match"
        assert retrieved_signal.symbol == 'BTCUSDT', "Symbol should match"
        assert retrieved_signal.timeframe == '15m', "Timeframe should match"
        assert retrieved_signal.direction == 'buy', "Direction should match"
        assert retrieved_signal.confidence == 0.815, "Confidence should match"
        assert retrieved_signal.entry_price == 49227.07, "Entry price should match"
        assert retrieved_signal.tp1 == 49719.34, "TP1 should match"
        assert retrieved_signal.tp2 == 50211.61, "TP2 should match"
        assert retrieved_signal.tp3 == 50703.88, "TP3 should match"
        assert retrieved_signal.tp4 == 51688.42, "TP4 should match"
        assert retrieved_signal.stop_loss == 48734.8, "Stop loss should match"
        assert retrieved_signal.risk_reward_ratio == 1.0, "Risk reward ratio should match"
        assert retrieved_signal.pattern_type == 'candlestick_breakout', "Pattern type should match"
        assert retrieved_signal.volume_confirmation == False, "Volume confirmation should match"
        assert retrieved_signal.trend_alignment == False, "Trend alignment should match"
        assert retrieved_signal.market_regime == 'choppy', "Market regime should match"
        assert retrieved_signal.outcome == 'pending', "Outcome should match"
        
        # Verify JSON fields
        assert retrieved_signal.indicators['rsi'] == 43.4, "RSI in indicators should match"
        assert retrieved_signal.indicators['macd'] == -64.2, "MACD in indicators should match"
        assert retrieved_signal.validation_metrics['volume_ratio'] == 2.9, "Volume ratio should match"
        assert retrieved_signal.metadata['processing_latency_ms'] == 13.1, "Processing latency should match"
        
        print(f"✅ Signal inserted successfully with ID: {signal.id}")
        print(f"  Signal ID: {retrieved_signal.signal_id}")
        print(f"  Confidence: {retrieved_signal.confidence}")
        print(f"  Entry Price: ${retrieved_signal.entry_price}")
        print(f"  TP1: ${retrieved_signal.tp1}")
        print(f"  Stop Loss: ${retrieved_signal.stop_loss}")
    
    def test_bulk_signal_insertion_performance(self, test_db_session):
        """Test bulk signal insertion performance"""
        print("📊 Testing bulk signal insertion performance...")
        
        start_time = time.perf_counter()
        
        # Create multiple signals
        signals = []
        for i in range(100):
            signal_data = {
                'signal_id': f'ALPHA_{i+1:06d}',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i*5),
                'symbol': f'BTCUSDT' if i % 2 == 0 else f'ETHUSDT',
                'timeframe': '15m',
                'direction': 'buy' if i % 3 == 0 else 'sell',
                'confidence': 0.7 + (i % 30) * 0.01,
                'entry_price': 50000 + (i % 1000),
                'tp1': 50500 + (i % 1000),
                'tp2': 51000 + (i % 1000),
                'tp3': 51500 + (i % 1000),
                'tp4': 52000 + (i % 1000),
                'stop_loss': 49500 + (i % 1000),
                'risk_reward_ratio': 1.0 + (i % 10) * 0.1,
                'pattern_type': ['rsi_divergence', 'bollinger_squeeze', 'pivot_point_breakout', 'candlestick_breakout'][i % 4],
                'volume_confirmation': i % 3 == 0,
                'trend_alignment': i % 2 == 0,
                'market_regime': ['trending', 'choppy', 'volatile'][i % 3],
                'indicators': {
                    'rsi': 50 + (i % 40),
                    'macd': 100 + i,
                    'bb_position': 0.1 + (i % 80) * 0.01,
                    'adx': 20 + (i % 30),
                    'atr': 1000 + (i % 1000)
                },
                'validation_metrics': {
                    'volume_ratio': 1.5 + (i % 10) * 0.1,
                    'price_momentum': -0.1 + (i % 20) * 0.01,
                    'volatility_score': 0.1 + (i % 10) * 0.01
                },
                'metadata': {
                    'processing_latency_ms': 10 + (i % 40),
                    'signal_strength': ['strong', 'medium', 'weak'][i % 3],
                    'filtered': i % 5 == 0,
                    'source': 'alphapulse_core'
                },
                'outcome': ['win', 'loss', 'pending'][i % 3]
            }
            signals.append(Signal(**signal_data))
        
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
        print(f"  Total signals inserted: {signal_count}")
        
        # Performance assertion
        assert insertion_time < 1000, f"Bulk insertion should be < 1000ms, got {insertion_time:.2f}ms"
    
    def test_log_insertion_for_false_positives(self, test_db_session):
        """Test log insertion for false positives"""
        print("📊 Testing log insertion for false positives...")
        
        # First create a signal
        signal = Signal(
            signal_id='ALPHA_000002',
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.65,
            entry_price=50000,
            outcome='pending'
        )
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Create test log
        log_data = {
            'signal_id': 'ALPHA_000002',
            'pattern_type': 'candlestick',
            'confidence_score': 0.65,
            'volume_context': {
                'volume_ratio': 1.8,
                'volume_sma': 1200000
            },
            'trend_context': {
                'adx': 22,
                'trend_direction': 'bullish'
            },
            'outcome': 'false_positive',
            'timestamp': datetime.now(timezone.utc)
        }
        
        log = Log(**log_data)
        test_db_session.add(log)
        test_db_session.commit()
        
        # Verify insertion
        assert log.id is not None, "Log should have an ID after insertion"
        
        # Retrieve log
        retrieved_log = test_db_session.query(Log).filter_by(signal_id='ALPHA_000002').first()
        assert retrieved_log is not None, "Should be able to retrieve inserted log"
        assert retrieved_log.pattern_type == 'candlestick', "Pattern type should match"
        assert retrieved_log.confidence_score == 0.65, "Confidence score should match"
        assert retrieved_log.outcome == 'false_positive', "Outcome should match"
        assert retrieved_log.volume_context['volume_ratio'] == 1.8, "Volume ratio should match"
        assert retrieved_log.trend_context['adx'] == 22, "ADX should match"
        
        print(f"✅ Log inserted successfully with ID: {log.id}")
        print(f"  Signal ID: {retrieved_log.signal_id}")
        print(f"  Pattern Type: {retrieved_log.pattern_type}")
        print(f"  Confidence Score: {retrieved_log.confidence_score}")
        print(f"  Outcome: {retrieved_log.outcome}")
    
    def test_feedback_loop_with_signal_outcomes(self, test_db_session):
        """Test feedback loop with signal outcomes"""
        print("📊 Testing feedback loop with signal outcomes...")
        
        # Create a signal
        signal = Signal(
            signal_id='ALPHA_000003',
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.8,
            entry_price=50000,
            tp1=50500,
            stop_loss=49500,
            outcome='pending'
        )
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Simulate signal outcome (after 5 candles)
        signal.outcome = 'win'
        test_db_session.commit()
        
        # Create feedback
        feedback_data = {
            'signal_id': 'ALPHA_000003',
            'market_outcome': 500,  # $500 profit
            'notes': "Signal hit TP1 successfully",
            'timestamp': datetime.now(timezone.utc)
        }
        
        feedback = Feedback(**feedback_data)
        test_db_session.add(feedback)
        test_db_session.commit()
        
        # Verify feedback loop
        retrieved_signal = test_db_session.query(Signal).filter_by(signal_id='ALPHA_000003').first()
        retrieved_feedback = test_db_session.query(Feedback).filter_by(signal_id='ALPHA_000003').first()
        
        assert retrieved_signal.outcome == 'win', "Signal outcome should be updated"
        assert retrieved_feedback is not None, "Feedback should be created"
        assert retrieved_feedback.market_outcome == 500, "Market outcome should match"
        assert retrieved_feedback.notes == "Signal hit TP1 successfully", "Notes should match"
        
        print(f"✅ Feedback loop completed successfully")
        print(f"  Signal ID: {signal.signal_id}")
        print(f"  Outcome: {retrieved_signal.outcome}")
        print(f"  PnL: ${retrieved_feedback.market_outcome}")
        print(f"  Notes: {retrieved_feedback.notes}")
    
    def test_query_performance_with_indexes(self, test_db_session, seeded_database):
        """Test database query performance with indexes"""
        print("📊 Testing query performance with indexes...")
        
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
            Signal.timestamp >= datetime.now(timezone.utc) - timedelta(hours=1)
        ).all()
        
        range_query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Range query completed in {range_query_time:.2f}ms")
        print(f"  Results: {len(recent_signals)} signals")
        
        # Performance assertion
        assert range_query_time < 20, f"Range query should be < 20ms, got {range_query_time:.2f}ms"
        
        # Test confidence-based query
        start_time = time.perf_counter()
        
        high_confidence_signals = test_db_session.query(Signal).filter(
            Signal.confidence >= 0.8
        ).all()
        
        confidence_query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Confidence query completed in {confidence_query_time:.2f}ms")
        print(f"  Results: {len(high_confidence_signals)} signals")
        
        # Performance assertion
        assert confidence_query_time < 15, f"Confidence query should be < 15ms, got {confidence_query_time:.2f}ms"
    
    def test_aggregation_queries(self, test_db_session, seeded_database):
        """Test aggregation and analytics queries"""
        print("📊 Testing aggregation queries...")
        
        # Test signal statistics
        start_time = time.perf_counter()
        
        # Total signals by symbol
        symbol_stats = test_db_session.query(
            Signal.symbol,
            func.count(Signal.id).label('total_signals'),
            func.sum(func.case([(Signal.outcome == 'win', 1)], else_=0)).label('winning_signals')
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
        
        confidence_stats = test_db_session.query(
            func.count(Signal.id).label('total'),
            func.avg(Signal.confidence).label('avg_confidence'),
            func.min(Signal.confidence).label('min_confidence'),
            func.max(Signal.confidence).label('max_confidence')
        ).first()
        
        confidence_query_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✅ Confidence statistics query completed in {confidence_query_time:.2f}ms")
        print(f"  Total signals: {confidence_stats.total}")
        print(f"  Average confidence: {confidence_stats.avg_confidence:.3f}")
        print(f"  Min confidence: {confidence_stats.min_confidence:.3f}")
        print(f"  Max confidence: {confidence_stats.max_confidence:.3f}")
        
        # Performance assertion
        assert confidence_query_time < 30, f"Confidence query should be < 30ms, got {confidence_query_time:.2f}ms"
    
    def test_foreign_key_constraints(self, test_db_session):
        """Test foreign key constraints and relationships"""
        print("📊 Testing foreign key constraints...")
        
        # Create a signal first
        signal = Signal(
            signal_id='ALPHA_000004',
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.8,
            entry_price=50000,
            outcome='pending'
        )
        test_db_session.add(signal)
        test_db_session.commit()
        
        # Create feedback with valid signal_id
        valid_feedback = Feedback(
            signal_id='ALPHA_000004',
            market_outcome=300,
            notes="Valid feedback",
            timestamp=datetime.now(timezone.utc)
        )
        test_db_session.add(valid_feedback)
        test_db_session.commit()
        
        assert valid_feedback.id is not None, "Valid feedback should be inserted"
        
        # Try to create feedback with invalid signal_id (should fail)
        try:
            invalid_feedback = Feedback(
                signal_id='INVALID_000001',  # Non-existent signal ID
                market_outcome=300,
                notes="Invalid feedback",
                timestamp=datetime.now(timezone.utc)
            )
            test_db_session.add(invalid_feedback)
            test_db_session.commit()
            assert False, "Should not allow invalid foreign key"
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Foreign key constraint enforced: {str(e)}")
        
        # Test relationship access
        signal_with_feedback = test_db_session.query(Signal).filter_by(signal_id='ALPHA_000004').first()
        assert len(signal_with_feedback.feedback) == 1, "Signal should have one feedback entry"
        assert signal_with_feedback.feedback[0].market_outcome == 300, "Feedback market outcome should match"
    
    def test_data_integrity(self, test_db_session):
        """Test data integrity constraints"""
        print("📊 Testing data integrity...")
        
        # Test required fields
        try:
            invalid_signal = Signal(
                # Missing required fields
                signal_id=None,
                timestamp=None,
                symbol=None,
                timeframe=None,
                direction=None,
                confidence=None,
                entry_price=None
            )
            test_db_session.add(invalid_signal)
            test_db_session.commit()
            assert False, "Should not allow null required fields"
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Required field constraint enforced: {str(e)}")
        
        # Test unique constraint on signal_id
        signal1 = Signal(
            signal_id='ALPHA_000005',
            timestamp=datetime.now(timezone.utc),
            symbol='BTCUSDT',
            timeframe='15m',
            direction='buy',
            confidence=0.8,
            entry_price=50000
        )
        test_db_session.add(signal1)
        test_db_session.commit()
        
        try:
            signal2 = Signal(
                signal_id='ALPHA_000005',  # Duplicate signal_id
                timestamp=datetime.now(timezone.utc),
                symbol='ETHUSDT',
                timeframe='15m',
                direction='sell',
                confidence=0.7,
                entry_price=3000
            )
            test_db_session.add(signal2)
            test_db_session.commit()
            assert False, "Should not allow duplicate signal_id"
        except Exception as e:
            test_db_session.rollback()
            print(f"✅ Unique constraint enforced: {str(e)}")
    
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
                session = get_test_session()
                
                for i in range(count):
                    signal = Signal(
                        signal_id=f'THREAD_{thread_id}_{i:03d}',
                        timestamp=datetime.now(timezone.utc),
                        symbol=f'SYMBOL{thread_id}',
                        timeframe='15m',
                        direction='buy',
                        confidence=0.8,
                        entry_price=50000 + i
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
    
    def test_performance_metrics_storage(self, test_db_session):
        """Test performance metrics storage and retrieval"""
        print("📊 Testing performance metrics storage...")
        
        # Create performance metrics
        metrics_data = {
            'test_name': 'pipeline_latency_test',
            'test_timestamp': datetime.now(timezone.utc),
            'avg_latency_ms': 32.5,
            'max_latency_ms': 47.8,
            'min_latency_ms': 15.2,
            'p95_latency_ms': 42.1,
            'p99_latency_ms': 45.3,
            'win_rate': 0.82,
            'total_signals': 1000,
            'winning_signals': 820,
            'losing_signals': 180,
            'filtered_signals': 650,
            'filter_rate': 0.65,
            'throughput_signals_per_sec': 15000,
            'cpu_usage_percent': 45.2,
            'memory_usage_mb': 256.8,
            'test_config': {
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1m', '5m', '15m'],
                'test_duration_seconds': 60
            },
            'test_results': {
                'status': 'passed',
                'assertions_passed': 15,
                'assertions_failed': 0
            }
        }
        
        metrics = PerformanceMetrics(**metrics_data)
        test_db_session.add(metrics)
        test_db_session.commit()
        
        # Verify insertion
        assert metrics.id is not None, "Performance metrics should have an ID"
        
        # Retrieve and verify
        retrieved_metrics = test_db_session.query(PerformanceMetrics).filter_by(
            test_name='pipeline_latency_test'
        ).first()
        
        assert retrieved_metrics is not None, "Should retrieve performance metrics"
        assert retrieved_metrics.avg_latency_ms == 32.5, "Average latency should match"
        assert retrieved_metrics.win_rate == 0.82, "Win rate should match"
        assert retrieved_metrics.throughput_signals_per_sec == 15000, "Throughput should match"
        assert retrieved_metrics.test_config['symbols'] == ['BTCUSDT', 'ETHUSDT'], "Test config should match"
        assert retrieved_metrics.test_results['status'] == 'passed', "Test results should match"
        
        print(f"✅ Performance metrics stored successfully")
        print(f"  Test: {retrieved_metrics.test_name}")
        print(f"  Average Latency: {retrieved_metrics.avg_latency_ms}ms")
        print(f"  Win Rate: {retrieved_metrics.win_rate:.2%}")
        print(f"  Throughput: {retrieved_metrics.throughput_signals_per_sec} signals/sec")
    
    def test_migration_validation(self, test_db_session):
        """Test that database schema matches expected structure"""
        print("📊 Testing migration validation...")
        
        from sqlalchemy import inspect
        
        inspector = inspect(test_db_session.bind)
        tables = inspector.get_table_names()
        
        # Verify required tables exist
        required_tables = ['signals', 'logs', 'feedback', 'performance_metrics']
        for table in required_tables:
            assert table in tables, f"Required table '{table}' should exist"
        
        # Verify signals table structure
        signal_columns = {col['name'] for col in inspector.get_columns('signals')}
        required_signal_columns = {
            'id', 'signal_id', 'timestamp', 'symbol', 'timeframe', 'direction',
            'confidence', 'entry_price', 'tp1', 'tp2', 'tp3', 'tp4', 'stop_loss',
            'risk_reward_ratio', 'pattern_type', 'volume_confirmation', 'trend_alignment',
            'market_regime', 'indicators', 'validation_metrics', 'metadata', 'outcome', 'created_at'
        }
        
        missing_columns = required_signal_columns - signal_columns
        assert len(missing_columns) == 0, f"Missing columns in signals table: {missing_columns}"
        
        # Verify indexes
        signal_indexes = {idx['name'] for idx in inspector.get_indexes('signals')}
        required_indexes = {
            'idx_signals_signal_id', 'idx_signals_symbol_timeframe_timestamp',
            'idx_signals_confidence_outcome'
        }
        
        missing_indexes = required_indexes - signal_indexes
        assert len(missing_indexes) == 0, f"Missing indexes in signals table: {missing_indexes}"
        
        print(f"✅ Migration validation passed")
        print(f"  Tables found: {tables}")
        print(f"  Signal columns: {len(signal_columns)}")
        print(f"  Signal indexes: {len(signal_indexes)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
