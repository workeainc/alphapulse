#!/usr/bin/env python3
"""
Consolidated Integration Tests for AlphaPulse
Comprehensive testing suite merging all phase*_integration.py files
"""

import asyncio
import pytest
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import redis
import sqlalchemy
from sqlalchemy.orm import sessionmaker

# Import core components
from ..src.core.alphapulse_core import AlphaPulseCore, Signal
from ..src.core.market_regime_detector import MarketRegimeDetector, MarketRegime
from ..src.core.indicators_engine import IndicatorsEngine
from ..src.core.ml_signal_generator import MLSignalGenerator
from ..src.database.models import Signal as DBSignal, Log, Feedback, PerformanceMetrics
from ..src.database.connection import get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAlphaPulseIntegration:
    """Comprehensive integration tests for AlphaPulse system"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.timeframes = ['1m', '15m']
        self.test_data = self._generate_test_data()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=15)  # Test DB
        
        # Mock database session
        self.mock_session = Mock()
        self.mock_session.add = Mock()
        self.mock_session.commit = Mock()
        self.mock_session.close = Mock()
    
    def _generate_test_data(self) -> List[Dict]:
        """Generate test candlestick data"""
        data = []
        base_price = 45000.0
        
        for i in range(100):
            timestamp = datetime.now() - timedelta(minutes=i)
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            current_price = base_price * (1 + price_change)
            
            candle = {
                'timestamp': timestamp,
                'open': current_price * (1 + np.random.normal(0, 0.005)),
                'high': current_price * (1 + abs(np.random.normal(0, 0.01))),
                'low': current_price * (1 - abs(np.random.normal(0, 0.01))),
                'close': current_price,
                'volume': np.random.uniform(1000000, 5000000)
            }
            data.append(candle)
        
        return data[::-1]  # Reverse to get chronological order
    
    @pytest.mark.asyncio
    async def test_alphapulse_core_initialization(self):
        """Test AlphaPulse Core initialization"""
        alphapulse = AlphaPulseCore(
            symbols=self.symbols,
            timeframes=self.timeframes,
            redis_url="redis://localhost:6379/15",
            enable_regime_detection=True,
            enable_ml=True,
            target_latency_ms=50.0
        )
        
        assert alphapulse.symbols == self.symbols
        assert alphapulse.timeframes == self.timeframes
        assert alphapulse.target_latency_ms == 50.0
        assert alphapulse.enable_regime_detection is True
        assert alphapulse.enable_ml is True
        assert len(alphapulse.regime_detectors) == len(self.symbols) * len(self.timeframes)
        assert len(alphapulse.websocket_connections) == len(self.symbols)
        
        logger.info("✅ AlphaPulse Core initialization test passed")
    
    @pytest.mark.asyncio
    async def test_market_regime_detection(self):
        """Test market regime detection functionality"""
        detector = MarketRegimeDetector(
            symbol='BTC/USDT',
            timeframe='15m',
            redis_client=self.redis_client
        )
        
        # Test regime classification
        for candle in self.test_data[:20]:
            regime = await detector.update_regime(candle)
            assert regime in MarketRegime
            assert detector.current_regime in MarketRegime
            assert 0.0 <= detector.regime_confidence <= 1.0
        
        # Test regime persistence
        stored_regime = self.redis_client.get(f"BTC/USDT_15m_regime")
        assert stored_regime is not None
        
        logger.info("✅ Market regime detection test passed")
    
    @pytest.mark.asyncio
    async def test_indicators_engine(self):
        """Test indicators engine functionality"""
        engine = IndicatorsEngine()
        
        # Test indicator calculation
        indicators = await engine.calculate_all(self.test_data)
        
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'bollinger_upper' in indicators
        assert 'bollinger_lower' in indicators
        assert 'atr' in indicators
        assert 'volume_ratio' in indicators
        
        # Validate indicator ranges
        assert 0 <= indicators['rsi'] <= 100
        assert indicators['atr'] > 0
        assert indicators['volume_ratio'] > 0
        
        logger.info("✅ Indicators engine test passed")
    
    @pytest.mark.asyncio
    async def test_signal_generation(self):
        """Test signal generation pipeline"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m'],
            enable_regime_detection=True,
            enable_ml=True
        )
        
        # Update candle buffers
        for candle in self.test_data:
            for timeframe in alphapulse.timeframes:
                key = f"BTC/USDT_{timeframe}"
                alphapulse.candle_buffers[key].append(candle)
        
        # Generate signals
        signals = await alphapulse._generate_signals('BTC/USDT', self.test_data[-1])
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert isinstance(signal, Signal)
            assert signal.symbol == 'BTC/USDT'
            assert signal.timeframe == '15m'
            assert signal.direction in ['long', 'short']
            assert 0.0 <= signal.confidence <= 1.0
            assert signal.entry_price > 0
            assert signal.risk_reward_ratio > 0
        
        logger.info("✅ Signal generation test passed")
    
    @pytest.mark.asyncio
    async def test_signal_validation(self):
        """Test signal validation and filtering"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m']
        )
        
        # Create test signals
        test_signals = [
            Signal(
                signal_id="test_signal_1",
                symbol="BTC/USDT",
                timeframe="15m",
                direction="long",
                confidence=0.8,
                entry_price=45000.0,
                tp1=45900.0,
                tp2=46800.0,
                tp3=47700.0,
                tp4=48600.0,
                stop_loss=44100.0,
                risk_reward_ratio=2.0,
                pattern_type="rsi_divergence",
                volume_confirmation=True,
                trend_alignment=True,
                market_regime="strong_trend_bull",
                indicators={"rsi": 35.0, "volume_ratio": 1.8},
                validation_metrics={"trend_strength": 0.75},
                timestamp=datetime.now()
            ),
            Signal(
                signal_id="test_signal_2",
                symbol="BTC/USDT",
                timeframe="15m",
                direction="short",
                confidence=0.3,  # Low confidence
                entry_price=45000.0,
                tp1=44100.0,
                tp2=43200.0,
                tp3=42300.0,
                tp4=41400.0,
                stop_loss=45900.0,
                risk_reward_ratio=1.0,  # Poor risk-reward
                pattern_type="breakout",
                volume_confirmation=False,  # No volume confirmation
                trend_alignment=False,  # No trend alignment
                market_regime="choppy",
                indicators={"rsi": 65.0, "volume_ratio": 0.5},
                validation_metrics={"trend_strength": 0.2},
                timestamp=datetime.now()
            )
        ]
        
        # Validate signals
        valid_signals = await alphapulse._validate_signals(test_signals)
        
        # Only the first signal should be valid
        assert len(valid_signals) == 1
        assert valid_signals[0].signal_id == "test_signal_1"
        
        logger.info("✅ Signal validation test passed")
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance metrics tracking"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m']
        )
        
        # Simulate performance tracking
        for i in range(10):
            latency_ms = np.random.uniform(30, 70)
            signals_count = np.random.randint(0, 3)
            alphapulse._track_performance(latency_ms, signals_count)
        
        # Check performance summary
        summary = alphapulse.get_performance_summary()
        
        assert 'uptime_seconds' in summary
        assert 'total_signals' in summary
        assert 'avg_latency_ms' in summary
        assert 'max_latency_ms' in summary
        assert 'min_latency_ms' in summary
        assert 'throughput_signals_sec' in summary
        assert 'target_latency_ms' in summary
        assert 'latency_target_met' in summary
        
        assert summary['avg_latency_ms'] > 0
        assert summary['max_latency_ms'] >= summary['avg_latency_ms']
        assert summary['min_latency_ms'] <= summary['avg_latency_ms']
        
        logger.info("✅ Performance tracking test passed")
    
    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test database integration"""
        # Test signal storage
        signal = DBSignal(
            signal_id="test_db_signal_1",
            symbol="BTC/USDT",
            timeframe="15m",
            direction="long",
            confidence=0.85,
            entry_price=45000.0,
            tp1=45900.0,
            tp2=46800.0,
            tp3=47700.0,
            tp4=48600.0,
            stop_loss=44100.0,
            risk_reward_ratio=2.5,
            pattern_type="rsi_divergence",
            volume_confirmation=True,
            trend_alignment=True,
            market_regime="strong_trend_bull",
            indicators={
                "rsi": 35.2,
                "macd": 0.85,
                "bollinger_position": 0.15,
                "volume_ratio": 1.8,
                "atr": 1200.0
            },
            validation_metrics={
                "volume_ratio": 1.8,
                "trend_strength": 0.75,
                "volatility": 1200.0,
                "momentum": 35.2,
                "breakout_strength": 0.85
            }
        )
        
        # Test log storage
        log = Log(
            signal_id=signal.signal_id,
            pattern_type="rsi_divergence",
            confidence_score=0.85,
            volume_context={
                "volume_sma": 1500000,
                "current_volume": 2700000,
                "volume_ratio": 1.8
            },
            trend_context={
                "ema_20": 44800,
                "ema_50": 44500,
                "trend_direction": "bullish",
                "trend_strength": 0.75
            }
        )
        
        # Test performance metrics storage
        perf_metrics = PerformanceMetrics(
            test_run_id="test_run_20250815_143022",
            latency_avg_ms=45.2,
            latency_max_ms=78.5,
            throughput_signals_sec=22.1,
            accuracy=0.82,
            filter_rate=0.65
        )
        
        # Verify data integrity
        assert signal.signal_id == "test_db_signal_1"
        assert signal.symbol == "BTC/USDT"
        assert signal.confidence == 0.85
        assert signal.risk_reward_ratio == 2.5
        assert signal.volume_confirmation is True
        assert signal.trend_alignment is True
        assert signal.market_regime == "strong_trend_bull"
        assert isinstance(signal.indicators, dict)
        assert isinstance(signal.validation_metrics, dict)
        
        assert log.signal_id == signal.signal_id
        assert log.pattern_type == "rsi_divergence"
        assert log.confidence_score == 0.85
        assert isinstance(log.volume_context, dict)
        assert isinstance(log.trend_context, dict)
        
        assert perf_metrics.test_run_id == "test_run_20250815_143022"
        assert perf_metrics.latency_avg_ms == 45.2
        assert perf_metrics.accuracy == 0.82
        assert perf_metrics.filter_rate == 0.65
        
        logger.info("✅ Database integration test passed")
    
    @pytest.mark.asyncio
    async def test_ml_signal_generation(self):
        """Test ML-based signal generation"""
        ml_generator = MLSignalGenerator()
        
        # Mock ML model prediction
        with patch.object(ml_generator, 'predict', return_value=0.75):
            signals = await ml_generator.generate_signals(
                symbol='BTC/USDT',
                timeframe='15m',
                candles=self.test_data,
                indicators={'rsi': 35.0, 'macd': 0.5}
            )
            
            assert isinstance(signals, list)
            
            if signals:
                signal = signals[0]
                assert isinstance(signal, Signal)
                assert signal.symbol == 'BTC/USDT'
                assert signal.timeframe == '15m'
                assert 0.0 <= signal.confidence <= 1.0
        
        logger.info("✅ ML signal generation test passed")
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m'],
            enable_regime_detection=True,
            enable_ml=True,
            target_latency_ms=50.0
        )
        
        # Simulate complete pipeline
        start_time = datetime.now()
        
        # Process candle data
        for candle in self.test_data[:50]:  # Process first 50 candles
            await alphapulse._process_candle_data('BTC/USDT', candle)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance
        summary = alphapulse.get_performance_summary()
        assert summary['total_signals'] >= 0
        assert summary['avg_latency_ms'] > 0
        
        # Verify regime detection
        regimes = await alphapulse.get_current_regimes()
        assert 'BTC/USDT_15m' in regimes
        assert regimes['BTC/USDT_15m'] in [r.value for r in MarketRegime]
        
        # Verify signal history
        signal_history = alphapulse.get_signal_history()
        assert isinstance(signal_history, list)
        
        logger.info(f"✅ End-to-end pipeline test passed in {processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and recovery"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m']
        )
        
        # Test with invalid data
        invalid_candle = {
            'timestamp': datetime.now(),
            'open': 'invalid',  # Invalid price
            'high': 'invalid',
            'low': 'invalid',
            'close': 'invalid',
            'volume': 'invalid'
        }
        
        # Should handle gracefully
        try:
            await alphapulse._process_candle_data('BTC/USDT', invalid_candle)
        except Exception as e:
            logger.info(f"Expected error handled: {e}")
        
        # Test with empty data
        empty_signals = await alphapulse._generate_signals('BTC/USDT', {})
        assert isinstance(empty_signals, list)
        
        # Test validation with invalid signals
        invalid_signals = [
            Signal(
                signal_id="invalid_signal",
                symbol="BTC/USDT",
                timeframe="15m",
                direction="invalid",  # Invalid direction
                confidence=-1.0,  # Invalid confidence
                entry_price=-1000.0,  # Invalid price
                tp1=0.0,
                tp2=0.0,
                tp3=0.0,
                tp4=0.0,
                stop_loss=0.0,
                risk_reward_ratio=-1.0,  # Invalid ratio
                pattern_type="invalid_pattern",
                volume_confirmation=False,
                trend_alignment=False,
                market_regime="invalid_regime",
                indicators={},
                validation_metrics={},
                timestamp=datetime.now()
            )
        ]
        
        valid_signals = await alphapulse._validate_signals(invalid_signals)
        assert len(valid_signals) == 0  # All should be filtered out
        
        logger.info("✅ Error handling test passed")
    
    @pytest.mark.asyncio
    async def test_latency_targets(self):
        """Test latency target compliance"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m'],
            target_latency_ms=50.0
        )
        
        # Process multiple candles to build up metrics
        for _ in range(20):
            candle = self.test_data[0]  # Use same candle for consistency
            await alphapulse._process_candle_data('BTC/USDT', candle)
        
        # Check latency compliance
        summary = alphapulse.get_performance_summary()
        
        if summary['total_signals'] > 0:
            assert summary['avg_latency_ms'] > 0
            assert summary['max_latency_ms'] > 0
            assert summary['min_latency_ms'] > 0
            
            # Log performance metrics
            logger.info(f"Average latency: {summary['avg_latency_ms']:.2f}ms")
            logger.info(f"Max latency: {summary['max_latency_ms']:.2f}ms")
            logger.info(f"Min latency: {summary['min_latency_ms']:.2f}ms")
            logger.info(f"Target latency: {summary['target_latency_ms']}ms")
            logger.info(f"Latency target met: {summary['latency_target_met']}")
        
        logger.info("✅ Latency targets test passed")
    
    @pytest.mark.asyncio
    async def test_redis_integration(self):
        """Test Redis integration for signal storage"""
        # Test signal storage in Redis
        test_signal = Signal(
            signal_id="test_redis_signal",
            symbol="BTC/USDT",
            timeframe="15m",
            direction="long",
            confidence=0.8,
            entry_price=45000.0,
            tp1=45900.0,
            tp2=46800.0,
            tp3=47700.0,
            tp4=48600.0,
            stop_loss=44100.0,
            risk_reward_ratio=2.0,
            pattern_type="rsi_divergence",
            volume_confirmation=True,
            trend_alignment=True,
            market_regime="strong_trend_bull",
            indicators={"rsi": 35.0},
            validation_metrics={"trend_strength": 0.75},
            timestamp=datetime.now()
        )
        
        # Store signal in Redis
        signal_key = f"signal:{test_signal.signal_id}"
        self.redis_client.setex(
            signal_key,
            3600,  # 1 hour TTL
            json.dumps(test_signal.__dict__, default=str)
        )
        
        # Retrieve signal from Redis
        stored_data = self.redis_client.get(signal_key)
        assert stored_data is not None
        
        # Parse stored data
        parsed_data = json.loads(stored_data)
        assert parsed_data['signal_id'] == test_signal.signal_id
        assert parsed_data['symbol'] == test_signal.symbol
        assert parsed_data['confidence'] == test_signal.confidence
        
        # Clean up
        self.redis_client.delete(signal_key)
        
        logger.info("✅ Redis integration test passed")


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Test system throughput under load"""
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m'],
            target_latency_ms=50.0
        )
        
        # Generate test data
        test_data = []
        for i in range(1000):
            candle = {
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': 45000.0 + np.random.normal(0, 100),
                'high': 45000.0 + np.random.normal(0, 150),
                'low': 45000.0 + np.random.normal(0, 150),
                'close': 45000.0 + np.random.normal(0, 100),
                'volume': np.random.uniform(1000000, 5000000)
            }
            test_data.append(candle)
        
        # Process data and measure performance
        start_time = datetime.now()
        
        for candle in test_data:
            await alphapulse._process_candle_data('BTC/USDT', candle)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate throughput
        throughput = len(test_data) / processing_time
        
        # Performance assertions
        assert throughput > 10  # At least 10 candles per second
        assert processing_time < 100  # Should complete within 100 seconds
        
        summary = alphapulse.get_performance_summary()
        logger.info(f"Throughput: {throughput:.2f} candles/sec")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Average latency: {summary.get('avg_latency_ms', 0):.2f}ms")
        
        logger.info("✅ Throughput benchmark test passed")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        alphapulse = AlphaPulseCore(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframes=['1m', '15m', '1h'],
            target_latency_ms=50.0
        )
        
        # Process large amount of data
        for i in range(5000):
            candle = {
                'timestamp': datetime.now() - timedelta(minutes=i),
                'open': 45000.0 + np.random.normal(0, 100),
                'high': 45000.0 + np.random.normal(0, 150),
                'low': 45000.0 + np.random.normal(0, 150),
                'close': 45000.0 + np.random.normal(0, 100),
                'volume': np.random.uniform(1000000, 5000000)
            }
            await alphapulse._process_candle_data('BTC/USDT', candle)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 500  # Should not increase by more than 500MB
        assert final_memory < 1000  # Total memory should be less than 1GB
        
        logger.info(f"Initial memory: {initial_memory:.2f}MB")
        logger.info(f"Final memory: {final_memory:.2f}MB")
        logger.info(f"Memory increase: {memory_increase:.2f}MB")
        
        logger.info("✅ Memory usage test passed")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
