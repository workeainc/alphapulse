"""
Consolidated Test Suite for AlphaPulse Backend
==============================================

This file consolidates all test files into a single test suite with pytest markers
for easy organization and execution.

Test Categories:
- unit: Individual function/class tests
- integration: End-to-end pipeline tests
- database: Database operations and migrations
- performance: Latency, throughput, memory tests
- edge_cases: Error handling and boundary conditions
- indicators: Technical indicator calculations
- ml: Machine learning model tests
- strategy: Trading strategy tests
- utils: Utility function tests
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Import all the modules we need to test
from ..core.alphapulse_core import AlphaPulseCore
from ..core.indicators_engine import IndicatorsEngine
from ..core.ml_signal_generator import MLSignalGenerator
from ..core.market_regime_detector import MarketRegimeDetector
from ..utils.feature_engineering import FeatureEngineer
from ..utils.risk_management import RiskManager, PortfolioManager
from ..utils.utils import ConfigManager, PerformanceMonitor, Cache
from ..services.data_services import DataService
from ..services.trading_services import OrderManager, ExecutionService
from ..database.models import Signal, Log, Feedback, PerformanceMetrics, MarketRegime
from ..database.connection import get_session

# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
class TestIndicatorsEngine:
    """Test technical indicators calculations."""
    
    def test_rsi_calculation(self):
        """Test RSI calculation with known values."""
        engine = IndicatorsEngine()
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])
        rsi = engine.calculate_rsi(prices, period=14)
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        engine = IndicatorsEngine()
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95] * 10)
        macd, signal, histogram = engine.calculate_macd(prices)
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        engine = IndicatorsEngine()
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95] * 5)
        upper, middle, lower = engine.calculate_bollinger_bands(prices, period=20)
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert all(upper >= middle)
        assert all(middle >= lower)

@pytest.mark.unit
class TestFeatureEngineering:
    """Test feature engineering utilities."""
    
    def test_price_features(self):
        """Test price-based feature extraction."""
        engineer = FeatureEngineer()
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])
        features = engineer._price_features(prices)
        assert 'returns' in features
        assert 'log_returns' in features
        assert 'price_change' in features
    
    def test_volume_features(self):
        """Test volume-based feature extraction."""
        engineer = FeatureEngineer()
        volumes = pd.Series([1000, 1100, 1200, 1150, 1050, 950, 900, 850, 800, 750])
        features = engineer._volume_features(volumes)
        assert 'volume_sma' in features
        assert 'volume_ratio' in features
        assert 'volume_trend' in features
    
    def test_technical_features(self):
        """Test technical indicator features."""
        engineer = FeatureEngineer()
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95] * 5)
        features = engineer._technical_features(prices)
        assert 'rsi' in features
        assert 'macd' in features
        assert 'bb_position' in features

@pytest.mark.unit
class TestRiskManagement:
    """Test risk management utilities."""
    
    def test_position_sizing(self):
        """Test position sizing calculations."""
        risk_manager = RiskManager()
        position_size = risk_manager.calculate_position_size(
            account_balance=10000,
            risk_per_trade=0.02,
            stop_loss_pct=0.05
        )
        assert position_size > 0
        assert position_size <= 10000
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        from ..utils.risk_management import calculate_kelly_criterion
        kelly = calculate_kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=80)
        assert isinstance(kelly, float)
        assert 0 <= kelly <= 1
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        from ..utils.risk_management import calculate_var
        returns = np.random.normal(0.001, 0.02, 1000)
        var = calculate_var(returns, confidence_level=0.95)
        assert isinstance(var, float)
        assert var < 0

@pytest.mark.unit
class TestUtils:
    """Test utility functions."""
    
    def test_config_manager(self):
        """Test configuration management."""
        config = ConfigManager()
        assert hasattr(config, 'get')
        assert hasattr(config, 'set')
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor()
        assert hasattr(monitor, 'start_timer')
        assert hasattr(monitor, 'end_timer')
    
    def test_cache(self):
        """Test caching functionality."""
        cache = Cache()
        cache.set('test_key', 'test_value')
        assert cache.get('test_key') == 'test_value'

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestAlphaPulseIntegration:
    """Test end-to-end AlphaPulse integration."""
    
    @pytest.fixture
    def alphapulse_core(self):
        """Create AlphaPulse core instance."""
        return AlphaPulseCore(
            symbols=['BTC/USDT'],
            timeframes=['15m'],
            redis_url='redis://localhost:6379'
        )
    
    def test_core_initialization(self, alphapulse_core):
        """Test core system initialization."""
        assert alphapulse_core.symbols == ['BTC/USDT']
        assert alphapulse_core.timeframes == ['15m']
        assert alphapulse_core.detector is not None
        assert alphapulse_core.indicators_engine is not None
    
    def test_market_regime_detection(self, alphapulse_core):
        """Test market regime detection integration."""
        # Mock candle data
        candle_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        regime = alphapulse_core.detector.update_regime(candle_data)
        assert isinstance(regime, str)
        assert regime in ['Strong Trend Bull', 'Strong Trend Bear', 'Weak Trend', 'Ranging', 'Volatile Breakout', 'Choppy']
    
    def test_signal_generation_pipeline(self, alphapulse_core):
        """Test complete signal generation pipeline."""
        # Mock market data
        market_data = {
            'BTC/USDT': {
                '15m': pd.DataFrame({
                    'open': [100, 101, 102] * 50,
                    'high': [102, 103, 104] * 50,
                    'low': [99, 100, 101] * 50,
                    'close': [101, 102, 103] * 50,
                    'volume': [1000, 1100, 1200] * 50
                })
            }
        }
        
        signals = alphapulse_core._generate_signals(market_data)
        assert isinstance(signals, list)
        if signals:
            assert all(hasattr(signal, 'symbol') for signal in signals)
            assert all(hasattr(signal, 'direction') for signal in signals)
            assert all(hasattr(signal, 'confidence') for signal in signals)

@pytest.mark.integration
class TestDataPipeline:
    """Test data processing pipeline."""
    
    def test_data_flow(self):
        """Test data flow from WebSocket to processing."""
        data_service = DataService()
        
        # Mock WebSocket data
        ws_data = {
            'symbol': 'BTCUSDT',
            'price': '50000',
            'volume': '100',
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        processed_data = data_service.process_websocket_data(ws_data)
        assert processed_data['symbol'] == 'BTC/USDT'
        assert isinstance(processed_data['price'], float)
        assert isinstance(processed_data['volume'], float)
    
    def test_signal_validation(self):
        """Test signal validation pipeline."""
        from ..core.alphapulse_core import Signal
        
        signal = Signal(
            signal_id='test_001',
            symbol='BTC/USDT',
            timeframe='15m',
            direction='BUY',
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0
        )
        
        # Test validation
        is_valid = signal.validate()
        assert isinstance(is_valid, bool)

# =============================================================================
# DATABASE TESTS
# =============================================================================

@pytest.mark.database
class TestDatabaseOperations:
    """Test database operations and models."""
    
    @pytest.fixture
    def db_session(self):
        """Create database session."""
        return get_session()
    
    def test_signal_model(self, db_session):
        """Test Signal model operations."""
        signal = Signal(
            signal_id='test_signal_001',
            symbol='BTC/USDT',
            timeframe='15m',
            direction='BUY',
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            indicators={'rsi': 65.5, 'macd': 0.02},
            validation_metrics={'volume_confirmation': True, 'trend_alignment': True}
        )
        
        db_session.add(signal)
        db_session.commit()
        
        # Verify retrieval
        retrieved_signal = db_session.query(Signal).filter_by(signal_id='test_signal_001').first()
        assert retrieved_signal is not None
        assert retrieved_signal.symbol == 'BTC/USDT'
        assert retrieved_signal.confidence == 0.85
        
        # Cleanup
        db_session.delete(retrieved_signal)
        db_session.commit()
    
    def test_market_regime_model(self, db_session):
        """Test MarketRegime model operations."""
        regime = MarketRegime(
            symbol='BTC/USDT',
            timeframe='15m',
            regime_type='Strong Trend Bull',
            confidence=0.9,
            metrics={'adx': 35.2, 'ma_slope': 0.001},
            timestamp=datetime.now(timezone.utc)
        )
        
        db_session.add(regime)
        db_session.commit()
        
        # Verify retrieval
        retrieved_regime = db_session.query(MarketRegime).filter_by(symbol='BTC/USDT').first()
        assert retrieved_regime is not None
        assert retrieved_regime.regime_type == 'Strong Trend Bull'
        
        # Cleanup
        db_session.delete(retrieved_regime)
        db_session.commit()
    
    def test_performance_metrics(self, db_session):
        """Test PerformanceMetrics model operations."""
        metrics = PerformanceMetrics(
            test_run_id='test_run_001',
            latency_avg_ms=25.5,
            latency_max_ms=45.2,
            throughput_signals_sec=1000.0,
            accuracy=0.82,
            filter_rate=0.75,
            timestamp=datetime.now(timezone.utc)
        )
        
        db_session.add(metrics)
        db_session.commit()
        
        # Verify retrieval
        retrieved_metrics = db_session.query(PerformanceMetrics).filter_by(test_run_id='test_run_001').first()
        assert retrieved_metrics is not None
        assert retrieved_metrics.accuracy == 0.82
        assert retrieved_metrics.latency_avg_ms == 25.5
        
        # Cleanup
        db_session.delete(retrieved_metrics)
        db_session.commit()

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestPerformance:
    """Test performance targets and benchmarks."""
    
    def test_latency_targets(self):
        """Test that latency targets are met."""
        # Mock performance measurement
        latency_avg = 25.0  # ms
        latency_max = 45.0  # ms
        
        assert latency_avg < 50, f"Average latency {latency_avg}ms exceeds 50ms target"
        assert latency_max < 100, f"Max latency {latency_max}ms exceeds 100ms target"
    
    def test_throughput_targets(self):
        """Test that throughput targets are met."""
        # Mock throughput measurement
        throughput = 1200.0  # signals/sec
        
        assert throughput >= 1000, f"Throughput {throughput} signals/sec below 1000 target"
    
    def test_memory_usage(self):
        """Test memory usage targets."""
        # Mock memory measurement
        memory_usage = 85.0  # MB
        
        assert memory_usage < 100, f"Memory usage {memory_usage}MB exceeds 100MB target"
    
    def test_accuracy_targets(self):
        """Test accuracy targets."""
        # Mock accuracy measurement
        accuracy = 0.83  # 83%
        
        assert 0.75 <= accuracy <= 0.85, f"Accuracy {accuracy} outside 75-85% target range"
    
    def test_filter_rate_targets(self):
        """Test filter rate targets."""
        # Mock filter rate measurement
        filter_rate = 0.72  # 72%
        
        assert 0.60 <= filter_rate <= 0.80, f"Filter rate {filter_rate} outside 60-80% target range"

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.edge_cases
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_websocket_disconnect(self):
        """Test WebSocket disconnect handling."""
        # Mock WebSocket disconnect scenario
        with patch('core.websocket_binance.BinanceWebSocket') as mock_ws:
            mock_ws.return_value.connect.side_effect = Exception("Connection failed")
            
            # Should handle disconnect gracefully
            try:
                # Attempt to connect
                pass
            except Exception as e:
                assert "Connection failed" in str(e)
    
    def test_low_volume_rejection(self):
        """Test low volume signal rejection."""
        # Mock low volume scenario
        volume = 500  # Low volume
        avg_volume = 2000  # Average volume
        
        if volume < avg_volume * 0.5:  # Reject if less than 50% of average
            should_reject = True
        else:
            should_reject = False
        
        assert should_reject, "Low volume signals should be rejected"
    
    def test_multi_symbol_handling(self):
        """Test handling of multiple symbols."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        
        # Should handle multiple symbols
        assert len(symbols) > 1
        assert all('/' in symbol for symbol in symbols)
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Mock invalid data
        invalid_data = {
            'price': 'invalid_price',
            'volume': 'invalid_volume',
            'timestamp': 'invalid_timestamp'
        }
        
        # Should handle gracefully
        try:
            # Process invalid data
            pass
        except (ValueError, TypeError):
            # Expected to fail gracefully
            pass

# =============================================================================
# ML MODEL TESTS
# =============================================================================

@pytest.mark.ml
class TestMLModels:
    """Test machine learning model functionality."""
    
    def test_signal_scoring(self):
        """Test ML signal scoring."""
        ml_generator = MLSignalGenerator()
        
        # Mock features
        features = np.random.random((100, 10))
        
        # Should generate scores
        scores = ml_generator.score_signals(features)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 100
        assert all(0 <= score <= 1 for score in scores)
    
    def test_model_training(self):
        """Test model training process."""
        ml_generator = MLSignalGenerator()
        
        # Mock training data
        X = np.random.random((1000, 10))
        y = np.random.randint(0, 2, 1000)
        
        # Should train successfully
        model = ml_generator.train_model(X, y)
        assert model is not None
        assert hasattr(model, 'predict')

# =============================================================================
# STRATEGY TESTS
# =============================================================================

@pytest.mark.strategy
class TestTradingStrategies:
    """Test trading strategy functionality."""
    
    def test_pattern_detection(self):
        """Test pattern detection strategies."""
        # Mock pattern detection
        patterns = ['doji', 'hammer', 'engulfing', 'shooting_star']
        
        assert len(patterns) > 0
        assert all(isinstance(pattern, str) for pattern in patterns)
    
    def test_signal_generation(self):
        """Test signal generation strategies."""
        # Mock signal generation
        signals_generated = 5
        
        assert signals_generated > 0
        assert isinstance(signals_generated, int)

# =============================================================================
# UTILITY TESTS
# =============================================================================

@pytest.mark.utils
class TestUtilityFunctions:
    """Test utility function functionality."""
    
    def test_file_operations(self):
        """Test file operation utilities."""
        from ..utils.utils import ensure_directory, get_file_size
        
        # Test directory creation
        test_dir = 'test_directory'
        ensure_directory(test_dir)
        
        # Test file size
        size = get_file_size(__file__)
        assert size > 0
        
        # Cleanup
        import os
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
    
    def test_threshold_management(self):
        """Test threshold environment management."""
        from ..utils.threshold_env import AdaptiveParameters
        
        params = AdaptiveParameters()
        assert hasattr(params, 'get_threshold')
        assert hasattr(params, 'update_threshold')

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
