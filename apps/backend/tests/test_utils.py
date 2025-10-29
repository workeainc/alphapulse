#!/usr/bin/env python3
"""
Test Utilities for AlphaPulse

This module contains tests for utility functions and common test fixtures.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import yaml

# Import utility functions to test
from ..src.utils.utils import (
    setup_logging, retry, async_retry, generate_id, hash_data,
    format_timestamp, parse_timestamp, load_json_file, save_json_file,
    load_yaml_file, save_yaml_file, run_in_threadpool, run_in_executor,
    ensure_directory, get_file_size, get_directory_size, format_bytes,
    ConfigManager, PerformanceMonitor, Cache, RateLimiter
)

from ..src.utils.risk_management import (
    RiskManager, PortfolioManager, calculate_kelly_criterion,
    calculate_var, calculate_max_drawdown, RiskLevel, RiskMetrics
)

from ..src.utils.threshold_env import (
    ThresholdManager, EnvironmentManager, AdaptiveParameters,
    Environment, ThresholdConfig
)


class TestUtilityFunctions:
    """Test basic utility functions."""
    
    def test_generate_id(self):
        """Test ID generation."""
        id1 = generate_id()
        id2 = generate_id()
        
        assert len(id1) == 8
        assert len(id2) == 8
        assert id1 != id2
        assert isinstance(id1, str)
    
    def test_hash_data(self):
        """Test data hashing."""
        data1 = {"test": "data"}
        data2 = {"test": "data"}
        data3 = {"different": "data"}
        
        hash1 = hash_data(data1)
        hash2 = hash_data(data2)
        hash3 = hash_data(data3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test datetime
        dt = datetime(2023, 1, 1, 12, 0, 0)
        formatted = format_timestamp(dt)
        assert "2023-01-01T12:00:00" in formatted
        
        # Test float
        timestamp = 1672574400.0
        formatted = format_timestamp(timestamp)
        assert "2023-01-01" in formatted
        
        # Test string
        formatted = format_timestamp("2023-01-01T12:00:00")
        assert formatted == "2023-01-01T12:00:00"
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        # Test ISO format
        dt = parse_timestamp("2023-01-01T12:00:00")
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 12
        
        # Test with timezone
        dt = parse_timestamp("2023-01-01T12:00:00Z")
        assert dt.year == 2023
    
    def test_retry_decorator(self):
        """Test retry decorator."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        assert call_count == 3
    
    def test_async_retry_decorator(self):
        """Test async retry decorator."""
        call_count = 0
        
        @async_retry(max_attempts=3, delay=0.1)
        async def failing_async_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            asyncio.run(failing_async_function())
        
        assert call_count == 3


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_save_and_load_json(self):
        """Test JSON file save and load."""
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            save_json_file(temp_file, test_data)
            loaded_data = load_json_file(temp_file)
            assert loaded_data == test_data
        finally:
            os.unlink(temp_file)
    
    def test_save_and_load_yaml(self):
        """Test YAML file save and load."""
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            save_yaml_file(temp_file, test_data)
            loaded_data = load_yaml_file(temp_file)
            assert loaded_data == test_data
        finally:
            os.unlink(temp_file)
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "test", "nested", "directory")
            ensure_directory(new_dir)
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)
    
    def test_get_file_size(self):
        """Test file size calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            temp_file = f.name
        
        try:
            size = get_file_size(temp_file)
            assert size > 0
        finally:
            os.unlink(temp_file)
    
    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(500) == "500.0 B"


class TestConfigManager:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager()
        assert config_manager.config.redis_host == "localhost"
        assert config_manager.config.redis_port == 6379
    
    def test_config_update(self):
        """Test configuration updates."""
        config_manager = ConfigManager()
        config_manager.update_config(redis_host="test-host", redis_port=6380)
        assert config_manager.config.redis_host == "test-host"
        assert config_manager.config.redis_port == 6380


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_timer_functionality(self):
        """Test timer functionality."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test")
        # Simulate some work
        import time
        time.sleep(0.01)
        duration = monitor.end_timer("test")
        
        assert duration > 0
        assert monitor.get_average_time("test") > 0
    
    def test_stats_calculation(self):
        """Test statistics calculation."""
        monitor = PerformanceMonitor()
        
        for i in range(5):
            monitor.start_timer("test")
            time.sleep(0.01)
            monitor.end_timer("test")
        
        stats = monitor.get_stats("test")
        assert stats['count'] == 5
        assert stats['avg'] > 0
        assert stats['min'] > 0
        assert stats['max'] > 0


class TestCache:
    """Test caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = Cache(ttl=1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = Cache(ttl=0.1)  # Very short TTL for testing
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        time.sleep(0.2)
        assert cache.get("key1") is None
    
    def test_cache_size(self):
        """Test cache size tracking."""
        cache = Cache()
        assert cache.size() == 0
        
        cache.set("key1", "value1")
        assert cache.size() == 1
        
        cache.set("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # First two calls should succeed
        assert limiter.can_call() is True
        assert limiter.can_call() is True
        
        # Third call should fail
        assert limiter.can_call() is False
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset over time."""
        limiter = RateLimiter(max_calls=1, time_window=0.1)
        
        assert limiter.can_call() is True
        assert limiter.can_call() is False
        
        # Wait for window to reset
        time.sleep(0.2)
        assert limiter.can_call() is True


class TestRiskManagement:
    """Test risk management functionality."""
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        risk_manager = RiskManager(account_balance=10000, risk_level=RiskLevel.MODERATE)
        assert risk_manager.account_balance == 10000
        assert risk_manager.risk_level == RiskLevel.MODERATE
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        risk_manager = RiskManager(account_balance=10000)
        
        metrics = risk_manager.calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            confidence=1.0
        )
        
        assert metrics.position_size > 0
        assert metrics.risk_amount > 0
        assert metrics.risk_percentage > 0
    
    def test_signal_validation(self):
        """Test signal validation."""
        risk_manager = RiskManager(account_balance=10000)
        
        valid_signal = {
            'risk_reward_ratio': 2.0,
            'risk_percentage': 1.0,
            'symbol': 'BTC/USDT'
        }
        
        is_valid, message = risk_manager.validate_signal(valid_signal)
        assert is_valid is True
    
    def test_portfolio_manager(self):
        """Test portfolio manager."""
        portfolio = PortfolioManager(initial_balance=10000)
        
        # Add position
        portfolio.add_position("test1", {
            'entry_price': 100.0,
            'position_size': 10.0,
            'symbol': 'BTC/USDT'
        })
        
        assert portfolio.get_position_count() == 1
        
        # Update position
        result = portfolio.update_position("test1", 105.0)
        assert result['pnl'] > 0
        
        # Close position
        result = portfolio.close_position("test1", 110.0)
        assert result['pnl'] > 0
        assert result['new_balance'] > 10000


class TestThresholdEnvironment:
    """Test threshold environment management."""
    
    def test_threshold_manager(self):
        """Test threshold manager."""
        threshold_manager = ThresholdManager()
        
        # Test default thresholds
        assert threshold_manager.get_threshold('rsi_oversold') == 30.0
        assert threshold_manager.get_threshold('rsi_overbought') == 70.0
        
        # Test setting thresholds
        threshold_manager.set_threshold('rsi_oversold', 25.0)
        assert threshold_manager.get_threshold('rsi_oversold') == 25.0
    
    def test_environment_manager(self):
        """Test environment manager."""
        env_manager = EnvironmentManager()
        
        # Test default environment
        assert env_manager.current_env == Environment.DEVELOPMENT
        
        # Test environment switching
        env_manager.set_environment(Environment.PRODUCTION)
        assert env_manager.is_production() is True
        assert env_manager.is_backtest() is False
    
    def test_adaptive_parameters(self):
        """Test adaptive parameters."""
        adaptive = AdaptiveParameters()
        
        # Add parameter
        adaptive.add_parameter("test_param", 1.0, 0.0, 10.0)
        assert adaptive.get_parameter("test_param") == 1.0
        
        # Set parameter
        adaptive.set_parameter("test_param", 5.0)
        assert adaptive.get_parameter("test_param") == 5.0
        
        # Test bounds
        adaptive.set_parameter("test_param", 15.0)  # Above max
        assert adaptive.get_parameter("test_param") == 10.0
        
        adaptive.set_parameter("test_param", -5.0)  # Below min
        assert adaptive.get_parameter("test_param") == 0.0


class TestAsyncUtilities:
    """Test async utility functions."""
    
    @pytest.mark.asyncio
    async def test_run_in_executor(self):
        """Test running functions in executor."""
        def sync_function(x, y):
            return x + y
        
        result = await run_in_executor(sync_function, 2, 3)
        assert result == 5
    
    def test_run_in_threadpool(self):
        """Test running functions in thread pool."""
        def test_function(x, y):
            return x * y
        
        result = run_in_threadpool(test_function, 4, 5)
        assert result == 20


# Test fixtures
@pytest.fixture
def sample_candle_data():
    """Provide sample candle data for tests."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })


@pytest.fixture
def mock_redis():
    """Provide mock Redis client."""
    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.ping.return_value = True
    return mock_client


@pytest.fixture
def mock_database():
    """Provide mock database session."""
    mock_session = Mock()
    mock_session.query.return_value = Mock()
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    return mock_session


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
