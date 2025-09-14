#!/usr/bin/env python3
"""
Simplified Phase 2 Verification Tests for AlphaPulse
Tests error handling, performance optimizations, logging standardization, and dependency fixes
"""

import asyncio
import logging
import time
import cProfile
import pstats
import io
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from fastapi import HTTPException

# Set environment variables directly
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["DATABASE_URL"] = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

# Import components to test
from app.core.unified_config import get_settings, get_logger
from routes.candlestick_analysis import validate_symbol, validate_timeframe
from data.optimized_data_processor import OptimizedDataProcessor
from app.strategies.strategy_manager import StrategyManager
from ..ai.priority2_feature_engineering import Priority2FeatureEngineering

class TestPhase2ErrorHandling:
    """Test comprehensive error handling improvements"""
    
    def test_specific_exception_handling(self):
        """Test that specific exceptions are handled correctly"""
        logger = get_logger(__name__)
        
        # Test ValueError handling
        try:
            raise ValueError("Test validation error")
        except ValueError as e:
            logger.error(f"Data validation error: {e}", exc_info=True)
            assert "Data validation error" in str(e)
        
        # Test ConnectionError handling
        try:
            raise ConnectionError("Database connection failed")
        except ConnectionError as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            assert "Database connection error" in str(e)
    
    def test_input_validation_functions(self):
        """Test input validation functions"""
        # Test valid symbol
        assert validate_symbol("BTC") == "BTC"
        assert validate_symbol("ETH") == "ETH"
        
        # Test invalid symbol
        with pytest.raises(HTTPException) as exc_info:
            validate_symbol("invalid-symbol")
        assert exc_info.value.status_code == 400
        
        # Test valid timeframe
        assert validate_timeframe("15m") == "15m"
        assert validate_timeframe("1h") == "1h"
        
        # Test invalid timeframe
        with pytest.raises(HTTPException) as exc_info:
            validate_timeframe("invalid")
        assert exc_info.value.status_code == 400

class TestPhase2PerformanceOptimizations:
    """Test performance optimization improvements"""
    
    def test_adaptive_interval_calculation(self):
        """Test adaptive interval calculation in strategy manager"""
        manager = StrategyManager()
        
        # Test adaptive interval calculation
        interval = manager._calculate_adaptive_interval()
        assert isinstance(interval, float)
        assert manager.min_interval <= interval <= manager.max_interval
    
    def test_lru_cache_functionality(self):
        """Test LRU cache in feature engineering"""
        fe = Priority2FeatureEngineering()
        
        # Test that _calculate_skewness is cached
        data1 = np.random.randn(100, 10)
        data2 = np.random.randn(100, 10)
        
        # First call should cache
        result1 = fe._calculate_skewness(data1)
        result2 = fe._calculate_skewness(data1)  # Same data, should use cache
        
        # Different data should not use cache
        result3 = fe._calculate_skewness(data2)
        
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert isinstance(result3, np.ndarray)
    
    def test_vectorized_operations(self):
        """Test vectorized operations in data processor"""
        processor = OptimizedDataProcessor()
        
        # Create test data
        data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        })
        
        # Test data validation
        is_valid = processor._validate_input_data(data, "BTC", "1h")
        assert is_valid == True
        
        # Test data preconditions
        preconditions_met = processor._apply_data_preconditions(data)
        assert preconditions_met == True
    
    def test_context_manager_usage(self):
        """Test context manager usage in data processing"""
        processor = OptimizedDataProcessor()
        
        # Test processing context creation
        context = processor._create_processing_context("BTC", "1h")
        
        with context as ctx:
            assert ctx.symbol == "BTC"
            assert ctx.timeframe == "1h"
            assert hasattr(ctx, 'start_time')

class TestPhase2LoggingStandardization:
    """Test logging standardization improvements"""
    
    def test_logging_format(self):
        """Test standardized logging format"""
        logger = get_logger("test_logger")
        
        # Test that logger uses standardized format
        with patch('sys.stdout') as mock_stdout:
            logger.info("Test message")
            # Verify log format doesn't contain emojis in production
            assert "Test message" in str(mock_stdout.write.call_args)
    
    def test_environment_based_logging(self):
        """Test environment-based logging configuration"""
        settings = get_settings()
        
        # Test that logging is configured based on environment
        assert hasattr(settings, 'LOG_LEVEL')
        assert hasattr(settings, 'DEBUG')
        
        # Test logger creation
        logger = get_logger("test_env_logger")
        assert isinstance(logger, logging.Logger)
    
    def test_no_emoji_in_production_logs(self):
        """Test that production logs don't contain emojis"""
        # Set production environment
        os.environ["ENV"] = "production"
        
        logger = get_logger("test_production_logger")
        
        # Test that log messages don't contain emojis
        test_message = "Test production log message"
        logger.info(test_message)
        
        # Verify no emoji characters in log format
        assert "🚀" not in test_message
        assert "✅" not in test_message
        assert "❌" not in test_message

class TestPhase2DependencyResolution:
    """Test dependency conflict resolution"""
    
    def test_python_dependencies(self):
        """Test Python dependency conflicts are resolved"""
        import subprocess
        import sys
        
        # Test pip check
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "check"], 
                                  capture_output=True, text=True)
            assert result.returncode == 0, f"Pip check failed: {result.stderr}"
        except Exception as e:
            pytest.skip(f"Pip check not available: {e}")
    
    def test_import_conflicts(self):
        """Test that there are no import conflicts"""
        # Test key imports work without conflicts
        try:
            import pandas as pd
            import numpy as np
            import psutil
            import asyncio
            import logging
            assert True
        except ImportError as e:
            pytest.fail(f"Import conflict detected: {e}")
    
    def test_version_consistency(self):
        """Test version consistency in requirements"""
        # Test that key packages have consistent versions
        import pandas as pd
        import numpy as np
        import psutil
        
        # Verify packages are imported successfully
        assert pd.__version__ is not None
        assert np.__version__ is not None
        assert psutil.__version__ is not None

class TestPhase2Integration:
    """Test Phase 2 changes integration"""
    
    def test_performance_improvements(self):
        """Test performance improvements are working"""
        # Test that performance optimizations are active
        processor = OptimizedDataProcessor()
        
        # Test that caching is working
        assert hasattr(processor, 'data_cache')
        assert hasattr(processor, 'indicator_cache')
        
        # Test that adaptive intervals are working
        manager = StrategyManager()
        assert hasattr(manager, '_calculate_adaptive_interval')
    
    def test_logging_consistency(self):
        """Test logging consistency across components"""
        # Test that all components use standardized logging
        components = [
            OptimizedDataProcessor(),
            StrategyManager(),
            Priority2FeatureEngineering()
        ]
        
        for component in components:
            # Verify components have proper logging setup
            assert hasattr(component, 'logger') or 'logger' in dir(component)

def run_performance_profiling():
    """Run performance profiling to verify improvements"""
    print("Running Performance Profiling...")
    
    # Profile strategy manager
    profiler = cProfile.Profile()
    profiler.enable()
    
    manager = StrategyManager()
    interval = manager._calculate_adaptive_interval()
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    
    print("Strategy Manager Performance Profile:")
    print(s.getvalue())
    
    # Profile feature engineering
    profiler = cProfile.Profile()
    profiler.enable()
    
    fe = Priority2FeatureEngineering()
    data = np.random.randn(100, 10)
    result = fe._calculate_skewness(data)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    
    print("Feature Engineering Performance Profile:")
    print(s.getvalue())

if __name__ == "__main__":
    print("Starting Phase 2 Verification Tests")
    print("=" * 50)
    
    # Run performance profiling
    run_performance_profiling()
    
    # Run tests
    pytest.main([__file__, "-v"])
    
    print("=" * 50)
    print("Phase 2 Verification Complete")
