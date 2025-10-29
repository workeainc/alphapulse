#!/usr/bin/env python3
"""
Simple working test to verify basic AlphaPulse setup
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from indicators_engine import TechnicalIndicators

class TestSimpleWorking:
    """Simple tests that work with existing code"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.indicators = TechnicalIndicators()
        self.prices = [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5, 107.0, 106.5, 108.0]
    
    def test_technical_indicators_initialization(self):
        """Test that TechnicalIndicators can be initialized"""
        assert self.indicators is not None
        assert hasattr(self.indicators, 'calculate_rsi')
        assert hasattr(self.indicators, 'calculate_macd')
        assert hasattr(self.indicators, 'calculate_bollinger_bands')
    
    def test_rsi_calculation_basic(self):
        """Test basic RSI calculation"""
        # Calculate RSI for each price point
        rsi_values = []
        for i in range(len(self.prices)):
            current_prices = self.prices[:i+1]
            if len(current_prices) >= 2:
                rsi = self.indicators.calculate_rsi(current_prices)
                rsi_values.append(rsi)
        
        # Check that we get RSI values
        assert len(rsi_values) > 0
        assert all(0 <= rsi <= 100 for rsi in rsi_values)
    
    def test_macd_calculation_basic(self):
        """Test basic MACD calculation"""
        # Calculate MACD for the full price series
        macd_line, signal_line, histogram = self.indicators.calculate_macd(self.prices)
        
        # Check that we get numeric values
        assert isinstance(macd_line, (int, float))
        assert isinstance(signal_line, (int, float))
        assert isinstance(histogram, (int, float))
    
    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation"""
        # Calculate Bollinger Bands for the full price series
        upper, middle, lower = self.indicators.calculate_bollinger_bands(self.prices)
        
        # Check that we get numeric values and proper relationships
        assert isinstance(upper, (int, float))
        assert isinstance(middle, (int, float))
        assert isinstance(lower, (int, float))
        assert upper >= middle >= lower  # Basic relationship check

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
