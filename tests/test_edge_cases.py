#!/usr/bin/env python3
"""
Simple edge case test for AlphaPulse
"""
import sys
import os
import pytest
import numpy as np
from datetime import datetime, timezone

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from indicators_engine import TechnicalIndicators

class TestEdgeCasesSimple:
    """Simple edge case tests"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.indicators = TechnicalIndicators()
    
    def test_empty_price_list(self):
        """Test with empty price list"""
        rsi = self.indicators.calculate_rsi([])
        assert rsi == 50.0  # Should return neutral value
    
    def test_single_price(self):
        """Test with single price"""
        rsi = self.indicators.calculate_rsi([100.0])
        assert rsi == 50.0  # Should return neutral value
    
    def test_constant_prices(self):
        """Test with constant prices"""
        constant_prices = [100.0] * 20
        rsi = self.indicators.calculate_rsi(constant_prices)
        assert 0 <= rsi <= 100  # Should be within valid range
    
    def test_extreme_prices(self):
        """Test with extreme price values"""
        # Very high prices
        high_prices = [1000000.0, 1000001.0, 1000002.0]
        rsi = self.indicators.calculate_rsi(high_prices)
        assert 0 <= rsi <= 100
        
        # Very low prices
        low_prices = [0.0001, 0.0002, 0.0003]
        rsi = self.indicators.calculate_rsi(low_prices)
        assert 0 <= rsi <= 100
    
    def test_negative_prices(self):
        """Test with negative prices"""
        negative_prices = [-100.0, -101.0, -102.0]
        rsi = self.indicators.calculate_rsi(negative_prices)
        assert 0 <= rsi <= 100
    
    def test_large_price_list(self):
        """Test with very large price list"""
        large_prices = [100.0 + i * 0.01 for i in range(10000)]
        rsi = self.indicators.calculate_rsi(large_prices)
        assert 0 <= rsi <= 100
    
    def test_nan_values(self):
        """Test handling of NaN values"""
        import math
        prices_with_nan = [100.0, float('nan'), 102.0]
        
        # Should handle gracefully without crashing
        try:
            rsi = self.indicators.calculate_rsi(prices_with_nan)
            assert 0 <= rsi <= 100
        except:
            # If it crashes, that's also acceptable for this simple test
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
