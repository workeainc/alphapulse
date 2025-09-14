#!/usr/bin/env python3
"""
Simple integration test for AlphaPulse
"""
import sys
import os
import pytest
import numpy as np
import time
from datetime import datetime, timezone

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from indicators_engine import TechnicalIndicators

class TestIntegrationSimple:
    """Simple integration tests"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.indicators = TechnicalIndicators()
        self.prices = [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0, 104.5, 106.0, 105.5, 107.0, 106.5, 108.0]
    
    def test_full_pipeline_basic(self):
        """Test basic full pipeline"""
        # Simulate processing multiple price points
        signals = []
        start_time = time.time()
        
        for i in range(len(self.prices)):
            current_prices = self.prices[:i+1]
            if len(current_prices) >= 2:
                # Calculate indicators
                rsi = self.indicators.calculate_rsi(current_prices)
                macd_line, signal_line, histogram = self.indicators.calculate_macd(current_prices)
                upper, middle, lower = self.indicators.calculate_bollinger_bands(current_prices)
                
                # Simple signal logic (without MLSignalGenerator)
                if rsi < 30 and macd_line > signal_line:
                    signal = {
                        'symbol': 'BTCUSDT',
                        'price': current_prices[-1],
                        'rsi': rsi,
                        'macd_line': macd_line,
                        'signal_line': signal_line,
                        'direction': 'BUY'
                    }
                    signals.append(signal)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Basic assertions
        assert latency < 1000  # Should be under 1 second
        assert isinstance(signals, list)
    
    def test_latency_measurement(self):
        """Test latency measurement"""
        start_time = time.time()
        
        # Perform a simple calculation
        rsi = self.indicators.calculate_rsi(self.prices)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        assert latency < 50  # Should be under 50ms
        assert 0 <= rsi <= 100
    
    def test_throughput_measurement(self):
        """Test throughput measurement"""
        start_time = time.time()
        operations = 0
        
        # Perform multiple operations
        for _ in range(100):
            rsi = self.indicators.calculate_rsi(self.prices)
            operations += 1
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = operations / duration
        
        assert throughput > 100  # Should be over 100 ops/sec
        assert duration < 1  # Should complete in under 1 second

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
