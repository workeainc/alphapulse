#!/usr/bin/env python3
"""
Unit tests for Candlestick Analysis API
Tests API endpoints, input validation, and error handling
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Update import paths for new structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from routes.candlestick_analysis import (
    router, 
    validate_symbol, 
    validate_timeframe, 
    validate_limit,
    get_processor
)
from data.real_time_processor import RealTimeCandlestickProcessor


class TestCandlestickAnalysisValidation:
    """Test input validation functions"""
    
    def test_validate_symbol_valid(self):
        """Test valid symbol validation"""
        valid_symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        
        for symbol in valid_symbols:
            result = validate_symbol(symbol)
            assert result == symbol.upper()
    
    def test_validate_symbol_invalid_format(self):
        """Test invalid symbol format validation"""
        invalid_symbols = ["btc", "ETH-USD", "BTC/USD", "btc-usd", "BTC_USD"]
        
        for symbol in invalid_symbols:
            with pytest.raises(HTTPException) as exc_info:
                validate_symbol(symbol)
            assert exc_info.value.status_code == 400
            assert "Invalid symbol format" in str(exc_info.value.detail)
    
    def test_validate_symbol_too_long(self):
        """Test symbol length validation"""
        long_symbol = "A" * 25  # 25 characters, exceeds 20 limit
        
        with pytest.raises(HTTPException) as exc_info:
            validate_symbol(long_symbol)
        assert exc_info.value.status_code == 400
        assert "Symbol too long" in str(exc_info.value.detail)
    
    def test_validate_timeframe_valid(self):
        """Test valid timeframe validation"""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        
        for timeframe in valid_timeframes:
            result = validate_timeframe(timeframe)
            assert result == timeframe
    
    def test_validate_timeframe_invalid(self):
        """Test invalid timeframe validation"""
        invalid_timeframes = ["2m", "10m", "2h", "6h", "2d", "invalid", ""]
        
        for timeframe in invalid_timeframes:
            with pytest.raises(HTTPException) as exc_info:
                validate_timeframe(timeframe)
            assert exc_info.value.status_code == 400
            assert "Invalid timeframe" in str(exc_info.value.detail)
    
    def test_validate_limit_valid(self):
        """Test valid limit validation"""
        valid_limits = [1, 10, 100, 500, 1000]
        
        for limit in valid_limits:
            result = validate_limit(limit)
            assert result == limit
    
    def test_validate_limit_invalid(self):
        """Test invalid limit validation"""
        invalid_limits = [0, -1, 1001, 2000]
        
        for limit in invalid_limits:
            with pytest.raises(HTTPException) as exc_info:
                validate_limit(limit)
            assert exc_info.value.status_code == 400
            assert "Limit must be between" in str(exc_info.value.detail)


class TestCandlestickAnalysisProcessor:
    """Test processor dependency injection"""
    
    @patch('routes.candlestick_analysis.RealTimeCandlestickProcessor')
    def test_get_processor_success(self, mock_processor_class):
        """Test successful processor creation"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        processor = get_processor()
        
        assert processor == mock_processor
        mock_processor_class.assert_called_once()
    
    @patch('routes.candlestick_analysis.RealTimeCandlestickProcessor')
    def test_get_processor_failure(self, mock_processor_class):
        """Test processor creation failure"""
        mock_processor_class.side_effect = Exception("Initialization error")
        
        with pytest.raises(HTTPException) as exc_info:
            get_processor()
        
        assert exc_info.value.status_code == 500
        assert "Failed to initialize processor" in str(exc_info.value.detail)


class TestCandlestickAnalysisEndpoints:
    """Test API endpoints"""
    
    def setup_method(self):
        """Setup test environment"""
        from fastapi import FastAPI
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_get_analysis_status_success(self, mock_get_processor):
        """Test successful analysis status endpoint"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.get_processing_stats.return_value = {
            "total_processed": 100,
            "avg_processing_time": 0.5
        }
        mock_processor.candlestick_data = {"BTC": [], "ETH": []}
        mock_processor.signal_history = {"BTC": [{"signal": "buy"}], "ETH": []}
        mock_get_processor.return_value = mock_processor
        
        response = self.client.get("/api/candlestick/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert "timestamp" in data
        assert "processing_stats" in data
        assert "active_symbols" in data
        assert "total_signals" in data
        assert data["total_signals"] == 1
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_get_analysis_status_database_error(self, mock_get_processor):
        """Test analysis status with database error"""
        mock_get_processor.side_effect = ConnectionError("Database connection failed")
        
        response = self.client.get("/api/candlestick/status")
        
        assert response.status_code == 503
        data = response.json()
        assert "Database temporarily unavailable" in data["detail"]
    
    @patch('routes.candlestick_analysis.get_processor')
    @patch('routes.candlestick_analysis.MLPatternDetector')
    def test_get_patterns_valid_input(self, mock_ml_detector, mock_get_processor):
        """Test patterns endpoint with valid input"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.get_symbol_data.return_value = {
            'candlesticks': [{'timestamp': '2023-01-01T00:00:00Z', 'open': 100, 'high': 105, 'low': 95, 'close': 103, 'volume': 1000}]
        }
        mock_processor._to_dataframe.return_value = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95], 'close': [103], 'volume': [1000]
        })
        mock_get_processor.return_value = mock_processor
        
        # Mock pattern detector
        mock_pattern = Mock()
        mock_pattern.pattern = "doji"
        mock_pattern.type = "reversal"
        mock_pattern.strength = 0.8
        mock_pattern.ml_confidence = 0.8
        mock_pattern.market_regime = "trending"
        mock_pattern.timestamp = "2023-01-01T00:00:00Z"
        mock_pattern.features = {}
        
        mock_ml_instance = Mock()
        mock_ml_instance.detect_patterns_ml.return_value = [mock_pattern]
        mock_ml_detector.return_value = mock_ml_instance
        
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m&limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "timeframe" in data
        assert "patterns" in data
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["pattern"] == "doji"
    
    def test_get_patterns_invalid_symbol(self):
        """Test patterns endpoint with invalid symbol"""
        response = self.client.get("/api/candlestick/patterns/btc-usd?timeframe=15m")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid symbol format" in data["detail"]
    
    def test_get_patterns_invalid_timeframe(self):
        """Test patterns endpoint with invalid timeframe"""
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=2h")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid timeframe" in data["detail"]
    
    def test_get_patterns_invalid_limit(self):
        """Test patterns endpoint with invalid limit"""
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m&limit=2000")
        
        assert response.status_code == 400
        data = response.json()
        assert "Limit must be between" in data["detail"]
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_get_patterns_processing_error(self, mock_get_processor):
        """Test patterns endpoint with processing error"""
        mock_get_processor.side_effect = Exception("Processing error")
        
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m")
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_get_signals_valid_input(self, mock_get_processor):
        """Test signals endpoint with valid input"""
        # Mock processor
        mock_processor = Mock()
        
        # Mock signal object with simple attributes
        class MockSignal:
            def __init__(self):
                self.signal_type = "buy"
                self.pattern = "doji"
                self.strength = 0.9
                self.confidence = 0.9
                self.price = 100.0
                self.timestamp = datetime(2023, 1, 1, 0, 0, 0)
                self.timeframe = "1h"
                self.stop_loss = 95.0
                self.take_profit = 105.0
                self.risk_reward_ratio = 2.0
                self.market_regime = "trending"
                self.volume_confirmation = True
                self.trend_alignment = True
                self.support_resistance_levels = {"support": 90, "resistance": 110}
                self.additional_indicators = {"rsi": 65, "macd": "bullish"}
        
        mock_processor.signal_history = {"BTC": [MockSignal()]}
        mock_get_processor.return_value = mock_processor
        
        response = self.client.get("/api/candlestick/signals/BTC?timeframe=1h&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "timeframe" in data
        assert "signals" in data
        assert len(data["signals"]) == 1
        assert data["signals"][0]["signal_type"] == "buy"
    
    def test_get_signals_invalid_symbol(self):
        """Test signals endpoint with invalid symbol"""
        response = self.client.get("/api/candlestick/signals/eth-usd?timeframe=1h")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid symbol format" in data["detail"]
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_get_signals_processing_error(self, mock_get_processor):
        """Test signals endpoint with processing error"""
        mock_get_processor.side_effect = ValueError("Invalid data")
        
        response = self.client.get("/api/candlestick/signals/BTC?timeframe=1h")
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
    
    @patch('routes.candlestick_analysis.get_processor')
    @patch('routes.candlestick_analysis.MLPatternDetector')
    def test_get_analysis_summary_valid_input(self, mock_ml_detector, mock_get_processor):
        """Test analysis summary endpoint with valid input"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.get_symbol_data.return_value = {
            'candlesticks': [{'timestamp': '2023-01-01T00:00:00Z', 'open': 100, 'high': 105, 'low': 95, 'close': 103, 'volume': 1000}],
            'processed': [Mock(open=100, high=105, low=95, close=103, volume=1000, indicators={'sma': 100})]
        }
        mock_processor._to_dataframe.return_value = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95], 'close': [103], 'volume': [1000]
        })
        mock_processor.signal_history = {"BTC": []}
        mock_processor.min_data_points = 50
        mock_get_processor.return_value = mock_processor
        
        # Mock pattern detector
        mock_pattern = Mock()
        mock_pattern.pattern = "doji"
        mock_pattern.type = "bullish"
        mock_pattern.market_regime = "trending"
        
        mock_ml_instance = Mock()
        mock_ml_instance.detect_patterns_ml.return_value = [mock_pattern]
        mock_ml_detector.return_value = mock_ml_instance
        
        response = self.client.get("/api/candlestick/analysis/BTC?timeframe=4h")
        
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "timeframe" in data
        assert "current_price" in data
        assert "pattern_summary" in data
        assert "signals" in data
    
    def test_get_analysis_summary_invalid_symbol(self):
        """Test analysis summary endpoint with invalid symbol"""
        response = self.client.get("/api/candlestick/analysis/ada-usd?timeframe=4h")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid symbol format" in data["detail"]
    
    def test_get_analysis_summary_invalid_timeframe(self):
        """Test analysis summary endpoint with invalid timeframe"""
        response = self.client.get("/api/candlestick/analysis/BTC?timeframe=2h")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid timeframe" in data["detail"]


class TestCandlestickAnalysisIntegration:
    """Integration tests for candlestick analysis"""
    
    def setup_method(self):
        """Setup test environment"""
        from fastapi import FastAPI
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
    
    @patch('routes.candlestick_analysis.get_processor')
    @patch('routes.candlestick_analysis.MLPatternDetector')
    def test_full_analysis_workflow(self, mock_ml_detector, mock_get_processor):
        """Test complete analysis workflow"""
        # Mock processor with realistic data
        mock_processor = Mock()
        
        # Mock status data
        mock_processor.get_processing_stats.return_value = {
            "total_processed": 1000,
            "avg_processing_time": 0.3,
            "cache_hit_rate": 0.85
        }
        mock_processor.candlestick_data = {"BTC": [], "ETH": [], "ADA": []}
        
        # Mock signal object with simple attributes
        class MockSignal:
            def __init__(self):
                self.signal_type = "buy"
                self.pattern = "hammer"
                self.strength = 0.8
                self.confidence = 0.8
                self.price = 46000.0
                self.timestamp = datetime.now()
                self.timeframe = "15m"
                self.stop_loss = 45000
                self.take_profit = 48000
                self.risk_reward_ratio = 2.0
                self.market_regime = "trending"
                self.volume_confirmation = True
                self.trend_alignment = True
                self.support_resistance_levels = {"support": 44000, "resistance": 49000}
                self.additional_indicators = {"rsi": 65, "macd": "bullish"}
        
        mock_processor.signal_history = {
            "BTC": [MockSignal()],
            "ETH": [],
            "ADA": []
        }
        
        # Mock analysis data
        class MockProcessedData:
            def __init__(self):
                self.open = 100
                self.high = 105
                self.low = 95
                self.close = 103
                self.volume = 1000
                self.indicators = {'sma': 100}
        
        mock_processor.get_symbol_data.return_value = {
            'candlesticks': [{'timestamp': '2023-01-01T00:00:00Z', 'open': 100, 'high': 105, 'low': 95, 'close': 103, 'volume': 1000}],
            'processed': [MockProcessedData()]
        }
        mock_processor._to_dataframe.return_value = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95], 'close': [103], 'volume': [1000]
        })
        mock_processor.min_data_points = 50
        
        mock_get_processor.return_value = mock_processor
        
        # Mock pattern detector
        class MockPattern:
            def __init__(self):
                self.pattern = "doji"
                self.type = "bullish"
                self.strength = 0.8
                self.ml_confidence = 0.75
                self.market_regime = "trending"
                self.timestamp = datetime.now()
                self.features = {"rsi": 65, "macd": "bullish"}
        
        class MockMLDetector:
            def detect_patterns_ml(self, df):
                return [MockPattern()]
        
        mock_ml_detector.return_value = MockMLDetector()
        
        # Test status endpoint
        status_response = self.client.get("/api/candlestick/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "active"
        assert status_data["total_signals"] == 1
        
        # Test patterns endpoint
        patterns_response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m&limit=10")
        assert patterns_response.status_code == 200
        patterns_data = patterns_response.json()
        assert len(patterns_data["patterns"]) == 1
        assert patterns_data["patterns"][0]["pattern"] == "doji"
        
        # Test signals endpoint
        signals_response = self.client.get("/api/candlestick/signals/BTC?timeframe=1h&limit=10")
        assert signals_response.status_code == 200
        signals_data = signals_response.json()
        assert len(signals_data["signals"]) == 1
        assert signals_data["signals"][0]["signal_type"] == "buy"
        
        # Test analysis endpoint
        analysis_response = self.client.get("/api/candlestick/analysis/BTC?timeframe=4h")
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        assert "symbol" in analysis_data
        assert "timeframe" in analysis_data
        assert "current_price" in analysis_data
    
    @patch('routes.candlestick_analysis.get_processor')
    def test_error_handling_scenarios(self, mock_get_processor):
        """Test various error handling scenarios"""
        # Test database connection error
        mock_get_processor.side_effect = ConnectionError("Database connection failed")
        
        response = self.client.get("/api/candlestick/status")
        assert response.status_code == 503
        
        # Test import error
        mock_get_processor.side_effect = ImportError("Module not found")
        
        response = self.client.get("/api/candlestick/status")
        assert response.status_code == 500
        
        # Test general exception
        mock_get_processor.side_effect = Exception("Unknown error")
        
        response = self.client.get("/api/candlestick/status")
        assert response.status_code == 500
    
    def test_input_validation_edge_cases(self):
        """Test input validation edge cases"""
        # Test empty symbol
        response = self.client.get("/api/candlestick/patterns/?timeframe=15m")
        assert response.status_code == 404  # FastAPI routing error
        
        # Test very long symbol
        long_symbol = "A" * 25
        response = self.client.get(f"/api/candlestick/patterns/{long_symbol}?timeframe=15m")
        assert response.status_code == 400
        
        # Test boundary limit values
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m&limit=0")
        assert response.status_code == 400
        
        response = self.client.get("/api/candlestick/patterns/BTC?timeframe=15m&limit=1001")
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
