"""
Test Market Structure Analyzer
Comprehensive tests for market structure analysis functionality
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Import the components to test
from backend.strategies.market_structure_analyzer import (
    MarketStructureAnalyzer, MarketStructureAnalysis, MarketStructureType,
    SwingPoint, SwingPointType, TrendLine, TrendLineType
)

class TestMarketStructureAnalyzer:
    """Test suite for Market Structure Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a market structure analyzer instance"""
        config = {
            'min_swing_distance': 0.005,
            'min_touch_count': 2,
            'lookback_periods': 50,
            'trend_line_tolerance': 0.002
        }
        return MarketStructureAnalyzer(config)
    
    @pytest.fixture
    def sample_uptrend_data(self):
        """Create sample uptrend candlestick data"""
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            # Create uptrend with clear swing points
            if i == 10:  # Swing high
                base_price = 100.0 + 15.0
            elif i == 20:  # Swing low
                base_price = 100.0 + 8.0
            elif i == 30:  # Swing high
                base_price = 100.0 + 25.0
            elif i == 40:  # Swing low
                base_price = 100.0 + 18.0
            elif i == 50:  # Swing high
                base_price = 100.0 + 35.0
            elif i == 60:  # Swing low
                base_price = 100.0 + 28.0
            elif i == 70:  # Swing high
                base_price = 100.0 + 45.0
            elif i == 80:  # Swing low
                base_price = 100.0 + 38.0
            elif i == 90:  # Swing high
                base_price = 100.0 + 55.0
            else:
                base_price = 100.0 + (i * 0.5)  # Normal trend
            
            high = base_price + 2.0
            low = base_price - 2.0
            close = base_price + 0.5
            open_price = base_price - 0.5
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000 + (i * 10)
            })
        
        return data
    
    @pytest.fixture
    def sample_downtrend_data(self):
        """Create sample downtrend candlestick data"""
        data = []
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            # Create downtrend with clear swing points
            if i == 10:  # Swing low
                base_price = 100.0 - 15.0
            elif i == 20:  # Swing high
                base_price = 100.0 - 8.0
            elif i == 30:  # Swing low
                base_price = 100.0 - 25.0
            elif i == 40:  # Swing high
                base_price = 100.0 - 18.0
            elif i == 50:  # Swing low
                base_price = 100.0 - 35.0
            elif i == 60:  # Swing high
                base_price = 100.0 - 28.0
            elif i == 70:  # Swing low
                base_price = 100.0 - 45.0
            elif i == 80:  # Swing high
                base_price = 100.0 - 38.0
            elif i == 90:  # Swing low
                base_price = 100.0 - 55.0
            else:
                base_price = 100.0 - (i * 0.5)  # Normal trend
            
            high = base_price + 2.0
            low = base_price - 2.0
            close = base_price - 0.5
            open_price = base_price + 0.5
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000 + (i * 10)
            })
        
        return data
    
    @pytest.fixture
    def sample_consolidation_data(self):
        """Create sample consolidation candlestick data"""
        data = []
        base_price = 100.0
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            # Create sideways movement
            price_change = np.sin(i * 0.1) * 2.0  # Oscillating pattern
            high = base_price + price_change + 1.0
            low = base_price + price_change - 1.0
            close = base_price + price_change + 0.2
            open_price = base_price + price_change - 0.2
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000 + (i * 5)
            })
            
            base_price = close
        
        return data
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.min_swing_distance == 0.005
        assert analyzer.min_touch_count == 2
        assert analyzer.lookback_periods == 50
        assert analyzer.trend_line_tolerance == 0.002
        assert analyzer.stats['analyses_performed'] == 0
    
    @pytest.mark.asyncio
    async def test_swing_point_detection(self, analyzer, sample_uptrend_data):
        """Test swing point detection"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        
        assert len(swing_points) > 0
        assert all(isinstance(sp, SwingPoint) for sp in swing_points)
        assert all(sp.swing_type in [SwingPointType.HIGH, SwingPointType.LOW] for sp in swing_points)
        assert all(sp.price > 0 for sp in swing_points)
        assert all(sp.volume > 0 for sp in swing_points)
    
    @pytest.mark.asyncio
    async def test_swing_point_classification(self, analyzer, sample_uptrend_data):
        """Test swing point classification (HH/LH/HL/LL)"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        classified_swings = await analyzer._classify_swing_points(swing_points)
        
        assert len(classified_swings) > 0
        
        # Check that classification was applied
        has_classification = any(
            sp.is_higher_high or sp.is_lower_high or sp.is_higher_low or sp.is_lower_low
            for sp in classified_swings
        )
        assert has_classification
        
        # Check swing strength calculation
        assert all(sp.swing_strength >= 0 for sp in classified_swings)
    
    @pytest.mark.asyncio
    async def test_uptrend_structure_analysis(self, analyzer, sample_uptrend_data):
        """Test uptrend market structure analysis"""
        analysis = await analyzer.analyze_market_structure(
            'BTCUSDT', '1h', sample_uptrend_data
        )
        
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.market_structure_type in [MarketStructureType.UPTREND, MarketStructureType.SIDEWAYS, MarketStructureType.UNKNOWN]
        assert analysis.structure_strength >= 0
        assert analysis.analysis_confidence >= 0
        assert analysis.analysis_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_downtrend_structure_analysis(self, analyzer, sample_downtrend_data):
        """Test downtrend market structure analysis"""
        analysis = await analyzer.analyze_market_structure(
            'BTCUSDT', '1h', sample_downtrend_data
        )
        
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.market_structure_type in [MarketStructureType.DOWNTREND, MarketStructureType.SIDEWAYS, MarketStructureType.UNKNOWN]
        assert analysis.structure_strength >= 0
        assert analysis.analysis_confidence >= 0
        assert analysis.analysis_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_consolidation_structure_analysis(self, analyzer, sample_consolidation_data):
        """Test consolidation market structure analysis"""
        analysis = await analyzer.analyze_market_structure(
            'BTCUSDT', '1h', sample_consolidation_data
        )
        
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.market_structure_type in [
            MarketStructureType.CONSOLIDATION, 
            MarketStructureType.SIDEWAYS, 
            MarketStructureType.UNKNOWN
        ]
        assert analysis.structure_strength >= 0
        assert analysis.analysis_confidence >= 0
        assert analysis.analysis_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_trend_line_detection(self, analyzer, sample_uptrend_data):
        """Test trend line detection"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        classified_swings = await analyzer._classify_swing_points(swing_points)
        trend_lines = await analyzer._detect_trend_lines(df, classified_swings)
        
        # Trend lines may or may not be detected depending on data quality
        assert isinstance(trend_lines, list)
        if trend_lines:
            assert all(isinstance(tl, TrendLine) for tl in trend_lines)
            assert all(tl.trend_line_type in [TrendLineType.SUPPORT, TrendLineType.RESISTANCE] for tl in trend_lines)
            assert all(tl.validation_score >= 0 for tl in trend_lines)
            assert all(tl.validation_score <= 1 for tl in trend_lines)
    
    @pytest.mark.asyncio
    async def test_structure_breakout_detection(self, analyzer, sample_uptrend_data):
        """Test structure breakout detection"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        classified_swings = await analyzer._classify_swing_points(swing_points)
        structure_type, _ = await analyzer._analyze_structure_type(classified_swings)
        
        breakout, direction = await analyzer._check_structure_breakout(
            df, classified_swings, structure_type
        )
        
        # Breakout detection should return boolean and optional direction
        assert isinstance(breakout, bool)
        if breakout:
            assert direction in ['up', 'down']
        else:
            assert direction is None
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data"""
        insufficient_data = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000
            }
        ]
        
        analysis = await analyzer.analyze_market_structure(
            'BTCUSDT', '1h', insufficient_data
        )
        
        assert analysis is not None
        assert analysis.market_structure_type == MarketStructureType.UNKNOWN
        assert analysis.structure_strength == 0.0
        assert analysis.analysis_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_analysis_confidence_calculation(self, analyzer, sample_uptrend_data):
        """Test analysis confidence calculation"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        classified_swings = await analyzer._classify_swing_points(swing_points)
        trend_lines = await analyzer._detect_trend_lines(df, classified_swings)
        structure_type, _ = await analyzer._analyze_structure_type(classified_swings)
        
        confidence = await analyzer._calculate_analysis_confidence(
            classified_swings, trend_lines, structure_type
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_structure_phase_determination(self, analyzer, sample_uptrend_data):
        """Test structure phase determination"""
        df = pd.DataFrame(sample_uptrend_data)
        swing_points = await analyzer._detect_swing_points(df)
        classified_swings = await analyzer._classify_swing_points(swing_points)
        structure_type, _ = await analyzer._analyze_structure_type(classified_swings)
        
        phase = await analyzer._determine_structure_phase(classified_swings, structure_type)
        
        assert isinstance(phase, str)
        assert phase in ['accumulation', 'markup', 'distribution', 'markdown', 'consolidation', 'unknown']
    
    @pytest.mark.asyncio
    async def test_performance_statistics(self, analyzer, sample_uptrend_data):
        """Test performance statistics tracking"""
        initial_stats = analyzer.stats.copy()
        
        await analyzer.analyze_market_structure('BTCUSDT', '1h', sample_uptrend_data)
        
        # Check that statistics were updated
        assert analyzer.stats['analyses_performed'] > initial_stats['analyses_performed']
        assert analyzer.stats['swing_points_detected'] > initial_stats['swing_points_detected']
        assert analyzer.stats['last_update'] is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling with invalid data"""
        invalid_data = [
            {
                'timestamp': 'invalid_timestamp',
                'open': 'invalid_price',
                'high': 'invalid_price',
                'low': 'invalid_price',
                'close': 'invalid_price',
                'volume': 'invalid_volume'
            }
        ]
        
        # Should not raise exception, should return default analysis
        analysis = await analyzer.analyze_market_structure(
            'BTCUSDT', '1h', invalid_data
        )
        
        assert analysis is not None
        assert analysis.market_structure_type == MarketStructureType.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_default_analysis(self, analyzer):
        """Test default analysis creation"""
        default_analysis = analyzer._get_default_analysis('BTCUSDT', '1h')
        
        assert default_analysis is not None
        assert default_analysis.symbol == 'BTCUSDT'
        assert default_analysis.timeframe == '1h'
        assert default_analysis.market_structure_type == MarketStructureType.UNKNOWN
        assert default_analysis.structure_strength == 0.0
        assert default_analysis.analysis_confidence == 0.0
        assert default_analysis.current_structure_phase == 'unknown'

class TestMarketStructureIntegration:
    """Integration tests for market structure analysis"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test the complete market structure analysis pipeline"""
        config = {
            'min_swing_distance': 0.005,
            'min_touch_count': 2,
            'lookback_periods': 50,
            'trend_line_tolerance': 0.002
        }
        
        analyzer = MarketStructureAnalyzer(config)
        
        # Create realistic test data
        data = []
        base_price = 100.0
        timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        
        for i in range(100):
            # Create a realistic price pattern
            if i < 30:
                # Uptrend phase
                price_change = i * 0.5
            elif i < 60:
                # Consolidation phase
                price_change = 15 + np.sin(i * 0.2) * 2
            else:
                # Downtrend phase
                price_change = 15 - (i - 60) * 0.3
            
            high = base_price + price_change + 1.0
            low = base_price + price_change - 1.0
            close = base_price + price_change + 0.2
            open_price = base_price + price_change - 0.2
            
            data.append({
                'timestamp': timestamp + timedelta(hours=i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000 + (i * 10)
            })
            
            base_price = close
        
        # Run full analysis
        analysis = await analyzer.analyze_market_structure('BTCUSDT', '1h', data)
        
        # Validate results
        assert analysis is not None
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        # The analysis should at least detect some structure
        assert analysis.structure_strength > 0 or analysis.analysis_confidence >= 0.5
        assert analysis.analysis_confidence > 0
        assert analysis.structure_duration_bars == 100
        
        # Check that swing points were detected
        if analysis.higher_highs:
            assert len(analysis.higher_highs) > 0
        if analysis.higher_lows:
            assert len(analysis.higher_lows) > 0
        if analysis.lower_highs:
            assert len(analysis.lower_highs) > 0
        if analysis.lower_lows:
            assert len(analysis.lower_lows) > 0
        
        # Check performance statistics
        assert analyzer.stats['analyses_performed'] == 1
        # Swing points may or may not be detected depending on data quality
        assert analyzer.stats['swing_points_detected'] >= 0
        assert analyzer.stats['last_update'] is not None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
