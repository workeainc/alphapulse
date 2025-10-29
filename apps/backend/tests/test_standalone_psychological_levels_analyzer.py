#!/usr/bin/env python3
"""
Test suite for Standalone Psychological Levels Analyzer
Tests the psychological levels analysis functionality
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.standalone_psychological_levels_analyzer import (
    StandalonePsychologicalLevelsAnalyzer,
    PsychologicalLevelType,
    PsychologicalLevel,
    PsychologicalAnalysis
)

class TestStandalonePsychologicalLevelsAnalyzer:
    """Test cases for Standalone Psychological Levels Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance for testing"""
        return StandalonePsychologicalLevelsAnalyzer()
    
    @pytest.fixture
    def mock_db_pool(self):
        """Create a mock database pool"""
        pool = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = AsyncMock()
        return pool
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data with psychological levels"""
        data = []
        base_price = 47000
        
        # Create data that touches psychological levels
        psychological_levels = [47000, 47500, 48000, 48500, 49000, 50000]
        
        for i in range(200):
            # Vary price around psychological levels
            level_index = i % len(psychological_levels)
            base_level = psychological_levels[level_index]
            
            # Add some variation around the level
            variation = np.random.uniform(-100, 100)
            price = base_level + variation
            
            data.append({
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 200),
                'low': price - np.random.uniform(0, 200),
                'close': price + np.random.uniform(-50, 50),
                'volume': 1000 + np.random.uniform(0, 500)
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.config['lookback_periods'] == 200
        assert analyzer.config['min_touch_count'] == 2
        assert analyzer.config['level_tolerance'] == 0.002
        assert analyzer.config['strength_threshold'] == 0.3
        assert analyzer.stats['total_analyses'] == 0
    
    @pytest.mark.asyncio
    async def test_get_recent_ohlcv_data(self, analyzer, mock_db_pool):
        """Test getting recent OHLCV data"""
        analyzer.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 47000,
                'high': 48000,
                'low': 46000,
                'close': 47500,
                'volume': 1000
            }
        ]
        
        data = await analyzer._get_recent_ohlcv_data('BTCUSDT', '1h')
        
        assert len(data) == 1
        assert data[0]['close'] == 47500
        mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_round_number_levels(self, analyzer, sample_ohlcv_data):
        """Test detecting round number levels"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        levels = await analyzer._detect_round_number_levels(df, 'BTCUSDT')
        
        assert isinstance(levels, list)
        # Should detect some round number levels
        for level in levels:
            assert level.level_type == PsychologicalLevelType.ROUND_NUMBER
            assert level.price_level > 0
            assert level.strength >= 0
            assert level.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_detect_fibonacci_levels(self, analyzer, sample_ohlcv_data):
        """Test detecting Fibonacci levels"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        levels = await analyzer._detect_fibonacci_levels(df, 'BTCUSDT')
        
        assert isinstance(levels, list)
        # Should detect some Fibonacci levels
        for level in levels:
            assert level.level_type in [PsychologicalLevelType.FIBONACCI_RETRACEMENT, 
                                      PsychologicalLevelType.FIBONACCI_EXTENSION]
            assert level.price_level > 0
            assert level.strength >= 0
            assert level.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_detect_milestone_levels(self, analyzer, sample_ohlcv_data):
        """Test detecting milestone levels"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        levels = await analyzer._detect_milestone_levels(df, 'BTCUSDT')
        
        assert isinstance(levels, list)
        # Should detect some milestone levels
        for level in levels:
            assert level.level_type == PsychologicalLevelType.PRICE_MILESTONE
            assert level.price_level > 0
            assert level.strength >= 0
            assert level.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_detect_percentage_levels(self, analyzer, sample_ohlcv_data):
        """Test detecting percentage levels"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        levels = await analyzer._detect_percentage_levels(df, 'BTCUSDT')
        
        assert isinstance(levels, list)
        # Should detect some percentage levels
        for level in levels:
            assert level.level_type == PsychologicalLevelType.PERCENTAGE_LEVEL
            assert level.price_level > 0
            assert level.strength >= 0
            assert level.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_count_level_touches(self, analyzer, sample_ohlcv_data):
        """Test counting level touches"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        # Test with a level that should have touches
        touch_count = await analyzer._count_level_touches(df, 47000)
        
        assert touch_count >= 0
        assert isinstance(touch_count, int)
    
    @pytest.mark.asyncio
    async def test_calculate_level_strength(self, analyzer, sample_ohlcv_data):
        """Test calculating level strength"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        strength = await analyzer._calculate_level_strength(df, 47000, 5)
        
        assert 0 <= strength <= 1
        assert isinstance(strength, float)
    
    @pytest.mark.asyncio
    async def test_calculate_level_confidence(self, analyzer, sample_ohlcv_data):
        """Test calculating level confidence"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        confidence = await analyzer._calculate_level_confidence(df, 47000, 5)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_calculate_volume_strength(self, analyzer, sample_ohlcv_data):
        """Test calculating volume strength"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        strength = await analyzer._calculate_volume_strength(df, 47000)
        
        assert strength >= 0
        assert isinstance(strength, float)
    
    @pytest.mark.asyncio
    async def test_calculate_time_strength(self, analyzer, sample_ohlcv_data):
        """Test calculating time strength"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        strength = await analyzer._calculate_time_strength(df, 47000)
        
        assert 0 <= strength <= 1
        assert isinstance(strength, float)
    
    @pytest.mark.asyncio
    async def test_calculate_level_consistency(self, analyzer, sample_ohlcv_data):
        """Test calculating level consistency"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        consistency = await analyzer._calculate_level_consistency(df, 47000)
        
        assert 0 <= consistency <= 1
        assert isinstance(consistency, float)
    
    @pytest.mark.asyncio
    async def test_get_first_touch_time(self, analyzer, sample_ohlcv_data):
        """Test getting first touch time"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        first_touch = await analyzer._get_first_touch_time(df, 47000)
        
        if first_touch:
            assert isinstance(first_touch, datetime)
    
    @pytest.mark.asyncio
    async def test_get_last_touch_time(self, analyzer, sample_ohlcv_data):
        """Test getting last touch time"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        last_touch = await analyzer._get_last_touch_time(df, 47000)
        
        if last_touch:
            assert isinstance(last_touch, datetime)
    
    @pytest.mark.asyncio
    async def test_get_market_context(self, analyzer, sample_ohlcv_data):
        """Test getting market context"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        context = analyzer._get_market_context(df, 47000)
        
        assert context in ['support', 'resistance', 'current', 'unknown']
    
    @pytest.mark.asyncio
    async def test_validate_and_score_levels(self, analyzer, sample_ohlcv_data):
        """Test validating and scoring levels"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        # Create test levels
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            ),
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=48000,
                strength=0.2,  # Below threshold
                confidence=0.3,  # Below threshold
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        validated_levels = await analyzer._validate_and_score_levels(df, test_levels)
        
        assert len(validated_levels) == 1  # Only the first level should pass
        assert validated_levels[0].price_level == 47000
    
    @pytest.mark.asyncio
    async def test_find_nearest_levels(self, analyzer):
        """Test finding nearest support and resistance levels"""
        current_price = 47500
        
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,  # Support
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            ),
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=48000,  # Resistance
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            ),
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=46000,  # Further support
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        nearest_support, nearest_resistance = await analyzer._find_nearest_levels(current_price, test_levels)
        
        assert nearest_support is not None
        assert nearest_resistance is not None
        assert nearest_support.price_level == 47000
        assert nearest_resistance.price_level == 48000
    
    @pytest.mark.asyncio
    async def test_analyze_level_interactions(self, analyzer, sample_ohlcv_data):
        """Test analyzing level interactions"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        interactions = await analyzer._analyze_level_interactions(df, test_levels)
        
        assert isinstance(interactions, list)
        # Each interaction should have required fields
        for interaction in interactions:
            assert 'level_price' in interaction
            assert 'level_type' in interaction
            assert 'timestamp' in interaction
            assert 'price' in interaction
            assert 'volume' in interaction
            assert 'reaction_type' in interaction
    
    @pytest.mark.asyncio
    async def test_determine_market_regime(self, analyzer, sample_ohlcv_data):
        """Test determining market regime"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        regime = await analyzer._determine_market_regime(df, test_levels)
        
        assert regime in ['trending_up', 'trending_down', 'ranging', 'volatile', 'unknown']
    
    @pytest.mark.asyncio
    async def test_calculate_analysis_confidence(self, analyzer, sample_ohlcv_data):
        """Test calculating analysis confidence"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        confidence = await analyzer._calculate_analysis_confidence(df, test_levels)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_prepare_algorithm_inputs(self, analyzer):
        """Test preparing algorithm inputs"""
        test_levels = [
            PsychologicalLevel(
                level_type=PsychologicalLevelType.ROUND_NUMBER,
                price_level=47000,
                strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        nearest_support = test_levels[0]
        nearest_resistance = PsychologicalLevel(
            level_type=PsychologicalLevelType.ROUND_NUMBER,
            price_level=48000,
            strength=0.8,
            confidence=0.7,
            timestamp=datetime.now(timezone.utc)
        )
        
        inputs = await analyzer._prepare_algorithm_inputs(test_levels, nearest_support, nearest_resistance, 'trending_up')
        
        assert 'psychological_levels' in inputs
        assert 'nearest_support' in inputs
        assert 'nearest_resistance' in inputs
        assert 'market_regime' in inputs
        assert 'total_levels' in inputs
        assert 'active_levels' in inputs
        assert inputs['market_regime'] == 'trending_up'
    
    @pytest.mark.asyncio
    async def test_analyze_psychological_levels(self, analyzer, mock_db_pool, sample_ohlcv_data):
        """Test complete psychological levels analysis"""
        analyzer.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = sample_ohlcv_data
        mock_conn.execute.return_value = "INSERT 0 1"
        
        analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, PsychologicalAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.current_price > 0
        assert isinstance(analysis.psychological_levels, list)
        assert analysis.market_regime in ['trending_up', 'trending_down', 'ranging', 'volatile', 'unknown']
        assert 0 <= analysis.analysis_confidence <= 1
        assert isinstance(analysis.algorithm_inputs, dict)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer, mock_db_pool):
        """Test handling of insufficient data"""
        analyzer.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []  # Empty data
        
        analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, PsychologicalAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.current_price == 0.0
        assert len(analysis.psychological_levels) == 0
        assert analysis.analysis_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling"""
        # Test with None database pool
        analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, PsychologicalAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analyzer.stats['failed_analyses'] == 1

# Integration tests
class TestStandalonePsychologicalLevelsAnalyzerIntegration:
    """Integration tests for Standalone Psychological Levels Analyzer"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test the complete analysis workflow"""
        analyzer = StandalonePsychologicalLevelsAnalyzer()
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create realistic mock data with clear psychological levels
            ohlcv_data = []
            psychological_levels = [47000, 47500, 48000, 48500, 49000, 50000]
            
            for i in range(200):
                level_index = i % len(psychological_levels)
                base_level = psychological_levels[level_index]
                
                # Create data that clearly touches psychological levels
                high = base_level + 50
                low = base_level - 50
                close = base_level + np.random.uniform(-25, 25)
                
                ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': base_level,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': 1000 + i * 10
                })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = ohlcv_data
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Run analysis
            await analyzer.initialize()
            analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
            
            # Verify results
            assert isinstance(analysis, PsychologicalAnalysis)
            assert analysis.symbol == 'BTCUSDT'
            assert analysis.current_price > 0
            assert len(analysis.psychological_levels) > 0
            assert analysis.analysis_confidence > 0
            assert analysis.market_regime != 'unknown'
            
            # Check that we detected different types of levels
            level_types = set(level.level_type for level in analysis.psychological_levels)
            assert len(level_types) > 0
            
            await analyzer.close()

# Performance tests
class TestStandalonePsychologicalLevelsAnalyzerPerformance:
    """Performance tests for Standalone Psychological Levels Analyzer"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        analyzer = StandalonePsychologicalLevelsAnalyzer()
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create large dataset
            large_ohlcv_data = []
            psychological_levels = [47000, 47500, 48000, 48500, 49000, 50000]
            
            for i in range(1000):
                level_index = i % len(psychological_levels)
                base_level = psychological_levels[level_index]
                
                large_ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': base_level + np.random.uniform(-100, 100),
                    'high': base_level + np.random.uniform(0, 200),
                    'low': base_level - np.random.uniform(0, 200),
                    'close': base_level + np.random.uniform(-50, 50),
                    'volume': 1000 + i * 10
                })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = large_ohlcv_data
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            start_time = time.time()
            
            await analyzer.initialize()
            analysis = await analyzer.analyze_psychological_levels('BTCUSDT', '1h')
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Performance assertions
            assert elapsed_time < 10.0  # Should complete within 10 seconds
            assert isinstance(analysis, PsychologicalAnalysis)
            assert analysis.current_price > 0
            
            await analyzer.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
