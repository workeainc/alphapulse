#!/usr/bin/env python3
"""
Test suite for Enhanced Volume-Weighted Levels Analyzer
Tests the volume-weighted levels analysis with HVN/LVN detection
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

from src.strategies.enhanced_volume_weighted_levels_analyzer import (
    EnhancedVolumeWeightedLevelsAnalyzer,
    VolumeNodeType,
    VolumeNode,
    VolumeProfileAnalysis
)

class TestEnhancedVolumeWeightedLevelsAnalyzer:
    """Test cases for Enhanced Volume-Weighted Levels Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance for testing"""
        return EnhancedVolumeWeightedLevelsAnalyzer()
    
    @pytest.fixture
    def mock_db_pool(self):
        """Create a mock database pool"""
        pool = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = AsyncMock()
        return pool
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data with volume patterns"""
        data = []
        base_price = 47000
        
        # Create data with clear volume patterns
        for i in range(200):
            # Create volume spikes at certain price levels
            if i % 20 == 0:  # High volume every 20 candles
                volume = 5000 + np.random.uniform(0, 2000)
                price_variation = np.random.uniform(-200, 200)
            elif i % 15 == 0:  # Low volume every 15 candles
                volume = 200 + np.random.uniform(0, 300)
                price_variation = np.random.uniform(-100, 100)
            else:
                volume = 1000 + np.random.uniform(0, 500)
                price_variation = np.random.uniform(-50, 50)
            
            price = base_price + price_variation
            
            data.append({
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                'open': price,
                'high': price + np.random.uniform(0, 100),
                'low': price - np.random.uniform(0, 100),
                'close': price + np.random.uniform(-25, 25),
                'volume': volume
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.config['lookback_periods'] == 200
        assert analyzer.config['volume_profile_periods'] == 100
        assert analyzer.config['poc_threshold'] == 0.7
        assert analyzer.config['value_area_percentage'] == 0.68
        assert analyzer.config['hvn_threshold'] == 1.5
        assert analyzer.config['lvn_threshold'] == 0.5
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
    async def test_create_volume_profile(self, analyzer, sample_ohlcv_data):
        """Test creating volume profile"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = await analyzer._create_volume_profile(df, 'BTCUSDT', '1h')
        
        assert 'volume_distribution' in volume_profile
        assert 'total_volume' in volume_profile
        assert 'avg_volume_per_level' in volume_profile
        assert 'price_range' in volume_profile
        assert 'volume_range' in volume_profile
        assert volume_profile['total_volume'] > 0
        assert len(volume_profile['volume_distribution']) > 0
    
    @pytest.mark.asyncio
    async def test_detect_high_volume_nodes(self, analyzer, sample_ohlcv_data):
        """Test detecting high volume nodes"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {47000: 5000, 47500: 2000, 48000: 1000},
            'avg_volume_per_level': 2000,
            'total_volume': 8000
        }
        
        hvn_nodes = await analyzer._detect_high_volume_nodes(df, volume_profile)
        
        assert isinstance(hvn_nodes, list)
        for node in hvn_nodes:
            assert node.node_type == VolumeNodeType.HIGH_VOLUME_NODE
            assert node.price_level > 0
            assert node.volume_at_level > 0
            assert node.node_strength >= 0
            assert node.confidence >= 0
            assert node.volume_trend in ['increasing', 'decreasing', 'stable']
    
    @pytest.mark.asyncio
    async def test_detect_low_volume_nodes(self, analyzer, sample_ohlcv_data):
        """Test detecting low volume nodes"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {47000: 500, 47500: 2000, 48000: 1000},
            'avg_volume_per_level': 2000,
            'total_volume': 3500
        }
        
        lvn_nodes = await analyzer._detect_low_volume_nodes(df, volume_profile)
        
        assert isinstance(lvn_nodes, list)
        for node in lvn_nodes:
            assert node.node_type == VolumeNodeType.LOW_VOLUME_NODE
            assert node.price_level > 0
            assert node.volume_at_level > 0
            assert node.node_strength >= 0
            assert node.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_detect_volume_gaps(self, analyzer, sample_ohlcv_data):
        """Test detecting volume gaps"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {47000: 1000, 47500: 100, 48000: 2000},  # Gap at 47500
            'total_volume': 3100
        }
        
        volume_gaps = await analyzer._detect_volume_gaps(df, volume_profile)
        
        assert isinstance(volume_gaps, list)
        for gap in volume_gaps:
            assert gap.node_type == VolumeNodeType.VOLUME_GAP
            assert gap.price_level > 0
            assert gap.node_strength >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_point_of_control(self, analyzer, sample_ohlcv_data):
        """Test calculating Point of Control"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {47000: 1000, 47500: 5000, 48000: 2000},  # POC at 47500
            'total_volume': 8000,
            'avg_volume_per_level': 2666.67
        }
        
        poc_node = await analyzer._calculate_point_of_control(df, volume_profile)
        
        assert poc_node.node_type == VolumeNodeType.POINT_OF_CONTROL
        assert poc_node.price_level == 47500
        assert poc_node.volume_at_level == 5000
        assert poc_node.node_strength > 0
        assert poc_node.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_value_area(self, analyzer, sample_ohlcv_data):
        """Test calculating Value Area"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {
                47000: 1000, 47500: 5000, 48000: 2000, 48500: 1500, 49000: 500
            },
            'total_volume': 10000
        }
        
        value_area_nodes = await analyzer._calculate_value_area(df, volume_profile)
        
        assert isinstance(value_area_nodes, list)
        assert len(value_area_nodes) == 2  # High and Low
        
        vah_node = next((node for node in value_area_nodes if node.node_type == VolumeNodeType.VALUE_AREA_HIGH), None)
        val_node = next((node for node in value_area_nodes if node.node_type == VolumeNodeType.VALUE_AREA_LOW), None)
        
        assert vah_node is not None
        assert val_node is not None
        assert vah_node.price_level >= val_node.price_level
    
    @pytest.mark.asyncio
    async def test_detect_institutional_activity(self, analyzer, sample_ohlcv_data):
        """Test detecting institutional activity"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = {
            'volume_distribution': {47000: 1000, 47500: 15000, 48000: 2000},  # Institutional at 47500
            'total_volume': 18000,
            'avg_volume_per_level': 6000
        }
        
        institutional_nodes = await analyzer._detect_institutional_activity(df, volume_profile)
        
        assert isinstance(institutional_nodes, list)
        for node in institutional_nodes:
            assert node.institutional_activity is True
            assert node.volume_at_level >= volume_profile['avg_volume_per_level'] * analyzer.config['institutional_threshold']
    
    @pytest.mark.asyncio
    async def test_calculate_node_confidence(self, analyzer, sample_ohlcv_data):
        """Test calculating node confidence"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        confidence = await analyzer._calculate_node_confidence(df, 47000, 5000)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_analyze_volume_trend(self, analyzer, sample_ohlcv_data):
        """Test analyzing volume trend"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        trend = await analyzer._analyze_volume_trend(df, 47000)
        
        assert trend in ['increasing', 'decreasing', 'stable']
    
    @pytest.mark.asyncio
    async def test_calculate_price_efficiency(self, analyzer, sample_ohlcv_data):
        """Test calculating price efficiency"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        efficiency = await analyzer._calculate_price_efficiency(df, 47000)
        
        assert 0 <= efficiency <= 1
        assert isinstance(efficiency, float)
    
    @pytest.mark.asyncio
    async def test_count_node_touches(self, analyzer, sample_ohlcv_data):
        """Test counting node touches"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        touch_count = await analyzer._count_node_touches(df, 47000)
        
        assert touch_count >= 0
        assert isinstance(touch_count, int)
    
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
    async def test_validate_and_score_nodes(self, analyzer, sample_ohlcv_data):
        """Test validating and scoring nodes"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        test_nodes = [
            VolumeNode(
                node_type=VolumeNodeType.HIGH_VOLUME_NODE,
                price_level=47000,
                volume_at_level=5000,
                volume_percentage=50.0,
                node_strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            ),
            VolumeNode(
                node_type=VolumeNodeType.HIGH_VOLUME_NODE,
                price_level=48000,
                volume_at_level=2000,
                volume_percentage=20.0,
                node_strength=0.2,  # Below threshold
                confidence=0.3,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        validated_nodes = await analyzer._validate_and_score_nodes(df, test_nodes)
        
        assert len(validated_nodes) == 1  # Only the first node should pass
        assert validated_nodes[0].price_level == 47000
        assert validated_nodes[0].touch_count >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_analysis_confidence(self, analyzer, sample_ohlcv_data):
        """Test calculating analysis confidence"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        test_nodes = [
            VolumeNode(
                node_type=VolumeNodeType.HIGH_VOLUME_NODE,
                price_level=47000,
                volume_at_level=5000,
                volume_percentage=50.0,
                node_strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        confidence = await analyzer._calculate_analysis_confidence(df, test_nodes)
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    @pytest.mark.asyncio
    async def test_prepare_algorithm_inputs(self, analyzer):
        """Test preparing algorithm inputs"""
        test_nodes = [
            VolumeNode(
                node_type=VolumeNodeType.HIGH_VOLUME_NODE,
                price_level=47000,
                volume_at_level=5000,
                volume_percentage=50.0,
                node_strength=0.8,
                confidence=0.7,
                timestamp=datetime.now(timezone.utc),
                institutional_activity=True,
                price_efficiency=0.8
            ),
            VolumeNode(
                node_type=VolumeNodeType.LOW_VOLUME_NODE,
                price_level=48000,
                volume_at_level=1000,
                volume_percentage=10.0,
                node_strength=0.6,
                confidence=0.6,
                timestamp=datetime.now(timezone.utc),
                institutional_activity=False,
                price_efficiency=0.5
            )
        ]
        
        volume_profile = {
            'total_volume': 10000
        }
        
        inputs = await analyzer._prepare_algorithm_inputs(test_nodes, volume_profile)
        
        assert 'volume_profile' in inputs
        assert 'volume_nodes' in inputs
        assert 'high_volume_nodes' in inputs
        assert 'low_volume_nodes' in inputs
        assert 'volume_gaps' in inputs
        assert 'institutional_nodes' in inputs
        assert 'total_nodes' in inputs
        assert 'active_nodes' in inputs
        assert inputs['total_nodes'] == 2
        assert inputs['active_nodes'] == 2
    
    @pytest.mark.asyncio
    async def test_analyze_volume_weighted_levels(self, analyzer, mock_db_pool, sample_ohlcv_data):
        """Test complete volume-weighted levels analysis"""
        analyzer.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = sample_ohlcv_data
        mock_conn.execute.return_value = "INSERT 0 1"
        
        analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, VolumeProfileAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.timeframe == '1h'
        assert analysis.poc_price > 0
        assert analysis.poc_volume > 0
        assert analysis.total_volume > 0
        assert isinstance(analysis.high_volume_nodes, list)
        assert isinstance(analysis.low_volume_nodes, list)
        assert isinstance(analysis.volume_gaps, list)
        assert 0 <= analysis.analysis_confidence <= 1
        assert isinstance(analysis.algorithm_inputs, dict)
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, analyzer, mock_db_pool):
        """Test handling of insufficient data"""
        analyzer.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []  # Empty data
        
        analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, VolumeProfileAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.poc_price == 0.0
        assert analysis.total_volume == 0.0
        assert analysis.analysis_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling"""
        # Test with None database pool
        analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
        
        assert isinstance(analysis, VolumeProfileAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analyzer.stats['failed_analyses'] == 1

# Integration tests
class TestEnhancedVolumeWeightedLevelsAnalyzerIntegration:
    """Integration tests for Enhanced Volume-Weighted Levels Analyzer"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test the complete analysis workflow"""
        analyzer = EnhancedVolumeWeightedLevelsAnalyzer()
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create realistic mock data with clear volume patterns
            ohlcv_data = []
            
            for i in range(200):
                # Create clear volume patterns
                if i % 20 == 0:  # High volume nodes
                    volume = 8000
                    price = 47000 + (i // 20) * 1000
                elif i % 15 == 0:  # Low volume nodes
                    volume = 300
                    price = 47500 + (i // 15) * 500
                else:
                    volume = 1500
                    price = 47000 + i * 10
                
                ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': price,
                    'high': price + 50,
                    'low': price - 50,
                    'close': price + np.random.uniform(-25, 25),
                    'volume': volume
                })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = ohlcv_data
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Run analysis
            await analyzer.initialize()
            analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
            
            # Verify results
            assert isinstance(analysis, VolumeProfileAnalysis)
            assert analysis.symbol == 'BTCUSDT'
            assert analysis.poc_price > 0
            assert analysis.poc_volume > 0
            assert analysis.total_volume > 0
            assert analysis.analysis_confidence > 0
            assert len(analysis.high_volume_nodes) > 0
            assert len(analysis.low_volume_nodes) > 0
            
            # Check algorithm inputs
            assert 'volume_profile' in analysis.algorithm_inputs
            assert 'high_volume_nodes' in analysis.algorithm_inputs
            assert 'low_volume_nodes' in analysis.algorithm_inputs
            
            await analyzer.close()

# Performance tests
class TestEnhancedVolumeWeightedLevelsAnalyzerPerformance:
    """Performance tests for Enhanced Volume-Weighted Levels Analyzer"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        analyzer = EnhancedVolumeWeightedLevelsAnalyzer()
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create large dataset
            large_ohlcv_data = []
            
            for i in range(1000):
                # Create varied volume patterns
                volume = 1000 + np.random.uniform(0, 5000)
                price = 47000 + i * 5 + np.random.uniform(-100, 100)
                
                large_ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': price,
                    'high': price + np.random.uniform(0, 200),
                    'low': price - np.random.uniform(0, 200),
                    'close': price + np.random.uniform(-50, 50),
                    'volume': volume
                })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.return_value = large_ohlcv_data
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            start_time = time.time()
            
            await analyzer.initialize()
            analysis = await analyzer.analyze_volume_weighted_levels('BTCUSDT', '1h')
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Performance assertions
            assert elapsed_time < 15.0  # Should complete within 15 seconds
            assert isinstance(analysis, VolumeProfileAnalysis)
            assert analysis.total_volume > 0
            
            await analyzer.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
