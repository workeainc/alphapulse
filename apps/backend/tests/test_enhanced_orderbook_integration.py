#!/usr/bin/env python3
"""
Test suite for Enhanced Order Book Integration
Tests the order book integration with volume profile analysis
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

from src.services.enhanced_orderbook_integration import (
    EnhancedOrderBookIntegration,
    OrderBookLevelType,
    VolumeProfile,
    OrderBookAnalysis
)

class TestEnhancedOrderBookIntegration:
    """Test cases for Enhanced Order Book Integration"""
    
    @pytest.fixture
    def integration(self):
        """Create an integration instance for testing"""
        return EnhancedOrderBookIntegration()
    
    @pytest.fixture
    def mock_db_pool(self):
        """Create a mock database pool"""
        pool = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = AsyncMock()
        return pool
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data"""
        return [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                'open': 47000 + i,
                'high': 48000 + i,
                'low': 46000 + i,
                'close': 47500 + i,
                'volume': 1000 + i * 10
            }
            for i in range(100)
        ]
    
    @pytest.fixture
    def sample_order_book_data(self):
        """Create sample order book data"""
        return [
            {
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                'bids': [[47000 - i, 100], [46900 - i, 200], [46800 - i, 150]],
                'asks': [[48000 + i, 120], [48100 + i, 180], [48200 + i, 160]],
                'best_bid': 47000 - i,
                'best_ask': 48000 + i,
                'spread': 1000
            }
            for i in range(10)
        ]
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, integration):
        """Test integration initialization"""
        assert integration.config['volume_profile_periods'] == 100
        assert integration.config['poc_threshold'] == 0.7
        assert integration.config['value_area_percentage'] == 0.68
        assert integration.stats['total_analyses'] == 0
    
    @pytest.mark.asyncio
    async def test_get_recent_ohlcv_data(self, integration, mock_db_pool):
        """Test getting recent OHLCV data"""
        integration.db_pool = mock_db_pool
        
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
        
        data = await integration._get_recent_ohlcv_data('BTCUSDT', '1m')
        
        assert len(data) == 1
        assert data[0]['symbol'] is None  # Not set in this method
        mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_recent_order_book_data(self, integration, mock_db_pool):
        """Test getting recent order book data"""
        integration.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = [
            {
                'timestamp': datetime.now(timezone.utc),
                'bids': [[47000, 100], [46900, 200]],
                'asks': [[48000, 120], [48100, 180]],
                'best_bid': 47000,
                'best_ask': 48000,
                'spread': 1000
            }
        ]
        
        data = await integration._get_recent_order_book_data('BTCUSDT')
        
        assert len(data) == 1
        assert data[0]['best_bid'] == 47000
        assert data[0]['best_ask'] == 48000
        mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_volume_profile(self, integration, sample_ohlcv_data):
        """Test creating volume profile"""
        df = pd.DataFrame(sample_ohlcv_data)
        
        volume_profile = await integration._create_volume_profile(df, 'BTCUSDT', '1m')
        
        assert 'volume_distribution' in volume_profile
        assert 'total_volume' in volume_profile
        assert 'avg_volume_per_level' in volume_profile
        assert volume_profile['total_volume'] > 0
        assert len(volume_profile['volume_distribution']) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_order_book_levels(self, integration, sample_order_book_data):
        """Test analyzing order book levels"""
        volume_profile = VolumeProfile(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime.now(timezone.utc),
            poc_price=47500,
            poc_volume=1000,
            value_area_high=48000,
            value_area_low=47000,
            value_area_volume=800,
            total_volume=1000,
            high_volume_nodes=[],
            low_volume_nodes=[],
            liquidity_walls=[],
            volume_distribution={47500: 1000},
            analysis_confidence=0.8
        )
        
        levels = await integration._analyze_order_book_levels(sample_order_book_data, volume_profile)
        
        assert isinstance(levels, list)
        # Should detect some liquidity walls if volume threshold is met
    
    @pytest.mark.asyncio
    async def test_calculate_market_microstructure(self, integration, sample_order_book_data):
        """Test calculating market microstructure"""
        volume_profile = VolumeProfile(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime.now(timezone.utc),
            poc_price=47500,
            poc_volume=1000,
            value_area_high=48000,
            value_area_low=47000,
            value_area_volume=800,
            total_volume=1000,
            high_volume_nodes=[],
            low_volume_nodes=[],
            liquidity_walls=[],
            volume_distribution={47500: 1000},
            analysis_confidence=0.8
        )
        
        microstructure = await integration._calculate_market_microstructure(sample_order_book_data, volume_profile)
        
        assert 'bid_ask_imbalance' in microstructure
        assert 'depth_pressure' in microstructure
        assert 'liquidity_score' in microstructure
        assert 'total_bid_volume' in microstructure
        assert 'total_ask_volume' in microstructure
        assert 'spread' in microstructure
        assert 'mid_price' in microstructure
    
    @pytest.mark.asyncio
    async def test_prepare_algorithm_inputs(self, integration):
        """Test preparing algorithm inputs"""
        volume_profile = VolumeProfile(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime.now(timezone.utc),
            poc_price=47500,
            poc_volume=1000,
            value_area_high=48000,
            value_area_low=47000,
            value_area_volume=800,
            total_volume=1000,
            high_volume_nodes=[],
            low_volume_nodes=[],
            liquidity_walls=[],
            volume_distribution={47500: 1000},
            analysis_confidence=0.8
        )
        
        order_book_levels = []
        microstructure = {
            'bid_ask_imbalance': 0.1,
            'depth_pressure': 0.5,
            'liquidity_score': 0.8
        }
        
        inputs = await integration._prepare_algorithm_inputs(volume_profile, order_book_levels, microstructure)
        
        assert 'volume_profile' in inputs
        assert 'volume_weighted_levels' in inputs
        assert 'supply_zones' in inputs
        assert 'demand_zones' in inputs
        assert 'market_microstructure' in inputs
        assert inputs['volume_profile']['poc_price'] == 47500
    
    @pytest.mark.asyncio
    async def test_analyze_order_book_with_volume_profile(self, integration, mock_db_pool, sample_ohlcv_data, sample_order_book_data):
        """Test complete order book analysis with volume profile"""
        integration.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.side_effect = [
            sample_ohlcv_data,  # First call for OHLCV data
            sample_order_book_data  # Second call for order book data
        ]
        mock_conn.execute.return_value = "INSERT 0 1"
        
        analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
        
        assert isinstance(analysis, OrderBookAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.volume_profile.symbol == 'BTCUSDT'
        assert analysis.bid_ask_imbalance is not None
        assert analysis.liquidity_score is not None
        assert analysis.algorithm_inputs is not None
    
    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, integration, mock_db_pool):
        """Test handling of insufficient data"""
        integration.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.return_value = []  # Empty data
        
        analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
        
        assert isinstance(analysis, OrderBookAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert analysis.volume_profile.poc_price == 0.0
        assert analysis.analysis_confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, integration):
        """Test error handling"""
        # Test with None database pool
        analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
        
        assert isinstance(analysis, OrderBookAnalysis)
        assert analysis.symbol == 'BTCUSDT'
        assert integration.stats['failed_analyses'] == 1
    
    @pytest.mark.asyncio
    async def test_get_volume_profile_strength(self, integration):
        """Test getting volume profile strength"""
        volume_profile = VolumeProfile(
            symbol='BTCUSDT',
            timeframe='1m',
            timestamp=datetime.now(timezone.utc),
            poc_price=47500,
            poc_volume=1000,
            value_area_high=48000,
            value_area_low=47000,
            value_area_volume=800,
            total_volume=1000,
            high_volume_nodes=[],
            low_volume_nodes=[],
            liquidity_walls=[],
            volume_distribution={47500: 1000, 47600: 500},
            analysis_confidence=0.8
        )
        
        # Test exact match
        strength = integration._get_volume_profile_strength(47500, volume_profile)
        assert strength == 1.0
        
        # Test close match
        strength = integration._get_volume_profile_strength(47501, volume_profile)
        assert strength == 1.0
        
        # Test no match
        strength = integration._get_volume_profile_strength(50000, volume_profile)
        assert strength == 0.0

# Integration tests
class TestEnhancedOrderBookIntegrationIntegration:
    """Integration tests for Enhanced Order Book Integration"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self):
        """Test the complete analysis workflow"""
        integration = EnhancedOrderBookIntegration()
        
        # Mock database and data
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create realistic mock data
            ohlcv_data = []
            order_book_data = []
            
            for i in range(100):
                ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': 47000 + i * 10,
                    'high': 48000 + i * 10,
                    'low': 46000 + i * 10,
                    'close': 47500 + i * 10,
                    'volume': 1000 + i * 20
                })
                
                if i < 10:
                    order_book_data.append({
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                        'bids': [[47000 + i * 10, 100 + i], [46900 + i * 10, 200 + i]],
                        'asks': [[48000 + i * 10, 120 + i], [48100 + i * 10, 180 + i]],
                        'best_bid': 47000 + i * 10,
                        'best_ask': 48000 + i * 10,
                        'spread': 1000
                    })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.side_effect = [ohlcv_data, order_book_data]
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Run analysis
            await integration.initialize()
            analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
            
            # Verify results
            assert isinstance(analysis, OrderBookAnalysis)
            assert analysis.symbol == 'BTCUSDT'
            assert analysis.volume_profile.poc_price > 0
            assert analysis.volume_profile.total_volume > 0
            assert analysis.bid_ask_imbalance is not None
            assert analysis.liquidity_score >= 0
            assert len(analysis.algorithm_inputs) > 0
            
            await integration.close()

# Performance tests
class TestEnhancedOrderBookIntegrationPerformance:
    """Performance tests for Enhanced Order Book Integration"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        integration = EnhancedOrderBookIntegration()
        
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            # Create large datasets
            large_ohlcv_data = []
            large_order_book_data = []
            
            for i in range(1000):
                large_ohlcv_data.append({
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': 47000 + i,
                    'high': 48000 + i,
                    'low': 46000 + i,
                    'close': 47500 + i,
                    'volume': 1000 + i * 10
                })
                
                if i < 100:
                    large_order_book_data.append({
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                        'bids': [[47000 + i, 100], [46900 + i, 200]],
                        'asks': [[48000 + i, 120], [48100 + i, 180]],
                        'best_bid': 47000 + i,
                        'best_ask': 48000 + i,
                        'spread': 1000
                    })
            
            mock_conn = AsyncMock()
            mock_conn.fetch.side_effect = [large_ohlcv_data, large_order_book_data]
            mock_conn.execute.return_value = "INSERT 0 1"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            start_time = time.time()
            
            await integration.initialize()
            analysis = await integration.analyze_order_book_with_volume_profile('BTCUSDT', '1m')
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Performance assertions
            assert elapsed_time < 5.0  # Should complete within 5 seconds
            assert isinstance(analysis, OrderBookAnalysis)
            assert analysis.volume_profile.total_volume > 0
            
            await integration.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
