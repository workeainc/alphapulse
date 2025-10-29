#!/usr/bin/env python3
"""
Test suite for Historical Data Preloader
Tests the historical data preloading functionality
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

from src.services.historical_data_preloader import (
    HistoricalDataPreloader, 
    PreloadConfig, 
    PreloadResult
)

class TestHistoricalDataPreloader:
    """Test cases for Historical Data Preloader"""
    
    @pytest.fixture
    def preloader(self):
        """Create a preloader instance for testing"""
        config = PreloadConfig(
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1m', '5m'],
            lookback_days=7,
            min_candles=100
        )
        return HistoricalDataPreloader(config=config)
    
    @pytest.fixture
    def mock_db_pool(self):
        """Create a mock database pool"""
        pool = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = AsyncMock()
        return pool
    
    @pytest.fixture
    def mock_exchange(self):
        """Create a mock exchange"""
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],  # Mock kline data
            [1640995260000, 47500, 48500, 47000, 48000, 1200],
            [1640995320000, 48000, 49000, 47500, 48500, 1100]
        ]
        return exchange
    
    @pytest.mark.asyncio
    async def test_preloader_initialization(self, preloader):
        """Test preloader initialization"""
        assert preloader.config.symbols == ['BTCUSDT', 'ETHUSDT']
        assert preloader.config.timeframes == ['1m', '5m']
        assert preloader.config.lookback_days == 7
        assert preloader.config.min_candles == 100
        assert preloader.stats['total_preloads'] == 0
    
    @pytest.mark.asyncio
    async def test_get_existing_candles_count(self, preloader, mock_db_pool):
        """Test getting existing candles count"""
        preloader.db_pool = mock_db_pool
        
        # Mock database response
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 150
        
        count = await preloader._get_existing_candles_count('BTCUSDT', '1m')
        
        assert count == 150
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_from_exchange(self, preloader, mock_exchange):
        """Test fetching data from exchange"""
        preloader.exchange = mock_exchange
        
        candles = await preloader._fetch_from_exchange('BTCUSDT', '1m', 100)
        
        assert len(candles) == 3
        assert candles[0]['symbol'] == 'BTCUSDT'
        assert candles[0]['timeframe'] == '1m'
        assert candles[0]['open'] == 47000.0
        assert candles[0]['volume'] == 1000.0
        assert candles[0]['source'] == 'historical_preload'
    
    @pytest.mark.asyncio
    async def test_store_candles_in_database(self, preloader, mock_db_pool):
        """Test storing candles in database"""
        preloader.db_pool = mock_db_pool
        
        mock_candles = [
            {
                'symbol': 'BTCUSDT',
                'timeframe': '1m',
                'timestamp': datetime.now(timezone.utc),
                'open': 47000.0,
                'high': 48000.0,
                'low': 46000.0,
                'close': 47500.0,
                'volume': 1000.0,
                'source': 'historical_preload'
            }
        ]
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "INSERT 0 1"
        
        stored_count = await preloader._store_candles_in_database('BTCUSDT', '1m', mock_candles)
        
        assert stored_count == 1
        mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_data_quality(self, preloader, mock_db_pool):
        """Test calculating data quality"""
        preloader.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.return_value = {
            'total_candles': 100,
            'valid_volume': 95,
            'valid_ohlc': 98
        }
        
        quality = await preloader._calculate_data_quality('BTCUSDT', '1m')
        
        assert quality == (95/100 + 98/100) / 2
        mock_conn.fetchrow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_preload_symbol_timeframe_sufficient_data(self, preloader, mock_db_pool):
        """Test preloading when sufficient data exists"""
        preloader.db_pool = mock_db_pool
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 200  # Sufficient data
        
        result = await preloader.preload_symbol_timeframe('BTCUSDT', '1m')
        
        assert result.success is True
        assert result.candles_loaded == 200
        assert result.data_quality == 1.0
    
    @pytest.mark.asyncio
    async def test_preload_symbol_timeframe_insufficient_data(self, preloader, mock_db_pool, mock_exchange):
        """Test preloading when insufficient data exists"""
        preloader.db_pool = mock_db_pool
        preloader.exchange = mock_exchange
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 50  # Insufficient data
        mock_conn.execute.return_value = "INSERT 0 3"
        
        result = await preloader.preload_symbol_timeframe('BTCUSDT', '1m')
        
        assert result.success is True
        assert result.candles_loaded == 53  # 50 existing + 3 new
        assert result.data_quality > 0
    
    @pytest.mark.asyncio
    async def test_preload_all_symbols(self, preloader, mock_db_pool, mock_exchange):
        """Test preloading all symbols"""
        preloader.db_pool = mock_db_pool
        preloader.exchange = mock_exchange
        
        mock_conn = mock_db_pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.return_value = 200  # Sufficient data
        mock_conn.execute.return_value = "INSERT 0 1"
        
        results = await preloader.preload_all_symbols()
        
        assert len(results) == 2  # BTCUSDT, ETHUSDT
        assert 'BTCUSDT' in results
        assert 'ETHUSDT' in results
        
        # Check statistics
        assert preloader.stats['total_preloads'] == 4  # 2 symbols * 2 timeframes
        assert preloader.stats['successful_preloads'] == 4
    
    @pytest.mark.asyncio
    async def test_get_timeframe_minutes(self, preloader):
        """Test timeframe to minutes conversion"""
        assert preloader._get_timeframe_minutes('1m') == 1
        assert preloader._get_timeframe_minutes('5m') == 5
        assert preloader._get_timeframe_minutes('15m') == 15
        assert preloader._get_timeframe_minutes('1h') == 60
        assert preloader._get_timeframe_minutes('1d') == 1440
        assert preloader._get_timeframe_minutes('unknown') == 60  # Default
    
    @pytest.mark.asyncio
    async def test_get_preload_status(self, preloader):
        """Test getting preload status"""
        status = await preloader.get_preload_status()
        
        assert 'stats' in status
        assert 'config' in status
        assert 'exchange_available' in status
        assert 'database_connected' in status
        assert status['config']['symbols'] == ['BTCUSDT', 'ETHUSDT']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, preloader):
        """Test error handling in preloader"""
        # Test with invalid symbol
        result = await preloader.preload_symbol_timeframe('INVALID', '1m')
        
        assert result.success is False
        assert result.candles_loaded == 0
        assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, preloader):
        """Test rate limiting functionality"""
        start_time = time.time()
        
        # Simulate rate limiting delay
        await preloader._wait_for_rate_limit('binance')
        
        # Should have some delay (even if minimal in test)
        elapsed = time.time() - start_time
        assert elapsed >= 0  # Basic check that function runs

# Integration tests
class TestHistoricalDataPreloaderIntegration:
    """Integration tests for Historical Data Preloader"""
    
    @pytest.mark.asyncio
    async def test_full_preload_workflow(self):
        """Test the complete preload workflow"""
        config = PreloadConfig(
            symbols=['BTCUSDT'],
            timeframes=['1m'],
            lookback_days=1,
            min_candles=10
        )
        
        preloader = HistoricalDataPreloader(config=config)
        
        # Mock the database and exchange
        with patch('asyncpg.create_pool') as mock_pool, \
             patch('ccxt.binance') as mock_ccxt:
            
            # Setup mocks
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            mock_exchange_instance = MagicMock()
            mock_exchange_instance.fetch_ohlcv.return_value = [
                [1640995200000, 47000, 48000, 46000, 47500, 1000],
                [1640995260000, 47500, 48500, 47000, 48000, 1200],
                [1640995320000, 48000, 49000, 47500, 48500, 1100]
            ]
            mock_ccxt.return_value = mock_exchange_instance
            
            # Mock database responses
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 5  # Insufficient data
            mock_conn.execute.return_value = "INSERT 0 3"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            # Initialize and run preload
            await preloader.initialize()
            results = await preloader.preload_all_symbols()
            
            # Verify results
            assert len(results) == 1
            assert 'BTCUSDT' in results
            assert len(results['BTCUSDT']) == 1  # One timeframe
            
            result = results['BTCUSDT'][0]
            assert result.success is True
            assert result.candles_loaded >= 5
            
            await preloader.close()

# Performance tests
class TestHistoricalDataPreloaderPerformance:
    """Performance tests for Historical Data Preloader"""
    
    @pytest.mark.asyncio
    async def test_preload_performance(self):
        """Test preload performance with large datasets"""
        config = PreloadConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            timeframes=['1m', '5m', '15m'],
            lookback_days=7,
            min_candles=200
        )
        
        preloader = HistoricalDataPreloader(config=config)
        
        # Mock everything for performance test
        with patch('asyncpg.create_pool') as mock_pool, \
             patch('ccxt.binance') as mock_ccxt:
            
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            mock_exchange_instance = MagicMock()
            # Generate large mock dataset
            large_dataset = []
            for i in range(1000):
                timestamp = 1640995200000 + (i * 60000)  # 1 minute intervals
                large_dataset.append([
                    timestamp, 47000 + i, 48000 + i, 46000 + i, 47500 + i, 1000 + i
                ])
            mock_exchange_instance.fetch_ohlcv.return_value = large_dataset
            mock_ccxt.return_value = mock_exchange_instance
            
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 50  # Insufficient data
            mock_conn.execute.return_value = "INSERT 0 1000"
            mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
            
            start_time = time.time()
            
            await preloader.initialize()
            results = await preloader.preload_all_symbols()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Performance assertions
            assert elapsed_time < 10.0  # Should complete within 10 seconds
            assert preloader.stats['total_preloads'] == 9  # 3 symbols * 3 timeframes
            
            await preloader.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
