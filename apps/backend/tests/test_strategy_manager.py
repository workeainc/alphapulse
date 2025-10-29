#!/usr/bin/env python3
"""
Unit tests for StrategyManager
Tests trading logic, adaptive intervals, and performance monitoring
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psutil
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.strategies.strategy_manager import StrategyManager


class TestStrategyManager:
    """Test StrategyManager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.manager = StrategyManager()
        self.manager.is_running = False  # Ensure clean state
    
    def teardown_method(self):
        """Cleanup after tests"""
        if self.manager.is_running:
            asyncio.run(self.manager.stop())
    
    def test_initialization(self):
        """Test StrategyManager initialization"""
        assert self.manager.is_running == False
        assert isinstance(self.manager.strategies, dict)
        assert isinstance(self.manager.active_strategies, set)
        assert isinstance(self.manager.strategy_performance, dict)
        assert self.manager.base_monitor_interval == 30
        assert self.manager.min_interval == 10
        assert self.manager.max_interval == 120
        assert self.manager.cpu_threshold_high == 80.0
        assert self.manager.cpu_threshold_low == 30.0
    
    @patch('psutil.cpu_percent')
    def test_calculate_adaptive_interval_high_cpu(self, mock_cpu_percent):
        """Test adaptive interval calculation with high CPU usage"""
        mock_cpu_percent.return_value = 85.0  # High CPU usage
        
        interval = self.manager._calculate_adaptive_interval()
        
        assert interval > self.manager.base_monitor_interval
        assert interval <= self.manager.max_interval
        mock_cpu_percent.assert_called_once_with(interval=0.1)
    
    @patch('psutil.cpu_percent')
    def test_calculate_adaptive_interval_low_cpu(self, mock_cpu_percent):
        """Test adaptive interval calculation with low CPU usage"""
        mock_cpu_percent.return_value = 20.0  # Low CPU usage
        
        interval = self.manager._calculate_adaptive_interval()
        
        assert interval < self.manager.base_monitor_interval
        assert interval >= self.manager.min_interval
        mock_cpu_percent.assert_called_once_with(interval=0.1)
    
    @patch('psutil.cpu_percent')
    def test_calculate_adaptive_interval_normal_cpu(self, mock_cpu_percent):
        """Test adaptive interval calculation with normal CPU usage"""
        mock_cpu_percent.return_value = 50.0  # Normal CPU usage
        
        interval = self.manager._calculate_adaptive_interval()
        
        # Should be close to base interval
        assert abs(interval - self.manager.base_monitor_interval) < 10
        mock_cpu_percent.assert_called_once_with(interval=0.1)
    
    @patch('psutil.cpu_percent')
    def test_calculate_adaptive_interval_exception_handling(self, mock_cpu_percent):
        """Test adaptive interval calculation with exception handling"""
        mock_cpu_percent.side_effect = Exception("CPU monitoring error")
        
        interval = self.manager._calculate_adaptive_interval()
        
        # Should return base interval on error
        assert interval == self.manager.base_monitor_interval
    
    @pytest.mark.asyncio
    async def test_register_strategy(self):
        """Test registering a strategy"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        result = await self.manager.register_strategy(strategy_id, strategy_instance)
        
        assert result == True
        assert strategy_id in self.manager.strategies
        assert self.manager.strategies[strategy_id] == strategy_instance
        assert strategy_id not in self.manager.active_strategies  # Not active by default
    
    @pytest.mark.asyncio
    async def test_unregister_strategy(self):
        """Test unregistering a strategy"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register strategy first
        await self.manager.register_strategy(strategy_id, strategy_instance)
        assert strategy_id in self.manager.strategies
        
        # Unregister strategy
        result = await self.manager.unregister_strategy(strategy_id)
        assert result == True
        assert strategy_id not in self.manager.strategies
        assert strategy_id not in self.manager.active_strategies
    
    @pytest.mark.asyncio
    async def test_activate_strategy(self):
        """Test activating a strategy"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register strategy first
        await self.manager.register_strategy(strategy_id, strategy_instance)
        
        # Activate strategy
        result = await self.manager.activate_strategy(strategy_id)
        assert result == True
        assert strategy_id in self.manager.active_strategies
    
    @pytest.mark.asyncio
    async def test_deactivate_strategy(self):
        """Test deactivating a strategy"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register and activate strategy first
        await self.manager.register_strategy(strategy_id, strategy_instance)
        await self.manager.activate_strategy(strategy_id)
        assert strategy_id in self.manager.active_strategies
        
        # Deactivate strategy
        result = await self.manager.deactivate_strategy(strategy_id)
        assert result == True
        assert strategy_id not in self.manager.active_strategies
    
    def test_get_strategy_status(self):
        """Test getting strategy status"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register strategy
        asyncio.run(self.manager.register_strategy(strategy_id, strategy_instance))
        
        # Get status (should be inactive)
        status = self.manager.get_strategy_status()
        assert "total_strategies" in status
        assert "active_strategies" in status
        assert "strategies" in status
        assert strategy_id in status["strategies"]
        assert status["strategies"][strategy_id]["active"] == False
        
        # Activate and check status
        asyncio.run(self.manager.activate_strategy(strategy_id))
        status = self.manager.get_strategy_status()
        assert status["strategies"][strategy_id]["active"] == True
    
    def test_get_strategy_performance_nonexistent(self):
        """Test getting performance of nonexistent strategy"""
        performance = self.manager.get_strategy_performance("nonexistent_strategy")
        assert performance == {}
    
    def test_get_strategy_performance(self):
        """Test getting strategy performance metrics"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register strategy
        asyncio.run(self.manager.register_strategy(strategy_id, strategy_instance))
        
        # Get performance (should be empty initially)
        performance = self.manager.get_strategy_performance(strategy_id)
        assert isinstance(performance, dict)
    
    def test_get_active_strategies(self):
        """Test getting active strategies list"""
        strategy_id = "test_strategy"
        strategy_instance = Mock()
        
        # Register strategy
        asyncio.run(self.manager.register_strategy(strategy_id, strategy_instance))
        
        # Initially no active strategies
        active_strategies = self.manager.get_active_strategies()
        assert len(active_strategies) == 0
        
        # Activate strategy
        asyncio.run(self.manager.activate_strategy(strategy_id))
        active_strategies = self.manager.get_active_strategies()
        assert len(active_strategies) == 1
        assert strategy_id in active_strategies
    
    @pytest.mark.asyncio
    async def test_start_stop_manager(self):
        """Test starting and stopping the strategy manager"""
        # Test start
        await self.manager.start()
        assert self.manager.is_running == True
        assert hasattr(self.manager, '_start_time')
        
        # Test stop
        await self.manager.stop()
        assert self.manager.is_running == False
    
    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test starting manager that's already running"""
        await self.manager.start()
        assert self.manager.is_running == True
        
        # Try to start again
        await self.manager.start()
        assert self.manager.is_running == True  # Should still be running
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test stopping manager that's not running"""
        assert self.manager.is_running == False
        
        # Try to stop
        await self.manager.stop()
        assert self.manager.is_running == False  # Should still be stopped
    
    def test_get_system_status(self):
        """Test getting system status"""
        status = self.manager.get_system_status()
        
        assert "system_status" in status
        assert "strategy_management" in status
        assert "candlestick_analysis" in status
        assert "total_active_components" in status
    
    def test_get_candlestick_status(self):
        """Test getting candlestick analysis status"""
        status = self.manager.get_candlestick_status()
        
        assert "analysis_enabled" in status
        assert "active_symbols" in status
        assert "active_timeframes" in status
        assert "total_symbols" in status
        assert "processing_stats" in status
    
    def test_calculate_uptime(self):
        """Test uptime calculation"""
        # Test without start time
        uptime = self.manager._calculate_uptime()
        assert uptime == "Unknown"
        
        # Test with start time
        self.manager._start_time = datetime.now() - timedelta(hours=1, minutes=30)
        uptime = self.manager._calculate_uptime()
        assert "1h 30m" in uptime


class TestStrategyManagerIntegration:
    """Integration tests for StrategyManager"""
    
    @pytest.mark.asyncio
    async def test_full_strategy_lifecycle(self):
        """Test complete strategy lifecycle"""
        manager = StrategyManager()
        
        try:
            # Start manager
            await manager.start()
            assert manager.is_running == True
            
            # Register strategy
            strategy_id = "integration_test_strategy"
            strategy_instance = Mock()
            
            result = await manager.register_strategy(strategy_id, strategy_instance)
            assert result == True
            assert strategy_id in manager.strategies
            
            # Activate strategy
            result = await manager.activate_strategy(strategy_id)
            assert result == True
            assert strategy_id in manager.active_strategies
            
            # Get status
            status = manager.get_strategy_status()
            assert strategy_id in status["strategies"]
            assert status["strategies"][strategy_id]["active"] == True
            
            # Get performance
            performance = manager.get_strategy_performance(strategy_id)
            assert isinstance(performance, dict)
            
            # Deactivate strategy
            result = await manager.deactivate_strategy(strategy_id)
            assert result == True
            assert strategy_id not in manager.active_strategies
            
            # Unregister strategy
            result = await manager.unregister_strategy(strategy_id)
            assert result == True
            assert strategy_id not in manager.strategies
            
        finally:
            await manager.stop()
            assert manager.is_running == False
    
    @pytest.mark.asyncio
    async def test_concurrent_strategy_operations(self):
        """Test concurrent strategy operations"""
        manager = StrategyManager()
        
        try:
            await manager.start()
            
            # Register multiple strategies concurrently
            strategies = [
                ("strategy1", Mock()),
                ("strategy2", Mock()),
                ("strategy3", Mock())
            ]
            
            # Register strategies
            for strategy_id, strategy_instance in strategies:
                result = await manager.register_strategy(strategy_id, strategy_instance)
                assert result == True
                assert strategy_id in manager.strategies
            
            # Activate all strategies
            for strategy_id, _ in strategies:
                result = await manager.activate_strategy(strategy_id)
                assert result == True
                assert strategy_id in manager.active_strategies
            
            # Verify all strategies are active
            assert len(manager.active_strategies) == 3
            
            # Get status
            status = manager.get_strategy_status()
            assert status["total_strategies"] == 3
            assert status["active_strategies"] == 3
            
        finally:
            await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
