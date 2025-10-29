#!/usr/bin/env python3
"""
Data Manager for AlphaPulse
Unified interface for all data operations
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .pipeline import DataPipeline, PipelineConfig
from .storage import DataStorage
from .validation import CandlestickValidator
from .exchange_connector import ExchangeConnector

logger = logging.getLogger(__name__)

class DataManager:
    """Unified data management interface for AlphaPulse"""
    
    def __init__(self, 
                 storage_path: str = "data",
                 storage_type: str = "postgresql",
                 auto_update: bool = True,
                 update_interval_minutes: int = 60):
        """
        Initialize data manager
        
        Args:
            storage_path: Path to data storage
            storage_type: Type of storage backend
            auto_update: Whether to enable automatic data updates
            update_interval_minutes: Interval for automatic updates
        """
        self.storage_path = Path(storage_path)
        self.storage_type = storage_type
        self.auto_update = auto_update
        self.update_interval_minutes = update_interval_minutes
        
        # Initialize components
        self.storage = DataStorage(storage_path, storage_type)
        self.validator = CandlestickValidator()
        self.connector = ExchangeConnector()
        
        # Pipeline configuration (can be updated)
        self.pipeline_config = PipelineConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
            intervals=['1h', '4h', '1d'],
            exchanges=['binance'],
            batch_size=1000,
            retry_attempts=3,
            validation_enabled=True,
            storage_type=storage_type,
            storage_path=storage_path,
            max_workers=5,
            update_frequency_minutes=update_interval_minutes
        )
        
        # Pipeline instance
        self.pipeline = DataPipeline(self.pipeline_config)
        
        # Auto-update thread
        self._auto_update_thread = None
        self._stop_auto_update = threading.Event()
        
        # Data cache
        self._data_cache = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_update': None,
            'errors': []
        }
        
        logger.info(f"Data manager initialized with {storage_type} storage at {storage_path}")
        
        # Start auto-update if enabled
        if self.auto_update:
            self.start_auto_update()
    
    def start_auto_update(self):
        """Start automatic data updates"""
        if self._auto_update_thread and self._auto_update_thread.is_alive():
            logger.warning("Auto-update thread is already running")
            return
        
        self._stop_auto_update.clear()
        self._auto_update_thread = threading.Thread(target=self._auto_update_worker, daemon=True)
        self._auto_update_thread.start()
        logger.info("Auto-update thread started")
    
    def stop_auto_update(self):
        """Stop automatic data updates"""
        if self._auto_update_thread and self._auto_update_thread.is_alive():
            self._stop_auto_update.set()
            self._auto_update_thread.join(timeout=5)
            logger.info("Auto-update thread stopped")
    
    def _auto_update_worker(self):
        """Worker thread for automatic updates"""
        while not self._stop_auto_update.is_set():
            try:
                logger.info("Running scheduled data update")
                asyncio.run(self.pipeline.run_pipeline(force_update=False))
                self.stats['last_update'] = datetime.now()
                
                # Wait for next update interval
                self._stop_auto_update.wait(self.update_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in auto-update worker: {e}")
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
                # Wait a bit before retrying
                time.sleep(60)
    
    async def update_data(self, 
                         symbols: Optional[List[str]] = None,
                         intervals: Optional[List[str]] = None,
                         exchanges: Optional[List[str]] = None,
                         force_update: bool = False) -> Dict[str, Any]:
        """
        Update data for specified symbols/intervals/exchanges
        
        Args:
            symbols: List of symbols to update (None for all)
            intervals: List of intervals to update (None for all)
            exchanges: List of exchanges to update (None for all)
            force_update: Force update even if recent data exists
            
        Returns:
            Update results
        """
        try:
            # Update pipeline configuration if specific parameters provided
            if symbols or intervals or exchanges:
                temp_config = PipelineConfig(
                    symbols=symbols or self.pipeline_config.symbols,
                    intervals=intervals or self.pipeline_config.intervals,
                    exchanges=exchanges or self.pipeline_config.exchanges,
                    **{k: v for k, v in self.pipeline_config.__dict__.items() 
                       if k not in ['symbols', 'intervals', 'exchanges']}
                )
                temp_pipeline = DataPipeline(temp_config)
                status = await temp_pipeline.run_pipeline(force_update=force_update)
            else:
                status = await self.pipeline.run_pipeline(force_update=force_update)
            
            # Clear cache after update
            self._clear_cache()
            
            # Update statistics
            self.stats['last_update'] = datetime.now()
            
            return {
                'success': True,
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            self.stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_candlestick_data(self, 
                            symbol: str,
                            interval: str,
                            exchange: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: Optional[int] = None,
                            use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get candlestick data for a symbol
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            exchange: Exchange name
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with candlestick data or None
        """
        self.stats['total_requests'] += 1
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{exchange}_{start_time}_{end_time}_{limit}"
        if use_cache and self._is_cache_valid(cache_key):
            self.stats['cache_hits'] += 1
            return self._data_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Retrieve from storage
        df = self.storage.retrieve_candlestick_data(
            symbol, interval, exchange, start_time, end_time, limit
        )
        
        # Cache the result
        if df is not None and use_cache:
            self._cache_data(cache_key, df)
        
        return df
    
    def get_multiple_symbols_data(self, 
                                 symbols: List[str],
                                 interval: str,
                                 exchange: str,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            interval: Time interval
            exchange: Exchange name
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records per symbol
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            future_to_symbol = {
                executor.submit(
                    self.get_candlestick_data,
                    symbol, interval, exchange, start_time, end_time, limit
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"Error retrieving data for {symbol}: {e}")
                    self.stats['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'error': f"Error retrieving data for {symbol}: {e}"
                    })
        
        return results
    
    def get_data_info(self, symbol: str, interval: str, exchange: str) -> Dict[str, Any]:
        """Get detailed information about available data"""
        return self.storage.get_data_info(symbol, interval, exchange)
    
    def get_all_symbols_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all available symbols"""
        symbols = self.storage.get_available_symbols()
        results = {}
        
        for symbol in symbols:
            symbol_info = {}
            for interval in self.pipeline_config.intervals:
                for exchange in self.pipeline_config.exchanges:
                    info = self.get_data_info(symbol, interval, exchange)
                    if info['available']:
                        symbol_info[f"{interval}_{exchange}"] = info
            
            if symbol_info:
                results[symbol] = symbol_info
        
        return results
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate candlestick data"""
        return self.validator.validate_candlestick_data(df)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return self.pipeline.get_pipeline_summary()
    
    def get_symbol_pipeline_status(self, symbol: str) -> Dict[str, Any]:
        """Get pipeline status for a specific symbol"""
        return self.pipeline.get_symbol_status(symbol)
    
    def reset_pipeline(self):
        """Reset pipeline status"""
        self.pipeline.reset_pipeline_status()
        logger.info("Pipeline status reset")
    
    def export_pipeline_report(self, filepath: str = None) -> str:
        """Export pipeline report"""
        return self.pipeline.export_pipeline_report(filepath)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.storage.get_storage_stats()
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get data manager statistics"""
        stats = self.stats.copy()
        stats['cache_size'] = len(self._data_cache)
        stats['cache_keys'] = list(self._data_cache.keys())
        stats['auto_update_running'] = (
            self._auto_update_thread is not None and 
            self._auto_update_thread.is_alive()
        )
        stats['update_interval_minutes'] = self.update_interval_minutes
        return stats
    
    def _cache_data(self, key: str, data: pd.DataFrame):
        """Cache data with TTL"""
        self._data_cache[key] = data
        self._cache_timestamps[key] = datetime.now()
        
        # Limit cache size
        if len(self._data_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            del self._data_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._data_cache or key not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[key]
        return age < self._cache_ttl
    
    def _clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Data cache cleared")
    
    def update_pipeline_config(self, **kwargs):
        """Update pipeline configuration"""
        for key, value in kwargs.items():
            if hasattr(self.pipeline_config, key):
                setattr(self.pipeline_config, key, value)
                logger.info(f"Updated pipeline config: {key} = {value}")
            else:
                logger.warning(f"Unknown pipeline config key: {key}")
        
        # Recreate pipeline with new config
        self.pipeline = DataPipeline(self.pipeline_config)
        logger.info("Pipeline recreated with updated configuration")
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to the pipeline"""
        if symbol not in self.pipeline_config.symbols:
            self.pipeline_config.symbols.append(symbol)
            self.pipeline = DataPipeline(self.pipeline_config)
            logger.info(f"Added symbol: {symbol}")
        else:
            logger.info(f"Symbol {symbol} already exists")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from the pipeline"""
        if symbol in self.pipeline_config.symbols:
            self.pipeline_config.symbols.remove(symbol)
            self.pipeline = DataPipeline(self.pipeline_config)
            logger.info(f"Removed symbol: {symbol}")
        else:
            logger.info(f"Symbol {symbol} not found")
    
    def get_available_intervals(self) -> List[str]:
        """Get list of available time intervals"""
        return self.pipeline_config.intervals.copy()
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return self.pipeline_config.exchanges.copy()
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to save storage space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for symbol in self.pipeline_config.symbols:
                for interval in self.pipeline_config.intervals:
                    for exchange in self.pipeline_config.exchanges:
                        # Get old data
                        old_data = self.storage.retrieve_candlestick_data(
                            symbol, interval, exchange,
                            end_time=cutoff_date
                        )
                        
                        if old_data is not None and not old_data.empty:
                            # Delete old data
                            if self.storage.delete_data(symbol, interval, exchange):
                                deleted_count += len(old_data)
                                logger.info(f"Deleted {len(old_data)} old records for {symbol} {interval} {exchange}")
            
            logger.info(f"Cleanup completed: {deleted_count} old records deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on data manager"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check storage
            storage_stats = self.get_storage_stats()
            health['components']['storage'] = {
                'status': 'healthy' if 'error' not in storage_stats else 'unhealthy',
                'stats': storage_stats
            }
            
            # Check pipeline
            pipeline_status = self.get_pipeline_status()
            health['components']['pipeline'] = {
                'status': 'healthy' if not pipeline_status.get('is_running', False) else 'running',
                'status_breakdown': pipeline_status
            }
            
            # Check auto-update
            health['components']['auto_update'] = {
                'status': 'healthy' if self._auto_update_thread and self._auto_update_thread.is_alive() else 'unhealthy',
                'enabled': self.auto_update,
                'interval_minutes': self.update_interval_minutes
            }
            
            # Check cache
            health['components']['cache'] = {
                'status': 'healthy',
                'size': len(self._data_cache),
                'ttl_minutes': self._cache_ttl.total_seconds() / 60
            }
            
            # Overall status
            unhealthy_components = [
                comp for comp, info in health['components'].items()
                if info['status'] != 'healthy'
            ]
            
            if unhealthy_components:
                health['status'] = 'degraded'
                health['unhealthy_components'] = unhealthy_components
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health

# Example usage and testing
def test_data_manager():
    """Test the data manager functionality"""
    # Initialize data manager
    manager = DataManager(
        storage_path="test_data",
        storage_type="postgresql",
        auto_update=False,
        update_interval_minutes=5
    )
    
    # Test data retrieval
    print("Testing data retrieval...")
    df = manager.get_candlestick_data("BTCUSDT", "1h", "binance", limit=10)
    print(f"Retrieved data: {len(df) if df is not None else 0} records")
    
    # Test pipeline status
    print("Testing pipeline status...")
    status = manager.get_pipeline_status()
    print(f"Pipeline status: {status}")
    
    # Test health check
    print("Testing health check...")
    health = manager.health_check()
    print(f"Health status: {health['status']}")
    
    # Test manager stats
    print("Testing manager stats...")
    stats = manager.get_manager_stats()
    print(f"Manager stats: {stats}")
    
    # Clean up
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")
    
    return manager

if __name__ == "__main__":
    # Run test if script is executed directly
    manager = test_data_manager()
