"""
Enhanced Data Storage Module for AlphaPlus
Handles data persistence and retrieval with database integration
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import os
import asyncio
from pathlib import Path

# Import our enhanced database connection
try:
    from ..database.connection import TimescaleDBConnection
except ImportError:
    # Fallback for testing
    TimescaleDBConnection = None

logger = logging.getLogger(__name__)

class DataStorage:
    """Enhanced data storage implementation with database integration"""
    
    def __init__(self, storage_path: str = "data", db_config: Dict[str, Any] = None):
        self.storage_path = storage_path
        self.db_config = db_config or {}
        self.logger = logger
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Database connection
        self.db_connection = None
        self.use_database = self.db_config.get('use_database', True)
        
        # File storage fallback
        self.file_storage_enabled = True
        
        # Performance tracking
        self.stats = {
            'database_operations': 0,
            'file_operations': 0,
            'errors': 0
        }
        
    async def initialize(self):
        """Initialize the data storage system"""
        try:
            self.logger.info("Initializing Enhanced Data Storage...")
            
            # Initialize database connection if enabled
            if self.use_database and TimescaleDBConnection:
                try:
                    self.db_connection = TimescaleDBConnection(self.db_config)
                    await self.db_connection.initialize()
                    self.logger.info("Database connection established")
                except Exception as e:
                    self.logger.warning(f"Database connection failed, using file storage: {e}")
                    self.use_database = False
                    self.db_connection = None
            else:
                self.logger.info("Using file storage only")
            
            self.logger.info("Enhanced Data Storage initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Data Storage: {e}")
            raise
    
    async def save_data(self, key: str, data: Any) -> bool:
        """Save data to storage (database preferred, file fallback)"""
        try:
            # Try database first if available
            if self.use_database and self.db_connection:
                success = await self._save_to_database(key, data)
                if success:
                    self.stats['database_operations'] += 1
                    return True
            
            # Fallback to file storage
            success = await self._save_to_file(key, data)
            if success:
                self.stats['file_operations'] += 1
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error saving data {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def _save_to_database(self, key: str, data: Any) -> bool:
        """Save data to database"""
        try:
            if not self.db_connection:
                return False
            
            # Determine data type and save accordingly
            if key.startswith('candlestick_'):
                return await self._save_candlestick_data(key, data)
            elif key.startswith('trade_'):
                return await self._save_trade_data(key, data)
            elif key.startswith('signal_'):
                return await self._save_signal_data(key, data)
            elif key.startswith('performance_'):
                return await self._save_performance_data(key, data)
            else:
                # Generic data storage
                return await self._save_generic_data(key, data)
                
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
            return False
    
    async def _save_candlestick_data(self, key: str, data: Any) -> bool:
        """Save candlestick data to database"""
        try:
            if not self.db_connection:
                return False
            
            # Extract symbol and timeframe from key
            # Format: candlestick_BTCUSDT_1m_20241201
            parts = key.split('_')
            if len(parts) >= 4:
                symbol = parts[1]
                timeframe = parts[2]
                
                candlestick_data = {
                    'symbol': symbol,
                    'timestamp': data.get('timestamp', datetime.now(timezone.utc)),
                    'open': data.get('open', 0.0),
                    'high': data.get('high', 0.0),
                    'low': data.get('low', 0.0),
                    'close': data.get('close', 0.0),
                    'volume': data.get('volume', 0.0),
                    'timeframe': timeframe,
                    'indicators': data.get('indicators', {}),
                    'patterns': data.get('patterns', [])
                }
                
                return await self.db_connection.save_candlestick(candlestick_data)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error saving candlestick data: {e}")
            return False
    
    async def _save_trade_data(self, key: str, data: Any) -> bool:
        """Save trade data to database"""
        try:
            if not self.db_connection:
                return False
            
            trade_data = {
                'signal_id': data.get('signal_id', key),
                'symbol': data.get('symbol', 'UNKNOWN'),
                'side': data.get('side', 'buy'),
                'entry_price': data.get('entry_price', 0.0),
                'quantity': data.get('quantity', 0.0),
                'timestamp': data.get('timestamp', datetime.now(timezone.utc)),
                'strategy': data.get('strategy', 'unknown'),
                'confidence': data.get('confidence', 0.0)
            }
            
            return await self.db_connection.save_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error saving trade data: {e}")
            return False
    
    async def _save_signal_data(self, key: str, data: Any) -> bool:
        """Save signal data to database"""
        try:
            if not self.db_connection:
                return False
            
            signal_data = {
                'id': data.get('id', key),
                'symbol': data.get('symbol', 'UNKNOWN'),
                'side': data.get('side', 'buy'),
                'strategy': data.get('strategy', 'unknown'),
                'confidence': data.get('confidence', 0.0),
                'strength': data.get('strength', 'weak'),
                'timestamp': data.get('timestamp', datetime.now(timezone.utc)),
                'price': data.get('price', 0.0),
                'stop_loss': data.get('stop_loss'),
                'take_profit': data.get('take_profit'),
                'metadata': data.get('metadata', {})
            }
            
            return await self.db_connection.save_signal(signal_data)
            
        except Exception as e:
            self.logger.error(f"Error saving signal data: {e}")
            return False
    
    async def _save_performance_data(self, key: str, data: Any) -> bool:
        """Save performance data to database"""
        try:
            if not self.db_connection:
                return False
            
            performance_data = {
                'timestamp': data.get('timestamp', datetime.now(timezone.utc)),
                'total_trades': data.get('total_trades', 0),
                'winning_trades': data.get('winning_trades', 0),
                'losing_trades': data.get('losing_trades', 0),
                'win_rate': data.get('win_rate', 0.0),
                'total_pnl': data.get('total_pnl', 0.0),
                'daily_pnl': data.get('daily_pnl', 0.0),
                'active_positions': data.get('active_positions', 0)
            }
            
            return await self.db_connection.save_performance_metrics(performance_data)
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
            return False
    
    async def _save_generic_data(self, key: str, data: Any) -> bool:
        """Save generic data to database (fallback)"""
        try:
            # For now, fall back to file storage for generic data
            return await self._save_to_file(key, data)
            
        except Exception as e:
            self.logger.error(f"Error saving generic data: {e}")
            return False
    
    async def _save_to_file(self, key: str, data: Any) -> bool:
        """Save data to file storage"""
        try:
            file_path = os.path.join(self.storage_path, f"{key}.json")
            
            # Convert datetime objects to ISO format
            if isinstance(data, dict):
                data = self._serialize_datetime(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved data to file: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to file {key}: {e}")
            return False
    
    async def load_data(self, key: str) -> Optional[Any]:
        """Load data from storage (database preferred, file fallback)"""
        try:
            # Try database first if available
            if self.use_database and self.db_connection:
                data = await self._load_from_database(key)
                if data is not None:
                    self.stats['database_operations'] += 1
                    return data
            
            # Fallback to file storage
            data = await self._load_from_file(key)
            if data is not None:
                self.stats['file_operations'] += 1
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def _load_from_database(self, key: str) -> Optional[Any]:
        """Load data from database"""
        try:
            if not self.db_connection:
                return None
            
            # Determine data type and load accordingly
            if key.startswith('candlestick_'):
                return await self._load_candlestick_data(key)
            elif key.startswith('trade_'):
                return await self._load_trade_data(key)
            elif key.startswith('signal_'):
                return await self._load_signal_data(key)
            elif key.startswith('performance_'):
                return await self._load_performance_data(key)
            else:
                # Generic data not stored in database
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}")
            return None
    
    async def _load_candlestick_data(self, key: str) -> Optional[Any]:
        """Load candlestick data from database"""
        try:
            if not self.db_connection:
                return None
            
            # Extract symbol and timeframe from key
            parts = key.split('_')
            if len(parts) >= 4:
                symbol = parts[1]
                timeframe = parts[2]
                
                # Get recent data
                data = await self.db_connection.get_candlestick_data(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    limit=100
                )
                
                return data if data else None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading candlestick data: {e}")
            return None
    
    async def _load_trade_data(self, key: str) -> Optional[Any]:
        """Load trade data from database"""
        try:
            if not self.db_connection:
                return None
            
            # Get recent trades
            data = await self.db_connection.get_trades(limit=100)
            return data if data else None
            
        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")
            return None
    
    async def _load_signal_data(self, key: str) -> Optional[Any]:
        """Load signal data from database"""
        try:
            if not self.db_connection:
                return None
            
            # For now, return empty list (signals are typically queried differently)
            return []
            
        except Exception as e:
            self.logger.error(f"Error loading signal data: {e}")
            return None
    
    async def _load_performance_data(self, key: str) -> Optional[Any]:
        """Load performance data from database"""
        try:
            if not self.db_connection:
                return None
            
            # Get performance summary
            data = await self.db_connection.get_performance_summary(days=30)
            return data if data else None
            
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
            return None
    
    async def _load_from_file(self, key: str) -> Optional[Any]:
        """Load data from file storage"""
        try:
            file_path = os.path.join(self.storage_path, f"{key}.json")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded data from file: {key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from file {key}: {e}")
            return None
    
    async def delete_data(self, key: str) -> bool:
        """Delete data from storage"""
        try:
            # Try database first if available
            if self.use_database and self.db_connection:
                # Note: Database deletion would require specific table operations
                # For now, just log the attempt
                self.logger.info(f"Database deletion requested for: {key}")
            
            # Delete from file storage
            file_path = os.path.join(self.storage_path, f"{key}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Deleted data file: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting data {key}: {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """List all available data keys"""
        try:
            keys = []
            
            # Get database keys if available
            if self.use_database and self.db_connection:
                try:
                    # Get available symbols and timeframes
                    # This is a simplified approach - in practice you'd query the database
                    db_keys = ['candlestick_BTCUSDT_1m', 'candlestick_ETHUSDT_1m']
                    keys.extend(db_keys)
                except Exception as e:
                    self.logger.warning(f"Could not get database keys: {e}")
            
            # Get file keys
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    keys.append(filename[:-5])  # Remove .json extension
            
            return list(set(keys))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error listing keys: {e}")
            return []
    
    def _serialize_datetime(self, obj: Any) -> Any:
        """Convert datetime objects to ISO format strings"""
        if isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = self.stats.copy()
            
            # Add database stats if available
            if self.db_connection:
                db_health = await self.db_connection.health_check()
                stats['database_health'] = db_health
                stats['database_connected'] = db_health.get('status') == 'healthy'
            
            # Add file storage stats
            try:
                file_count = len([f for f in os.listdir(self.storage_path) if f.endswith('.json')])
                stats['file_count'] = file_count
                stats['storage_path'] = self.storage_path
            except Exception as e:
                stats['file_count'] = 0
                stats['storage_error'] = str(e)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for data storage"""
        try:
            health_status = {
                'status': 'healthy',
                'file_storage_enabled': self.file_storage_enabled,
                'database_enabled': self.use_database,
                'database_connected': False,
                'storage_path': self.storage_path,
                'stats': self.stats
            }
            
            # Check database health if available
            if self.db_connection:
                try:
                    db_health = await self.db_connection.health_check()
                    health_status['database_connected'] = db_health.get('status') == 'healthy'
                    health_status['database_health'] = db_health
                    
                    if not health_status['database_connected']:
                        health_status['status'] = 'degraded'
                        health_status['warnings'] = ['Database connection issues']
                except Exception as e:
                    health_status['database_connected'] = False
                    health_status['database_error'] = str(e)
                    health_status['status'] = 'degraded'
            
            # Check file storage
            try:
                if not os.path.exists(self.storage_path):
                    os.makedirs(self.storage_path, exist_ok=True)
                
                test_file = os.path.join(self.storage_path, '.health_check')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                health_status['file_storage_healthy'] = True
            except Exception as e:
                health_status['file_storage_healthy'] = False
                health_status['file_storage_error'] = str(e)
                health_status['status'] = 'unhealthy'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close data storage and cleanup"""
        try:
            if self.db_connection:
                await self.db_connection.close()
            
            self.logger.info("Data storage closed")
            
        except Exception as e:
            self.logger.error(f"Error closing data storage: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
