"""
Unified Database Connection Manager for AlphaPlus
Provides centralized database connection management with proper error handling
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import asyncpg
from contextlib import asynccontextmanager
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Unified database connection manager for AlphaPlus
    Handles TimescaleDB/PostgreSQL connections with proper pooling and error handling
    """
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._connection_config: Dict[str, Any] = {}
        self._is_initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the database connection pool
        
        Args:
            config: Database configuration dictionary
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Use provided config or environment variables
            if config is None:
                config = self._get_default_config()
            
            self._connection_config = config
            
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=config.get('host', 'postgres'),
                port=config.get('port', 5432),
                database=config.get('database', 'alphapulse'),
                user=config.get('username', 'alpha_emon'),
                password=config.get('password', 'Emon_@17711'),
                min_size=config.get('min_size', 5),
                max_size=config.get('max_size', 20),
                command_timeout=config.get('command_timeout', 60),
                server_settings={
                    'application_name': 'alphapulse_trading_system'
                }
            )
            
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self._is_initialized = True
            logger.info("✅ Database connection pool initialized successfully")
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            self._is_initialized = False
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default database configuration from environment variables"""
        return {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'alphapulse'),
            'username': os.getenv('DB_USER', 'alpha_emon'),
            'password': os.getenv('DB_PASSWORD', 'Emon_@17711'),
            'min_size': int(os.getenv('DB_MIN_SIZE', '5')),
            'max_size': int(os.getenv('DB_MAX_SIZE', '20')),
            'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '60'))
        }
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the database connection pool"""
        return self._pool
    
    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._is_initialized
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool
        
        Yields:
            asyncpg.Connection: Database connection
        """
        if not self._is_initialized or self._pool is None:
            raise RuntimeError("Database not initialized")
        
        try:
            async with self._pool.acquire() as conn:
                yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    async def execute(self, query: str, *args, **kwargs) -> str:
        """
        Execute a database query
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            str: Query result
        """
        async with self.get_connection() as conn:
            return await conn.execute(query, *args, **kwargs)
    
    async def fetch(self, query: str, *args, **kwargs) -> list:
        """
        Fetch data from database
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            list: Query results
        """
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    async def fetchrow(self, query: str, *args, **kwargs) -> Optional[asyncpg.Record]:
        """
        Fetch a single row from database
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            **kwargs: Additional arguments
            
        Returns:
            Optional[asyncpg.Record]: Single row result
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args, **kwargs)
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._is_initialized:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self):
        """Perform database health check"""
        try:
            async with self.get_connection() as conn:
                await conn.execute('SELECT 1')
            logger.debug("Database health check passed")
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            # Could implement reconnection logic here
    
    async def health_check(self) -> bool:
        """Health check method for service manager"""
        try:
            if not self._is_initialized or not self._pool:
                return False
            
            async with self.get_connection() as conn:
                await conn.execute('SELECT 1')
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
        
        self._is_initialized = False

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
async def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager

async def initialize_database(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global database manager"""
    return await db_manager.initialize(config)

async def close_database():
    """Close the global database manager"""
    await db_manager.close()
