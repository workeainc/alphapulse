"""
Enhanced Database Connection for AlphaPlus
Supports both PostgreSQL and TimescaleDB with connection pooling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)

class AsyncPGConnectionWrapper:
    """Wrapper to provide asyncpg-style interface for SQLAlchemy async engine"""
    
    def __init__(self, engine):
        self.engine = engine
        self.connection = None
    
    async def __aenter__(self):
        self.connection = await self.engine.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            await self.connection.close()
    
    async def fetchrow(self, query, *args):
        """Execute query and return single row"""
        result = await self.connection.execute(text(query), args)
        return result.fetchone()
    
    async def fetch(self, query, *args):
        """Execute query and return all rows"""
        result = await self.connection.execute(text(query), args)
        return result.fetchall()
    
    async def execute(self, query, *args):
        """Execute query"""
        return await self.connection.execute(text(query), args)

class SQLiteConnectionWrapper:
    """Wrapper to provide asyncpg-style interface for SQLite"""
    
    def __init__(self, connection):
        self.connection = connection
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # SQLite connection is managed by the main class
    
    async def fetchrow(self, query, *args):
        """Execute query and return single row"""
        cursor = await self.connection.execute(query, args)
        row = await cursor.fetchone()
        await cursor.close()
        return row
    
    async def fetch(self, query, *args):
        """Execute query and return all rows"""
        cursor = await self.connection.execute(query, args)
        rows = await cursor.fetchall()
        await cursor.close()
        return rows
    
    async def execute(self, query, *args):
        """Execute query"""
        cursor = await self.connection.execute(query, args)
        await cursor.close()
        return cursor

class TimescaleDBConnection:
    """Enhanced TimescaleDB connection with connection pooling"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Database configuration - Updated with actual credentials
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 5432)
        self.database = self.config.get('database', 'alphapulse')
        self.username = self.config.get('username', 'alpha_emon')
        self.password = self.config.get('password', 'Emon_@17711')
        
        # Connection pooling - Phase 4.3 Optimized
        self.pool_size = self.config.get('pool_size', 30)  # Increased for better performance
        self.max_overflow = self.config.get('max_overflow', 50)  # Increased overflow
        self.pool_timeout = self.config.get('pool_timeout', 60)  # Increased timeout
        self.pool_recycle = self.config.get('pool_recycle', 1800)  # Reduced recycle time
        
        # Phase 4.3: TimescaleDB Optimization Settings
        self.batch_size = self.config.get('batch_size', 1000)  # Batch operations size
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.retention_days = self.config.get('retention_days', 90)  # Data retention
        self.chunk_time_interval = self.config.get('chunk_time_interval', '1 day')
        self.parallel_workers = self.config.get('parallel_workers', 4)
        
        # Phase 4.3: Performance Monitoring
        self.query_timeout = self.config.get('query_timeout', 30)
        self.max_connections_per_worker = self.config.get('max_connections_per_worker', 5)
        self.connection_health_check_interval = self.config.get('health_check_interval', 300)
        
        # Phase 4: Data Lifecycle Management
        self.lifecycle_enabled = self.config.get('lifecycle_enabled', True)
        self.retention_policies = self.config.get('retention_policies', {})
        self.compression_policies = self.config.get('compression_policies', {})
        self.cleanup_policies = self.config.get('cleanup_policies', {})
        self.archive_policies = self.config.get('archive_policies', {})
        
        # Phase 5: Security Enhancement
        self.security_enabled = self.config.get('security_enabled', True)
        self.audit_logging = self.config.get('audit_logging', {})
        self.access_control = self.config.get('access_control', {})
        self.secrets_management = self.config.get('secrets_management', {})
        self.security_monitoring = self.config.get('security_monitoring', {})
        
        # Connection objects
        self.async_engine = None
        self.async_session = None
        self.pool = None
        
        # Fallback support
        self.use_fallback = False
        self.fallback_db_path = self.config.get('fallback_db_path', 'data/alphapulse_fallback.db')
        self.sqlite_connection = None
        
        # Connection state
        self.connected = False
        self.connection_count = 0
        
        # Phase 4: Lifecycle Management Components
        self.lifecycle_manager = None
        
        # Phase 5: Security Components
        self.security_manager = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.connected:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def get_session(self):
        """Get a database session context manager"""
        if not self.async_session:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        # Return a context manager that properly handles async sessions
        class AsyncSessionContextManager:
            def __init__(self, session_factory):
                self.session_factory = session_factory
            
            async def __aenter__(self):
                self.session = self.session_factory()
                return self.session
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.session.close()
        
        return AsyncSessionContextManager(self.async_session)
    
    async def get_connection(self):
        """Get a raw database connection for asyncpg-style operations"""
        if not self.connected:
            await self.initialize()
        
        if self.use_fallback:
            # Return SQLite connection wrapper
            return SQLiteConnectionWrapper(self.sqlite_connection)
        else:
            # Return asyncpg connection wrapper
            return AsyncPGConnectionWrapper(self.async_engine)
    
    async def close(self):
        """Close database connections"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                self.async_engine = None
            if self.async_session:
                self.async_session.close_all()
                self.async_session = None
            if self.sqlite_connection:
                await self.sqlite_connection.close()
                self.sqlite_connection = None
            self.connected = False
            self.logger.info("Database connections closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
        
    async def initialize(self, create_tables: bool = False):
        """Initialize database connection and optionally create tables"""
        try:
            self.logger.info("Initializing TimescaleDB Connection...")
            
            # Try TimescaleDB first
            try:
                await self._initialize_timescaledb()
                self.logger.info("✅ Connected to TimescaleDB")
                self.use_fallback = False
            except Exception as e:
                self.logger.warning(f"TimescaleDB connection failed: {e}")
                self.logger.info("⚠️ Falling back to SQLite")
                await self._initialize_sqlite_fallback()
                self.use_fallback = True
            
            # Create tables if requested
            if create_tables:
                try:
                    await self._create_tables()
                except Exception as e:
                    self.logger.warning(f"Table creation failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _initialize_timescaledb(self):
        """Initialize TimescaleDB connection"""
        # Create async engine with URL-encoded password
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(self.password)
        database_url = f"postgresql+asyncpg://{self.username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
        
        self.async_engine = create_async_engine(
            database_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            echo=self.config.get('echo', False)
        )
        
        # Create async session factory
        self.async_session = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with self.async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        self.connected = True
    
    async def _initialize_sqlite_fallback(self):
        """Initialize SQLite fallback connection"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.fallback_db_path), exist_ok=True)
        
        # Create SQLite connection
        self.sqlite_connection = await aiosqlite.connect(self.fallback_db_path)
        
        # Create basic tables for SQLite fallback
        await self._create_sqlite_tables()
        
        self.connected = True
        self.logger.info(f"SQLite fallback initialized at {self.fallback_db_path}")
    
    async def _create_sqlite_tables(self):
        """Create basic tables for SQLite fallback"""
        if not self.sqlite_connection:
            return
            
        # Basic table creation for SQLite fallback
        tables = [
            """CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data JSON,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        ]
        
        for table_sql in tables:
            await self.sqlite_connection.execute(table_sql)
        
        await self.sqlite_connection.commit()
    
    def get_async_session(self):
        """Get async session factory for database operations"""
        if not self.connected:
            raise RuntimeError("Database connection not initialized. Call initialize() first.")
        
        # Return a context manager that properly handles async sessions
        class AsyncSessionContextManager:
            def __init__(self, session_factory):
                self.session_factory = session_factory
            
            async def __aenter__(self):
                self.session = self.session_factory()
                return self.session
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self.session.close()
        
        return AsyncSessionContextManager(self.async_session)
    
    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            async with self.async_session() as session:
                # Create candlestick data table with composite primary key for TimescaleDB
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS candlestick_data (
                        id SERIAL,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        open DECIMAL(20, 8) NOT NULL,
                        high DECIMAL(20, 8) NOT NULL,
                        low DECIMAL(20, 8) NOT NULL,
                        close DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        indicators JSONB,
                        patterns TEXT[],
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, id)
                    );
                """))
                
                # Create hypertable for time-series data
                await session.execute(text("""
                    SELECT create_hypertable('candlestick_data', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """))
                
                # Create trades table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        signal_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20, 8) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        status VARCHAR(20) DEFAULT 'open',
                        exit_price DECIMAL(20, 8),
                        exit_timestamp TIMESTAMPTZ,
                        pnl DECIMAL(20, 8),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create signals table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id VARCHAR(50) PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        strength VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        stop_loss DECIMAL(20, 8),
                        take_profit DECIMAL(20, 8),
                        metadata JSONB,
                        status VARCHAR(20) DEFAULT 'active',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create enhanced_signals table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS enhanced_signals (
                        id VARCHAR(50) PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        strength VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        stop_loss DECIMAL(20, 8),
                        take_profit DECIMAL(20, 8),
                        metadata JSONB,
                        ichimoku_data JSONB,
                        fibonacci_data JSONB,
                        volume_analysis JSONB,
                        advanced_indicators JSONB,
                        smc_analysis JSONB,
                        order_blocks_data JSONB[],
                        fair_value_gaps_data JSONB[],
                        liquidity_sweeps_data JSONB[],
                        market_structures_data JSONB[],
                        smc_confidence DECIMAL(5, 4),
                        smc_bias VARCHAR(20),
                        dl_analysis JSONB,
                        lstm_prediction DECIMAL(5, 4),
                        cnn_prediction DECIMAL(5, 4),
                        lstm_cnn_prediction DECIMAL(5, 4),
                        ensemble_prediction DECIMAL(5, 4),
                        dl_confidence DECIMAL(5, 4),
                        dl_bias VARCHAR(20),
                        rl_analysis JSONB,
                        rl_action_type VARCHAR(20),
                        rl_position_size DECIMAL(10, 4),
                        rl_stop_loss DECIMAL(10, 4),
                        rl_take_profit DECIMAL(10, 4),
                        rl_confidence_threshold DECIMAL(5, 4),
                        rl_risk_allocation DECIMAL(5, 4),
                        rl_optimization_params JSONB,
                        rl_bias VARCHAR(20),
                        rl_action_strength DECIMAL(5, 4),
                        rl_training_episodes INTEGER DEFAULT 0,
                        rl_avg_reward DECIMAL(10, 6),
                        rl_best_reward DECIMAL(10, 6),
                        nlp_analysis JSONB,
                        nlp_overall_sentiment_score DECIMAL(5, 4),
                        nlp_overall_confidence DECIMAL(5, 4),
                        nlp_news_sentiment DECIMAL(5, 4),
                        nlp_news_confidence DECIMAL(5, 4),
                        nlp_twitter_sentiment DECIMAL(5, 4),
                        nlp_twitter_confidence DECIMAL(5, 4),
                        nlp_reddit_sentiment DECIMAL(5, 4),
                        nlp_reddit_confidence DECIMAL(5, 4),
                        nlp_bias VARCHAR(20),
                        nlp_sentiment_strength DECIMAL(5, 4),
                        nlp_high_confidence_sentiment BOOLEAN,
                        nlp_analyses_performed INTEGER DEFAULT 0,
                        nlp_cache_hit_rate DECIMAL(5, 4),
                        nlp_models_available JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create performance metrics table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        win_rate DECIMAL(5, 4) NOT NULL,
                        total_pnl DECIMAL(20, 8) NOT NULL,
                        daily_pnl DECIMAL(20, 8) NOT NULL,
                        active_positions INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create real-time market data table for high-frequency updates
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS real_time_market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(20, 8) NOT NULL,
                        bid DECIMAL(20, 8),
                        ask DECIMAL(20, 8),
                        bid_volume DECIMAL(20, 8),
                        ask_volume DECIMAL(20, 8),
                        exchange VARCHAR(20) NOT NULL,
                        data_type VARCHAR(20) NOT NULL, -- 'tick', 'order_book', 'trade'
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for real-time data
                await session.execute(text("""
                    SELECT create_hypertable('real_time_market_data', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create order book snapshots table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS order_book_snapshots (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        bids JSONB NOT NULL, -- Array of [price, volume] pairs
                        asks JSONB NOT NULL, -- Array of [price, volume] pairs
                        spread DECIMAL(20, 8),
                        total_bid_volume DECIMAL(20, 8),
                        total_ask_volume DECIMAL(20, 8),
                        depth_levels INTEGER NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for order book data
                await session.execute(text("""
                    SELECT create_hypertable('order_book_snapshots', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create liquidation events table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS liquidation_events (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL, -- 'long' or 'short'
                        price DECIMAL(20, 8) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        quote_quantity DECIMAL(20, 8) NOT NULL,
                        liquidation_type VARCHAR(20), -- 'partial', 'full'
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for liquidation data
                await session.execute(text("""
                    SELECT create_hypertable('liquidation_events', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create market depth analysis table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_depth_analysis (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        analysis_type VARCHAR(20) NOT NULL, -- 'liquidity_walls', 'order_clusters', 'imbalance'
                        price_level DECIMAL(20, 8) NOT NULL,
                        volume_at_level DECIMAL(20, 8) NOT NULL,
                        side VARCHAR(10) NOT NULL, -- 'bid' or 'ask'
                        confidence DECIMAL(5, 4) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for market depth analysis
                await session.execute(text("""
                    SELECT create_hypertable('market_depth_analysis', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create on-chain events table for blockchain data
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS on_chain_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        chain VARCHAR(20) NOT NULL,
                        tx_hash VARCHAR(66) NOT NULL,
                        from_address VARCHAR(42),
                        to_address VARCHAR(42),
                        value DECIMAL(30, 18),
                        gas_used BIGINT,
                        event_type VARCHAR(50),
                        symbol VARCHAR(20),
                        block_number BIGINT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for on-chain data
                await session.execute(text("""
                    SELECT create_hypertable('on_chain_events', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create social sentiment table for social media data
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS social_sentiment (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        tweet_id BIGINT,
                        text TEXT,
                        sentiment_score FLOAT,
                        cluster_label INT,
                        source VARCHAR(20) DEFAULT 'twitter',
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for social sentiment data
                await session.execute(text("""
                    SELECT create_hypertable('social_sentiment', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create anomaly events table for Week 7.2 Phase 3
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS anomaly_events (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        anomaly_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        value DECIMAL(20, 8) NOT NULL,
                        threshold DECIMAL(20, 8) NOT NULL,
                        description TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))

                # Create hypertable for anomaly events
                await session.execute(text("""
                    SELECT create_hypertable('anomaly_events', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))

                # Create anomaly alerts table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS anomaly_alerts (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        message TEXT NOT NULL,
                        data JSONB,
                        action_required BOOLEAN DEFAULT FALSE,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        acknowledged_by VARCHAR(50),
                        acknowledged_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))

                # Create hypertable for anomaly alerts
                await session.execute(text("""
                    SELECT create_hypertable('anomaly_alerts', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))

                # Create funding rates table for Week 7.3 Phase 1
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS funding_rates (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        funding_rate DECIMAL(10, 8) NOT NULL,
                        next_funding_time TIMESTAMPTZ,
                        estimated_rate DECIMAL(10, 8),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))

                # Create hypertable for funding rates
                await session.execute(text("""
                    SELECT create_hypertable('funding_rates', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create indexes for performance
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_candlestick_symbol_time 
                    ON candlestick_data (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_real_time_symbol_time 
                    ON real_time_market_data (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_order_book_symbol_time 
                    ON order_book_snapshots (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_liquidation_symbol_time 
                    ON liquidation_events (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_market_depth_symbol_time 
                    ON market_depth_analysis (symbol, timestamp DESC);
                """))

                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_exchange_time 
                    ON funding_rates (symbol, exchange, timestamp DESC);
                """))

                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_funding_rates_timestamp 
                    ON funding_rates (timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_on_chain_chain_timestamp 
                    ON on_chain_events (chain, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_on_chain_symbol_timestamp 
                    ON on_chain_events (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_on_chain_tx_hash 
                    ON on_chain_events (tx_hash);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_social_symbol_timestamp 
                    ON social_sentiment (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_social_sentiment_score 
                    ON social_sentiment (sentiment_score);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_social_cluster_label 
                    ON social_sentiment (cluster_label);
                """))
                
                # Create indexes for anomaly tables
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomaly_symbol_timestamp 
                    ON anomaly_events (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomaly_type_severity 
                    ON anomaly_events (anomaly_type, severity);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomaly_alert_symbol_timestamp 
                    ON anomaly_alerts (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomaly_alert_type_severity 
                    ON anomaly_alerts (alert_type, severity);
                """))
                
                # Create signal predictions table for Week 7.4
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS signal_predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        signal_type VARCHAR(50) NOT NULL,
                        confidence FLOAT NOT NULL,
                        predicted_pnl FLOAT,
                        features JSONB,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for signal predictions
                await session.execute(text("""
                    SELECT create_hypertable('signal_predictions', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create indexes for signal predictions
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_signal_predictions_symbol_time 
                    ON signal_predictions (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_signal_predictions_confidence 
                    ON signal_predictions (confidence DESC, timestamp DESC);
                """))
                
                # Create strategy configurations table for Week 8
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS strategy_configs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        strategy_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20),
                        max_loss_pct FLOAT DEFAULT 0.05,
                        take_profit_pct FLOAT DEFAULT 0.03,
                        stop_loss_pct FLOAT DEFAULT 0.02,
                        max_leverage FLOAT DEFAULT 10.0,
                        position_size_pct FLOAT DEFAULT 0.1,
                        risk_reward_ratio FLOAT DEFAULT 2.0,
                        max_daily_trades INTEGER DEFAULT 10,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for strategy configs
                await session.execute(text("""
                    SELECT create_hypertable('strategy_configs', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """))
                
                # Create indexes for strategy configs
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_strategy_configs_strategy_symbol 
                    ON strategy_configs (strategy_id, symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_strategy_configs_risk_params 
                    ON strategy_configs (max_loss_pct, stop_loss_pct, timestamp DESC);
                """))
                
                # Create performance metrics table for Week 8
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        strategy_id VARCHAR(50) NOT NULL,
                        pnl FLOAT DEFAULT 0.0,
                        win_rate FLOAT DEFAULT 0.0,
                        drawdown FLOAT DEFAULT 0.0,
                        execution_time FLOAT DEFAULT 0.0,
                        trade_count INTEGER DEFAULT 0,
                        avg_position_size FLOAT DEFAULT 0.0,
                        max_position_size FLOAT DEFAULT 0.0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for performance metrics
                await session.execute(text("""
                    SELECT create_hypertable('performance_metrics', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create indexes for performance metrics
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol_strategy 
                    ON performance_metrics (symbol, strategy_id, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_performance_metrics_pnl_drawdown 
                    ON performance_metrics (pnl, drawdown, timestamp DESC);
                """))
                
                # Create anomalies table for Week 8
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS anomalies (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        data_type VARCHAR(50) NOT NULL,
                        value FLOAT NOT NULL,
                        z_score FLOAT NOT NULL,
                        threshold FLOAT DEFAULT 3.0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for anomalies
                await session.execute(text("""
                    SELECT create_hypertable('anomalies', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create indexes for anomalies
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomalies_symbol_time 
                    ON anomalies (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_anomalies_z_score 
                    ON anomalies (z_score DESC, timestamp DESC);
                """))
                
                # Create system metrics table for Week 8
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        metric_unit VARCHAR(20),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for system metrics
                await session.execute(text("""
                    SELECT create_hypertable('system_metrics', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create compliance reports table for Week 9
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS compliance_reports (
                        id SERIAL PRIMARY KEY,
                        report_id VARCHAR(100) UNIQUE NOT NULL,
                        report_type VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        period_start TIMESTAMPTZ NOT NULL,
                        period_end TIMESTAMPTZ NOT NULL,
                        risk_metrics JSONB,
                        position_data JSONB,
                        stress_test_results JSONB,
                        compliance_status VARCHAR(50) NOT NULL,
                        recommendations JSONB,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for compliance reports
                await session.execute(text("""
                    SELECT create_hypertable('compliance_reports', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """))
                
                # Create audit trail table for Week 9
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id SERIAL PRIMARY KEY,
                        entry_id VARCHAR(100) UNIQUE NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        action VARCHAR(200) NOT NULL,
                        user VARCHAR(100) NOT NULL,
                        details JSONB,
                        risk_impact FLOAT,
                        compliance_status VARCHAR(50) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """))
                
                # Create hypertable for audit trail
                await session.execute(text("""
                    SELECT create_hypertable('audit_trail', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    );
                """))
                
                # Create indexes for system metrics
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time 
                    ON system_metrics (metric_name, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                    ON system_metrics (timestamp DESC);
                """))
                
                # Create indexes for compliance reports (Week 9)
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_reports_symbol_time 
                    ON compliance_reports (symbol, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_compliance_reports_status 
                    ON compliance_reports (compliance_status, timestamp DESC);
                """))
                
                # Create indexes for audit trail (Week 9)
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_trail_user_time 
                    ON audit_trail (user, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_trail_action_time 
                    ON audit_trail (action, timestamp DESC);
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_audit_trail_compliance_status 
                    ON audit_trail (compliance_status, timestamp DESC);
                """))
                
                await session.commit()
                self.logger.info("Database tables created successfully")

        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    async def get_session(self) -> AsyncSession:
        """Get a database session"""
        if not self.connected:
            raise ConnectionError("Database not connected")
        
        return self.async_session()
    
    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save trade data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO trades (signal_id, symbol, side, entry_price, quantity, 
                                     timestamp, strategy, confidence)
                    VALUES (:signal_id, :symbol, :side, :entry_price, :quantity, 
                           :timestamp, :strategy, :confidence)
                    RETURNING id
                """)
                
                result = await session.execute(query, trade_data)
                await session.commit()
                
                trade_id = result.scalar()
                self.logger.info(f"Trade saved with ID: {trade_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            return False
    
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Save signal data to database with advanced indicators and SMC"""
        try:
            async with self.async_session() as session:
                # Extract advanced indicator data
                indicators = signal_data.get('metadata', {}).get('indicators', {})
                
                # Prepare advanced indicator data
                ichimoku_data = indicators.get('ichimoku', {})
                fibonacci_data = indicators.get('fibonacci', {})
                volume_analysis = indicators.get('volume_analysis', {})
                advanced_indicators = indicators.get('advanced_indicators', {})
                
                # Prepare Smart Money Concepts data
                smc_analysis = indicators.get('smc_analysis', {})
                order_blocks_data = smc_analysis.get('order_blocks', [])
                fair_value_gaps_data = smc_analysis.get('fair_value_gaps', [])
                liquidity_sweeps_data = smc_analysis.get('liquidity_sweeps', [])
                market_structures_data = smc_analysis.get('market_structures', [])
                smc_confidence = smc_analysis.get('overall_confidence', 0.0)
                smc_bias = smc_analysis.get('smc_bias', 'neutral')
                
                # Prepare Deep Learning data
                dl_analysis = indicators.get('dl_analysis', {})
                lstm_prediction = dl_analysis.get('lstm_prediction', 0.0)
                cnn_prediction = dl_analysis.get('cnn_prediction', 0.0)
                lstm_cnn_prediction = dl_analysis.get('lstm_cnn_prediction', 0.0)
                ensemble_prediction = dl_analysis.get('ensemble_prediction', 0.0)
                dl_confidence = dl_analysis.get('dl_confidence', 0.0)
                dl_bias = dl_analysis.get('dl_bias', 'neutral')
                
                # Prepare Reinforcement Learning data
                rl_analysis = indicators.get('rl_analysis', {})
                rl_action_type = rl_analysis.get('action_type', 'hold')
                rl_position_size = rl_analysis.get('position_size', 0.0)
                rl_stop_loss = rl_analysis.get('stop_loss', 0.0)
                rl_take_profit = rl_analysis.get('take_profit', 0.0)
                rl_confidence_threshold = rl_analysis.get('confidence_threshold', 0.0)
                rl_risk_allocation = rl_analysis.get('risk_allocation', 0.0)
                rl_optimization_params = rl_analysis.get('optimization_params', {})
                rl_bias = rl_analysis.get('rl_bias', 'neutral')
                rl_action_strength = rl_analysis.get('action_strength', 0.0)
                rl_training_episodes = rl_analysis.get('training_episodes', 0)
                rl_avg_reward = rl_analysis.get('avg_reward', 0.0)
                rl_best_reward = rl_analysis.get('best_reward', 0.0)
                
                # Prepare Natural Language Processing data
                nlp_analysis = indicators.get('nlp_analysis', {})
                nlp_overall_sentiment_score = nlp_analysis.get('overall_sentiment_score', 0.0)
                nlp_overall_confidence = nlp_analysis.get('overall_confidence', 0.0)
                nlp_news_sentiment = nlp_analysis.get('news_sentiment', 0.0)
                nlp_news_confidence = nlp_analysis.get('news_confidence', 0.0)
                nlp_twitter_sentiment = nlp_analysis.get('twitter_sentiment', 0.0)
                nlp_twitter_confidence = nlp_analysis.get('twitter_confidence', 0.0)
                nlp_reddit_sentiment = nlp_analysis.get('reddit_sentiment', 0.0)
                nlp_reddit_confidence = nlp_analysis.get('reddit_confidence', 0.0)
                nlp_bias = nlp_analysis.get('nlp_bias', 'neutral')
                nlp_sentiment_strength = nlp_analysis.get('sentiment_strength', 0.0)
                nlp_high_confidence_sentiment = nlp_analysis.get('high_confidence_sentiment', False)
                nlp_analyses_performed = nlp_analysis.get('analyses_performed', 0)
                nlp_cache_hit_rate = nlp_analysis.get('cache_hit_rate', 0.0)
                nlp_models_available = nlp_analysis.get('models_available', {})
                
                query = text("""
                    INSERT INTO enhanced_signals (
                        id, symbol, side, strategy, confidence, strength, timestamp, 
                        price, stop_loss, take_profit, metadata, ichimoku_data, 
                        fibonacci_data, volume_analysis, advanced_indicators,
                        smc_analysis, order_blocks_data, fair_value_gaps_data,
                        liquidity_sweeps_data, market_structures_data, smc_confidence, smc_bias,
                        dl_analysis, lstm_prediction, cnn_prediction, lstm_cnn_prediction,
                        ensemble_prediction, dl_confidence, dl_bias,
                        rl_analysis, rl_action_type, rl_position_size, rl_stop_loss, rl_take_profit,
                        rl_confidence_threshold, rl_risk_allocation, rl_optimization_params,
                        rl_bias, rl_action_strength, rl_training_episodes, rl_avg_reward, rl_best_reward,
                        nlp_analysis, nlp_overall_sentiment_score, nlp_overall_confidence,
                        nlp_news_sentiment, nlp_news_confidence, nlp_twitter_sentiment, nlp_twitter_confidence,
                        nlp_reddit_sentiment, nlp_reddit_confidence, nlp_bias, nlp_sentiment_strength,
                        nlp_high_confidence_sentiment, nlp_analyses_performed, nlp_cache_hit_rate, nlp_models_available
                    )
                    VALUES (
                        :id, :symbol, :side, :strategy, :confidence, :strength, :timestamp,
                        :price, :stop_loss, :take_profit, :metadata, :ichimoku_data,
                        :fibonacci_data, :volume_analysis, :advanced_indicators,
                        :smc_analysis, :order_blocks_data, :fair_value_gaps_data,
                        :liquidity_sweeps_data, :market_structures_data, :smc_confidence, :smc_bias,
                        :dl_analysis, :lstm_prediction, :cnn_prediction, :lstm_cnn_prediction,
                        :ensemble_prediction, :dl_confidence, :dl_bias,
                        :rl_analysis, :rl_action_type, :rl_position_size, :rl_stop_loss, :rl_take_profit,
                        :rl_confidence_threshold, :rl_risk_allocation, :rl_optimization_params,
                        :rl_bias, :rl_action_strength, :rl_training_episodes, :rl_avg_reward, :rl_best_reward,
                        :nlp_analysis, :nlp_overall_sentiment_score, :nlp_overall_confidence,
                        :nlp_news_sentiment, :nlp_news_confidence, :nlp_twitter_sentiment, :nlp_twitter_confidence,
                        :nlp_reddit_sentiment, :nlp_reddit_confidence, :nlp_bias, :nlp_sentiment_strength,
                        :nlp_high_confidence_sentiment, :nlp_analyses_performed, :nlp_cache_hit_rate, :nlp_models_available
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        price = EXCLUDED.price,
                        metadata = EXCLUDED.metadata,
                        ichimoku_data = EXCLUDED.ichimoku_data,
                        fibonacci_data = EXCLUDED.fibonacci_data,
                        volume_analysis = EXCLUDED.volume_analysis,
                        advanced_indicators = EXCLUDED.advanced_indicators,
                        smc_analysis = EXCLUDED.smc_analysis,
                        order_blocks_data = EXCLUDED.order_blocks_data,
                        fair_value_gaps_data = EXCLUDED.fair_value_gaps_data,
                        liquidity_sweeps_data = EXCLUDED.liquidity_sweeps_data,
                        market_structures_data = EXCLUDED.market_structures_data,
                        smc_confidence = EXCLUDED.smc_confidence,
                        smc_bias = EXCLUDED.smc_bias,
                        dl_analysis = EXCLUDED.dl_analysis,
                        lstm_prediction = EXCLUDED.lstm_prediction,
                        cnn_prediction = EXCLUDED.cnn_prediction,
                        lstm_cnn_prediction = EXCLUDED.lstm_cnn_prediction,
                        ensemble_prediction = EXCLUDED.ensemble_prediction,
                        dl_confidence = EXCLUDED.dl_confidence,
                        dl_bias = EXCLUDED.dl_bias,
                        rl_analysis = EXCLUDED.rl_analysis,
                        rl_action_type = EXCLUDED.rl_action_type,
                        rl_position_size = EXCLUDED.rl_position_size,
                        rl_stop_loss = EXCLUDED.rl_stop_loss,
                        rl_take_profit = EXCLUDED.rl_take_profit,
                        rl_confidence_threshold = EXCLUDED.rl_confidence_threshold,
                        rl_risk_allocation = EXCLUDED.rl_risk_allocation,
                        rl_optimization_params = EXCLUDED.rl_optimization_params,
                        rl_bias = EXCLUDED.rl_bias,
                        rl_action_strength = EXCLUDED.rl_action_strength,
                        rl_training_episodes = EXCLUDED.rl_training_episodes,
                        rl_avg_reward = EXCLUDED.rl_avg_reward,
                        rl_best_reward = EXCLUDED.rl_best_reward,
                        nlp_analysis = EXCLUDED.nlp_analysis,
                        nlp_overall_sentiment_score = EXCLUDED.nlp_overall_sentiment_score,
                        nlp_overall_confidence = EXCLUDED.nlp_overall_confidence,
                        nlp_news_sentiment = EXCLUDED.nlp_news_sentiment,
                        nlp_news_confidence = EXCLUDED.nlp_news_confidence,
                        nlp_twitter_sentiment = EXCLUDED.nlp_twitter_sentiment,
                        nlp_twitter_confidence = EXCLUDED.nlp_twitter_confidence,
                        nlp_reddit_sentiment = EXCLUDED.nlp_reddit_sentiment,
                        nlp_reddit_confidence = EXCLUDED.nlp_reddit_confidence,
                        nlp_bias = EXCLUDED.nlp_bias,
                        nlp_sentiment_strength = EXCLUDED.nlp_sentiment_strength,
                        nlp_high_confidence_sentiment = EXCLUDED.nlp_high_confidence_sentiment,
                        nlp_analyses_performed = EXCLUDED.nlp_analyses_performed,
                        nlp_cache_hit_rate = EXCLUDED.nlp_cache_hit_rate,
                        nlp_models_available = EXCLUDED.nlp_models_available,
                        updated_at = NOW()
                """)
                
                await session.execute(query, {
                    'id': signal_data.get('id'),
                    'symbol': signal_data.get('symbol'),
                    'side': signal_data.get('side'),
                    'strategy': signal_data.get('strategy'),
                    'confidence': signal_data.get('confidence'),
                    'strength': signal_data.get('strength'),
                    'timestamp': signal_data.get('timestamp'),
                    'price': signal_data.get('price'),
                    'stop_loss': signal_data.get('stop_loss'),
                    'take_profit': signal_data.get('take_profit'),
                    'metadata': signal_data.get('metadata'),
                    'ichimoku_data': ichimoku_data,
                    'fibonacci_data': fibonacci_data,
                    'volume_analysis': volume_analysis,
                    'advanced_indicators': advanced_indicators,
                    'smc_analysis': smc_analysis,
                    'order_blocks_data': order_blocks_data,
                    'fair_value_gaps_data': fair_value_gaps_data,
                    'liquidity_sweeps_data': liquidity_sweeps_data,
                    'market_structures_data': market_structures_data,
                    'smc_confidence': smc_confidence,
                    'smc_bias': smc_bias,
                    'dl_analysis': dl_analysis,
                    'lstm_prediction': lstm_prediction,
                    'cnn_prediction': cnn_prediction,
                    'lstm_cnn_prediction': lstm_cnn_prediction,
                    'ensemble_prediction': ensemble_prediction,
                    'dl_confidence': dl_confidence,
                    'dl_bias': dl_bias,
                    'rl_analysis': rl_analysis,
                    'rl_action_type': rl_action_type,
                    'rl_position_size': rl_position_size,
                    'rl_stop_loss': rl_stop_loss,
                    'rl_take_profit': rl_take_profit,
                    'rl_confidence_threshold': rl_confidence_threshold,
                    'rl_risk_allocation': rl_risk_allocation,
                    'rl_optimization_params': rl_optimization_params,
                    'rl_bias': rl_bias,
                    'rl_action_strength': rl_action_strength,
                    'rl_training_episodes': rl_training_episodes,
                    'rl_avg_reward': rl_avg_reward,
                    'rl_best_reward': rl_best_reward,
                    'nlp_analysis': nlp_analysis,
                    'nlp_overall_sentiment_score': nlp_overall_sentiment_score,
                    'nlp_overall_confidence': nlp_overall_confidence,
                    'nlp_news_sentiment': nlp_news_sentiment,
                    'nlp_news_confidence': nlp_news_confidence,
                    'nlp_twitter_sentiment': nlp_twitter_sentiment,
                    'nlp_twitter_confidence': nlp_twitter_confidence,
                    'nlp_reddit_sentiment': nlp_reddit_sentiment,
                    'nlp_reddit_confidence': nlp_reddit_confidence,
                    'nlp_bias': nlp_bias,
                    'nlp_sentiment_strength': nlp_sentiment_strength,
                    'nlp_high_confidence_sentiment': nlp_high_confidence_sentiment,
                    'nlp_analyses_performed': nlp_analyses_performed,
                    'nlp_cache_hit_rate': nlp_cache_hit_rate,
                    'nlp_models_available': nlp_models_available
                })
                
                await session.commit()
                
                self.logger.info(f"Signal saved with advanced indicators, SMC, Deep Learning, RL, and NLP: {signal_data.get('id')}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal with advanced indicators, SMC, Deep Learning, RL, and NLP: {e}")
            return False
    
    async def get_latest_signals(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest signals from enhanced_signals table"""
        try:
            async with self.async_session() as session:
                if symbol:
                    query = text("""
                        SELECT id, symbol, side, strategy, confidence, strength, timestamp, 
                               price, stop_loss, take_profit, metadata
                        FROM enhanced_signals
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """)
                    result = await session.execute(query, {'symbol': symbol, 'limit': limit})
                else:
                    query = text("""
                        SELECT id, symbol, side, strategy, confidence, strength, timestamp, 
                               price, stop_loss, take_profit, metadata
                        FROM enhanced_signals
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """)
                    result = await session.execute(query, {'limit': limit})
                
                rows = result.fetchall()
                signals = []
                for row in rows:
                    signal = dict(row._mapping)
                    # Convert timestamp to string for JSON serialization
                    if signal.get('timestamp'):
                        signal['timestamp'] = signal['timestamp'].isoformat()
                    signals.append(signal)
                
                self.logger.info(f"Retrieved {len(signals)} signals from database")
                return signals
                
        except Exception as e:
            self.logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def save_signal_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save signal prediction data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO signal_predictions (timestamp, symbol, signal_type, 
                                                 confidence, predicted_pnl, features, metadata)
                    VALUES (:timestamp, :symbol, :signal_type, :confidence, 
                           :predicted_pnl, :features, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, prediction_data)
                await session.commit()
                
                prediction_id = result.scalar()
                self.logger.info(f"Signal prediction saved with ID: {prediction_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal prediction: {e}")
            return False
    
    async def save_anomaly(self, anomaly_data: Dict[str, Any]) -> bool:
        """Save anomaly data to database for Week 8 dashboard"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO anomalies (timestamp, symbol, data_type, value, z_score, threshold, metadata)
                    VALUES (:timestamp, :symbol, :data_type, :value, :z_score, :threshold, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, anomaly_data)
                await session.commit()
                
                anomaly_id = result.scalar()
                self.logger.info(f"Anomaly saved with ID: {anomaly_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving anomaly: {e}")
            return False
    
    async def save_system_metric(self, metric_data: Dict[str, Any]) -> bool:
        """Save system metric data to database for Week 8 dashboard"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO system_metrics (timestamp, metric_name, metric_value, metric_unit, metadata)
                    VALUES (:timestamp, :metric_name, :metric_value, :metric_unit, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, metric_data)
                await session.commit()
                
                metric_id = result.scalar()
                self.logger.info(f"System metric saved with ID: {metric_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving system metric: {e}")
            return False
    
    async def save_strategy_config(self, config_data: Dict[str, Any]) -> bool:
        """Save strategy configuration to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO strategy_configs (timestamp, strategy_id, symbol, max_loss_pct,
                                                 take_profit_pct, stop_loss_pct, max_leverage,
                                                 position_size_pct, risk_reward_ratio, max_daily_trades, metadata)
                    VALUES (:timestamp, :strategy_id, :symbol, :max_loss_pct, :take_profit_pct,
                           :stop_loss_pct, :max_leverage, :position_size_pct, :risk_reward_ratio,
                           :max_daily_trades, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, config_data)
                await session.commit()
                
                config_id = result.scalar()
                self.logger.info(f"Strategy config saved with ID: {config_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving strategy config: {e}")
            return False
    
    async def get_strategy_config(self, strategy_id: str, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get strategy configuration from database"""
        try:
            async with self.async_session() as session:
                if symbol:
                    query = text("""
                        SELECT * FROM strategy_configs 
                        WHERE strategy_id = :strategy_id AND symbol = :symbol
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    result = await session.execute(query, {'strategy_id': strategy_id, 'symbol': symbol})
                else:
                    query = text("""
                        SELECT * FROM strategy_configs 
                        WHERE strategy_id = :strategy_id
                        ORDER BY timestamp DESC LIMIT 1
                    """)
                    result = await session.execute(query, {'strategy_id': strategy_id})
                
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting strategy config: {e}")
            return None
    
    async def update_strategy_config(self, config_data: Dict[str, Any]) -> bool:
        """Update existing strategy configuration"""
        try:
            async with self.async_session() as session:
                query = text("""
                    UPDATE strategy_configs 
                    SET max_loss_pct = :max_loss_pct, take_profit_pct = :take_profit_pct,
                        stop_loss_pct = :stop_loss_pct, max_leverage = :max_leverage,
                        position_size_pct = :position_size_pct, risk_reward_ratio = :risk_reward_ratio,
                        max_daily_trades = :max_daily_trades, metadata = :metadata,
                        timestamp = NOW()
                    WHERE strategy_id = :strategy_id AND symbol = :symbol
                """)
                
                await session.execute(query, config_data)
                await session.commit()
                
                self.logger.info(f"Strategy config updated for {config_data['strategy_id']} - {config_data['symbol']}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error updating strategy config: {e}")
            return False
    
    async def save_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Save performance metrics to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO performance_metrics (timestamp, symbol, strategy_id, pnl, win_rate,
                                                   drawdown, execution_time, trade_count,
                                                   avg_position_size, max_position_size, metadata)
                    VALUES (:timestamp, :symbol, :strategy_id, :pnl, :win_rate, :drawdown,
                           :execution_time, :trade_count, :avg_position_size, :max_position_size, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, metrics_data)
                await session.commit()
                
                metric_id = result.scalar()
                self.logger.info(f"Performance metrics saved with ID: {metric_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            return False
    
    async def get_performance_metrics(self, symbol: str = None, strategy_id: str = None, 
                                     limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance metrics from database"""
        try:
            async with self.async_session() as session:
                where_conditions = []
                params = {}
                
                if symbol:
                    where_conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if strategy_id:
                    where_conditions.append("strategy_id = :strategy_id")
                    params['strategy_id'] = strategy_id
                
                if start_time:
                    where_conditions.append("timestamp >= :start_time")
                    params['start_time'] = start_time
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                params['limit'] = limit
                
                query = text(f"""
                    SELECT * FROM performance_metrics 
                    WHERE {where_clause}
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return []
    
    async def get_performance_summary(self, symbol: str = None, strategy_id: str = None,
                                     days: int = 30) -> Optional[Dict[str, Any]]:
        """Get performance summary for specified period"""
        try:
            async with self.async_session() as session:
                where_conditions = []
                params = {}
                
                if symbol:
                    where_conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if strategy_id:
                    where_conditions.append("strategy_id = :strategy_id")
                    params['strategy_id'] = strategy_id
                
                params['days'] = days
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = text(f"""
                    SELECT 
                        COUNT(*) as total_trades,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        AVG(win_rate) as avg_win_rate,
                        MAX(drawdown) as max_drawdown,
                        AVG(execution_time) as avg_execution_time,
                        AVG(avg_position_size) as avg_position_size
                    FROM performance_metrics 
                    WHERE {where_clause} AND timestamp >= NOW() - INTERVAL ':days days'
                """)
                
                result = await session.execute(query, params)
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return None
    
    async def save_candlestick(self, candlestick_data: Dict[str, Any]) -> bool:
        """Save candlestick data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO candlestick_data (symbol, timestamp, open, high, low, close, 
                                                volume, timeframe, indicators, patterns)
                    VALUES (:symbol, :timestamp, :open, :high, :low, :close, 
                           :volume, :timeframe, :indicators, :patterns)
                    ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        indicators = EXCLUDED.indicators,
                        patterns = EXCLUDED.patterns,
                        updated_at = NOW()
                """)
                
                await session.execute(query, candlestick_data)
                await session.commit()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving candlestick: {e}")
            return False
    
    async def get_candlestick_data(self, symbol: str, timeframe: str, 
                                  limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get candlestick data from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT symbol, timestamp, open, high, low, close, volume, 
                           timeframe, indicators, patterns
                    FROM candlestick_data
                    WHERE symbol = :symbol AND timeframe = :timeframe
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting candlestick data: {e}")
            return []
    
    async def get_trades(self, symbol: Optional[str] = None, 
                        start_time: Optional[datetime] = None, 
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """Get trades from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT id, signal_id, symbol, side, entry_price, quantity, 
                           timestamp, strategy, confidence, status, exit_price, 
                           exit_timestamp, pnl
                    FROM trades
                    WHERE (:symbol IS NULL OR symbol = :symbol)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return []
    
    async def update_trade_status(self, trade_id: int, status: str, 
                                 exit_price: Optional[float] = None, 
                                 pnl: Optional[float] = None) -> bool:
        """Update trade status"""
        try:
            async with self.async_session() as session:
                query = text("""
                    UPDATE trades 
                    SET status = :status, 
                        exit_price = :exit_price,
                        exit_timestamp = CASE WHEN :exit_price IS NOT NULL THEN NOW() ELSE exit_timestamp END,
                        pnl = :pnl
                    WHERE id = :trade_id
                """)
                
                await session.execute(query, {
                    'trade_id': trade_id,
                    'status': status,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                
                await session.commit()
                return True
            
        except Exception as e:
            self.logger.error(f"Error updating trade status: {e}")
            return False
    
    async def save_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Save performance metrics to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO performance_metrics (timestamp, total_trades, winning_trades, 
                                                   losing_trades, win_rate, total_pnl, daily_pnl, active_positions)
                    VALUES (:timestamp, :total_trades, :winning_trades, :losing_trades, 
                           :win_rate, :total_pnl, :daily_pnl, :active_positions)
                """)
                
                await session.execute(query, metrics_data)
                await session.commit()
                
                return True
                    
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            return False

    async def save_real_time_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Save real-time market data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO real_time_market_data (symbol, timestamp, price, volume, bid, ask, 
                                                     bid_volume, ask_volume, exchange, data_type, metadata)
                    VALUES (:symbol, :timestamp, :price, :volume, :bid, :ask, 
                           :bid_volume, :ask_volume, :exchange, :data_type, :metadata)
                """)
                
                await session.execute(query, market_data)
                await session.commit()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving real-time market data: {e}")
            return False
    
    async def save_order_book_snapshot(self, order_book_data: Dict[str, Any]) -> bool:
        """Save order book snapshot to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO order_book_snapshots (symbol, timestamp, exchange, bids, asks, 
                                                     spread, total_bid_volume, total_ask_volume, depth_levels)
                    VALUES (:symbol, :timestamp, :exchange, :bids, :asks, 
                           :spread, :total_bid_volume, :total_ask_volume, :depth_levels)
                """)
                
                await session.execute(query, order_book_data)
                await session.commit()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving order book snapshot: {e}")
            return False
    
    async def save_liquidation_event(self, liquidation_data: Dict[str, Any]) -> bool:
        """Save liquidation event to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO liquidation_events (symbol, timestamp, exchange, side, price, 
                                                   quantity, quote_quantity, liquidation_type, metadata)
                    VALUES (:symbol, :timestamp, :exchange, :side, :price, 
                           :quantity, :quote_quantity, :liquidation_type, :metadata)
                """)
                
                await session.execute(query, liquidation_data)
                await session.commit()
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving liquidation event: {e}")
            return False
    
    async def save_market_depth_analysis(self, depth_analysis: Dict[str, Any]) -> bool:
        """Save market depth analysis to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO market_depth_analysis (symbol, timestamp, exchange, analysis_type, 
                                                      price_level, volume_at_level, side, confidence, metadata)
                    VALUES (:symbol, :timestamp, :exchange, :analysis_type, 
                           :price_level, :volume_at_level, :side, :confidence, :metadata)
                """)
                
                await session.execute(query, depth_analysis)
                await session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving market depth analysis: {e}")
            return False
    
    async def save_on_chain_event(self, on_chain_data: Dict[str, Any]) -> bool:
        """Save on-chain event to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO on_chain_events (timestamp, chain, tx_hash, from_address, to_address, 
                                               value, gas_used, event_type, symbol, block_number, metadata)
                    VALUES (:timestamp, :chain, :tx_hash, :from_address, :to_address, 
                           :value, :gas_used, :event_type, :symbol, :block_number, :metadata)
                """)
                
                await session.execute(query, on_chain_data)
                await session.commit()
                
                return True

        except Exception as e:
            self.logger.error(f"Error saving on-chain event: {e}")
            return False
    
    async def get_on_chain_events(self, chain: str = None, symbol: str = None, 
                                 limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get on-chain events from database"""
        try:
            async with self.async_session() as session:
                where_conditions = []
                params = {'limit': limit, 'start_time': start_time}
                
                if chain:
                    where_conditions.append("chain = :chain")
                    params['chain'] = chain
                
                if symbol:
                    where_conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if start_time:
                    where_conditions.append("timestamp >= :start_time")
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = text(f"""
                    SELECT timestamp, chain, tx_hash, from_address, to_address, 
                           value, gas_used, event_type, symbol, block_number, metadata
                    FROM on_chain_events
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting on-chain events: {e}")
            return []
    
    async def save_social_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """Save social sentiment data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO social_sentiment (timestamp, symbol, tweet_id, text, 
                                               sentiment_score, cluster_label, source, metadata)
                    VALUES (:timestamp, :symbol, :tweet_id, :text, 
                           :sentiment_score, :cluster_label, :source, :metadata)
                """)
                
                await session.execute(query, sentiment_data)
                await session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving social sentiment: {e}")
            return False
    
    async def get_social_sentiment(self, symbol: str = None, 
                                 limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get social sentiment data from database"""
        try:
            async with self.async_session() as session:
                where_conditions = []
                params = {'limit': limit, 'start_time': start_time}
                
                if symbol:
                    where_conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if start_time:
                    where_conditions.append("timestamp >= :start_time")
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = text(f"""
                    SELECT timestamp, symbol, tweet_id, text, sentiment_score, 
                           cluster_label, source, metadata
                    FROM social_sentiment
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting social sentiment: {e}")
            return []
    
    async def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trades
                    WHERE timestamp >= NOW() - INTERVAL ':days days'
                    AND status = 'closed'
                """)
                
                result = await session.execute(query, {'days': days})
                row = result.fetchone()
                
                if row:
                    data = dict(row._mapping)
                    win_rate = (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0
                    
                    return {
                        'total_trades': data['total_trades'],
                        'winning_trades': data['winning_trades'],
                        'losing_trades': data['losing_trades'],
                        'win_rate': round(win_rate, 2),
                        'avg_win': float(data['avg_win'] or 0),
                        'avg_loss': float(data['avg_loss'] or 0),
                        'total_pnl': float(data['total_pnl'] or 0),
                        'best_trade': float(data['best_trade'] or 0),
                        'worst_trade': float(data['worst_trade'] or 0)
                    }
                
                return {}
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def get_real_time_market_data(self, symbol: str, limit: int = 1000, 
                                       start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get real-time market data from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT symbol, timestamp, price, volume, bid, ask, bid_volume, ask_volume, 
                           exchange, data_type, metadata
                    FROM real_time_market_data
                    WHERE symbol = :symbol
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting real-time market data: {e}")
            return []
    
    async def get_order_book_snapshots(self, symbol: str, limit: int = 100, 
                                      start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get order book snapshots from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT symbol, timestamp, exchange, bids, asks, spread, 
                           total_bid_volume, total_ask_volume, depth_levels
                    FROM order_book_snapshots
                    WHERE symbol = :symbol
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting order book snapshots: {e}")
            return []
    
    async def get_liquidation_events(self, symbol: str, limit: int = 1000, 
                                    start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get liquidation events from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT symbol, timestamp, exchange, side, price, quantity, 
                           quote_quantity, liquidation_type, metadata
                    FROM liquidation_events
                    WHERE symbol = :symbol
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting liquidation events: {e}")
            return []
    
    async def get_market_depth_analysis(self, symbol: str, analysis_type: Optional[str] = None,
                                       limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get market depth analysis from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT symbol, timestamp, exchange, analysis_type, price_level, 
                           volume_at_level, side, confidence, metadata
                    FROM market_depth_analysis
                    WHERE symbol = :symbol
                    AND (:analysis_type IS NULL OR analysis_type = :analysis_type)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'analysis_type': analysis_type,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting market depth analysis: {e}")
            return []
    
    async def save_anomaly_event(self, anomaly_data: Dict[str, Any]) -> bool:
        """Save anomaly event to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO anomaly_events (timestamp, symbol, anomaly_type, severity, 
                                             confidence, value, threshold, description, metadata)
                    VALUES (:timestamp, :symbol, :anomaly_type, :severity, 
                           :confidence, :value, :threshold, :description, :metadata)
                    RETURNING id
                """)
                
                result = await session.execute(query, anomaly_data)
                await session.commit()
                
                anomaly_id = result.scalar()
                self.logger.info(f"Anomaly event saved with ID: {anomaly_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving anomaly event: {e}")
            return False
    
    async def save_anomaly_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Save anomaly alert to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO anomaly_alerts (timestamp, symbol, alert_type, severity, 
                                             message, data, action_required)
                    VALUES (:timestamp, :symbol, :alert_type, :severity, 
                           :message, :data, :action_required)
                    RETURNING id
                """)
                
                result = await session.execute(query, alert_data)
                await session.commit()
                
                alert_id = result.scalar()
                self.logger.info(f"Anomaly alert saved with ID: {alert_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error saving anomaly alert: {e}")
            return False
    
    async def get_anomaly_events(self, symbol: str, anomaly_type: Optional[str] = None,
                                 severity: Optional[str] = None, limit: int = 1000, 
                                 start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get anomaly events from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, symbol, anomaly_type, severity, confidence, 
                           value, threshold, description, metadata
                    FROM anomaly_events
                    WHERE symbol = :symbol
                    AND (:anomaly_type IS NULL OR anomaly_type = :anomaly_type)
                    AND (:severity IS NULL OR severity = :severity)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'anomaly_type': anomaly_type,
                    'severity': severity,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting anomaly events: {e}")
            return []
    
    async def get_anomaly_alerts(self, symbol: str, alert_type: Optional[str] = None,
                                 severity: Optional[str] = None, limit: int = 1000, 
                                 start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get anomaly alerts from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, symbol, alert_type, severity, message, 
                           data, action_required, acknowledged, acknowledged_by, acknowledged_at
                    FROM anomaly_alerts
                    WHERE symbol = :symbol
                    AND (:alert_type IS NULL OR alert_type = :alert_type)
                    AND (:severity IS NULL OR severity = :severity)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'alert_type': alert_type,
                    'severity': severity,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting anomaly alerts: {e}")
            return []
    
    # ===== Week 7.3 Phase 1: Funding Rate Methods =====
    
    async def save_funding_rate(self, symbol: str, exchange: str, funding_rate: float, 
                                next_funding_time: Optional[datetime] = None,
                                estimated_rate: Optional[float] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save funding rate data to database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    INSERT INTO funding_rates (timestamp, symbol, exchange, funding_rate, 
                                             next_funding_time, estimated_rate, metadata)
                    VALUES (:timestamp, :symbol, :exchange, :funding_rate, 
                           :next_funding_time, :estimated_rate, :metadata)
                """)
                
                await session.execute(query, {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': symbol,
                    'exchange': exchange,
                    'funding_rate': funding_rate,
                    'next_funding_time': next_funding_time,
                    'estimated_rate': estimated_rate,
                    'metadata': metadata or {}
                })
                
                await session.commit()
                self.logger.info(f"Funding rate saved: {symbol} on {exchange} = {funding_rate}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving funding rate: {e}")
            return False
    
    async def get_funding_rates(self, symbol: str, exchange: str = None, 
                                limit: int = 1000, start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get funding rate data from database"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, symbol, exchange, funding_rate, 
                           next_funding_time, estimated_rate, metadata
                    FROM funding_rates
                    WHERE symbol = :symbol
                    AND (:exchange IS NULL OR exchange = :exchange)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'exchange': exchange,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting funding rates: {e}")
            return []
    
    async def get_latest_funding_rate(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """Get the latest funding rate for a symbol on a specific exchange"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, symbol, exchange, funding_rate, 
                           next_funding_time, estimated_rate, metadata
                    FROM funding_rates
                    WHERE symbol = :symbol AND exchange = :exchange
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'exchange': exchange
                })
                
                row = result.fetchone()
                return dict(row._mapping) if row else None
                
        except Exception as e:
            self.logger.error(f"Error getting latest funding rate: {e}")
            return None
    
    async def get_anomalies(self, symbol: str = None, start_time: Optional[datetime] = None, 
                           limit: int = 1000) -> List[Dict[str, Any]]:
        """Get anomaly data from database for Week 8 dashboard"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, symbol, data_type, value, z_score, threshold, metadata
                    FROM anomalies
                    WHERE (:symbol IS NULL OR symbol = :symbol)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'symbol': symbol,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting anomalies: {e}")
            return []
    
    async def get_system_metrics(self, metric_name: str = None, start_time: Optional[datetime] = None,
                                limit: int = 1000) -> List[Dict[str, Any]]:
        """Get system metrics data from database for Week 8 dashboard"""
        try:
            async with self.async_session() as session:
                query = text("""
                    SELECT timestamp, metric_name, metric_value, metric_unit, metadata
                    FROM system_metrics
                    WHERE (:metric_name IS NULL OR metric_name = :metric_name)
                    AND (:start_time IS NULL OR timestamp >= :start_time)
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {
                    'metric_name': metric_name,
                    'start_time': start_time,
                    'limit': limit
                })
                
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            
            self.connected = False
            self.logger.info("Database connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for database connection"""
        try:
            if not self.connected:
                return {'status': 'disconnected', 'error': 'Not connected'}
            
            # Test connection with a simple query
            async with self.async_session() as session:
                result = await session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
                
                if test_value == 1:
                    return {
                        'status': 'healthy',
                        'connected': True,
                        'pool_size': self.pool_size,
                        'connection_count': self.connection_count
                    }
                else:
                    return {'status': 'error', 'error': 'Connection test failed'}
                    
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def execute_query(self, query: str, params: Dict = None) -> Any:
        """Execute a query and return results"""
        try:
            async with self.async_session() as session:
                result = await session.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    async def get_signals_count(self) -> int:
        """Get count of signals in database"""
        try:
            result = await self.execute_query("SELECT COUNT(*) FROM signals")
            return result[0][0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting signals count: {e}")
            return 0
    
    async def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals from database"""
        try:
            # First check what columns exist
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'signals' 
                ORDER BY ordinal_position
            """
            columns_result = await self.execute_query(columns_query)
            available_columns = [row[0] for row in columns_result]
            
            # Build query based on available columns
            base_columns = ['signal_id', 'symbol', 'timestamp']
            optional_columns = ['direction', 'confidence', 'entry_price']
            
            select_columns = []
            for col in base_columns + optional_columns:
                if col in available_columns:
                    select_columns.append(col)
            
            if not select_columns:
                select_columns = ['*']  # fallback to all columns
            
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM signals 
                ORDER BY timestamp DESC 
                LIMIT :limit
            """
            result = await self.execute_query(query, {'limit': limit})
            
            signals = []
            for row in result:
                signal_dict = {}
                for i, col in enumerate(select_columns):
                    if col == '*':
                        # If using *, we need to handle differently
                        signal_dict = dict(zip(available_columns, row))
                        break
                    else:
                        value = row[i] if i < len(row) else None
                        if col in ['confidence', 'entry_price'] and value is not None:
                            signal_dict[col] = float(value)
                        elif col == 'timestamp' and value is not None:
                            signal_dict[col] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                        else:
                            signal_dict[col] = value
                
                signals.append(signal_dict)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting recent signals: {e}")
            return []

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    # Phase 4.3: TimescaleDB Optimization Methods
    
    async def optimize_timescaledb_settings(self):
        """Apply TimescaleDB optimization settings"""
        try:
            async with self.async_session() as session:
                # Enable TimescaleDB features
                await session.execute(text("""
                    SET timescaledb.max_background_workers = :workers;
                    SET timescaledb.max_background_workers_per_gather = :workers_per_gather;
                    SET timescaledb.telemetry_level = 'off';
                """), {
                    'workers': self.parallel_workers,
                    'workers_per_gather': min(2, self.parallel_workers)
                })
                
                # Set query optimization parameters
                await session.execute(text("""
                    SET random_page_cost = 1.1;
                    SET effective_cache_size = '4GB';
                    SET shared_buffers = '256MB';
                    SET work_mem = '64MB';
                    SET maintenance_work_mem = '256MB';
                """))
                
                self.logger.info("✅ TimescaleDB optimization settings applied")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to apply TimescaleDB optimizations: {e}")
    
    async def setup_hypertable_optimizations(self, table_name: str = 'enhanced_signals'):
        """Setup hypertable optimizations for better performance"""
        try:
            async with self.async_session() as session:
                # Create hypertable if it doesn't exist
                await session.execute(text(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '{self.chunk_time_interval}'
                    );
                """))
                
                # Enable compression if enabled
                if self.compression_enabled:
                    await session.execute(text(f"""
                        ALTER TABLE {table_name} SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'symbol',
                            timescaledb.compress_orderby = 'timestamp DESC'
                        );
                    """))
                    
                    # Add compression policy
                    await session.execute(text(f"""
                        SELECT add_compression_policy('{table_name}', INTERVAL '7 days');
                    """))
                
                # Add retention policy
                await session.execute(text(f"""
                    SELECT add_retention_policy('{table_name}', INTERVAL '{self.retention_days} days');
                """))
                
                self.logger.info(f"✅ Hypertable optimizations applied to {table_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup hypertable optimizations: {e}")
    
    async def create_performance_indexes(self):
        """Create performance-optimized indexes"""
        try:
            async with self.async_session() as session:
                # Create composite indexes for common query patterns
                indexes = [
                    # Time-based queries
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_time_symbol ON enhanced_signals (timestamp DESC, symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_symbol_time ON enhanced_signals (symbol, timestamp DESC)",
                    
                    # Performance queries
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_confidence_time ON enhanced_signals (confidence DESC, timestamp DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_side_time ON enhanced_signals (side, timestamp DESC)",
                    
                    # JSONB indexes for metadata queries
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_metadata_gin ON enhanced_signals USING GIN (metadata)",
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_advanced_indicators_gin ON enhanced_signals USING GIN (advanced_indicators)",
                    
                    # Phase 4.3 specific indexes
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_phase4_3_performance ON enhanced_signals (timestamp DESC, symbol, confidence DESC) WHERE confidence > 0.7",
                    "CREATE INDEX IF NOT EXISTS idx_enhanced_signals_batch_processing ON enhanced_signals (timestamp DESC, symbol) WHERE timestamp > NOW() - INTERVAL '24 hours'"
                ]
                
                for index_sql in indexes:
                    try:
                        await session.execute(text(index_sql))
                    except Exception as e:
                        self.logger.warning(f"Index creation warning: {e}")
                
                self.logger.info("✅ Performance indexes created")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create performance indexes: {e}")
    
    async def batch_insert_signals(self, signals_data: List[Dict]) -> bool:
        """Efficient batch insert for signals data"""
        try:
            if not signals_data:
                return True
            
            # Prepare batch data
            batch_size = self.batch_size
            total_inserted = 0
            
            for i in range(0, len(signals_data), batch_size):
                batch = signals_data[i:i + batch_size]
                
                async with self.async_session() as session:
                    # Use COPY for efficient batch insert
                    await session.execute(text("""
                        INSERT INTO enhanced_signals (
                            id, symbol, timestamp, side, confidence, price, 
                            strategy, strength, stop_loss, take_profit, metadata,
                            advanced_indicators, smc_analysis, dl_analysis, rl_analysis,
                            nlp_analysis, sentiment_analysis, ensemble_analysis,
                            news_events, social_media_sentiment, volume_analysis,
                            ichimoku_data, fibonacci_data, fair_value_gaps_data,
                            liquidity_sweeps_data, market_structures_data, order_blocks_data,
                            event_categories, event_keywords, event_sentiment_analysis,
                            event_filtered_sentiment, event_enhanced_confidence,
                            event_filtered_confidence, event_impact_score, event_relevance_score,
                            high_impact_events, medium_impact_events, low_impact_events,
                            news_aware_signal, social_aware_signal, news_events_last_updated,
                            social_media_last_updated, sentiment_last_updated, nlp_analyses_performed,
                            nlp_models_available, nlp_cache_hit_rate, nlp_overall_confidence,
                            nlp_overall_sentiment_score, nlp_sentiment_strength, nlp_bias,
                            nlp_news_confidence, nlp_news_sentiment, nlp_twitter_confidence,
                            nlp_twitter_sentiment, nlp_reddit_confidence, nlp_reddit_sentiment,
                            nlp_high_confidence_sentiment, social_confidence, social_impact_score,
                            social_enhanced_confidence, social_sentiment_score, social_sentiment_history,
                            social_trends, social_volume, social_momentum, social_volatility,
                            social_engagement, social_correlation, social_filtered_sentiment,
                            ensemble_prediction, ensemble_confidence, ensemble_diversity_score,
                            ensemble_agreement_ratio, ensemble_bias, ensemble_model_count,
                            ensemble_performance_score, ensemble_voting_method, ensemble_model_weights,
                            ensemble_individual_predictions, ensemble_last_updated, ensemble_analysis,
                            lstm_prediction, cnn_prediction, lstm_cnn_prediction, dl_bias,
                            dl_confidence, dl_analysis, rl_action_type, rl_action_strength,
                            rl_position_size, rl_stop_loss, rl_take_profit, rl_risk_allocation,
                            rl_avg_reward, rl_best_reward, rl_confidence_threshold, rl_training_episodes,
                            rl_bias, rl_analysis, rl_optimization_params, smc_bias, smc_confidence,
                            smc_analysis, confirmation_count, signal_quality_score, discord_sentiment,
                            telegram_sentiment, twitter_sentiment, reddit_sentiment, news_sentiment,
                            sentiment_score, sentiment_label, sentiment_trend, sentiment_volatility,
                            sentiment_momentum, sentiment_correlation, sentiment_confidence,
                            sentiment_sources, sentiment_analysis, event_count, created_at,
                            processing_time_ms, cache_hit, cache_key, performance_score,
                            memory_usage_mb, cpu_usage_percent, queue_size, async_processing,
                            parallel_analysis, optimization_level, target_latency_ms, actual_latency_ms,
                            latency_percentile_50, latency_percentile_95, latency_percentile_99,
                            throughput_per_second, phase_4_1_features, performance_metadata,
                            memory_cleanups, cache_size, gc_collections, gc_objects_freed,
                            cache_hit_rate, latency_percentile_95, resource_alerts, optimization_enabled,
                            memory_pressure_level, cpu_throttling_applied, cache_compression_applied,
                            gc_optimization_applied, phase_4_2_features, memory_optimization_metadata
                        ) VALUES (
                            :id, :symbol, :timestamp, :side, :confidence, :price,
                            :strategy, :strength, :stop_loss, :take_profit, :metadata,
                            :advanced_indicators, :smc_analysis, :dl_analysis, :rl_analysis,
                            :nlp_analysis, :sentiment_analysis, :ensemble_analysis,
                            :news_events, :social_media_sentiment, :volume_analysis,
                            :ichimoku_data, :fibonacci_data, :fair_value_gaps_data,
                            :liquidity_sweeps_data, :market_structures_data, :order_blocks_data,
                            :event_categories, :event_keywords, :event_sentiment_analysis,
                            :event_filtered_sentiment, :event_enhanced_confidence,
                            :event_filtered_confidence, :event_impact_score, :event_relevance_score,
                            :high_impact_events, :medium_impact_events, :low_impact_events,
                            :news_aware_signal, :social_aware_signal, :news_events_last_updated,
                            :social_media_last_updated, :sentiment_last_updated, :nlp_analyses_performed,
                            :nlp_models_available, :nlp_cache_hit_rate, :nlp_overall_confidence,
                            :nlp_overall_sentiment_score, :nlp_sentiment_strength, :nlp_bias,
                            :nlp_news_confidence, :nlp_news_sentiment, :nlp_twitter_confidence,
                            :nlp_twitter_sentiment, :nlp_reddit_confidence, :nlp_reddit_sentiment,
                            :nlp_high_confidence_sentiment, :social_confidence, :social_impact_score,
                            :social_enhanced_confidence, :social_sentiment_score, :social_sentiment_history,
                            :social_trends, :social_volume, :social_momentum, :social_volatility,
                            :social_engagement, :social_correlation, :social_filtered_sentiment,
                            :ensemble_prediction, :ensemble_confidence, :ensemble_diversity_score,
                            :ensemble_agreement_ratio, :ensemble_bias, :ensemble_model_count,
                            :ensemble_performance_score, :ensemble_voting_method, :ensemble_model_weights,
                            :ensemble_individual_predictions, :ensemble_last_updated, :ensemble_analysis,
                            :lstm_prediction, :cnn_prediction, :lstm_cnn_prediction, :dl_bias,
                            :dl_confidence, :dl_analysis, :rl_action_type, :rl_action_strength,
                            :rl_position_size, :rl_stop_loss, :rl_take_profit, :rl_risk_allocation,
                            :rl_avg_reward, :rl_best_reward, :rl_confidence_threshold, :rl_training_episodes,
                            :rl_bias, :rl_analysis, :rl_optimization_params, :smc_bias, :smc_confidence,
                            :smc_analysis, :confirmation_count, :signal_quality_score, :discord_sentiment,
                            :telegram_sentiment, :twitter_sentiment, :reddit_sentiment, :news_sentiment,
                            :sentiment_score, :sentiment_label, :sentiment_trend, :sentiment_volatility,
                            :sentiment_momentum, :sentiment_correlation, :sentiment_confidence,
                            :sentiment_sources, :sentiment_analysis, :event_count, :created_at,
                            :processing_time_ms, :cache_hit, :cache_key, :performance_score,
                            :memory_usage_mb, :cpu_usage_percent, :queue_size, :async_processing,
                            :parallel_analysis, :optimization_level, :target_latency_ms, :actual_latency_ms,
                            :latency_percentile_50, :latency_percentile_95, :latency_percentile_99,
                            :throughput_per_second, :phase_4_1_features, :performance_metadata,
                            :memory_cleanups, :cache_size, :gc_collections, :gc_objects_freed,
                            :cache_hit_rate, :latency_percentile_95, :resource_alerts, :optimization_enabled,
                            :memory_pressure_level, :cpu_throttling_applied, :cache_compression_applied,
                            :gc_optimization_applied, :phase_4_2_features, :memory_optimization_metadata
                        )
                    """), batch)
                    
                    await session.commit()
                    total_inserted += len(batch)
            
            self.logger.info(f"✅ Batch inserted {total_inserted} signals")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Batch insert failed: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        try:
            async with self.async_session() as session:
                # Get table statistics
                stats_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE tablename = 'enhanced_signals'
                    ORDER BY n_distinct DESC
                """
                stats_result = await session.execute(text(stats_query))
                stats = [dict(row) for row in stats_result.fetchall()]
                
                # Get index usage statistics
                index_query = """
                    SELECT 
                        indexrelname,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE relname = 'enhanced_signals'
                    ORDER BY idx_tup_read DESC
                """
                index_result = await session.execute(text(index_query))
                index_stats = [dict(row) for row in index_result.fetchall()]
                
                # Get hypertable information
                hypertable_query = """
                    SELECT 
                        hypertable_name,
                        num_chunks,
                        compression_enabled
                    FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = 'enhanced_signals'
                """
                hypertable_result = await session.execute(text(hypertable_query))
                hypertable_info = [dict(row) for row in hypertable_result.fetchall()]
                
                return {
                    'table_stats': stats,
                    'index_stats': index_stats,
                    'hypertable_info': hypertable_info,
                    'connection_pool': {
                        'pool_size': self.pool_size,
                        'max_overflow': self.max_overflow,
                        'connection_count': self.connection_count
                    }
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}
    
    # =====================================================
    # PHASE 4: DATA LIFECYCLE MANAGEMENT METHODS
    # =====================================================
    
    async def initialize_lifecycle_management(self):
        """Initialize data lifecycle management components"""
        try:
            if not self.lifecycle_enabled:
                self.logger.info("Data lifecycle management is disabled")
                return True
            
            # Import lifecycle manager
            from .lifecycle_manager import DataLifecycleManager
            self.lifecycle_manager = DataLifecycleManager(self.async_engine)
            await self.lifecycle_manager.initialize()
            
            self.logger.info("✅ Data lifecycle management initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize lifecycle management: {e}")
            return False
    
    async def create_retention_policy(self, table_name: str, retention_days: int, policy_name: str = None) -> bool:
        """Create a retention policy for a table"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.create_retention_policy(table_name, retention_days, policy_name)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create retention policy: {e}")
            return False
    
    async def create_compression_policy(self, table_name: str, compress_after_days: int = 7, policy_name: str = None) -> bool:
        """Create a compression policy for a table"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.create_compression_policy(table_name, compress_after_days, policy_name)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create compression policy: {e}")
            return False
    
    async def execute_cleanup_operation(self, table_name: str, cleanup_type: str, criteria: dict = None) -> int:
        """Execute a cleanup operation on a table"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.execute_cleanup_operation(table_name, cleanup_type, criteria)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute cleanup operation: {e}")
            return 0
    
    async def get_lifecycle_statistics(self, table_name: str = None, days_back: int = 30) -> dict:
        """Get lifecycle management statistics"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.get_statistics(table_name, days_back)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get lifecycle statistics: {e}")
            return {}
    
    async def archive_table_data(self, table_name: str, date_range: tuple, archive_format: str = 'parquet') -> str:
        """Archive table data to external storage"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.archive_table_data(table_name, date_range, archive_format)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to archive table data: {e}")
            return None
    
    async def restore_archived_data(self, archive_name: str, target_table: str = None) -> bool:
        """Restore archived data to a table"""
        try:
            if not self.lifecycle_manager:
                await self.initialize_lifecycle_management()
            
            return await self.lifecycle_manager.restore_archived_data(archive_name, target_table)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore archived data: {e}")
            return False

    # =====================================================
    # PHASE 5: SECURITY ENHANCEMENT METHODS
    # =====================================================
    
    async def initialize_security_management(self):
        """Initialize security management components"""
        try:
            if not self.security_enabled:
                self.logger.info("Security management is disabled")
                return True
            
            # Import security manager
            from .security_manager import SecurityManager
            self.security_manager = SecurityManager(self.async_engine)
            await self.security_manager.initialize()
            
            self.logger.info("✅ Security management initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize security management: {e}")
            return False
    
    async def log_security_audit(self, user_id: str, session_id: str, action_type: str, 
                               resource_type: str = None, resource_id: str = None,
                               action_details: dict = None, ip_address: str = None,
                               user_agent: str = None, request_method: str = None,
                               request_path: str = None, response_status: int = None,
                               execution_time_ms: int = None, success: bool = True,
                               error_message: str = None, metadata: dict = None) -> int:
        """Log security audit event"""
        try:
            if not self.security_manager:
                await self.initialize_security_management()
            
            return await self.security_manager.log_audit_event(
                user_id, session_id, action_type, resource_type, resource_id,
                action_details, ip_address, user_agent, request_method, request_path,
                response_status, execution_time_ms, success, error_message, metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log security audit: {e}")
            return 0
    
    async def check_user_permission(self, user_id: str, permission: str, 
                                  resource_type: str = None, resource_id: str = None) -> bool:
        """Check if user has specific permission"""
        try:
            if not self.security_manager:
                await self.initialize_security_management()
            
            return await self.security_manager.check_permission(user_id, permission, resource_type, resource_id)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check user permission: {e}")
            return False
    
    async def log_security_event(self, event_type: str, severity: str, source: str,
                               user_id: str = None, session_id: str = None,
                               event_details: dict = None, ip_address: str = None,
                               user_agent: str = None, threat_level: str = None) -> int:
        """Log security event"""
        try:
            if not self.security_manager:
                await self.initialize_security_management()
            
            return await self.security_manager.log_security_event(
                event_type, severity, source, user_id, session_id,
                event_details, ip_address, user_agent, threat_level
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log security event: {e}")
            return 0
    
    async def rotate_secret(self, secret_name: str, new_version: str) -> bool:
        """Rotate a secret"""
        try:
            if not self.security_manager:
                await self.initialize_security_management()
            
            return await self.security_manager.rotate_secret(secret_name, new_version)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rotate secret: {e}")
            return False
    
    async def get_security_statistics(self, days_back: int = 30) -> dict:
        """Get security statistics"""
        try:
            if not self.security_manager:
                await self.initialize_security_management()
            
            return await self.security_manager.get_statistics(days_back)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get security statistics: {e}")
            return {}
