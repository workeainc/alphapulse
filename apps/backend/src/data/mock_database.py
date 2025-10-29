#!/usr/bin/env python3
"""
Mock Database Implementation for Testing
Provides database-like functionality without requiring PostgreSQL
"""

import asyncio
import logging
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDatabase:
    """Mock database implementation using SQLite for testing"""
    
    def __init__(self, db_path: str = "test_database.db"):
        self.db_path = db_path
        self.conn = None
        
    async def initialize(self):
        """Initialize the mock database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create tables
            await self._create_tables()
            logger.info("✅ Mock database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize mock database: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("🔌 Mock database connection closed")
    
    async def _create_tables(self):
        """Create necessary tables"""
        cursor = self.conn.cursor()
        
        # OHLCV Data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL,
                trades_count INTEGER,
                source TEXT DEFAULT 'websocket',
                data_quality_score REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Technical Indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_value REAL NOT NULL,
                indicator_params TEXT,
                calculation_method TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                pattern_type TEXT,
                volume_confirmation BOOLEAN DEFAULT FALSE,
                trend_alignment BOOLEAN DEFAULT FALSE,
                market_regime TEXT,
                indicators TEXT,
                validation_metrics TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                outcome TEXT DEFAULT 'pending'
            )
        """)
        
        # ML Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                prediction REAL NOT NULL,
                confidence REAL,
                features_used TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data (symbol, timeframe)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON technical_indicators (symbol)")
        
        self.conn.commit()
        logger.info("📊 Mock database tables created")
    
    async def execute(self, query: str, *params):
        """Execute a query"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Query execution error: {e}")
            raise
    
    async def fetch(self, query: str, *params):
        """Fetch multiple rows"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"❌ Fetch error: {e}")
            return []
    
    async def fetchval(self, query: str, *params):
        """Fetch single value"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"❌ Fetchval error: {e}")
            return None
    
    async def executemany(self, query: str, params_list):
        """Execute query with multiple parameter sets"""
        try:
            cursor = self.conn.cursor()
            cursor.executemany(query, params_list)
            self.conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Executemany error: {e}")
            raise

class MockDataPipeline:
    """Mock data pipeline for testing without real database"""
    
    def __init__(self):
        self.db = MockDatabase()
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'db_inserts': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize the pipeline"""
        await self.db.initialize()
        logger.info("✅ Mock data pipeline initialized")
    
    async def close(self):
        """Close the pipeline"""
        await self.db.close()
    
    async def process_websocket_message(self, message: Dict[str, Any]) -> bool:
        """Process WebSocket message"""
        try:
            self.stats['messages_received'] += 1
            
            if message.get('type') == 'kline':
                await self._process_kline_data(message)
            
            self.stats['messages_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            self.stats['errors'] += 1
            return False
    
    async def _process_kline_data(self, message: Dict[str, Any]):
        """Process kline data"""
        try:
            await self.db.execute("""
                INSERT INTO ohlcv_data (
                    symbol, timeframe, timestamp, open, high, low, close, volume,
                    quote_volume, trades_count, source, data_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            message.get('symbol', 'BTCUSDT'),
            message.get('timeframe', '1m'),
            message.get('timestamp').isoformat() if isinstance(message.get('timestamp'), datetime) else str(message.get('timestamp')),
            float(message.get('open', 0)),
            float(message.get('high', 0)),
            float(message.get('low', 0)),
            float(message.get('close', 0)),
            float(message.get('volume', 0)),
            float(message.get('quote_volume', 0)) if message.get('quote_volume') else None,
            int(message.get('trades', 0)) if message.get('trades') else None,
            message.get('source', 'websocket'),
            1.0
            )
            
            self.stats['db_inserts'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing kline data: {e}")
            raise
    
    async def calculate_technical_indicators(self, symbol: str, timeframe: str):
        """Calculate technical indicators"""
        try:
            # Get latest OHLCV data
            data = await self.db.fetch("""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, symbol, timeframe)
            
            if len(data) < 20:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate indicators
            indicators = {}
            
            # SMA 20
            if len(df) >= 20:
                indicators['SMA_20'] = df['close'].rolling(window=20).mean().iloc[-1]
            
            # SMA 50
            if len(df) >= 50:
                indicators['SMA_50'] = df['close'].rolling(window=50).mean().iloc[-1]
            
            # RSI 14
            if len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['RSI_14'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Store indicators
            current_time = datetime.now(timezone.utc).isoformat()
            for indicator_name, value in indicators.items():
                if pd.notna(value):
                    await self.db.execute("""
                        INSERT INTO technical_indicators (
                            symbol, timeframe, timestamp, indicator_name, indicator_value, calculation_method
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, symbol, timeframe, current_time, indicator_name, float(value), 'python_calculation')
            
            logger.info(f"✅ Calculated indicators for {symbol} {timeframe}: {list(indicators.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            raise
    
    async def get_latest_ohlcv_data(self, symbol: str, timeframe: str, periods: int = 100) -> List[Dict[str, Any]]:
        """Get latest OHLCV data"""
        try:
            data = await self.db.fetch("""
                SELECT timestamp, open, high, low, close, volume, quote_volume, trades_count
                FROM ohlcv_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, symbol, timeframe, periods)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error getting OHLCV data: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'messages_received': self.stats['messages_received'],
            'messages_processed': self.stats['messages_processed'],
            'db_inserts': self.stats['db_inserts'],
            'errors': self.stats['errors'],
            'success_rate': (self.stats['messages_processed'] / max(self.stats['messages_received'], 1)) * 100
        }

class MockSDEIntegration:
    """Mock SDE integration for testing"""
    
    def __init__(self):
        self.db = MockDatabase()
        self.stats = {
            'signals_generated': 0,
            'signals_stored': 0,
            'consensus_reached': 0,
            'consensus_failed': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize SDE integration"""
        await self.db.initialize()
        logger.info("✅ Mock SDE integration initialized")
    
    async def close(self):
        """Close SDE integration"""
        await self.db.close()
    
    async def generate_signal(self, request) -> Optional[Dict[str, Any]]:
        """Generate mock signal"""
        try:
            # Mock signal generation
            signal_id = f"{request.symbol}_{request.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Mock consensus result
            consensus_reached = True  # Always reach consensus for testing
            direction = "BUY" if request.market_data.get('current_price', 0) > 45000 else "SELL"
            confidence = 0.75
            strength = 0.65
            
            # Store signal
            await self.db.execute("""
                INSERT INTO signals (
                    signal_id, symbol, timeframe, direction, confidence, entry_price,
                    pattern_type, volume_confirmation, trend_alignment, market_regime,
                    indicators, validation_metrics, timestamp, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            signal_id,
            request.symbol,
            request.timeframe,
            direction,
            confidence,
            request.market_data.get('current_price', 0),
            'mock_sde_consensus',
            True,
            True,
            'trending',
            json.dumps(request.market_data.get('indicators', {})),
            json.dumps({
                'strength': strength,
                'consensus_reached': consensus_reached,
                'model_head_count': 4,
                'market_conditions': request.analysis_results
            }),
            datetime.now(timezone.utc).isoformat(),
            'pending'
            )
            
            # Update stats
            self.stats['signals_generated'] += 1
            self.stats['signals_stored'] += 1
            self.stats['consensus_reached'] += 1
            
            # Return mock result
            result = type('SignalGenerationResult', (), {
                'signal_id': signal_id,
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'direction': direction,
                'confidence': confidence,
                'strength': strength,
                'timestamp': datetime.now(timezone.utc),
                'consensus_result': {'consensus_reached': consensus_reached},
                'model_head_results': [{'head_type': 'mock_head'} for _ in range(4)]
            })()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating mock signal: {e}")
            self.stats['errors'] += 1
            return None
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            'signals_generated': self.stats['signals_generated'],
            'signals_stored': self.stats['signals_stored'],
            'consensus_reached': self.stats['consensus_reached'],
            'consensus_failed': self.stats['consensus_failed'],
            'errors': self.stats['errors'],
            'consensus_rate': (self.stats['consensus_reached'] / max(self.stats['consensus_reached'] + self.stats['consensus_failed'], 1)) * 100
        }

# Export mock classes for use in tests
__all__ = ['MockDatabase', 'MockDataPipeline', 'MockSDEIntegration']
