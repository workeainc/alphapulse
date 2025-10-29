#!/usr/bin/env python3
"""
TimescaleDB ML Integration Service for AlphaPulse
Optimized database integration for storing ML predictions and signals
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import time
from sqlalchemy import text, insert, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB

# Local imports
from ..src.database.connection import TimescaleDBConnection
from ..src.ai.ultra_low_latency_inference import UltraLowLatencyInference, InferenceResult
from ..src.ai.knowledge_distillation import KnowledgeDistillation
from ..advanced_logging_system import redis_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

@dataclass
class MLSignalData:
    """ML signal data for database storage"""
    signal_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    prediction: float
    confidence: float
    model_used: str
    features: Dict[str, float]
    inference_latency_ms: float
    signal_direction: str  # 'buy', 'sell', 'hold'
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    market_regime: Optional[str] = None
    pattern_type: Optional[str] = None
    volume_confirmation: Optional[bool] = None
    trend_alignment: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str = 'generated'  # 'generated', 'active', 'completed', 'cancelled'
    outcome: Optional[str] = None  # 'win', 'loss', 'pending'
    pnl: Optional[float] = None
    created_at: Optional[datetime] = None

@dataclass
class MLPredictionData:
    """ML prediction data for database storage"""
    prediction_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    prediction: float
    confidence: float
    model_used: str
    features: Dict[str, float]
    inference_latency_ms: float
    ensemble_predictions: Optional[Dict[str, float]] = None
    model_weights: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

class TimescaleDBMLIntegration:
    """
    TimescaleDB integration service for ML predictions and signals
    """
    
    def __init__(self, 
                 database_config: Dict = None,
                 inference_engine: UltraLowLatencyInference = None):
        """
        Initialize TimescaleDB ML integration service
        
        Args:
            database_config: Database configuration
            inference_engine: Ultra-low latency inference engine
        """
        self.database_config = database_config or {
            'host': 'postgres',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'alpha_emon',
            'password': 'Emon_@17711'
        }
        
        # Initialize database connection
        self.db_connection = TimescaleDBConnection(self.database_config)
        
        # Initialize inference engine
        self.inference_engine = inference_engine or UltraLowLatencyInference()
        
        # Performance tracking
        self.storage_times: List[float] = []
        self.storage_errors: int = 0
        self.signals_stored: int = 0
        self.predictions_stored: int = 0
        
        logger.info(f"✅ TimescaleDB ML integration service initialized")
    
    async def initialize(self):
        """Initialize the service and create necessary tables"""
        try:
            logger.info("🔄 Initializing TimescaleDB ML integration...")
            
            # Initialize database connection
            await self.db_connection.initialize()
            
            # Create ML-specific tables
            await self._create_ml_tables()
            
            # Initialize inference engine
            await self.inference_engine.initialize()
            
            logger.info("✅ TimescaleDB ML integration initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing TimescaleDB ML integration: {e}")
            raise
    
    async def _create_ml_tables(self):
        """Create ML-specific tables with TimescaleDB optimizations"""
        try:
            async with self.db_connection.get_session() as session:
                # Create ML predictions table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        id BIGSERIAL,
                        prediction_id VARCHAR(100) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        prediction DECIMAL(10,6) NOT NULL,
                        confidence DECIMAL(10,6) NOT NULL,
                        model_used VARCHAR(50) NOT NULL,
                        features JSONB NOT NULL,
                        inference_latency_ms DECIMAL(10,3) NOT NULL,
                        ensemble_predictions JSONB,
                        model_weights JSONB,
                        feature_importance JSONB,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, id)
                    )
                """))
                
                # Create ML signals table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS ml_signals (
                        id BIGSERIAL,
                        signal_id VARCHAR(100) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        prediction DECIMAL(10,6) NOT NULL,
                        confidence DECIMAL(10,6) NOT NULL,
                        model_used VARCHAR(50) NOT NULL,
                        features JSONB NOT NULL,
                        inference_latency_ms DECIMAL(10,3) NOT NULL,
                        signal_direction VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20,8) NOT NULL,
                        stop_loss DECIMAL(20,8),
                        take_profit DECIMAL(20,8),
                        risk_reward_ratio DECIMAL(10,4),
                        market_regime VARCHAR(20),
                        pattern_type VARCHAR(50),
                        volume_confirmation BOOLEAN,
                        trend_alignment BOOLEAN,
                        metadata JSONB,
                        status VARCHAR(20) DEFAULT 'generated',
                        outcome VARCHAR(20),
                        pnl DECIMAL(20,8),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, id)
                    )
                """))
                
                # Create ML model performance table
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS ml_model_performance (
                        id BIGSERIAL,
                        model_name VARCHAR(100) NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        avg_latency_ms DECIMAL(10,3) NOT NULL,
                        p50_latency_ms DECIMAL(10,3) NOT NULL,
                        p90_latency_ms DECIMAL(10,3) NOT NULL,
                        p99_latency_ms DECIMAL(10,3) NOT NULL,
                        total_predictions INTEGER NOT NULL,
                        error_count INTEGER NOT NULL,
                        accuracy DECIMAL(10,6),
                        cache_hit_rate DECIMAL(10,6),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, id)
                    )
                """))
                
                # Convert to TimescaleDB hypertables
                await session.execute(text("""
                    SELECT create_hypertable('ml_predictions', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    )
                """))
                
                await session.execute(text("""
                    SELECT create_hypertable('ml_signals', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 hour'
                    )
                """))
                
                await session.execute(text("""
                    SELECT create_hypertable('ml_model_performance', 'timestamp', 
                        if_not_exists => TRUE, 
                        chunk_time_interval => INTERVAL '1 day'
                    )
                """))
                
                # Create indexes for optimal query performance
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time 
                    ON ml_predictions (symbol, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_used 
                    ON ml_predictions (model_used, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence 
                    ON ml_predictions (confidence DESC, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_signals_symbol_time 
                    ON ml_signals (symbol, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_signals_direction 
                    ON ml_signals (signal_direction, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_signals_status 
                    ON ml_signals (status, timestamp DESC)
                """))
                
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_ml_signals_high_confidence 
                    ON ml_signals (confidence DESC, timestamp DESC) 
                    WHERE confidence >= 0.8
                """))
                
                await session.commit()
                
                logger.info("✅ ML tables created successfully")
                
        except Exception as e:
            logger.error(f"❌ Error creating ML tables: {e}")
            raise
    
    async def store_prediction(self, prediction_data: MLPredictionData) -> bool:
        """
        Store ML prediction in TimescaleDB
        
        Args:
            prediction_data: Prediction data to store
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            async with self.db_connection.get_session() as session:
                # Insert prediction
                await session.execute(text("""
                    INSERT INTO ml_predictions (
                        prediction_id, symbol, timeframe, timestamp, prediction, 
                        confidence, model_used, features, inference_latency_ms,
                        ensemble_predictions, model_weights, feature_importance, metadata
                    ) VALUES (
                        :prediction_id, :symbol, :timeframe, :timestamp, :prediction,
                        :confidence, :model_used, :features, :inference_latency_ms,
                        :ensemble_predictions, :model_weights, :feature_importance, :metadata
                    )
                """), {
                    'prediction_id': prediction_data.prediction_id,
                    'symbol': prediction_data.symbol,
                    'timeframe': prediction_data.timeframe,
                    'timestamp': prediction_data.timestamp,
                    'prediction': prediction_data.prediction,
                    'confidence': prediction_data.confidence,
                    'model_used': prediction_data.model_used,
                    'features': json.dumps(prediction_data.features),
                    'inference_latency_ms': prediction_data.inference_latency_ms,
                    'ensemble_predictions': json.dumps(prediction_data.ensemble_predictions) if prediction_data.ensemble_predictions else None,
                    'model_weights': json.dumps(prediction_data.model_weights) if prediction_data.model_weights else None,
                    'feature_importance': json.dumps(prediction_data.feature_importance) if prediction_data.feature_importance else None,
                    'metadata': json.dumps(prediction_data.metadata) if prediction_data.metadata else None
                })
                
                await session.commit()
                
                # Track performance
                storage_time = time.time() - start_time
                self.storage_times.append(storage_time)
                self.predictions_stored += 1
                
                logger.debug(f"✅ Stored prediction {prediction_data.prediction_id} in {storage_time:.3f}s")
                return True
                
        except Exception as e:
            self.storage_errors += 1
            logger.error(f"❌ Error storing prediction: {e}")
            return False
    
    async def store_signal(self, signal_data: MLSignalData) -> bool:
        """
        Store ML signal in TimescaleDB
        
        Args:
            signal_data: Signal data to store
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            async with self.db_connection.get_session() as session:
                # Insert signal
                await session.execute(text("""
                    INSERT INTO ml_signals (
                        signal_id, symbol, timeframe, timestamp, prediction, 
                        confidence, model_used, features, inference_latency_ms,
                        signal_direction, entry_price, stop_loss, take_profit,
                        risk_reward_ratio, market_regime, pattern_type,
                        volume_confirmation, trend_alignment, metadata, status
                    ) VALUES (
                        :signal_id, :symbol, :timeframe, :timestamp, :prediction,
                        :confidence, :model_used, :features, :inference_latency_ms,
                        :signal_direction, :entry_price, :stop_loss, :take_profit,
                        :risk_reward_ratio, :market_regime, :pattern_type,
                        :volume_confirmation, :trend_alignment, :metadata, :status
                    )
                """), {
                    'signal_id': signal_data.signal_id,
                    'symbol': signal_data.symbol,
                    'timeframe': signal_data.timeframe,
                    'timestamp': signal_data.timestamp,
                    'prediction': signal_data.prediction,
                    'confidence': signal_data.confidence,
                    'model_used': signal_data.model_used,
                    'features': json.dumps(signal_data.features),
                    'inference_latency_ms': signal_data.inference_latency_ms,
                    'signal_direction': signal_data.signal_direction,
                    'entry_price': signal_data.entry_price,
                    'stop_loss': signal_data.stop_loss,
                    'take_profit': signal_data.take_profit,
                    'risk_reward_ratio': signal_data.risk_reward_ratio,
                    'market_regime': signal_data.market_regime,
                    'pattern_type': signal_data.pattern_type,
                    'volume_confirmation': signal_data.volume_confirmation,
                    'trend_alignment': signal_data.trend_alignment,
                    'metadata': json.dumps(signal_data.metadata) if signal_data.metadata else None,
                    'status': signal_data.status
                })
                
                await session.commit()
                
                # Track performance
                storage_time = time.time() - start_time
                self.storage_times.append(storage_time)
                self.signals_stored += 1
                
                logger.debug(f"✅ Stored signal {signal_data.signal_id} in {storage_time:.3f}s")
                return True
                
        except Exception as e:
            self.storage_errors += 1
            logger.error(f"❌ Error storing signal: {e}")
            return False
    
    async def store_model_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        Store model performance metrics
        
        Args:
            performance_data: Performance data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_connection.get_session() as session:
                await session.execute(text("""
                    INSERT INTO ml_model_performance (
                        model_name, timestamp, avg_latency_ms, p50_latency_ms,
                        p90_latency_ms, p99_latency_ms, total_predictions,
                        error_count, accuracy, cache_hit_rate, metadata
                    ) VALUES (
                        :model_name, :timestamp, :avg_latency_ms, :p50_latency_ms,
                        :p90_latency_ms, :p99_latency_ms, :total_predictions,
                        :error_count, :accuracy, :cache_hit_rate, :metadata
                    )
                """), {
                    'model_name': performance_data.get('model_name', 'ensemble'),
                    'timestamp': datetime.now(),
                    'avg_latency_ms': performance_data.get('avg_latency_ms', 0),
                    'p50_latency_ms': performance_data.get('p50_latency_ms', 0),
                    'p90_latency_ms': performance_data.get('p90_latency_ms', 0),
                    'p99_latency_ms': performance_data.get('p99_latency_ms', 0),
                    'total_predictions': performance_data.get('total_predictions', 0),
                    'error_count': performance_data.get('error_count', 0),
                    'accuracy': performance_data.get('accuracy'),
                    'cache_hit_rate': performance_data.get('cache_hit_rate'),
                    'metadata': json.dumps(performance_data.get('metadata', {}))
                })
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Error storing model performance: {e}")
            return False
    
    async def make_prediction_and_store(self, 
                                      symbol: str, 
                                      timeframe: str,
                                      candlestick_data: pd.DataFrame,
                                      generate_signal: bool = True) -> Tuple[InferenceResult, bool]:
        """
        Make prediction and store in database
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            candlestick_data: OHLCV data
            generate_signal: Whether to generate and store a signal
            
        Returns:
            Tuple of (InferenceResult, storage_success)
        """
        try:
            # Make prediction
            inference_result = await self.inference_engine.predict(symbol, timeframe, candlestick_data)
            
            # Store prediction
            prediction_data = MLPredictionData(
                prediction_id=f"PRED_{symbol}_{timeframe}_{int(time.time())}",
                symbol=symbol,
                timeframe=timeframe,
                timestamp=inference_result.timestamp,
                prediction=inference_result.prediction,
                confidence=inference_result.confidence,
                model_used=inference_result.model_used,
                features=inference_result.features_used,
                inference_latency_ms=inference_result.latency_ms,
                metadata=inference_result.metadata
            )
            
            storage_success = await self.store_prediction(prediction_data)
            
            # Generate and store signal if requested
            if generate_signal and inference_result.confidence > 0.7:
                signal_data = await self._create_signal_from_prediction(
                    inference_result, symbol, timeframe, candlestick_data
                )
                if signal_data:
                    signal_storage_success = await self.store_signal(signal_data)
                    storage_success = storage_success and signal_storage_success
            
            return inference_result, storage_success
            
        except Exception as e:
            logger.error(f"❌ Error in make_prediction_and_store: {e}")
            return None, False
    
    async def _create_signal_from_prediction(self, 
                                           inference_result: InferenceResult,
                                           symbol: str,
                                           timeframe: str,
                                           candlestick_data: pd.DataFrame) -> Optional[MLSignalData]:
        """Create signal from inference result"""
        try:
            # Determine signal direction based on prediction
            if inference_result.prediction > 0.6:
                signal_direction = 'buy'
            elif inference_result.prediction < 0.4:
                signal_direction = 'sell'
            else:
                signal_direction = 'hold'
            
            # Skip if hold signal
            if signal_direction == 'hold':
                return None
            
            # Get latest price
            latest_candle = candlestick_data.iloc[-1]
            entry_price = float(latest_candle['close'])
            
            # Calculate stop loss and take profit
            atr = inference_result.features_used.get('atr', 0.01)
            if atr == 0:
                atr = entry_price * 0.01  # 1% default
            
            if signal_direction == 'buy':
                stop_loss = entry_price - (atr * 2)  # 2 ATR below entry
                take_profit = entry_price + (atr * 3)  # 3 ATR above entry
            else:
                stop_loss = entry_price + (atr * 2)  # 2 ATR above entry
                take_profit = entry_price - (atr * 3)  # 3 ATR below entry
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Create signal data
            signal_data = MLSignalData(
                signal_id=f"SIGNAL_{symbol}_{timeframe}_{int(time.time())}",
                symbol=symbol,
                timeframe=timeframe,
                timestamp=inference_result.timestamp,
                prediction=inference_result.prediction,
                confidence=inference_result.confidence,
                model_used=inference_result.model_used,
                features=inference_result.features_used,
                inference_latency_ms=inference_result.latency_ms,
                signal_direction=signal_direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                metadata=inference_result.metadata
            )
            
            return signal_data
            
        except Exception as e:
            logger.error(f"❌ Error creating signal from prediction: {e}")
            return None
    
    async def get_recent_predictions(self, 
                                   symbol: str = None, 
                                   timeframe: str = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions from database"""
        try:
            async with self.db_connection.get_session() as session:
                query = """
                    SELECT * FROM ml_predictions 
                    WHERE 1=1
                """
                params = {}
                
                if symbol:
                    query += " AND symbol = :symbol"
                    params['symbol'] = symbol
                
                if timeframe:
                    query += " AND timeframe = :timeframe"
                    params['timeframe'] = timeframe
                
                query += " ORDER BY timestamp DESC LIMIT :limit"
                params['limit'] = limit
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                predictions = []
                for row in rows:
                    prediction = dict(row._mapping)
                    # Parse JSON fields
                    if prediction['features']:
                        prediction['features'] = json.loads(prediction['features'])
                    if prediction['ensemble_predictions']:
                        prediction['ensemble_predictions'] = json.loads(prediction['ensemble_predictions'])
                    if prediction['model_weights']:
                        prediction['model_weights'] = json.loads(prediction['model_weights'])
                    if prediction['feature_importance']:
                        prediction['feature_importance'] = json.loads(prediction['feature_importance'])
                    if prediction['metadata']:
                        prediction['metadata'] = json.loads(prediction['metadata'])
                    
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"❌ Error getting recent predictions: {e}")
            return []
    
    async def get_recent_signals(self, 
                               symbol: str = None, 
                               timeframe: str = None,
                               status: str = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent signals from database"""
        try:
            async with self.db_connection.get_session() as session:
                query = """
                    SELECT * FROM ml_signals 
                    WHERE 1=1
                """
                params = {}
                
                if symbol:
                    query += " AND symbol = :symbol"
                    params['symbol'] = symbol
                
                if timeframe:
                    query += " AND timeframe = :timeframe"
                    params['timeframe'] = timeframe
                
                if status:
                    query += " AND status = :status"
                    params['status'] = status
                
                query += " ORDER BY timestamp DESC LIMIT :limit"
                params['limit'] = limit
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                signals = []
                for row in rows:
                    signal = dict(row._mapping)
                    # Parse JSON fields
                    if signal['features']:
                        signal['features'] = json.loads(signal['features'])
                    if signal['metadata']:
                        signal['metadata'] = json.loads(signal['metadata'])
                    
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            # Get inference engine stats
            inference_stats = await self.inference_engine.get_performance_stats()
            
            # Calculate storage stats
            storage_stats = {
                'avg_storage_time_ms': np.mean(self.storage_times) * 1000 if self.storage_times else 0,
                'total_storage_operations': len(self.storage_times),
                'storage_errors': self.storage_errors,
                'signals_stored': self.signals_stored,
                'predictions_stored': self.predictions_stored,
                'error_rate': self.storage_errors / (len(self.storage_times) + self.storage_errors) if (len(self.storage_times) + self.storage_errors) > 0 else 0
            }
            
            return {
                'inference_stats': inference_stats,
                'storage_stats': storage_stats,
                'database_connected': self.db_connection.is_connected()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}
    
    async def close(self):
        """Close database connection"""
        try:
            await self.db_connection.close()
            logger.info("✅ TimescaleDB ML integration closed")
        except Exception as e:
            logger.error(f"❌ Error closing TimescaleDB ML integration: {e}")

# Global instance
timescaledb_ml_integration = TimescaleDBMLIntegration()
