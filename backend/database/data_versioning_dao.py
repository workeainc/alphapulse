#!/usr/bin/env python3
"""
Data Access Object for Data Versioning Tables
Phase 1: Database Schema Implementation
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from sqlalchemy import text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import json

logger = logging.getLogger(__name__)

class DataVersioningDAO:
    """Data Access Object for data versioning tables"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    # ==================== SIGNALS TABLE OPERATIONS ====================
    
    async def create_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new signal record"""
        try:
            query = text("""
                INSERT INTO signals (
                    label, pred, proba, ts, symbol, tf, features, 
                    model_id, outcome, realized_rr, latency_ms
                ) VALUES (
                    :label, :pred, :proba, :ts, :symbol, :tf, :features,
                    :model_id, :outcome, :realized_rr, :latency_ms
                ) RETURNING id, created_at
            """)
            
            result = await self.session.execute(query, signal_data)
            row = result.fetchone()
            
            await self.session.commit()
            
            return {
                'id': row[0],
                'created_at': row[1],
                'status': 'created'
            }
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating signal: {e}")
            raise
    
    async def get_signals(
        self, 
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get signals with optional filtering"""
        try:
            conditions = []
            params = {}
            
            if symbol:
                conditions.append("symbol = :symbol")
                params['symbol'] = symbol
            
            if timeframe:
                conditions.append("tf = :timeframe")
                params['timeframe'] = timeframe
            
            if model_id:
                conditions.append("model_id = :model_id")
                params['model_id'] = model_id
            
            if start_time:
                conditions.append("ts >= :start_time")
                params['start_time'] = start_time
            
            if end_time:
                conditions.append("ts <= :end_time")
                params['end_time'] = end_time
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = text(f"""
                SELECT id, label, pred, proba, ts, symbol, tf, features,
                       model_id, outcome, realized_rr, latency_ms,
                       created_at, updated_at
                FROM signals
                WHERE {where_clause}
                ORDER BY ts DESC
                LIMIT :limit
            """)
            
            params['limit'] = limit
            result = await self.session.execute(query, params)
            
            signals = []
            for row in result:
                signal = {
                    'id': row[0],
                    'label': row[1],
                    'pred': row[2],
                    'proba': row[3],
                    'ts': row[4],
                    'symbol': row[5],
                    'tf': row[6],
                    'features': row[7],
                    'model_id': row[8],
                    'outcome': row[9],
                    'realized_rr': row[10],
                    'latency_ms': row[11],
                    'created_at': row[12],
                    'updated_at': row[13]
                }
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            raise
    
    async def update_signal_outcome(
        self, 
        signal_id: int, 
        outcome: str, 
        realized_rr: Optional[float] = None
    ) -> bool:
        """Update signal outcome and realized risk/reward"""
        try:
            query = text("""
                UPDATE signals 
                SET outcome = :outcome, 
                    realized_rr = :realized_rr,
                    updated_at = NOW()
                WHERE id = :signal_id
            """)
            
            result = await self.session.execute(query, {
                'signal_id': signal_id,
                'outcome': outcome,
                'realized_rr': realized_rr
            })
            
            await self.session.commit()
            return result.rowcount > 0
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating signal outcome: {e}")
            raise
    
    # ==================== CANDLES TABLE OPERATIONS ====================
    
    async def create_candle(self, candle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new candle record"""
        try:
            query = text("""
                INSERT INTO candles (
                    symbol, tf, ts, o, h, l, c, v, vwap, taker_buy_vol, features
                ) VALUES (
                    :symbol, :tf, :ts, :o, :h, :l, :c, :v, :vwap, :taker_buy_vol, :features
                ) RETURNING id, created_at
            """)
            
            result = await self.session.execute(query, candle_data)
            row = result.fetchone()
            
            await self.session.commit()
            
            return {
                'id': row[0],
                'created_at': row[1],
                'status': 'created'
            }
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating candle: {e}")
            raise
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get candles for a specific symbol and timeframe"""
        try:
            conditions = [
                "symbol = :symbol",
                "tf = :timeframe"
            ]
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'limit': limit
            }
            
            if start_time:
                conditions.append("ts >= :start_time")
                params['start_time'] = start_time
            
            if end_time:
                conditions.append("ts <= :end_time")
                params['end_time'] = end_time
            
            where_clause = " AND ".join(conditions)
            
            query = text(f"""
                SELECT id, symbol, tf, ts, o, h, l, c, v, vwap, taker_buy_vol, features,
                       created_at, updated_at
                FROM candles
                WHERE {where_clause}
                ORDER BY ts ASC
                LIMIT :limit
            """)
            
            result = await self.session.execute(query, params)
            
            candles = []
            for row in result:
                candle = {
                    'id': row[0],
                    'symbol': row[1],
                    'tf': row[2],
                    'ts': row[3],
                    'o': row[4],
                    'h': row[5],
                    'l': row[6],
                    'c': row[7],
                    'v': row[8],
                    'vwap': row[9],
                    'taker_buy_vol': row[10],
                    'features': row[11],
                    'created_at': row[12],
                    'updated_at': row[13]
                }
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            raise
    
    async def update_candle_features(
        self, 
        candle_id: int, 
        features: Dict[str, Any]
    ) -> bool:
        """Update candle features"""
        try:
            query = text("""
                UPDATE candles 
                SET features = :features,
                    updated_at = NOW()
                WHERE id = :candle_id
            """)
            
            result = await self.session.execute(query, {
                'candle_id': candle_id,
                'features': json.dumps(features)
            })
            
            await self.session.commit()
            return result.rowcount > 0
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating candle features: {e}")
            raise
    
    # ==================== RETRAIN QUEUE OPERATIONS ====================
    
    async def add_to_retrain_queue(
        self, 
        signal_id: int, 
        reason: str, 
        priority: int = 1
    ) -> Dict[str, Any]:
        """Add a signal to the retrain queue"""
        try:
            query = text("""
                INSERT INTO retrain_queue (
                    signal_id, reason, priority
                ) VALUES (
                    :signal_id, :reason, :priority
                ) RETURNING id, inserted_at
            """)
            
            result = await self.session.execute(query, {
                'signal_id': signal_id,
                'reason': reason,
                'priority': priority
            })
            
            row = result.fetchone()
            await self.session.commit()
            
            return {
                'id': row[0],
                'inserted_at': row[1],
                'status': 'queued'
            }
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error adding to retrain queue: {e}")
            raise
    
    async def get_retrain_queue(
        self, 
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get retrain queue items"""
        try:
            conditions = []
            params = {'limit': limit}
            
            if status:
                conditions.append("status = :status")
                params['status'] = status
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = text(f"""
                SELECT rq.id, rq.signal_id, rq.reason, rq.inserted_at,
                       rq.status, rq.priority, rq.started_at, rq.completed_at,
                       rq.error_message, rq.created_at, rq.updated_at,
                       s.symbol, s.tf, s.model_id
                FROM retrain_queue rq
                LEFT JOIN signals s ON rq.signal_id = s.id
                WHERE {where_clause}
                ORDER BY rq.priority DESC, rq.inserted_at ASC
                LIMIT :limit
            """)
            
            result = await self.session.execute(query, params)
            
            queue_items = []
            for row in result:
                item = {
                    'id': row[0],
                    'signal_id': row[1],
                    'reason': row[2],
                    'inserted_at': row[3],
                    'status': row[4],
                    'priority': row[5],
                    'started_at': row[6],
                    'completed_at': row[7],
                    'error_message': row[8],
                    'created_at': row[9],
                    'updated_at': row[10],
                    'symbol': row[11],
                    'timeframe': row[12],
                    'model_id': row[13]
                }
                queue_items.append(item)
            
            return queue_items
            
        except Exception as e:
            logger.error(f"Error getting retrain queue: {e}")
            raise
    
    async def update_retrain_status(
        self, 
        queue_id: int, 
        status: str, 
        error_message: Optional[str] = None
    ) -> bool:
        """Update retrain queue item status"""
        try:
            if status == 'processing':
                query = text("""
                    UPDATE retrain_queue 
                    SET status = :status, started_at = NOW(), updated_at = NOW()
                    WHERE id = :queue_id
                """)
                params = {'queue_id': queue_id, 'status': status}
            elif status == 'completed':
                query = text("""
                    UPDATE retrain_queue 
                    SET status = :status, completed_at = NOW(), updated_at = NOW()
                    WHERE id = :queue_id
                """)
                params = {'queue_id': queue_id, 'status': status}
            elif status == 'failed':
                query = text("""
                    UPDATE retrain_queue 
                    SET status = :status, error_message = :error_message, 
                        completed_at = NOW(), updated_at = NOW()
                    WHERE id = :queue_id
                """)
                params = {
                    'queue_id': queue_id, 
                    'status': status, 
                    'error_message': error_message
                }
            else:
                query = text("""
                    UPDATE retrain_queue 
                    SET status = :status, updated_at = NOW()
                    WHERE id = :queue_id
                """)
                params = {'queue_id': queue_id, 'status': status}
            
            result = await self.session.execute(query, params)
            await self.session.commit()
            
            return result.rowcount > 0
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating retrain status: {e}")
            raise
    
    # ==================== ANALYTICS QUERIES ====================
    
    async def get_signal_performance_summary(
        self, 
        model_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get signal performance summary"""
        try:
            conditions = ["ts >= NOW() - INTERVAL ':days days'"]
            params = {'days': days}
            
            if model_id:
                conditions.append("model_id = :model_id")
                params['model_id'] = model_id
            
            if symbol:
                conditions.append("symbol = :symbol")
                params['symbol'] = symbol
            
            where_clause = " AND ".join(conditions)
            
            query = text(f"""
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN outcome = 'win' THEN 1 END) as wins,
                    COUNT(CASE WHEN outcome = 'loss' THEN 1 END) as losses,
                    AVG(CASE WHEN outcome = 'win' THEN realized_rr END) as avg_win_rr,
                    AVG(CASE WHEN outcome = 'loss' THEN realized_rr END) as avg_loss_rr,
                    AVG(latency_ms) as avg_latency_ms
                FROM signals
                WHERE {where_clause}
                  AND outcome IS NOT NULL
            """)
            
            result = await self.session.execute(query, params)
            row = result.fetchone()
            
            if row and row[0] > 0:
                win_rate = (row[1] / row[0]) * 100
                profit_factor = abs(row[3] / row[4]) if row[4] and row[4] != 0 else 0
                
                return {
                    'total_signals': row[0],
                    'wins': row[1],
                    'losses': row[2],
                    'win_rate': round(win_rate, 2),
                    'avg_win_rr': round(row[3] or 0, 2),
                    'avg_loss_rr': round(row[4] or 0, 2),
                    'profit_factor': round(profit_factor, 2),
                    'avg_latency_ms': round(row[5] or 0, 2)
                }
            else:
                return {
                    'total_signals': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'avg_win_rr': 0.0,
                    'avg_loss_rr': 0.0,
                    'profit_factor': 0.0,
                    'avg_latency_ms': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting signal performance summary: {e}")
            raise
    
    async def get_feature_importance_analysis(
        self, 
        model_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze feature importance based on signal outcomes"""
        try:
            query = text("""
                SELECT features, outcome, realized_rr
                FROM signals
                WHERE model_id = :model_id
                  AND ts >= NOW() - INTERVAL ':days days'
                  AND features IS NOT NULL
                  AND outcome IS NOT NULL
                ORDER BY ts DESC
            """)
            
            result = await self.session.execute(query, {
                'model_id': model_id,
                'days': days
            })
            
            feature_stats = {}
            total_signals = 0
            
            for row in result:
                features = row[0] or {}
                outcome = row[1]
                rr = row[2] or 0
                
                total_signals += 1
                
                for feature_name, feature_value in features.items():
                    if feature_name not in feature_stats:
                        feature_stats[feature_name] = {
                            'total_occurrences': 0,
                            'win_occurrences': 0,
                            'loss_occurrences': 0,
                            'total_rr': 0.0,
                            'values': []
                        }
                    
                    feature_stats[feature_name]['total_occurrences'] += 1
                    feature_stats[feature_name]['total_rr'] += rr
                    feature_stats[feature_name]['values'].append(feature_value)
                    
                    if outcome == 'win':
                        feature_stats[feature_name]['win_occurrences'] += 1
                    elif outcome == 'loss':
                        feature_stats[feature_name]['loss_occurrences'] += 1
            
            # Calculate feature importance metrics
            feature_importance = {}
            for feature_name, stats in feature_stats.items():
                if stats['total_occurrences'] >= 10:  # Minimum sample size
                    win_rate = (stats['win_occurrences'] / stats['total_occurrences']) * 100
                    avg_rr = stats['total_rr'] / stats['total_occurrences']
                    
                    # Simple importance score (can be enhanced)
                    importance_score = (win_rate * avg_rr) / 100
                    
                    feature_importance[feature_name] = {
                        'occurrences': stats['total_occurrences'],
                        'win_rate': round(win_rate, 2),
                        'avg_rr': round(avg_rr, 2),
                        'importance_score': round(importance_score, 4)
                    }
            
            # Sort by importance score
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )
            
            return {
                'total_signals': total_signals,
                'feature_importance': dict(sorted_features),
                'analysis_period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise
