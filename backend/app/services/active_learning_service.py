#!/usr/bin/env python3
"""
Active Learning Service for AlphaPulse
Phase 3 - Priority 7: Active Learning Loop Implementation

Captures low-confidence predictions (0.45-0.55), provides manual labeling interface,
and feeds labeled samples into retrain queue for model improvement.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Database imports
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Import existing services
try:
    from ..database.connection_simple import SimpleTimescaleDBConnection
    from ..database.data_versioning_dao import DataVersioningDAO
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database components not available")

logger = logging.getLogger(__name__)


class LabelStatus(Enum):
    """Status of labeling process"""
    PENDING = "pending"
    LABELED = "labeled"
    PROCESSED = "processed"
    SKIPPED = "skipped"


class PredictionLabel(Enum):
    """Trading signal labels"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ActiveLearningItem:
    """Active learning queue item"""
    id: int
    signal_id: Optional[int]
    symbol: str
    timeframe: str
    prediction_confidence: float
    predicted_label: Optional[str]
    predicted_probability: Optional[float]
    features: Dict[str, Any]
    market_data: Dict[str, Any]
    model_id: Optional[str]
    timestamp: datetime
    manual_label: Optional[str] = None
    labeled_by: Optional[str] = None
    labeled_at: Optional[datetime] = None
    labeling_notes: Optional[str] = None
    status: LabelStatus = LabelStatus.PENDING
    priority: int = 1
    retrain_queue_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ActiveLearningStats:
    """Statistics for active learning queue"""
    total_items: int
    pending_items: int
    labeled_items: int
    processed_items: int
    skipped_items: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    label_distribution: Dict[str, int]
    model_distribution: Dict[str, int]


class ActiveLearningService:
    """
    Active Learning Service for capturing and labeling low-confidence predictions
    """
    
    def __init__(self, 
                 confidence_low: float = 0.45,
                 confidence_high: float = 0.55,
                 max_queue_size: int = 1000,
                 auto_cleanup_days: int = 30):
        """
        Initialize the Active Learning Service
        
        Args:
            confidence_low: Lower bound for low-confidence predictions
            confidence_high: Upper bound for low-confidence predictions
            max_queue_size: Maximum number of items in queue
            auto_cleanup_days: Days after which to cleanup old items
        """
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.max_queue_size = max_queue_size
        self.auto_cleanup_days = auto_cleanup_days
        
        # Database connection
        self.db_connection = SimpleTimescaleDBConnection() if DATABASE_AVAILABLE else None
        
        # Service state
        self.is_running = False
        self.stats = {
            'items_captured': 0,
            'items_labeled': 0,
            'items_processed': 0,
            'items_skipped': 0,
            'last_capture': None,
            'last_cleanup': None
        }
        
        logger.info(f"ActiveLearningService initialized with confidence range: {confidence_low}-{confidence_high}")
    
    async def start(self):
        """Start the active learning service"""
        if self.is_running:
            logger.warning("Active learning service is already running")
            return
        
        logger.info("🚀 Starting Active Learning Service...")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_items())
        asyncio.create_task(self._monitor_queue_size())
        
        logger.info("✅ Active Learning Service started successfully")
    
    async def stop(self):
        """Stop the active learning service"""
        if not self.is_running:
            logger.warning("Active learning service is not running")
            return
        
        logger.info("🛑 Stopping Active Learning Service...")
        self.is_running = False
        logger.info("✅ Active Learning Service stopped successfully")
    
    async def capture_low_confidence_prediction(self,
                                              signal_id: Optional[int],
                                              symbol: str,
                                              timeframe: str,
                                              prediction_confidence: float,
                                              predicted_label: Optional[str],
                                              predicted_probability: Optional[float],
                                              features: Dict[str, Any],
                                              market_data: Dict[str, Any],
                                              model_id: Optional[str],
                                              timestamp: Optional[datetime] = None) -> Optional[int]:
        """
        Capture a low-confidence prediction for manual labeling
        
        Args:
            signal_id: ID of the signal in signals table
            symbol: Trading symbol
            timeframe: Timeframe of the prediction
            prediction_confidence: Confidence score (0.0-1.0)
            predicted_label: Predicted label (BUY/SELL/HOLD)
            predicted_probability: Predicted probability
            features: Feature vector used for prediction
            market_data: Market data at prediction time
            model_id: ID of the model that made the prediction
            timestamp: Timestamp of the prediction
            
        Returns:
            Queue item ID if captured, None otherwise
        """
        try:
            # Check if confidence is in the low-confidence range
            if not (self.confidence_low <= prediction_confidence <= self.confidence_high):
                return None
            
            # Use current time if timestamp not provided
            if timestamp is None:
                timestamp = datetime.now()
            
            # Calculate priority based on confidence
            if 0.48 <= prediction_confidence <= 0.52:
                priority = 3  # Highest priority (closest to 0.5)
            elif 0.46 <= prediction_confidence <= 0.54:
                priority = 2  # Medium priority
            else:
                priority = 1  # Lower priority
            
            if not self.db_connection:
                logger.error("❌ Database connection not available")
                return None
                
            session_factory = await self.db_connection.get_async_session()
            async with session_factory as session:
                # Use the database function to add the prediction
                result = await session.execute(text("""
                    SELECT add_low_confidence_prediction(
                        :signal_id, :symbol, :timeframe, :prediction_confidence,
                        :predicted_label, :predicted_probability, :features,
                        :market_data, :model_id, :timestamp
                    )
                """), {
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'prediction_confidence': prediction_confidence,
                    'predicted_label': predicted_label,
                    'predicted_probability': predicted_probability,
                    'features': json.dumps(features),
                    'market_data': json.dumps(market_data),
                    'model_id': model_id,
                    'timestamp': timestamp
                })
                
                queue_id = result.scalar()
                
                # Explicitly commit the transaction
                await session.commit()
                
                if queue_id:
                    self.stats['items_captured'] += 1
                    self.stats['last_capture'] = datetime.now()
                    logger.info(f"✅ Captured low-confidence prediction (ID: {queue_id}, Confidence: {prediction_confidence:.3f})")
                    return queue_id
                else:
                    logger.debug(f"⏭️ Skipped prediction (Confidence: {prediction_confidence:.3f} outside range)")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error capturing low-confidence prediction: {e}")
            return None
    
    async def get_pending_items(self, 
                               limit: int = 50,
                               symbol: Optional[str] = None,
                               model_id: Optional[str] = None) -> List[ActiveLearningItem]:
        """
        Get pending items for manual labeling
        
        Args:
            limit: Maximum number of items to return
            symbol: Filter by symbol
            model_id: Filter by model ID
            
        Returns:
            List of pending active learning items
        """
        try:
            if not self.db_connection:
                logger.error("❌ Database connection not available")
                return []
                
            session_factory = await self.db_connection.get_async_session()
            async with session_factory as session:
                query = """
                    SELECT * FROM active_learning_pending
                    WHERE 1=1
                """
                params = {}
                
                if symbol:
                    query += " AND symbol = :symbol"
                    params['symbol'] = symbol
                
                if model_id:
                    query += " AND model_id = :model_id"
                    params['model_id'] = model_id
                
                query += " LIMIT :limit"
                params['limit'] = limit
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                items = []
                for row in rows:
                    # Handle features JSON parsing safely
                    features = {}
                    if row.features:
                        try:
                            if isinstance(row.features, str):
                                features = json.loads(row.features)
                            elif isinstance(row.features, dict):
                                features = row.features
                        except (json.JSONDecodeError, TypeError):
                            features = {}
                    
                    item = ActiveLearningItem(
                        id=row.id,
                        signal_id=None,  # Not in view
                        symbol=row.symbol,
                        timeframe=row.timeframe,
                        prediction_confidence=row.prediction_confidence,
                        predicted_label=row.predicted_label,
                        predicted_probability=row.predicted_probability,
                        features=features,
                        market_data={},  # Not in view
                        model_id=row.model_id,
                        timestamp=row.timestamp,
                        created_at=row.created_at,
                        priority=row.priority
                    )
                    items.append(item)
                
                logger.info(f"📋 Retrieved {len(items)} pending items for labeling")
                return items
                
        except Exception as e:
            logger.error(f"❌ Error getting pending items: {e}")
            return []
    
    async def label_item(self,
                        queue_id: int,
                        manual_label: str,
                        labeled_by: str,
                        labeling_notes: Optional[str] = None) -> bool:
        """
        Manually label an active learning item
        
        Args:
            queue_id: ID of the queue item to label
            manual_label: Manual label (BUY/SELL/HOLD)
            labeled_by: Name/ID of the person doing the labeling
            labeling_notes: Optional notes about the labeling
            
        Returns:
            True if labeling successful, False otherwise
        """
        try:
            # Validate label
            if manual_label not in [label.value for label in PredictionLabel]:
                logger.error(f"❌ Invalid label: {manual_label}")
                return False
            
            if not self.db_connection:
                logger.error("❌ Database connection not available")
                return False
                
            session_factory = await self.db_connection.get_async_session()
            async with session_factory as session:
                # Use the database function to process the labeled item
                result = await session.execute(text("""
                    SELECT process_labeled_item(:queue_id, :manual_label, :labeled_by, :labeling_notes)
                """), {
                    'queue_id': queue_id,
                    'manual_label': manual_label,
                    'labeled_by': labeled_by,
                    'labeling_notes': labeling_notes or ''  # Ensure not None
                })
                
                retrain_id = result.scalar()
                
                if retrain_id is not None:  # Function returns 0 for success without retrain queue, or retrain_id for success with retrain queue
                    self.stats['items_labeled'] += 1
                    if retrain_id > 0:
                        logger.info(f"✅ Labeled item {queue_id} as {manual_label} (Retrain ID: {retrain_id})")
                    else:
                        logger.info(f"✅ Labeled item {queue_id} as {manual_label} (no retrain queue entry)")
                    return True
                else:
                    logger.error(f"❌ Failed to label item {queue_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error labeling item {queue_id}: {e}")
            return False
    
    async def skip_item(self, queue_id: int, reason: Optional[str] = None) -> bool:
        """
        Skip an active learning item (mark as skipped)
        
        Args:
            queue_id: ID of the queue item to skip
            reason: Optional reason for skipping
            
        Returns:
            True if skip successful, False otherwise
        """
        try:
            if not self.db_connection:
                logger.error("❌ Database connection not available")
                return False
                
            session_factory = await self.db_connection.get_async_session()
            async with session_factory as session:
                await session.execute(text("""
                    UPDATE active_learning_queue 
                    SET status = 'skipped', updated_at = NOW()
                    WHERE id = :queue_id
                """), {'queue_id': queue_id})
                
                await session.commit()
                
                self.stats['items_skipped'] += 1
                logger.info(f"⏭️ Skipped item {queue_id} (Reason: {reason or 'No reason provided'})")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error skipping item {queue_id}: {e}")
            return False
    
    async def get_statistics(self) -> ActiveLearningStats:
        """
        Get statistics about the active learning queue
        
        Returns:
            ActiveLearningStats object with queue statistics
        """
        try:
            if not self.db_connection:
                logger.error("❌ Database connection not available")
                return ActiveLearningStats(
                    total_items=0, pending_items=0, labeled_items=0, processed_items=0, skipped_items=0,
                    avg_confidence=0.0, min_confidence=0.0, max_confidence=0.0,
                    label_distribution={}, model_distribution={}
                )
                
            session_factory = await self.db_connection.get_async_session()
            async with session_factory as session:
                # Get overall statistics
                result = await session.execute(text("""
                    SELECT * FROM active_learning_stats
                """))
                stats_rows = result.fetchall()
                
                # Get label distribution
                result = await session.execute(text("""
                    SELECT manual_label, COUNT(*) as count
                    FROM active_learning_queue 
                    WHERE manual_label IS NOT NULL
                    GROUP BY manual_label
                """))
                label_dist = {row.manual_label: row.count for row in result.fetchall()}
                
                # Get model distribution
                result = await session.execute(text("""
                    SELECT model_id, COUNT(*) as count
                    FROM active_learning_queue 
                    WHERE model_id IS NOT NULL
                    GROUP BY model_id
                """))
                model_dist = {row.model_id: row.count for row in result.fetchall()}
                
                # Build statistics object
                stats = ActiveLearningStats(
                    total_items=sum(row.count for row in stats_rows),
                    pending_items=next((row.count for row in stats_rows if row.status == 'pending'), 0),
                    labeled_items=next((row.count for row in stats_rows if row.status == 'labeled'), 0),
                    processed_items=next((row.count for row in stats_rows if row.status == 'processed'), 0),
                    skipped_items=next((row.count for row in stats_rows if row.status == 'skipped'), 0),
                    avg_confidence=next((row.avg_confidence for row in stats_rows), 0.0),
                    min_confidence=next((row.min_confidence for row in stats_rows), 0.0),
                    max_confidence=next((row.max_confidence for row in stats_rows), 0.0),
                    label_distribution=label_dist,
                    model_distribution=model_dist
                )
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return ActiveLearningStats(
                total_items=0, pending_items=0, labeled_items=0, processed_items=0, skipped_items=0,
                avg_confidence=0.0, min_confidence=0.0, max_confidence=0.0,
                label_distribution={}, model_distribution={}
            )
    
    async def _cleanup_old_items(self):
        """Background task to cleanup old items"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_date = datetime.now() - timedelta(days=self.auto_cleanup_days)
                
                if not self.db_connection:
                    logger.error("❌ Database connection not available")
                    continue
                    
                session_factory = await self.db_connection.get_async_session()
                async with session_factory as session:
                    # Clean up old processed/skipped items
                    result = await session.execute(text("""
                        DELETE FROM active_learning_queue 
                        WHERE status IN ('processed', 'skipped')
                        AND updated_at < :cutoff_date
                    """), {'cutoff_date': cutoff_date})
                    
                    deleted_count = result.rowcount
                    if deleted_count > 0:
                        logger.info(f"🗑️ Cleaned up {deleted_count} old items")
                        self.stats['last_cleanup'] = datetime.now()
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup task: {e}")
    
    async def _monitor_queue_size(self):
        """Background task to monitor queue size"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                stats = await self.get_statistics()
                
                if stats.total_items > self.max_queue_size:
                    logger.warning(f"⚠️ Active learning queue size ({stats.total_items}) exceeds limit ({self.max_queue_size})")
                    
                    # Skip oldest low-priority items
                    async with get_async_session() as session:
                        result = await session.execute(text("""
                            UPDATE active_learning_queue 
                            SET status = 'skipped', updated_at = NOW()
                            WHERE id IN (
                                SELECT id FROM active_learning_queue 
                                WHERE status = 'pending' AND priority = 1
                                ORDER BY created_at ASC
                                LIMIT :skip_count
                            )
                        """), {'skip_count': stats.total_items - self.max_queue_size})
                        
                        skipped_count = result.rowcount
                        if skipped_count > 0:
                            logger.info(f"⏭️ Skipped {skipped_count} low-priority items to maintain queue size")
                
            except Exception as e:
                logger.error(f"❌ Error in queue monitoring task: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'service_running': self.is_running,
            'confidence_range': f"{self.confidence_low}-{self.confidence_high}",
            'max_queue_size': self.max_queue_size,
            'auto_cleanup_days': self.auto_cleanup_days,
            **self.stats
        }
