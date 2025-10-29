#!/usr/bin/env python3
"""
Hard Example Buffer Service for AlphaPulse
Phase 5C: Misclassification Capture Implementation

Implements:
1. Outcome computation after trade closure
2. Misclassification detection and categorization
3. Balanced hard example buffer (60% hard negatives, 40% near-positives)
4. Integration with existing retrain_queue and retraining pipeline
5. Automated workflow with Prefect integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time

# Database imports
from ..src.database.data_versioning_dao import DataVersioningDAO
from ..src.database.connection import get_enhanced_connection

# Prefect imports
try:
    from prefect import task, flow, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

logger = logging.getLogger(__name__)

class BufferType(Enum):
    """Hard example buffer types"""
    HARD_NEGATIVE = "hard_negative"  # 60% - misclassified or low-quality
    NEAR_POSITIVE = "near_positive"   # 40% - near decision boundary

class OutcomeStatus(Enum):
    """Trade outcome status"""
    WIN = "win"
    LOSS = "loss"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

@dataclass
class TradeOutcome:
    """Computed trade outcome metrics"""
    signal_id: int
    outcome: OutcomeStatus
    realized_rr: float
    max_drawdown: float
    confidence: float
    prediction_correct: bool
    buffer_type: Optional[BufferType] = None
    reason: Optional[str] = None

@dataclass
class BufferStats:
    """Hard example buffer statistics"""
    total_examples: int
    hard_negatives: int
    near_positives: int
    hard_negative_ratio: float
    near_positive_ratio: float
    last_updated: datetime
    buffer_size_mb: float

class HardExampleBufferService:
    """
    Service for managing hard examples and misclassification capture
    Integrates with existing TimescaleDB infrastructure
    """
    
    def __init__(self):
        self.db_connection = get_enhanced_connection()
        # DAO will be created with session when needed
        
        # Buffer configuration
        self.buffer_config = {
            'max_size': 10000,  # Maximum buffer size
            'target_hard_negative_ratio': 0.60,  # 60% hard negatives
            'target_near_positive_ratio': 0.40,  # 40% near positives
            'balance_tolerance': 0.05,  # ±5% tolerance
            'min_realized_rr_threshold': 0.5,  # Minimum R/R for quality
            'max_drawdown_threshold': 0.5,  # Maximum drawdown threshold
            'confidence_boundary_low': 0.4,  # Low confidence boundary
            'confidence_boundary_high': 0.6,  # High confidence boundary
        }
        
        # Performance tracking
        self.performance_metrics = {
            'outcome_computation_time': 0.0,
            'buffer_update_time': 0.0,
            'total_trades_processed': 0,
            'total_hard_examples_captured': 0,
            'last_metrics_reset': datetime.now()
        }
        
        logger.info("🚀 Hard Example Buffer Service initialized")
    
    async def compute_trade_outcomes(self, 
                                   symbols: List[str] = None,
                                   batch_size: int = 1000) -> List[TradeOutcome]:
        """
        Compute outcomes for closed trades
        Efficiently processes batches to minimize database overhead
        """
        start_time = time.time()
        logger.info(f"🔄 Computing trade outcomes for {len(symbols) if symbols else 'all'} symbols")
        
        try:
            # Get closed trades that need outcome computation
            closed_trades = await self._get_trades_needing_outcomes(symbols, batch_size)
            
            if not closed_trades:
                logger.info("✅ No trades need outcome computation")
                return []
            
            outcomes = []
            
            # Process trades in batches for efficiency
            for i in range(0, len(closed_trades), batch_size):
                batch = closed_trades[i:i + batch_size]
                batch_outcomes = await self._compute_batch_outcomes(batch)
                outcomes.extend(batch_outcomes)
                
                # Update signals table with computed outcomes
                await self._update_signals_with_outcomes(batch_outcomes)
                
                logger.info(f"📊 Processed batch {i//batch_size + 1}: {len(batch_outcomes)} trades")
            
            # Update performance metrics
            computation_time = time.time() - start_time
            self.performance_metrics['outcome_computation_time'] = computation_time
            self.performance_metrics['total_trades_processed'] += len(outcomes)
            
            logger.info(f"✅ Computed outcomes for {len(outcomes)} trades in {computation_time:.2f}s")
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Error computing trade outcomes: {e}")
            raise
    
    async def _get_trades_needing_outcomes(self, 
                                         symbols: List[str] = None,
                                         limit: int = 1000) -> List[Dict[str, Any]]:
        """Get trades that need outcome computation"""
        try:
            # Query signals table for trades without outcomes
            query = """
                SELECT s.id, s.symbol, s.tf, s.ts, s.label, s.pred, s.proba,
                       s.features, s.model_id, s.outcome, s.realized_rr
                FROM signals s
                WHERE s.outcome IS NULL 
                  AND s.ts < NOW() - INTERVAL '1 hour'  -- Ensure trade has time to close
            """
            
            params = {}
            if symbols:
                placeholders = ','.join([f"'{s}'" for s in symbols])
                query += f" AND s.symbol IN ({placeholders})"
            
            query += " ORDER BY s.ts DESC LIMIT :limit"
            params['limit'] = limit
            
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(query, params)
                trades = result.fetchall()
                
                return [dict(trade) for trade in trades]
                
        except Exception as e:
            logger.error(f"❌ Error getting trades needing outcomes: {e}")
            raise
    
    async def _compute_batch_outcomes(self, trades: List[Dict[str, Any]]) -> List[TradeOutcome]:
        """Compute outcomes for a batch of trades"""
        outcomes = []
        
        for trade in trades:
            try:
                # Get price data for R/R and drawdown calculation
                price_data = await self._get_trade_price_data(trade)
                
                if not price_data:
                    # Skip trades without sufficient price data
                    continue
                
                # Compute outcome metrics
                outcome = await self._compute_single_trade_outcome(trade, price_data)
                outcomes.append(outcome)
                
            except Exception as e:
                logger.warning(f"⚠️ Error computing outcome for trade {trade.get('id')}: {e}")
                continue
        
        return outcomes
    
    async def _get_trade_price_data(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get price data for trade outcome computation"""
        try:
            # Query candles table for price data around trade time
            query = """
                SELECT o, h, l, c, v, ts
                FROM candles 
                WHERE symbol = :symbol 
                  AND tf = :tf
                  AND ts BETWEEN :start_time AND :end_time
                ORDER BY ts ASC
            """
            
            # Get price data from 1 hour before to 1 hour after trade
            trade_time = trade['ts']
            start_time = trade_time - timedelta(hours=1)
            end_time = trade_time + timedelta(hours=1)
            
            params = {
                'symbol': trade['symbol'],
                'tf': trade['tf'],
                'start_time': start_time,
                'end_time': end_time
            }
            
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(query, params)
                price_data = result.fetchall()
                
                if len(price_data) < 2:  # Need at least 2 data points
                    return None
                
                return {
                    'prices': [dict(row) for row in price_data],
                    'trade_time': trade_time
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting price data for trade {trade.get('id')}: {e}")
            return None
    
    async def _compute_single_trade_outcome(self, 
                                          trade: Dict[str, Any], 
                                          price_data: Dict[str, Any]) -> TradeOutcome:
        """Compute outcome for a single trade"""
        try:
            # Extract trade information
            signal_id = trade['id']
            label = trade['label']
            pred = trade['pred']
            proba = trade['proba']
            
            # Calculate realized R/R (simplified - in production use actual entry/exit prices)
            realized_rr = await self._calculate_realized_rr(trade, price_data)
            
            # Calculate max drawdown
            max_drawdown = await self._calculate_max_drawdown(trade, price_data)
            
            # Determine outcome status
            outcome = self._determine_outcome_status(realized_rr, max_drawdown)
            
            # Check if prediction was correct
            prediction_correct = label == pred if label and pred else False
            
            # Determine buffer type (will be set later in categorization)
            buffer_type = None
            
            return TradeOutcome(
                signal_id=signal_id,
                outcome=outcome,
                realized_rr=realized_rr,
                max_drawdown=max_drawdown,
                confidence=proba or 0.0,
                prediction_correct=prediction_correct
            )
            
        except Exception as e:
            logger.error(f"❌ Error computing outcome for trade {trade.get('id')}: {e}")
            raise
    
    async def _calculate_realized_rr(self, trade: Dict[str, Any], price_data: Dict[str, Any]) -> float:
        """Calculate realized risk/reward ratio"""
        try:
            # Simplified R/R calculation
            # In production, this would use actual entry/exit prices from trade execution
            prices = price_data['prices']
            trade_time = price_data['trade_time']
            
            if len(prices) < 2:
                return 0.0
            
            # Find entry and exit prices (simplified logic)
            entry_price = prices[0]['c']  # Close price at start
            exit_price = prices[-1]['c']  # Close price at end
            
            # Calculate simple return
            if entry_price > 0:
                return (exit_price - entry_price) / entry_price
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating realized R/R: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self, trade: Dict[str, Any], price_data: Dict[str, Any]) -> float:
        """Calculate maximum drawdown during trade"""
        try:
            prices = price_data['prices']
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate drawdown as percentage from highest to lowest
            high_price = max(p['h'] for p in prices)
            low_price = min(p['l'] for p in prices)
            
            if high_price > 0:
                return (high_price - low_price) / high_price
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating max drawdown: {e}")
            return 0.0
    
    def _determine_outcome_status(self, realized_rr: float, max_drawdown: float) -> OutcomeStatus:
        """Determine trade outcome status"""
        if realized_rr > 0.1:  # 10% profit threshold
            return OutcomeStatus.WIN
        elif realized_rr < -0.1:  # 10% loss threshold
            return OutcomeStatus.LOSS
        elif max_drawdown > 0.5:  # High drawdown
            return OutcomeStatus.LOSS
        else:
            return OutcomeStatus.UNKNOWN
    
    async def _update_signals_with_outcomes(self, outcomes: List[TradeOutcome]):
        """Update signals table with computed outcomes"""
        try:
            if not outcomes:
                return
            
            # Batch update for efficiency
            update_query = """
                UPDATE signals 
                SET outcome = :outcome,
                    realized_rr = :realized_rr,
                    updated_at = NOW()
                WHERE id = :signal_id
            """
            
            async with self.db_connection.get_async_session() as session:
                for outcome in outcomes:
                    await session.execute(update_query, {
                        'outcome': outcome.outcome.value,
                        'realized_rr': outcome.realized_rr,
                        'signal_id': outcome.signal_id
                    })
                
                await session.commit()
                
            logger.info(f"✅ Updated {len(outcomes)} signals with outcomes")
            
        except Exception as e:
            logger.error(f"❌ Error updating signals with outcomes: {e}")
            raise
    
    async def categorize_hard_examples(self, outcomes: List[TradeOutcome]) -> List[TradeOutcome]:
        """
        Categorize trade outcomes into hard examples
        Maintains 60/40 balance between hard negatives and near positives
        """
        start_time = time.time()
        logger.info(f"🏷️ Categorizing {len(outcomes)} trade outcomes into hard examples")
        
        try:
            # Get current buffer statistics
            buffer_stats = await self.get_buffer_statistics()
            
            # Categorize each outcome
            for outcome in outcomes:
                outcome.buffer_type = self._categorize_single_outcome(outcome, buffer_stats)
                
                # Update buffer statistics
                if outcome.buffer_type == BufferType.HARD_NEGATIVE:
                    buffer_stats.hard_negatives += 1
                elif outcome.buffer_type == BufferType.NEAR_POSITIVE:
                    buffer_stats.near_positives += 1
                
                buffer_stats.total_examples += 1
            
            # Add hard examples to retrain queue
            hard_examples = [o for o in outcomes if o.buffer_type]
            if hard_examples:
                await self._add_hard_examples_to_queue(hard_examples)
            
            # Update performance metrics
            categorization_time = time.time() - start_time
            self.performance_metrics['buffer_update_time'] = categorization_time
            self.performance_metrics['total_hard_examples_captured'] += len(hard_examples)
            
            logger.info(f"✅ Categorized {len(hard_examples)} hard examples in {categorization_time:.2f}s")
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Error categorizing hard examples: {e}")
            raise
    
    def _categorize_single_outcome(self, 
                                  outcome: TradeOutcome, 
                                  buffer_stats: BufferStats) -> Optional[BufferType]:
        """Categorize a single trade outcome"""
        try:
            # Check if this is a hard negative (misclassified or low-quality)
            is_hard_negative = (
                not outcome.prediction_correct or  # Wrong prediction
                outcome.realized_rr < self.buffer_config['min_realized_rr_threshold'] or  # Low R/R
                outcome.max_drawdown > self.buffer_config['max_drawdown_threshold']  # High drawdown
            )
            
            # Check if this is a near positive (near decision boundary)
            is_near_positive = (
                outcome.prediction_correct and  # Correct prediction
                (self.buffer_config['confidence_boundary_low'] <= outcome.confidence <= 
                 self.buffer_config['confidence_boundary_high'])  # Low confidence
            )
            
            # Determine buffer type based on current balance
            if is_hard_negative:
                # Check if we can add more hard negatives
                current_hard_ratio = buffer_stats.hard_negative_ratio
                if current_hard_ratio < (self.buffer_config['target_hard_negative_ratio'] + 
                                       self.buffer_config['balance_tolerance']):
                    return BufferType.HARD_NEGATIVE
            
            elif is_near_positive:
                # Check if we can add more near positives
                current_near_ratio = buffer_stats.near_positive_ratio
                if current_near_ratio < (self.buffer_config['target_near_positive_ratio'] + 
                                       self.buffer_config['balance_tolerance']):
                    return BufferType.NEAR_POSITIVE
            
            # Not categorized as hard example
            return None
            
        except Exception as e:
            logger.error(f"❌ Error categorizing outcome: {e}")
            return None
    
    async def _add_hard_examples_to_queue(self, hard_examples: List[TradeOutcome]):
        """Add hard examples to retrain queue"""
        try:
            async with self.db_connection.get_async_session() as session:
                dao = DataVersioningDAO(session)
                
                for example in hard_examples:
                    # Determine reason for retraining
                    reason = self._determine_retrain_reason(example)
                    
                    # Add to retrain queue with appropriate priority
                    priority = 1 if example.buffer_type == BufferType.HARD_NEGATIVE else 2
                    
                    await dao.add_to_retrain_queue(
                        signal_id=example.signal_id,
                        reason=reason,
                        priority=priority
                    )
            
            logger.info(f"✅ Added {len(hard_examples)} hard examples to retrain queue")
            
        except Exception as e:
            logger.error(f"❌ Error adding hard examples to queue: {e}")
            raise
    
    def _determine_retrain_reason(self, example: TradeOutcome) -> str:
        """Determine reason for adding to retrain queue"""
        if not example.prediction_correct:
            return "misclassified"
        elif example.realized_rr < self.buffer_config['min_realized_rr_threshold']:
            return "low_rr"
        elif example.max_drawdown > self.buffer_config['max_drawdown_threshold']:
            return "high_drawdown"
        elif example.confidence <= self.buffer_config['confidence_boundary_high']:
            return "low_confidence"
        else:
            return "hard_example"
    
    async def get_buffer_statistics(self) -> BufferStats:
        """Get current hard example buffer statistics"""
        try:
            # Query retrain queue for current buffer state
            query = """
                SELECT 
                    COUNT(*) as total_examples,
                    COUNT(CASE WHEN reason IN ('misclassified', 'low_rr', 'high_drawdown') THEN 1 END) as hard_negatives,
                    COUNT(CASE WHEN reason IN ('low_confidence') THEN 1 END) as near_positives
                FROM retrain_queue 
                WHERE status = 'pending'
            """
            
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(query)
                row = result.fetchone()
                
                total = row[0] or 0
                hard_negatives = row[1] or 0
                near_positives = row[2] or 0
                
                # Calculate ratios
                hard_negative_ratio = hard_negatives / total if total > 0 else 0.0
                near_positive_ratio = near_positives / total if total > 0 else 0.0
                
                # Estimate buffer size in MB
                buffer_size_mb = total * 0.001  # Rough estimate: 1KB per example
                
                return BufferStats(
                    total_examples=total,
                    hard_negatives=hard_negatives,
                    near_positives=near_positives,
                    hard_negative_ratio=hard_negative_ratio,
                    near_positive_ratio=near_positive_ratio,
                    last_updated=datetime.now(),
                    buffer_size_mb=buffer_size_mb
                )
                
        except Exception as e:
            logger.error(f"❌ Error getting buffer statistics: {e}")
            # Return default stats
            return BufferStats(
                total_examples=0,
                hard_negatives=0,
                near_positives=0,
                hard_negative_ratio=0.0,
                near_positive_ratio=0.0,
                last_updated=datetime.now(),
                buffer_size_mb=0.0
            )
    
    async def maintain_buffer_balance(self):
        """Maintain buffer balance by adjusting categorization thresholds"""
        try:
            buffer_stats = await self.get_buffer_statistics()
            
            if buffer_stats.total_examples == 0:
                return
            
            # Check if balance is within tolerance
            hard_negative_balanced = (
                abs(buffer_stats.hard_negative_ratio - self.buffer_config['target_hard_negative_ratio']) 
                <= self.buffer_config['balance_tolerance']
            )
            
            near_positive_balanced = (
                abs(buffer_stats.near_positive_ratio - self.buffer_config['target_near_positive_ratio']) 
                <= self.buffer_config['balance_tolerance']
            )
            
            if hard_negative_balanced and near_positive_balanced:
                logger.info("✅ Buffer balance is within tolerance")
                return
            
            # Adjust thresholds to rebalance
            if buffer_stats.hard_negative_ratio > self.buffer_config['target_hard_negative_ratio']:
                # Too many hard negatives, make it harder to qualify
                self.buffer_config['min_realized_rr_threshold'] = min(
                    0.8, self.buffer_config['min_realized_rr_threshold'] + 0.1
                )
                logger.info(f"🔧 Adjusted R/R threshold to {self.buffer_config['min_realized_rr_threshold']}")
            
            elif buffer_stats.near_positive_ratio > self.buffer_config['target_near_positive_ratio']:
                # Too many near positives, make it harder to qualify
                self.buffer_config['confidence_boundary_high'] = max(
                    0.5, self.buffer_config['confidence_boundary_high'] - 0.05
                )
                logger.info(f"🔧 Adjusted confidence boundary to {self.buffer_config['confidence_boundary_high']}")
            
            logger.info("🔧 Buffer balance thresholds adjusted")
            
        except Exception as e:
            logger.error(f"❌ Error maintaining buffer balance: {e}")
    
    async def cleanup_old_examples(self, max_age_days: int = 30):
        """Clean up old examples from retrain queue"""
        try:
            # Remove processed examples older than max_age_days
            cleanup_query = """
                DELETE FROM retrain_queue 
                WHERE status IN ('completed', 'failed') 
                  AND updated_at < NOW() - INTERVAL ':max_age_days days'
            """
            
            async with self.db_connection.get_async_session() as session:
                result = await session.execute(cleanup_query, {'max_age_days': max_age_days})
                deleted_count = result.rowcount
                await session.commit()
            
            logger.info(f"🧹 Cleaned up {deleted_count} old examples from retrain queue")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old examples: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            **self.performance_metrics,
            'buffer_stats': await self.get_buffer_statistics(),
            'config': self.buffer_config
        }
    
    async def reset_performance_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'outcome_computation_time': 0.0,
            'buffer_update_time': 0.0,
            'total_trades_processed': 0,
            'total_hard_examples_captured': 0,
            'last_metrics_reset': datetime.now()
        }
        logger.info("🔄 Performance metrics reset")

# Prefect tasks for workflow integration
if PREFECT_AVAILABLE:
    @task(name="compute_trade_outcomes")
    async def compute_trade_outcomes_task(symbols: List[str] = None, batch_size: int = 1000):
        """Prefect task for computing trade outcomes"""
        service = HardExampleBufferService()
        return await service.compute_trade_outcomes(symbols, batch_size)
    
    @task(name="categorize_hard_examples")
    async def categorize_hard_examples_task(outcomes: List[TradeOutcome]):
        """Prefect task for categorizing hard examples"""
        service = HardExampleBufferService()
        return await service.categorize_hard_examples(outcomes)
    
    @task(name="maintain_buffer_balance")
    async def maintain_buffer_balance_task():
        """Prefect task for maintaining buffer balance"""
        service = HardExampleBufferService()
        await service.maintain_buffer_balance()
    
    @task(name="cleanup_old_examples")
    async def cleanup_old_examples_task(max_age_days: int = 30):
        """Prefect task for cleaning up old examples"""
        service = HardExampleBufferService()
        await service.cleanup_old_examples(max_age_days)
    
    @flow(name="hard_example_buffer_workflow")
    async def hard_example_buffer_workflow(symbols: List[str] = None):
        """Complete workflow for hard example buffer management"""
        logger = get_run_logger()
        
        try:
            logger.info("🚀 Starting hard example buffer workflow")
            
            # 1. Compute trade outcomes
            outcomes = await compute_trade_outcomes_task(symbols)
            
            if not outcomes:
                logger.info("✅ No outcomes to process")
                return
            
            # 2. Categorize hard examples
            categorized_outcomes = await categorize_hard_examples_task(outcomes)
            
            # 3. Maintain buffer balance
            await maintain_buffer_balance_task()
            
            # 4. Cleanup old examples (weekly)
            if datetime.now().weekday() == 0:  # Monday
                await cleanup_old_examples_task()
            
            logger.info(f"✅ Hard example buffer workflow completed: {len(categorized_outcomes)} outcomes processed")
            
        except Exception as e:
            logger.error(f"❌ Hard example buffer workflow failed: {e}")
            raise

# Global service instance
hard_example_buffer_service = HardExampleBufferService()

# Export for use in other modules
__all__ = [
    'HardExampleBufferService',
    'TradeOutcome',
    'BufferType',
    'OutcomeStatus',
    'BufferStats',
    'hard_example_buffer_service'
]
