#!/usr/bin/env python3
"""
Pattern Performance Tracker for Phase 4B ML Retraining
Tracks pattern outcomes and calculates performance metrics for ML model retraining
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import uuid
from decimal import Decimal

import pandas as pd
import numpy as np
from sqlalchemy import text, select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..src.database.connection import TimescaleDBConnection
from ..src.strategies.phase3_enhanced_pattern_detector import Phase3PatternResult

logger = logging.getLogger(__name__)

@dataclass
class PatternOutcome:
    """Represents the outcome of a pattern detection"""
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    detection_timestamp: datetime
    outcome_timestamp: Optional[datetime] = None
    outcome_type: Optional[str] = None  # success, failure, partial
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    holding_period_hours: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_profit: Optional[float] = None
    market_regime: Optional[str] = None
    confidence_at_detection: Optional[float] = None
    volume_confirmation: Optional[bool] = None
    noise_filter_passed: Optional[bool] = None
    validation_passed: Optional[bool] = None
    model_version: Optional[str] = None
    detection_method: Optional[str] = None
    performance_metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for patterns"""
    total_signals: int
    successful_signals: int
    failed_signals: int
    success_rate: float
    avg_profit_loss: float
    avg_profit_loss_pct: float
    total_profit_loss: float
    max_drawdown: float
    avg_holding_period: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    max_consecutive_wins: Optional[int] = None

@dataclass
class RegimePerformance:
    """Performance metrics by market regime"""
    regime_type: str
    total_signals: int
    successful_signals: int
    success_rate: float
    avg_profit_loss: float
    total_profit_loss: float
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None

class PatternPerformanceTracker:
    """
    Tracks pattern performance and calculates metrics for ML retraining
    """
    
    def __init__(self):
        self.db_connection = TimescaleDBConnection()
        self.is_running = False
        self.tracking_config = {
            'min_holding_period_hours': 0.1,  # Minimum 6 minutes
            'max_holding_period_hours': 168,  # Maximum 1 week
            'success_threshold_pct': 0.5,     # 0.5% profit for success
            'failure_threshold_pct': -0.5,    # -0.5% loss for failure
            'partial_threshold_pct': 0.1,     # 0.1% for partial success
            'tracking_window_days': 30,       # Track last 30 days
            'min_samples_for_metrics': 10     # Minimum samples for reliable metrics
        }
        
        # Performance tracking state
        self.pending_outcomes = {}
        self.performance_cache = {}
        self.last_metrics_update = None
        
        logger.info("📊 Pattern Performance Tracker initialized")
    
    async def start(self):
        """Start the performance tracker"""
        if self.is_running:
            logger.warning("Pattern Performance Tracker is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Pattern Performance Tracker started")
    
    async def stop(self):
        """Stop the performance tracker"""
        self.is_running = False
        logger.info("🛑 Pattern Performance Tracker stopped")
    
    async def record_pattern_detection(self, pattern_result: Phase3PatternResult) -> str:
        """
        Record a new pattern detection for performance tracking
        
        Args:
            pattern_result: The pattern detection result
            
        Returns:
            str: Tracking ID for the pattern
        """
        try:
            tracking_id = str(uuid.uuid4())
            
            # Create initial outcome record
            outcome = PatternOutcome(
                pattern_id=pattern_result.pattern_id,
                symbol=pattern_result.symbol,
                timeframe=pattern_result.timeframe,
                pattern_type=pattern_result.pattern_type,
                detection_timestamp=pattern_result.timestamp,
                confidence_at_detection=pattern_result.final_confidence,
                volume_confirmation=pattern_result.volume_confirmation_score > 0.7,
                noise_filter_passed=pattern_result.passed_filters,
                validation_passed=pattern_result.validation_passed,
                model_version=getattr(pattern_result, 'model_version', None),
                detection_method='hybrid',  # Phase 3 uses hybrid detection
                market_regime=getattr(pattern_result, 'market_regime', None),
                performance_metadata={
                    'quality_score': pattern_result.quality_score,
                    'quality_level': pattern_result.quality_level.value,
                    'priority_rank': pattern_result.priority_rank,
                    'signal_strength': pattern_result.signal_strength,
                    'calibrated_confidence': getattr(pattern_result, 'calibrated_confidence', None),
                    'multi_timeframe_alignment': getattr(pattern_result, 'multi_timeframe_alignment', None)
                }
            )
            
            # Store in pending outcomes
            self.pending_outcomes[tracking_id] = outcome
            
            # Store in database
            await self._store_pattern_detection(outcome)
            
            logger.debug(f"📝 Recorded pattern detection: {tracking_id} for {pattern_result.symbol} {pattern_result.pattern_type}")
            return tracking_id
            
        except Exception as e:
            logger.error(f"❌ Error recording pattern detection: {e}")
            raise
    
    async def record_pattern_outcome(self, tracking_id: str, outcome_data: Dict[str, Any]) -> bool:
        """
        Record the outcome of a pattern detection
        
        Args:
            tracking_id: The tracking ID from record_pattern_detection
            outcome_data: Dictionary containing outcome information
            
        Returns:
            bool: True if successfully recorded
        """
        try:
            if tracking_id not in self.pending_outcomes:
                logger.warning(f"⚠️ Tracking ID not found: {tracking_id}")
                return False
            
            outcome = self.pending_outcomes[tracking_id]
            
            # Update outcome with results
            outcome.outcome_timestamp = outcome_data.get('outcome_timestamp', datetime.utcnow())
            outcome.profit_loss = outcome_data.get('profit_loss')
            outcome.profit_loss_pct = outcome_data.get('profit_loss_pct')
            outcome.holding_period_hours = outcome_data.get('holding_period_hours')
            outcome.max_drawdown = outcome_data.get('max_drawdown')
            outcome.max_profit = outcome_data.get('max_profit')
            
            # Determine outcome type based on profit/loss
            if outcome.profit_loss_pct is not None:
                if outcome.profit_loss_pct >= self.tracking_config['success_threshold_pct']:
                    outcome.outcome_type = 'success'
                elif outcome.profit_loss_pct <= self.tracking_config['failure_threshold_pct']:
                    outcome.outcome_type = 'failure'
                elif outcome.profit_loss_pct >= self.tracking_config['partial_threshold_pct']:
                    outcome.outcome_type = 'partial'
                else:
                    outcome.outcome_type = 'failure'
            
            # Update database
            await self._update_pattern_outcome(outcome)
            
            # Remove from pending
            del self.pending_outcomes[tracking_id]
            
            # Clear performance cache to force recalculation
            self.performance_cache.clear()
            
            logger.debug(f"✅ Recorded pattern outcome: {tracking_id} - {outcome.outcome_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording pattern outcome: {e}")
            return False
    
    async def get_pattern_performance_metrics(
        self, 
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_version: Optional[str] = None,
        days: int = 30
    ) -> PerformanceMetrics:
        """
        Get performance metrics for patterns
        
        Args:
            symbol: Filter by symbol
            pattern_type: Filter by pattern type
            timeframe: Filter by timeframe
            model_version: Filter by model version
            days: Number of days to analyze
            
        Returns:
            PerformanceMetrics: Calculated performance metrics
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{pattern_type}_{timeframe}_{model_version}_{days}"
            if cache_key in self.performance_cache:
                return self.performance_cache[cache_key]
            
            async with self.db_connection.get_session() as session:
                # Build query
                query = select([
                    func.count().label('total_signals'),
                    func.count().filter(text("outcome_type = 'success'")).label('successful_signals'),
                    func.count().filter(text("outcome_type = 'failure'")).label('failed_signals'),
                    func.avg(text('profit_loss')).label('avg_profit_loss'),
                    func.avg(text('profit_loss_pct')).label('avg_profit_loss_pct'),
                    func.sum(text('profit_loss')).label('total_profit_loss'),
                    func.min(text('profit_loss')).label('max_drawdown'),
                    func.avg(text('holding_period_hours')).label('avg_holding_period')
                ]).select_from(text('pattern_performance_history')).where(
                    and_(
                        text('detection_timestamp >= NOW() - INTERVAL :days DAYS'),
                        text('outcome_type IS NOT NULL')
                    )
                )
                
                # Add filters
                params = {'days': days}
                if symbol:
                    query = query.where(text('symbol = :symbol'))
                    params['symbol'] = symbol
                if pattern_type:
                    query = query.where(text('pattern_type = :pattern_type'))
                    params['pattern_type'] = pattern_type
                if timeframe:
                    query = query.where(text('timeframe = :timeframe'))
                    params['timeframe'] = timeframe
                if model_version:
                    query = query.where(text('model_version = :model_version'))
                    params['model_version'] = model_version
                
                result = await session.execute(query, params)
                row = result.fetchone()
                
                if not row or row.total_signals == 0:
                    return PerformanceMetrics(
                        total_signals=0, successful_signals=0, failed_signals=0,
                        success_rate=0.0, avg_profit_loss=0.0, avg_profit_loss_pct=0.0,
                        total_profit_loss=0.0, max_drawdown=0.0, avg_holding_period=0.0
                    )
                
                # Calculate basic metrics
                success_rate = row.successful_signals / row.total_signals if row.total_signals > 0 else 0.0
                
                # Get detailed data for advanced metrics
                detailed_query = select([
                    text('profit_loss'),
                    text('profit_loss_pct'),
                    text('outcome_type')
                ]).select_from(text('pattern_performance_history')).where(
                    and_(
                        text('detection_timestamp >= NOW() - INTERVAL :days DAYS'),
                        text('outcome_type IS NOT NULL'),
                        text('profit_loss IS NOT NULL')
                    )
                )
                
                # Add same filters
                for key, value in params.items():
                    if key != 'days':
                        detailed_query = detailed_query.where(text(f'{key} = :{key}'))
                
                detailed_result = await session.execute(detailed_query, params)
                detailed_data = detailed_result.fetchall()
                
                # Calculate advanced metrics
                profits = [row.profit_loss for row in detailed_data if row.profit_loss > 0]
                losses = [row.profit_loss for row in detailed_data if row.profit_loss < 0]
                
                avg_win = np.mean(profits) if profits else 0.0
                avg_loss = np.mean(losses) if losses else 0.0
                profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
                
                # Calculate Sharpe ratio (simplified)
                returns = [row.profit_loss_pct for row in detailed_data if row.profit_loss_pct is not None]
                sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
                
                # Calculate consecutive wins/losses
                consecutive_wins, consecutive_losses = self._calculate_consecutive_outcomes(detailed_data)
                
                metrics = PerformanceMetrics(
                    total_signals=row.total_signals,
                    successful_signals=row.successful_signals,
                    failed_signals=row.failed_signals,
                    success_rate=success_rate,
                    avg_profit_loss=float(row.avg_profit_loss) if row.avg_profit_loss else 0.0,
                    avg_profit_loss_pct=float(row.avg_profit_loss_pct) if row.avg_profit_loss_pct else 0.0,
                    total_profit_loss=float(row.total_profit_loss) if row.total_profit_loss else 0.0,
                    max_drawdown=float(row.max_drawdown) if row.max_drawdown else 0.0,
                    avg_holding_period=float(row.avg_holding_period) if row.avg_holding_period else 0.0,
                    sharpe_ratio=sharpe_ratio,
                    profit_factor=profit_factor,
                    win_rate=len(profits) / len(detailed_data) if detailed_data else 0.0,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    max_consecutive_losses=consecutive_losses,
                    max_consecutive_wins=consecutive_wins
                )
                
                # Cache the result
                self.performance_cache[cache_key] = metrics
                
                return metrics
                
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            raise
    
    async def get_regime_performance(
        self, 
        model_version: Optional[str] = None,
        days: int = 30
    ) -> List[RegimePerformance]:
        """
        Get performance metrics by market regime
        
        Args:
            model_version: Filter by model version
            days: Number of days to analyze
            
        Returns:
            List[RegimePerformance]: Performance by regime
        """
        try:
            async with self.db_connection.get_session() as session:
                query = text("""
                    SELECT 
                        market_regime,
                        COUNT(*) as total_signals,
                        COUNT(*) FILTER (WHERE outcome_type = 'success') as successful_signals,
                        ROUND(COUNT(*) FILTER (WHERE outcome_type = 'success')::NUMERIC / COUNT(*), 4) as success_rate,
                        ROUND(AVG(profit_loss), 4) as avg_profit_loss,
                        ROUND(SUM(profit_loss), 4) as total_profit_loss
                    FROM pattern_performance_history
                    WHERE detection_timestamp >= NOW() - INTERVAL :days DAYS
                        AND outcome_type IS NOT NULL
                        AND market_regime IS NOT NULL
                """)
                
                params = {'days': days}
                if model_version:
                    query = text(str(query) + " AND model_version = :model_version")
                    params['model_version'] = model_version
                
                query = text(str(query) + " GROUP BY market_regime ORDER BY total_signals DESC")
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                regime_performances = []
                for row in rows:
                    regime_perf = RegimePerformance(
                        regime_type=row.market_regime,
                        total_signals=row.total_signals,
                        successful_signals=row.successful_signals,
                        success_rate=float(row.success_rate),
                        avg_profit_loss=float(row.avg_profit_loss) if row.avg_profit_loss else 0.0,
                        total_profit_loss=float(row.total_profit_loss) if row.total_profit_loss else 0.0
                    )
                    regime_performances.append(regime_perf)
                
                return regime_performances
                
        except Exception as e:
            logger.error(f"❌ Error getting regime performance: {e}")
            raise
    
    async def get_retraining_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for model retraining based on performance
        
        Returns:
            Dict[str, Any]: Retraining recommendations
        """
        try:
            recommendations = {
                'retraining_needed': False,
                'reasons': [],
                'priority': 'low',
                'model_types': [],
                'performance_issues': []
            }
            
            # Check overall performance
            overall_metrics = await self.get_pattern_performance_metrics(days=7)
            
            if overall_metrics.total_signals < self.tracking_config['min_samples_for_metrics']:
                recommendations['reasons'].append(f"Insufficient data: {overall_metrics.total_signals} signals in last 7 days")
                return recommendations
            
            # Check success rate
            if overall_metrics.success_rate < 0.4:  # Less than 40% success rate
                recommendations['retraining_needed'] = True
                recommendations['reasons'].append(f"Low success rate: {overall_metrics.success_rate:.2%}")
                recommendations['priority'] = 'high'
                recommendations['model_types'].append('pattern_detector')
            
            # Check profit factor
            if overall_metrics.profit_factor and overall_metrics.profit_factor < 1.2:
                recommendations['retraining_needed'] = True
                recommendations['reasons'].append(f"Low profit factor: {overall_metrics.profit_factor:.2f}")
                recommendations['priority'] = 'high'
                recommendations['model_types'].append('quality_scorer')
            
            # Check regime performance
            regime_performances = await self.get_regime_performance(days=7)
            for regime in regime_performances:
                if regime.total_signals >= 5 and regime.success_rate < 0.3:
                    recommendations['retraining_needed'] = True
                    recommendations['reasons'].append(f"Poor {regime.regime_type} regime performance: {regime.success_rate:.2%}")
                    recommendations['performance_issues'].append({
                        'regime': regime.regime_type,
                        'success_rate': regime.success_rate,
                        'total_signals': regime.total_signals
                    })
            
            # Check for consecutive losses
            if overall_metrics.max_consecutive_losses and overall_metrics.max_consecutive_losses > 5:
                recommendations['retraining_needed'] = True
                recommendations['reasons'].append(f"High consecutive losses: {overall_metrics.max_consecutive_losses}")
                recommendations['priority'] = 'medium'
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error getting retraining recommendations: {e}")
            raise
    
    async def _store_pattern_detection(self, outcome: PatternOutcome) -> None:
        """Store pattern detection in database"""
        try:
            async with self.db_connection.get_session() as session:
                query = text("""
                    INSERT INTO pattern_performance_history (
                        pattern_id, symbol, timeframe, pattern_type, detection_timestamp,
                        confidence_at_detection, volume_confirmation, noise_filter_passed,
                        validation_passed, model_version, detection_method, market_regime,
                        performance_metadata
                    ) VALUES (
                        :pattern_id, :symbol, :timeframe, :pattern_type, :detection_timestamp,
                        :confidence_at_detection, :volume_confirmation, :noise_filter_passed,
                        :validation_passed, :model_version, :detection_method, :market_regime,
                        :performance_metadata
                    )
                """)
                
                await session.execute(query, {
                    'pattern_id': outcome.pattern_id,
                    'symbol': outcome.symbol,
                    'timeframe': outcome.timeframe,
                    'pattern_type': outcome.pattern_type,
                    'detection_timestamp': outcome.detection_timestamp,
                    'confidence_at_detection': outcome.confidence_at_detection,
                    'volume_confirmation': outcome.volume_confirmation,
                    'noise_filter_passed': outcome.noise_filter_passed,
                    'validation_passed': outcome.validation_passed,
                    'model_version': outcome.model_version,
                    'detection_method': outcome.detection_method,
                    'market_regime': outcome.market_regime,
                    'performance_metadata': json.dumps(outcome.performance_metadata) if outcome.performance_metadata else None
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error storing pattern detection: {e}")
            raise
    
    async def _update_pattern_outcome(self, outcome: PatternOutcome) -> None:
        """Update pattern outcome in database"""
        try:
            async with self.db_connection.get_session() as session:
                query = text("""
                    UPDATE pattern_performance_history
                    SET outcome_timestamp = :outcome_timestamp,
                        outcome_type = :outcome_type,
                        profit_loss = :profit_loss,
                        profit_loss_pct = :profit_loss_pct,
                        holding_period_hours = :holding_period_hours,
                        max_drawdown = :max_drawdown,
                        max_profit = :max_profit
                    WHERE pattern_id = :pattern_id
                """)
                
                await session.execute(query, {
                    'outcome_timestamp': outcome.outcome_timestamp,
                    'outcome_type': outcome.outcome_type,
                    'profit_loss': outcome.profit_loss,
                    'profit_loss_pct': outcome.profit_loss_pct,
                    'holding_period_hours': outcome.holding_period_hours,
                    'max_drawdown': outcome.max_drawdown,
                    'max_profit': outcome.max_profit,
                    'pattern_id': outcome.pattern_id
                })
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error updating pattern outcome: {e}")
            raise
    
    def _calculate_consecutive_outcomes(self, data: List) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for row in data:
            if row.outcome_type == 'success':
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif row.outcome_type == 'failure':
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return max_consecutive_wins, max_consecutive_losses

# Global instance
pattern_performance_tracker = PatternPerformanceTracker()
